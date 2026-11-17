"""End-to-end experiment runs against scripted fake agents (no network), and the summary table."""
import json
import subprocess
from pathlib import Path

import pytest

from dreamwake import agent, designs, llm, run, tasks

TASK = "settings_gen"
SESSIONS = {s["prompt"]: s["id"] for s in tasks.load_task(TASK)["sessions"]}
REFERENCE = tasks.load_task(TASK)["reference"]
NOTE = "Settings live in settings.tmpl.json; durations are in _MS (replaces seconds)."


def writes(name):
    """read_file then write_file for every file of a reference state (existing files must be read first)."""
    root = REFERENCE / name
    calls = []
    for p in sorted(root.rglob("*")):
        if p.is_file():
            path = p.relative_to(root).as_posix()
            calls += [("read_file", {"path": path}), ("write_file", {"path": path, "content": p.read_text(encoding="utf-8")})]
    return calls


def perfect(name):
    return writes(name) + [("run_tests", {}), ("finish", {"summary": "done"})]


def scripted_agent(scripts, note=NOTE):
    """A fake chat: picks the script for a session from its prompt; sleep calls (no tools) get `note`."""
    state = {"queue": []}

    def chat(messages, tools=None, *, max_tokens, seed=None, **kwargs):
        if tools is None:
            chat.sleep_seeds.append(seed)
            return {"content": note, "tool_calls": [], "usage": {"prompt_tokens": 80, "completion_tokens": 30},
                    "seconds": 2.0, "finish_reason": "stop"}
        chat.seeds.append(seed)
        if len(messages) == 2:  # a fresh session: [system, request]
            chat.systems.append(messages[0]["content"])
            state["queue"] = list(scripts[SESSIONS[messages[1]["content"]]])
        name, args = state["queue"].pop(0)
        return {"content": "", "tool_calls": [{"name": name, "arguments": args, "raw": json.dumps(args)}],
                "usage": {"prompt_tokens": 500, "completion_tokens": 50}, "seconds": 1.0,
                "finish_reason": "tool_calls"}
    chat.seeds, chat.systems, chat.sleep_seeds = [], [], []
    return chat


def rows(out):
    return [json.loads(line) for line in (out / "results.jsonl").read_text(encoding="utf-8").splitlines()]


def test_perfect_agent_gives_a_valid_wake_and_passes_every_design(tmp_path, monkeypatch):
    monkeypatch.setenv("DW_MODEL", "fake-model")
    out = tmp_path / "out"
    chat = scripted_agent({"s1": perfect("s1"), "s2": perfect("s2"), "final": perfect("final")})
    run.run_experiment(TASK, ["workspace", "handoff"], 1, 4096, 700, out, chat)
    workspace, handoff = rows(out)
    for row in (workspace, handoff):
        assert row["valid"] and row["passed"], row
        assert row["task"] == TASK and row["seed"] == 0 and row["model"] == "fake-model"
        assert row["window"] == 4096 and row["reserve"] == 700 and row["wake_attempts"] == 1
        assert row["wake"] == {"s1_passed": True, "s2_passed": True, "trap_experienced": False}
        assert all(row["checks"].values()) and len(row["checks"]) == 5
        steps = len(perfect("final"))
        assert row["steps"] == steps and not row["ran_out_of_steps"] and not row["repeated_mistake"]
        assert row["ended_by"] == "finish"
        assert row["prompt_tokens"] == 500 * steps and row["completion_tokens"] == 50 * steps
        assert row["max_prompt_tokens"] == 500 and row["seconds"] > 0 and row["overflow"] is False
    assert workspace["design"] == "workspace" and workspace["memory"] == "" and workspace["memory_chars"] == 0
    assert workspace["sleep_cost"] == {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "seconds": 0.0}
    assert handoff["memory"] == NOTE and handoff["memory_chars"] == len(NOTE)
    assert handoff["memory_tokens"] == llm.estimate_tokens(agent.MEMORY.format(memory=NOTE))  # header included
    assert workspace["memory_tokens"] == 0
    assert handoff["sleep_cost"] == {"calls": 2, "prompt_tokens": 160, "completion_tokens": 60, "seconds": 4.0}
    assert NOTE in chat.systems[-1] and NOTE not in chat.systems[-2]  # only the handoff final got the note
    wake_seeds = chat.seeds[:len(perfect("s1")) + len(perfect("s2"))]
    assert len(set(wake_seeds)) == len(wake_seeds)
    seed_dir = out / "seed0"
    for name in ("wake_s1.jsonl", "wake_s2.jsonl", "final_workspace.jsonl", "final_handoff.jsonl"):
        assert (seed_dir / name).read_text(encoding="utf-8").strip(), name
    config = json.loads((out / "config.json").read_text(encoding="utf-8"))
    assert config["args"]["designs"] == ["workspace", "handoff"] and config["model"] == "fake-model"
    assert config["git_commit"] and config["started"] and isinstance(config["git_dirty"], bool)
    assert len(chat.sleep_seeds) == 2 and None not in chat.sleep_seeds and len(set(chat.sleep_seeds)) == 2
    assert not set(chat.sleep_seeds) & set(chat.seeds)  # sleep seeds do not collide with any session step seed


def test_consolidated_design_runs_end_to_end_and_injects_its_rules_and_lessons(tmp_path):
    out = tmp_path / "out"
    memory = ('{"kind": "rule", "text": "Store durations in milliseconds with the _MS suffix."}\n'
              '{"kind": "lesson", "text": "src/settings.py is generated; edit settings.tmpl.json instead."}')
    chat = scripted_agent({"s1": perfect("s1"), "s2": perfect("s2"), "final": perfect("final")}, memory)
    run.run_experiment(TASK, ["workspace", "handoff", "consolidated"], 1, 4096, 700, out, chat)
    workspace, handoff, consolidated = rows(out)
    expected = ("- RULE: Store durations in milliseconds with the _MS suffix.\n"
                "- LESSON: src/settings.py is generated; edit settings.tmpl.json instead.")
    assert consolidated["design"] == "consolidated" and consolidated["valid"] and consolidated["passed"]
    assert consolidated["memory"] == expected and consolidated["memory_chars"] == len(expected)
    assert consolidated["memory_tokens"] == llm.estimate_tokens(agent.MEMORY.format(memory=expected))
    assert expected in chat.systems[-1]
    sleep_calls = len(chat.sleep_seeds) - handoff["sleep_cost"]["calls"]
    assert sleep_calls >= 4 and consolidated["sleep_cost"]["calls"] == sleep_calls  # >= 1 extract + 1 merge each
    assert consolidated["sleep_cost"]["prompt_tokens"] == 80 * sleep_calls


def tree(root):
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in Path(root).rglob("*") if p.is_file()}


def test_dreaming_and_replay_practise_on_snapshots_of_the_wake_and_log_it(tmp_path, monkeypatch):
    out = tmp_path / "out"
    memory = '{"kind": "rule", "text": "Store durations in milliseconds with the _MS suffix."}'
    dreams = ["Add `CACHE_TTL_SECONDS` of 5 seconds.", "Add a poll interval of 9 seconds."]  # they add nothing
    inner = scripted_agent({"s1": perfect("s1"), "s2": perfect("s2"), "final": perfect("final")}, memory)
    practised = []
    real = agent.run_session

    def spy(task_id, workspace, prompt, **kwargs):
        if Path(workspace).parent.name.startswith("practice_"):
            practised.append((Path(workspace).parent.name, prompt))
        return real(task_id, workspace, prompt, **kwargs)
    monkeypatch.setattr(run.agent, "run_session", spy)

    def chat(messages, tools=None, **kwargs):
        prompt = messages[-1]["content"] if tools is None else messages[1]["content"]
        if tools is None and prompt.endswith(designs.DREAM_INSTRUCTION):
            content = "\n".join(json.dumps({"request": d, "must_add": ["_MS"], "must_not_add": []}) for d in dreams)
        elif tools is not None and prompt in dreams:
            return {"content": "", "tool_calls": [{"name": "finish", "arguments": {"summary": "ok"}, "raw": "{}"}],
                    "usage": {"prompt_tokens": 500, "completion_tokens": 50}, "seconds": 1.0,
                    "finish_reason": "tool_calls"}
        else:
            return inner(messages, tools, **kwargs)
        return {"content": content, "tool_calls": [], "usage": {"prompt_tokens": 80, "completion_tokens": 30},
                "seconds": 2.0, "finish_reason": "stop"}
    run.run_experiment(TASK, ["workspace", "handoff", "consolidated", "dreaming", "replay"], 1, 4096, 700, out, chat)
    result = {r["design"]: r for r in rows(out)}
    assert list(result) == ["workspace", "handoff", "consolidated", "dreaming", "replay"]
    assert all(r["valid"] and r["passed"] for r in result.values())
    prompts = {sid: s["prompt"] for s in tasks.load_task(TASK)["sessions"] for sid in [s["id"]]}
    for design in ("workspace", "handoff", "consolidated"):
        assert result[design]["practice_log"] == [] and result[design]["contamination"] == []
    replay, dreaming = result["replay"]["practice_log"], result["dreaming"]["practice_log"]
    assert len(replay) == len(dreaming) == 2  # one record per sleep, s1 first
    assert all(set(r) == {"exercises", "refined", "accepted"} and len(r["exercises"]) == 2 for r in replay + dreaming)
    entries = [(r, e) for r in replay + dreaming for e in r["exercises"]]
    assert all(set(e) == {"request", "hypothetical", "must_add", "must_not_add", "score", "score2", "check_passed",
                          "check_passed2"} for _, e in entries)
    assert [[e["request"] for e in r["exercises"]] for r in replay] == [[prompts["s1"]] * 2, [prompts["s2"]] * 2]
    assert [(e["score"], e["score2"], e["check_passed"]) for r in replay for e in r["exercises"]] == [
        (0, None, True)] * 4  # the scripted agent passes its real sessions
    assert not any(r["refined"] for r in replay)
    assert [[e["request"] for e in r["exercises"]] for r in dreaming] == [dreams, dreams]
    assert all(e["hypothetical"] and e["must_add"] == ["_MS"] for r in dreaming for e in r["exercises"])
    assert [(e["score"], e["score2"]) for r in dreaming for e in r["exercises"]] == [(1, 1)] * 4  # adds no _MS
    assert all(r["refined"] and not r["accepted"] for r in dreaming)
    for design, sessions, calls in (("dreaming", 8, 4), ("replay", 4, 0)):  # dream + refine per sleep
        assert result[design]["memory"] == result["consolidated"]["memory"]
        assert result[design]["sleep_cost"]["practice_sessions"] == sessions
        assert result[design]["sleep_cost"]["calls"] == result["consolidated"]["sleep_cost"]["calls"] + calls
        assert result[design]["contamination"] == []
    assert [name for name, _ in practised] == (["practice_dreaming_s1"] * 4 + ["practice_dreaming_s2"] * 4
                                               + ["practice_replay_s1"] * 2 + ["practice_replay_s2"] * 2)
    seed_dir = out / "seed0"
    fresh = tasks.make_workspace(TASK, tmp_path / "fresh")
    assert tree(seed_dir / "snap_s1_before") == tree(fresh)
    assert tree(seed_dir / "snap_s1_after") == tree(seed_dir / "snap_s2_before") != tree(fresh)
    assert tree(seed_dir / "snap_s2_after") == tree(seed_dir / "ws_wake") == tree(seed_dir / "baseline")
    assert (seed_dir / "practice_replay_s1" / "attempt0" / "CHANGES.md").exists()


def test_editing_the_generated_file_in_the_final_session_fails_and_counts_as_repeated(tmp_path):
    out = tmp_path / "out"
    trap = [w for w in writes("trap") if w[1]["path"] == "src/settings.py"]
    chat = scripted_agent({"s1": perfect("s1"), "s2": perfect("s2"),
                           "final": trap + [("finish", {"summary": "done"})]})
    run.run_experiment(TASK, ["workspace"], 1, 4096, 700, out, chat)
    [row] = rows(out)
    assert row["valid"] and not row["passed"] and row["repeated_mistake"]
    assert row["checks"]["test_trap_avoided"] is False and row["wake"]["trap_experienced"] is False


def test_trap_write_during_the_wake_is_recorded(tmp_path):
    out = tmp_path / "out"
    trap = [("read_file", {"path": "src/settings.py"}),
            ("write_file", {"path": "./src/settings.py", "content": "TIMEOUT_S = 30\n"})]
    chat = scripted_agent({"s1": trap + perfect("s1"), "s2": perfect("s2"), "final": perfect("final")})
    run.run_experiment(TASK, ["workspace"], 1, 4096, 700, out, chat)
    [row] = rows(out)
    assert row["valid"] and row["wake"]["trap_experienced"] and row["passed"] and not row["repeated_mistake"]


def test_invalid_wake_is_redone_then_the_seed_is_marked_invalid(tmp_path):
    out = tmp_path / "out"
    give_up = [("finish", {"summary": "nothing"})]
    chat = scripted_agent({"s1": give_up, "s2": give_up, "final": give_up})
    run.run_experiment(TASK, ["workspace", "handoff"], 1, 4096, 700, out, chat, wake_retries=1)
    result = rows(out)
    assert [r["design"] for r in result] == ["workspace", "handoff"]
    assert all(not r["valid"] and r["wake_attempts"] == 2 for r in result)
    assert result[0]["wake"] == {"s1_passed": False, "s2_passed": False, "trap_experienced": False}
    assert len(chat.seeds) == 4 and len(set(chat.seeds)) == 4  # seeds differ per attempt and session
    assert (out / "seed0" / "wake_s1.jsonl").exists() and (out / "seed0" / "wake_s1.attempt2.jsonl").exists()
    assert "| n valid | 0/1 | 0/1 |" in run.summarize(out)


WAKES = {0: (1, False), 1: (3, True), 2: (2, False)}  # seed -> (wake attempts, trap experienced)


def summary_row(design, passed=True, valid=True, seed=0, **extra):
    attempts, trap = WAKES.get(seed, (1, False))
    row = {"design": design, "valid": valid, "seed": seed, "wake_attempts": attempts,
           "wake": {"s1_passed": True, "s2_passed": valid, "trap_experienced": trap}}
    if valid:
        row.update(passed=passed, checks={"test_a": passed, "test_b": True}, repeated_mistake=not passed,
                   steps=4, prompt_tokens=900, completion_tokens=100, seconds=10.0, ran_out_of_steps=False,
                   sleep_cost={"seconds": 3.0}, memory_tokens=40, eval_error=None)
    row.update(extra)
    return row


def write_results(out, lines):
    (out / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in lines), encoding="utf-8")


def test_summarize_counts_rates_and_means(tmp_path):
    lines = [summary_row("workspace", passed=False, ran_out_of_steps=True), summary_row("workspace", seed=1),
             summary_row("workspace", valid=False, seed=2),
             summary_row("handoff", steps=8), summary_row("handoff", steps=6, seed=1),
             summary_row("handoff", valid=False, seed=2)]
    write_results(tmp_path, lines)
    table = run.summarize(tmp_path)
    assert table.splitlines()[0] == "| metric | workspace | handoff |"
    assert "| n valid | 2/3 | 2/3 |" in table
    assert "| wake validity | 0.67 | 0.67 |" in table and "| mean wake attempts | 2.0 | 2.0 |" in table
    assert "| trap-experienced rate | 0.33 | 0.33 |" in table
    assert "| completion rate | 0.50 | 1.00 |" in table
    assert "| ran-out-of-steps rate | 0.50 | 0.00 |" in table
    assert "| test_a pass rate | 0.50 | 1.00 |" in table and "| test_b pass rate | 1.00 | 1.00 |" in table
    assert "| repeated-mistake rate | 0.50 | 0.00 |" in table
    assert "| mean steps | 4.0 | 7.0 |" in table
    assert "| mean final tokens | 1000.0 | 1000.0 |" in table
    assert "| mean final seconds | 10.0 | 10.0 |" in table
    assert "| mean sleep seconds | 3.0 | 3.0 |" in table and "| mean injected tokens | 40.0 | 40.0 |" in table
    assert "workspace eval errors: 0" in table and "errored seeds: 0" in table


def test_summarize_uses_only_seeds_that_have_a_row_for_every_design(tmp_path):
    write_results(tmp_path, [summary_row("workspace"), summary_row("handoff", passed=False),
                             summary_row("workspace", seed=1)])  # seed 1 never got its handoff row
    table = run.summarize(tmp_path)
    assert "| n valid | 1/1 | 1/1 |" in table and "| completion rate | 1.00 | 0.00 |" in table


def test_summarize_leaves_evaluator_errors_out_of_every_rate_and_counts_them(tmp_path):
    broken = dict(passed=False, checks={"test_a": False, "test_b": False}, eval_error="pytest exit 3")
    write_results(tmp_path, [summary_row("workspace"), summary_row("workspace", seed=1, **broken),
                             summary_row("handoff"), summary_row("handoff", seed=1)])
    table = run.summarize(tmp_path)
    assert "| completion rate | 1.00 | 1.00 |" in table and "| test_a pass rate | 1.00 | 1.00 |" in table
    assert "workspace eval errors: 1" in table and "handoff eval errors: 0" in table


def test_cli_defaults(monkeypatch):
    seen = {}
    monkeypatch.setenv("DW_MODEL", "fake-model")
    monkeypatch.setattr(run, "run_experiment", lambda *args: seen.update(args=args))
    monkeypatch.setattr(run, "summarize", lambda out: "table")
    run.main(["--out", "somewhere"])
    task, designs_, seeds, window, reserve, out, chat = seen["args"]
    assert (task, designs_, seeds, window, reserve, str(out)) == (
        "settings_gen", ["workspace", "handoff"], 3, 4096, 1024, "somewhere")


def test_cli_summarize_prints_the_table(tmp_path, capsys):
    (tmp_path / "results.jsonl").write_text(json.dumps(summary_row("handoff")) + "\n", encoding="utf-8")
    run.main(["--summarize", str(tmp_path)])
    assert "| n valid | 1/1 |" in capsys.readouterr().out


# --- final-session directory, trap detection, git ---------------------------------------------------

def test_every_designs_final_session_runs_in_the_same_directory_and_is_archived_after(tmp_path, monkeypatch):
    final_prompt = tasks.load_task(TASK)["sessions"][2]["prompt"]
    real, seen = agent.run_session, []

    def spy(task_id, workspace, prompt, **kwargs):
        if prompt == final_prompt:
            seen.append(Path(workspace))
        return real(task_id, workspace, prompt, **kwargs)
    monkeypatch.setattr(run.agent, "run_session", spy)
    out = tmp_path / "out"
    chat = scripted_agent({"s1": perfect("s1"), "s2": perfect("s2"), "final": perfect("final")})
    run.run_experiment(TASK, ["workspace", "handoff"], 1, 4096, 700, out, chat)
    seed_dir = out / "seed0"
    assert seen == [seed_dir / "ws_final", seed_dir / "ws_final"]
    assert not (seed_dir / "ws_final").exists()
    for design in ("workspace", "handoff"):
        assert (seed_dir / f"final_{design}_ws" / "src" / "retry.py").exists()


def write_event(step, path, result="wrote 3 characters to x"):
    return [{"type": "call", "step": step, "tool": "write_file", "args": {"path": path, "content": "x"}},
            {"type": "result", "step": step, "text": result}]


@pytest.mark.parametrize("spelling", ["src/settings.py", "./src/settings.py", "src\\settings.py", "SRC/Settings.py",
                                      "src/../src/settings.py", "ABSOLUTE"])
def test_wrote_trap_counts_any_spelling_of_the_trap_path(tmp_path, spelling):
    spelling = str(tmp_path / "src" / "settings.py") if spelling == "ABSOLUTE" else spelling
    assert run._wrote_trap(write_event(1, spelling), {"src/settings.py"}, tmp_path)


def test_wrote_trap_ignores_failed_writes_other_files_and_other_tools(tmp_path):
    trap = {"src/settings.py"}
    failed = write_event(1, "src/settings.py", "error: src/settings.py already exists - read_file it first")
    assert not run._wrote_trap(failed, trap, tmp_path)
    assert not run._wrote_trap(write_event(1, "src/settings_2.py"), trap, tmp_path)
    assert not run._wrote_trap(write_event(1, "../src/settings.py"), trap, tmp_path)
    read = [{"type": "call", "step": 1, "tool": "read_file", "args": {"path": "src/settings.py"}},
            {"type": "result", "step": 1, "text": "TIMEOUT"}]
    assert not run._wrote_trap(read, trap, tmp_path)
    assert run._wrote_trap(failed + write_event(2, "src/settings.py"), trap, tmp_path)  # a later success counts


def test_git_dirty_is_true_only_when_the_work_tree_has_changes(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "REPO", tmp_path)
    git = ["git", "-c", "user.name=t", "-c", "user.email=t@example.invalid"]
    subprocess.run(git + ["init", "-q"], cwd=tmp_path, check=True)
    assert run._git_dirty() is False
    (tmp_path / "a.txt").write_text("x", encoding="utf-8")
    assert run._git_dirty() is True
    subprocess.run(git + ["add", "."], cwd=tmp_path, check=True)
    subprocess.run(git + ["commit", "-qm", "x"], cwd=tmp_path, check=True)
    assert run._git_dirty() is False


# --- partial results, evaluator faults, transport faults -------------------------------------------

def test_a_seed_that_dies_midway_leaves_no_rows_and_earlier_seeds_stay(tmp_path):
    inner = scripted_agent({"s1": perfect("s1"), "s2": perfect("s2"), "final": perfect("final")})
    sleeps = []

    def chat(messages, tools=None, **kwargs):
        if tools is None:
            sleeps.append(1)
            if len(sleeps) == 3:  # seed 1, handoff: after the workspace design already finished
                raise ValueError("not a transport fault")
        return inner(messages, tools, **kwargs)
    out = tmp_path / "out"
    with pytest.raises(ValueError):
        run.run_experiment(TASK, ["workspace", "handoff"], 2, 4096, 700, out, chat)
    assert [(r["seed"], r["design"]) for r in rows(out)] == [(0, "workspace"), (0, "handoff")]


def test_an_existing_results_file_is_never_appended_to(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / "results.jsonl").write_text("old\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="results.jsonl"):
        run.run_experiment(TASK, ["workspace"], 1, 4096, 700, out, scripted_agent({}))
    assert (out / "results.jsonl").read_text(encoding="utf-8") == "old\n" and not (out / "config.json").exists()


def test_an_evaluator_error_during_the_wake_aborts_instead_of_failing_the_wake(tmp_path, monkeypatch):
    monkeypatch.setattr(run.tasks, "evaluate", lambda *a, **k: {"passed": False, "checks": {}, "error": "gen boom"})
    sessions = {s["id"]: s for s in tasks.load_task(TASK)["sessions"]}
    chat = scripted_agent({"s1": perfect("s1"), "s2": perfect("s2"), "final": perfect("final")})
    with pytest.raises(RuntimeError, match="gen boom"):
        run._wake(TASK, sessions, tmp_path, 0, chat, 4096, 700, 3)


def test_an_evaluator_error_on_the_final_keeps_the_row_and_the_summary_leaves_it_out(tmp_path, monkeypatch):
    real = tasks.evaluate

    def evaluate(task_id, stage, *args, **kwargs):
        if stage == "final":
            return {"passed": False, "checks": {"test_x": False}, "error": "pytest exit 3"}
        return real(task_id, stage, *args, **kwargs)
    monkeypatch.setattr(run.tasks, "evaluate", evaluate)
    out = tmp_path / "out"
    chat = scripted_agent({"s1": perfect("s1"), "s2": perfect("s2"), "final": perfect("final")})
    run.run_experiment(TASK, ["workspace", "handoff"], 1, 4096, 700, out, chat)
    assert [r["eval_error"] for r in rows(out)] == ["pytest exit 3"] * 2
    table = run.summarize(out)
    assert "workspace eval errors: 1" in table and "handoff eval errors: 1" in table
    assert "| completion rate | - | - |" in table


@pytest.mark.parametrize("fault", [RuntimeError("HTTP 500 from server"), ConnectionResetError("connection reset")])
def test_a_transport_fault_costs_only_its_own_seed(tmp_path, fault):
    inner = scripted_agent({"s1": perfect("s1"), "s2": perfect("s2"), "final": perfect("final")})
    calls = []

    def chat(messages, tools=None, **kwargs):
        calls.append(1)
        if len(calls) == 1:  # the very first request of seed 0
            raise fault
        return inner(messages, tools, **kwargs)
    out = tmp_path / "out"
    run.run_experiment(TASK, ["workspace", "handoff"], 2, 4096, 700, out, chat)
    assert [(r["seed"], r["design"]) for r in rows(out)] == [(1, "workspace"), (1, "handoff")]
    [error] = [json.loads(line) for line in (out / "errors.jsonl").read_text(encoding="utf-8").splitlines()]
    assert error["seed"] == 0 and str(fault) in error["error"]
    assert "errored seeds: 1" in run.summarize(out)
