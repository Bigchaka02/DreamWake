"""End-to-end experiment runs against scripted fake agents (no network), and the summary table."""
import json

from dreamwake import run, tasks

TASK = "settings_gen"
SESSIONS = {s["prompt"]: s["id"] for s in tasks.load_task(TASK)["sessions"]}
REFERENCE = tasks.load_task(TASK)["reference"]
NOTE = "Settings live in settings.tmpl.json; durations are in _MS (replaces seconds)."


def writes(name):
    root = REFERENCE / name
    return [("write_file", {"path": p.relative_to(root).as_posix(), "content": p.read_text(encoding="utf-8")})
            for p in sorted(root.rglob("*")) if p.is_file()]


def perfect(name):
    return writes(name) + [("run_tests", {}), ("finish", {"summary": "done"})]


def scripted_agent(scripts):
    """A fake chat: picks the script for a session from its prompt; sleep calls (no tools) get NOTE."""
    state = {"queue": []}

    def chat(messages, tools=None, *, max_tokens, seed=None, **kwargs):
        if tools is None:
            return {"content": NOTE, "tool_calls": [], "usage": {"prompt_tokens": 80, "completion_tokens": 30},
                    "seconds": 2.0, "finish_reason": "stop"}
        chat.seeds.append(seed)
        if len(messages) == 2:  # a fresh session: [system, request]
            chat.systems.append(messages[0]["content"])
            state["queue"] = list(scripts[SESSIONS[messages[1]["content"]]])
        name, args = state["queue"].pop(0)
        return {"content": "", "tool_calls": [{"name": name, "arguments": args, "raw": json.dumps(args)}],
                "usage": {"prompt_tokens": 500, "completion_tokens": 50}, "seconds": 1.0,
                "finish_reason": "tool_calls"}
    chat.seeds, chat.systems = [], []
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
        assert row["prompt_tokens"] == 500 * steps and row["completion_tokens"] == 50 * steps
        assert row["max_prompt_tokens"] == 500 and row["seconds"] > 0 and row["overflow"] is False
    assert workspace["design"] == "workspace" and workspace["memory"] == "" and workspace["memory_chars"] == 0
    assert workspace["sleep_cost"] == {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "seconds": 0.0}
    assert handoff["memory"] == NOTE and handoff["memory_chars"] == len(NOTE) and handoff["memory_tokens"] > 0
    assert handoff["sleep_cost"] == {"calls": 2, "prompt_tokens": 160, "completion_tokens": 60, "seconds": 4.0}
    assert NOTE in chat.systems[-1] and NOTE not in chat.systems[-2]  # only the handoff final got the note
    wake_seeds = chat.seeds[:len(perfect("s1")) + len(perfect("s2"))]
    assert len(set(wake_seeds)) == len(wake_seeds)
    seed_dir = out / "seed0"
    for name in ("wake_s1.jsonl", "wake_s2.jsonl", "final_workspace.jsonl", "final_handoff.jsonl"):
        assert (seed_dir / name).read_text(encoding="utf-8").strip(), name
    config = json.loads((out / "config.json").read_text(encoding="utf-8"))
    assert config["args"]["designs"] == ["workspace", "handoff"] and config["model"] == "fake-model"
    assert config["git_commit"] and config["started"]


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
    trap = [("write_file", {"path": "./src/settings.py", "content": "TIMEOUT_S = 30\n"})]
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


def summary_row(design, passed=True, valid=True, **extra):
    row = {"design": design, "valid": valid}
    if valid:
        row.update(passed=passed, checks={"test_a": passed, "test_b": True}, repeated_mistake=not passed,
                   steps=4, prompt_tokens=900, completion_tokens=100, seconds=10.0,
                   sleep_cost={"seconds": 3.0}, memory_tokens=40)
    row.update(extra)
    return row


def test_summarize_counts_rates_and_means(tmp_path):
    lines = [summary_row("workspace", passed=False), summary_row("workspace"), summary_row("workspace", valid=False),
             summary_row("handoff", steps=8), summary_row("handoff", steps=6), summary_row("handoff", valid=False)]
    (tmp_path / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in lines), encoding="utf-8")
    table = run.summarize(tmp_path)
    assert table.splitlines()[0] == "| metric | workspace | handoff |"
    assert "| n valid | 2/3 | 2/3 |" in table
    assert "| completion rate | 0.50 | 1.00 |" in table
    assert "| test_a pass rate | 0.50 | 1.00 |" in table and "| test_b pass rate | 1.00 | 1.00 |" in table
    assert "| repeated-mistake rate | 0.50 | 0.00 |" in table
    assert "| mean steps | 4.0 | 7.0 |" in table
    assert "| mean final tokens | 1000.0 | 1000.0 |" in table
    assert "| mean final seconds | 10.0 | 10.0 |" in table
    assert "| mean sleep seconds | 3.0 | 3.0 |" in table and "| mean injected tokens | 40.0 | 40.0 |" in table


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
