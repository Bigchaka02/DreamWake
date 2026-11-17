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


@pytest.mark.parametrize("store", ["handoff", "consolidated", "replay", "dreaming", "oracle", "requests"])
@pytest.mark.parametrize("placement", [None, "system", "request", "tail"])
@pytest.mark.parametrize("gate", [False, True])
def test_parse_design_valid_shapes(store, placement, gate):
    spec = store + ("@" + placement if placement else "") + ("+gate" if gate else "")
    assert run.parse_design(spec) == (store, placement or "system", gate)


def test_parse_design_workspace():
    assert run.parse_design("workspace") == ("workspace", "system", False)


@pytest.mark.parametrize("spec", ["", "nearby", "unknown", "handoff@", "handoff@user", "handoff@SYSTEM",
    "handoff+", "handoff+check", "handoff+gate+gate", "handoff@tail+gate+extra", "handoff@@tail",
    "handoff@tail@system", "handoff+gate@tail", "handoff@tail+", "handoff@tail+check", "handoff ",
    "workspace@system", "workspace@request", "workspace@tail", "workspace+gate", "workspace@tail+gate"])
def test_parse_design_rejects_invalid_shapes_before_model_calls(tmp_path, spec):
    with pytest.raises(ValueError, match="store|placement|suffix|workspace"):
        run.parse_design(spec)

    def no_chat(*args, **kwargs):
        pytest.fail("invalid spec reached the model")
    with pytest.raises(ValueError):
        run.run_experiment(TASK, ["workspace", spec], 1, 4096, 700, tmp_path / "out", no_chat)


@pytest.mark.parametrize("text", [None, "x" * (designs.NOTE_CAP + 1)])
def test_oracle_missing_or_too_long_fails_before_model_calls(tmp_path, monkeypatch, text):
    task_dir = tmp_path / TASK
    task_dir.mkdir()
    (task_dir / "task.json").write_text((tasks.TASKS_DIR / TASK / "task.json").read_text(encoding="utf-8"),
                                        encoding="utf-8")
    if text is not None:
        (task_dir / "oracle_memory.txt").write_text(text, encoding="utf-8")
    monkeypatch.setattr(tasks, "TASKS_DIR", tmp_path)

    def no_chat(*args, **kwargs):
        pytest.fail("invalid oracle reached the model")
    with pytest.raises(ValueError, match="oracle_memory.txt"):
        run.run_experiment(TASK, ["workspace", "oracle@request"], 1, 4096, 700, tmp_path / "out", no_chat)


def test_oracle_uses_pinned_rules_without_sleep_cost_or_practice(tmp_path):
    expected = (
        "- RULE: Every duration setting is stored in milliseconds and its name ends in _MS. This also holds "
        "when a request gives the value in seconds: convert it (30 seconds is stored as TIMEOUT_MS = 30000). "
        "Functions that return seconds divide by 1000. Counts and sizes are not durations.\n"
        "- RULE: Settings are defined in settings.tmpl.json. src/settings.py is generated from it by tools/gen.py; "
        "never edit src/settings.py by hand.\n"
        "- RULE: Every new or renamed setting must be listed in CHANGES.md, or tools/audit.py fails.")
    assert (tasks.TASKS_DIR / TASK / "oracle_memory.txt").read_text(encoding="utf-8").strip() == expected
    out = tmp_path / "out"
    chat = scripted_agent({"s1": perfect("s1"), "s2": perfect("s2"), "final": perfect("final")})
    run.run_experiment(TASK, ["oracle"], 1, 4096, 700, out, chat, final_repeats=2)
    assert chat.sleep_seeds == []
    for row in rows(out):
        assert row["memory"] == expected and row["practice_log"] == [] and row["passed"]
        assert row["sleep_cost"] == {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "seconds": 0.0}
        assert (row["design"], row["store"], row["placement"], row["gate"]) == ("oracle", "oracle", "system", False)
        assert row["memory_tokens"] == llm.estimate_tokens(agent.MEMORY.format(memory=expected))


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


def tiny_task(tmp_path, monkeypatch, sessions, passing=None):
    """Real tiny workspaces and hidden tests, driven only by a fake chat transport."""
    task_dir = tmp_path / "tasks" / "tiny"
    workspace = task_dir / "workspace"
    workspace.mkdir(parents=True)
    (workspace / "state.txt").write_text("", encoding="utf-8")
    task = {"id": "tiny", "sessions": sessions, "check_cmd": [], "protected": []}
    (task_dir / "task.json").write_text(json.dumps(task), encoding="utf-8")
    states, state = {}, ""
    for session in sessions:
        state += session["id"] + "\n"
        states[session["prompt"]] = state
        stage = session.get("stage", session["id"])
        if stage is not None:
            hidden = task_dir / "hidden_tests" / stage
            hidden.mkdir(parents=True, exist_ok=True)
            expected = state if (passing or {}).get(session["id"], True) else "wrong"
            (hidden / "test_state.py").write_text(
                "from pathlib import Path\n\ndef test_state():\n"
                f"    assert Path('state.txt').read_text(encoding='utf-8') == {expected!r}\n",
                encoding="utf-8")
    monkeypatch.setattr(tasks, "TASKS_DIR", task_dir.parent)
    queue = []

    def chat(messages, tools=None, *, seed=None, **kwargs):
        if tools is None:
            return {"content": NOTE, "tool_calls": [], "usage": {"prompt_tokens": 10, "completion_tokens": 5},
                    "seconds": 1.0, "finish_reason": "stop"}
        if len(messages) == 2:
            chat.starts.append((messages, seed))
            prompt = messages[1]["content"]
            queue[:] = [("read_file", {"path": "state.txt"}),
                        ("write_file", {"path": "state.txt", "content": states[prompt]})]
        name, args = queue.pop(0) if queue else ("finish", {"summary": "done"})
        return {"content": "", "tool_calls": [{"name": name, "arguments": args, "raw": json.dumps(args)}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 10}, "seconds": 1.0,
                "finish_reason": "tool_calls"}
    chat.starts = []
    return chat


def test_four_wakes_keep_order_snapshots_sleep_seeds_and_stages(tmp_path, monkeypatch):
    sessions = [{"id": "first", "prompt": "one", "max_steps": 4},
                {"id": "second", "stage": "revision", "prompt": "two", "max_steps": 4},
                {"id": "third", "stage": None, "prompt": "three", "max_steps": 4},
                {"id": "fourth", "stage": None, "prompt": "four", "max_steps": 4},
                {"id": "last", "stage": "score", "prompt": "end", "max_steps": 4}]
    chat = tiny_task(tmp_path, monkeypatch, sessions, {"first": False})
    real_sleep, sleeps = designs.sleep, []

    def sleep(store, artifact, events, **kwargs):
        sleeps.append((store, artifact, events, kwargs))
        return real_sleep(store, artifact, events, **kwargs)
    monkeypatch.setattr(designs, "sleep", sleep)
    out = tmp_path / "out"
    run.run_experiment("tiny", ["workspace", "handoff"], 2, 4096, 700, out, chat, wake_retries=0)
    assert len(chat.starts) == 12
    assert len(sleeps) == 16
    for k in range(2):
        seed_dir = out / f"seed{k}"
        starts = chat.starts[k * 6:(k + 1) * 6]
        assert [messages[1]["content"] for messages, _ in starts] == [s["prompt"] for s in sessions] + ["end"]
        assert [seed for _, seed in starts[:4]] == [
            k * run.SEED_STRIDE + run.ATTEMPT_STRIDE + i * run.SESSION_STRIDE + 1 for i in range(4)]
        assert all(messages == [{"role": "system", "content": agent.SYSTEM.format(max_steps=4)},
                                {"role": "user", "content": s["prompt"]}]
                   for (messages, _), s in zip(starts, sessions[:4]))
        state = ""
        for i, s in enumerate(sessions[:4]):
            sid = s["id"]
            before, after = seed_dir / f"snap_{sid}_before", seed_dir / f"snap_{sid}_after"
            assert (before / "state.txt").read_text(encoding="utf-8") == state
            state += sid + "\n"
            assert (after / "state.txt").read_text(encoding="utf-8") == state
            for j, store in enumerate(("workspace", "handoff")):
                actual_store, artifact, events, kwargs = sleeps[k * 8 + j * 4 + i]
                assert actual_store == store and events[0] == {"type": "user", "text": s["prompt"]}
                assert artifact == (NOTE if store == "handoff" and i else None)
                assert kwargs["seed"] == k * run.SEED_STRIDE + run.SLEEP_OFFSET + i * run.SESSION_STRIDE
                assert kwargs["practice"] == {
                    "task_id": "tiny", "request": s["prompt"], "ws_before": before, "ws_after": after,
                    "scratch": seed_dir / f"practice_{store}_{sid}", "max_steps": run.PRACTICE_STEPS,
                    "memory_at": "system", "finish_gate": False}
        for row in rows(out)[k * 2:(k + 1) * 2]:
            assert row["valid"] and row["passed"] and row["checks"] == {"test_state": True}
            assert row["wake"] == {"first_passed": False, "second_passed": True, "trap_experienced": False}


@pytest.mark.parametrize("stages, passing, valid", [
    (["early", "later", None], {"early": False, "later": True}, True),
    (["early", "later", None], {"early": True, "later": False}, False),
    ([None, None, None], {}, True),
])
def test_wake_validity_uses_last_scored_stage(tmp_path, monkeypatch, stages, passing, valid):
    sessions = [{"id": sid, "stage": stage, "prompt": sid, "max_steps": 3}
                for sid, stage in zip(("early", "later", "cutoff"), stages)]
    sessions.append({"id": "end", "prompt": "end", "max_steps": 3})
    chat = tiny_task(tmp_path, monkeypatch, sessions, passing)
    out = tmp_path / "out"
    run.run_experiment("tiny", ["workspace"], 1, 4096, 700, out, chat, wake_retries=1)
    [row] = rows(out)
    assert row["valid"] is valid
    assert row["wake_attempts"] == (1 if valid else 2)
    assert row["wake"] == {**{f"{sid}_passed": passed for sid, passed in passing.items()},
                           "trap_experienced": False}
    assert len(chat.starts) == (4 if valid else 6)
    if not valid:
        assert [seed for _, seed in chat.starts] == [
            attempt * run.ATTEMPT_STRIDE + i * run.SESSION_STRIDE + 1
            for attempt in (1, 2) for i in range(3)]


@pytest.mark.parametrize("count, null_final", [(0, False), (1, False), (11, False), (3, True)])
def test_invalid_session_layout_rejected_before_model(tmp_path, monkeypatch, count, null_final):
    sessions = [{"id": f"s{i}", "prompt": f"session {i}", "max_steps": 3} for i in range(count)]
    if null_final:
        sessions[-1]["stage"] = None
    tiny_task(tmp_path, monkeypatch, sessions)

    def no_chat(*args, **kwargs):
        pytest.fail("invalid sessions reached the model")
    with pytest.raises(ValueError, match="wake|final"):
        run.run_experiment("tiny", ["workspace"], 1, 4096, 700, tmp_path / "out", no_chat)


@pytest.mark.parametrize("count", [1, 9])
def test_one_or_nine_wakes_run(tmp_path, monkeypatch, count):
    sessions = [{"id": f"wake{i}", "stage": None, "prompt": f"wake {i}", "max_steps": 3}
                for i in range(count)]
    sessions.append({"id": "end", "prompt": "end", "max_steps": 3})
    chat = tiny_task(tmp_path, monkeypatch, sessions)
    out = tmp_path / "out"
    run.run_experiment("tiny", ["workspace"], 1, 4096, 700, out, chat)
    [row] = rows(out)
    assert row["valid"] and row["passed"] and row["wake"] == {"trap_experienced": False}
    assert len(chat.starts) == count + 1


@pytest.mark.parametrize("spec, gated", [("workspace", False), ("handoff+gate", True)])
def test_final_row_records_finish_gate(tmp_path, monkeypatch, spec, gated):
    sessions = [{"id": "s1", "prompt": "one", "max_steps": 4},
                {"id": "s2", "prompt": "two", "max_steps": 4},
                {"id": "final", "prompt": "end", "max_steps": 4}]
    chat = tiny_task(tmp_path, monkeypatch, sessions)
    out = tmp_path / "out"
    run.run_experiment("tiny", [spec], 1, 4096, 700, out, chat)
    [row] = rows(out)
    assert row["passed"] and row["gated"] is gated


def inherited_task(tmp_path, monkeypatch):
    sessions = [{"id": "s1", "prompt": "one", "max_steps": 3},
                {"id": "final", "prompt": "end", "max_steps": 3}]
    chat = tiny_task(tmp_path, monkeypatch, sessions)
    base_dir = tasks.TASKS_DIR / "tiny"
    base = json.loads((base_dir / "task.json").read_text(encoding="utf-8"))
    base.update(check_cmd=["python check.py"], trap_paths=["trap.txt"],
                protected=["checks"], eval_setup=["python setup.py"])
    (base_dir / "task.json").write_text(json.dumps(base), encoding="utf-8")
    (base_dir / "workspace" / "checks").mkdir()
    (base_dir / "workspace" / "checks" / "pristine.txt").write_text("base", encoding="utf-8")
    (base_dir / "workspace" / "check.py").write_text("print('base check')\n", encoding="utf-8")
    (base_dir / "workspace" / "setup.py").write_text(
        "from pathlib import Path\nPath('setup.txt').write_text('ready', encoding='utf-8')\n", encoding="utf-8")
    reference = base_dir / "reference"
    (reference / "ready").mkdir(parents=True)
    (reference / "parents.json").write_text('{"ready": null}', encoding="utf-8")
    (reference / "ready" / "state.txt").write_text("base reference", encoding="utf-8")
    (base_dir / "oracle_memory.txt").write_text("base memory", encoding="utf-8")
    child_dir = tasks.TASKS_DIR / "child"
    child_dir.mkdir()
    child = {"id": "child", "base": "tiny", "sessions": sessions}
    (child_dir / "task.json").write_text(json.dumps(child), encoding="utf-8")
    return base_dir, child_dir, chat


def test_task_inherits_defaults_and_paths(tmp_path, monkeypatch):
    base_dir, child_dir, _ = inherited_task(tmp_path, monkeypatch)
    base, child = tasks.load_task("tiny"), tasks.load_task("child")
    assert child["id"] == "child" and child["dir"] == child_dir
    for key in ("check_cmd", "trap_paths", "protected", "eval_setup", "workspace", "reference"):
        assert child[key] == base[key]
    ws = tasks.make_workspace("child", tmp_path / "ws")
    code, out = tasks.run_check("child", ws)
    assert code == 0 and "base check" in out
    tasks.apply_reference("child", "ready", ws)
    assert (ws / "state.txt").read_text(encoding="utf-8") == "base reference"


@pytest.mark.parametrize("local", [("workspace",), ("reference",), ("workspace", "reference")])
def test_child_values_and_folders_override_base(tmp_path, monkeypatch, local):
    base_dir, child_dir, _ = inherited_task(tmp_path, monkeypatch)
    child = json.loads((child_dir / "task.json").read_text(encoding="utf-8"))
    for key in ("check_cmd", "trap_paths", "protected", "eval_setup"):
        child[key] = []
    child["sessions"] = [{"id": "child session"}]
    (child_dir / "task.json").write_text(json.dumps(child), encoding="utf-8")
    if "workspace" in local:
        (child_dir / "workspace").mkdir()
        (child_dir / "workspace" / "state.txt").write_text("child workspace", encoding="utf-8")
    if "reference" in local:
        reference = child_dir / "reference"
        (reference / "ready").mkdir(parents=True)
        (reference / "parents.json").write_text('{"ready": null}', encoding="utf-8")
        (reference / "ready" / "state.txt").write_text("child reference", encoding="utf-8")
    loaded = tasks.load_task("child")
    assert loaded["sessions"] == child["sessions"]
    assert all(loaded[key] == [] for key in ("check_cmd", "trap_paths", "protected", "eval_setup"))
    for key in ("workspace", "reference"):
        assert loaded[key] == (child_dir if key in local else base_dir) / key
    ws = tasks.make_workspace("child", tmp_path / "ws")
    assert (ws / "state.txt").read_text(encoding="utf-8") == ("child workspace" if "workspace" in local else "")
    tasks.apply_reference("child", "ready", ws)
    expected = "child reference" if "reference" in local else "base reference"
    assert (ws / "state.txt").read_text(encoding="utf-8") == expected


@pytest.mark.parametrize("child_stage", ["s1", "new_stage"])
def test_hidden_tests_merge_with_child_winning_clash(tmp_path, monkeypatch, child_stage):
    base_dir, child_dir, _ = inherited_task(tmp_path, monkeypatch)
    (base_dir / "hidden_tests" / "s1" / "test_clash.py").write_text(
        "def test_replaced_base():\n    assert False\n", encoding="utf-8")
    hidden = child_dir / "hidden_tests" / child_stage
    hidden.mkdir(parents=True)
    (hidden / "test_clash.py").write_text("def test_child_override():\n    assert True\n", encoding="utf-8")
    (hidden / "test_child.py").write_text(
        "from pathlib import Path\n\ndef test_child():\n"
        "    assert Path('setup.txt').read_text(encoding='utf-8') == 'ready'\n"
        "    assert Path('checks/pristine.txt').read_text(encoding='utf-8') == 'base'\n"
        "    assert not Path('checks/extra.txt').exists()\n", encoding="utf-8")
    (hidden / "ignored.py").write_text("raise RuntimeError('not a hidden test')\n", encoding="utf-8")
    ws = tasks.make_workspace("tiny", tmp_path / "ws")
    (ws / "state.txt").write_text("s1\n", encoding="utf-8")
    (ws / "checks" / "pristine.txt").write_text("tampered", encoding="utf-8")
    (ws / "checks" / "extra.txt").write_text("extra", encoding="utf-8")
    expected = {"test_child_override": True, "test_child": True}
    if child_stage == "s1":
        expected["test_state"] = True
    assert tasks.evaluate("child", child_stage, ws) == {"passed": True, "checks": expected}
    if child_stage == "new_stage":  # a stage with no child directory still inherits its base files
        result = tasks.evaluate("child", "final", ws)
        assert result == {"passed": False, "checks": {"test_state": False}}


def test_unbased_task_keeps_hidden_support_files(tmp_path, monkeypatch):
    tiny_task(tmp_path, monkeypatch, [{"id": "s1", "prompt": "one", "max_steps": 3}])
    hidden = tasks.TASKS_DIR / "tiny" / "hidden_tests" / "s1"
    (hidden / "support.py").write_text("VALUE = 'support'\n", encoding="utf-8")
    (hidden / "test_support.py").write_text(
        "from support import VALUE\n\ndef test_support():\n    assert VALUE == 'support'\n", encoding="utf-8")
    ws = tasks.make_workspace("tiny", tmp_path / "ws")
    (ws / "state.txt").write_text("s1\n", encoding="utf-8")
    assert tasks.evaluate("tiny", "s1", ws) == {
        "passed": True, "checks": {"test_state": True, "test_support": True}}


@pytest.mark.parametrize("child_memory", [None, "child memory", ""])
def test_oracle_memory_inherits_with_child_precedence(tmp_path, monkeypatch, child_memory):
    _, child_dir, chat = inherited_task(tmp_path, monkeypatch)
    if child_memory is not None:
        (child_dir / "oracle_memory.txt").write_text(child_memory, encoding="utf-8")
    out = tmp_path / "out"
    run.run_experiment("child", ["oracle"], 1, 4096, 700, out, chat)
    [row] = rows(out)
    assert row["valid"] and row["passed"]
    assert row["memory"] == (child_memory if child_memory is not None else "base memory")
    assert row["sleep_cost"]["calls"] == 0 and row["practice_log"] == []


def test_two_level_inheritance_is_rejected_before_model(tmp_path, monkeypatch):
    base_dir, _, _ = inherited_task(tmp_path, monkeypatch)
    base = json.loads((base_dir / "task.json").read_text(encoding="utf-8"))
    base["base"] = "child"
    (base_dir / "task.json").write_text(json.dumps(base), encoding="utf-8")

    def no_chat(*args, **kwargs):
        pytest.fail("two-level inheritance reached the model")
    with pytest.raises(ValueError, match="base|inherit"):
        tasks.load_task("child")
    with pytest.raises(ValueError, match="base|inherit"):
        run.run_experiment("child", ["workspace"], 1, 4096, 700, tmp_path / "out", no_chat)


def test_perfect_agent_gives_a_valid_wake_and_passes_every_design(tmp_path, monkeypatch):
    monkeypatch.setenv("DW_MODEL", "fake-model")
    out = tmp_path / "out"
    chat = scripted_agent({"s1": perfect("s1"), "s2": perfect("s2"), "final": perfect("final")})
    run.run_experiment(TASK, ["workspace", "handoff"], 1, 4096, 700, out, chat)
    workspace, handoff = rows(out)
    for row in (workspace, handoff):
        assert row["repeat"] == 0
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
    assert config["args"]["final_repeats"] == 1
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


def test_request_spec_uses_safe_labels_and_preserves_design(tmp_path):
    out = tmp_path / "out"
    rule = "Store durations in milliseconds with the _MS suffix."
    inner = scripted_agent({"s1": perfect("s1"), "s2": perfect("s2"), "final": perfect("final")},
                           json.dumps({"kind": "rule", "text": rule}))
    starts = []

    def chat(messages, tools=None, **kwargs):
        if tools is not None:
            if len(messages) == 2:
                starts.append(messages[:2])
            prompt = next(p for p in SESSIONS if messages[1]["content"].startswith(p))
            messages = [messages[0], {"role": "user", "content": prompt}, *messages[2:]]
        return inner(messages, tools, **kwargs)

    run.run_experiment(TASK, ["consolidated", "consolidated@request"], 1, 4096, 700, out, chat, final_repeats=2)
    result = rows(out)
    assert len(result) == 4 and all(r["valid"] and r["passed"] for r in result)
    assert [r["memory"] for r in result] == [f"- RULE: {rule}"] * 4
    assert len({r["memory_tokens"] for r in result}) == 1
    assert result[0]["sleep_cost"] == result[2]["sleep_cost"]
    assert [(r["design"], r["store"], r["placement"], r["gate"]) for r in result] == [
        (d, "consolidated", p, False) for d, p in
        (("consolidated", "system"), ("consolidated@request", "request")) for _ in range(2)]
    prompts = {sid: p for p, sid in SESSIONS.items()}
    note = agent.MEMORY.format(memory=result[0]["memory"])
    sessions = tasks.load_task(TASK)["sessions"]
    assert starts[:2] == [[{"role": "system", "content": agent.SYSTEM.format(max_steps=s["max_steps"])},
                          {"role": "user", "content": s["prompt"]}] for s in sessions[:2]]
    system = agent.SYSTEM.format(max_steps=sessions[2]["max_steps"])
    assert starts[2:4] == [[{"role": "system", "content": system + note},
                           {"role": "user", "content": prompts["final"]}]] * 2
    assert starts[4:] == [[{"role": "system", "content": system},
                          {"role": "user", "content": prompts["final"] + note}]] * 2
    for design in ("consolidated", "consolidated_at_request"):
        for suffix in ("", ".r1"):
            log = out / "seed0" / f"final_{design}{suffix}.jsonl"
            assert json.loads(log.read_text(encoding="utf-8").splitlines()[0]) == {
                "type": "user", "text": prompts["final"]}
            assert (out / "seed0" / f"final_{design}_ws{suffix}").is_dir()


@pytest.mark.parametrize("spec", ["handoff+gate", "oracle@request+gate", "requests@tail+gate",
                                 "replay@request+gate", "dreaming@tail+gate"])
def test_specs_wire_final_and_practice_placement_and_gate_with_safe_paths(tmp_path, monkeypatch, spec):
    store, placement, gate = run.parse_design(spec)
    label = spec.replace("@", "_at_").replace("+", "_")
    real, seen = agent.run_session, []

    def spy(task_id, workspace, prompt, **kwargs):
        result = real(task_id, workspace, prompt, **kwargs)
        seen.append((Path(workspace), prompt, kwargs, result))
        return result
    monkeypatch.setattr(agent, "run_session", spy)
    note = json.dumps({"kind": "rule", "text": "Use milliseconds."})
    inner = scripted_agent({"s1": perfect("s1"), "s2": perfect("s2"),
                            "final": perfect("final") + [("finish", {"summary": "checked"})]}, note)

    def chat(messages, tools=None, **kwargs):
        if tools is not None:
            prompt = next(p for p in SESSIONS if messages[1]["content"].startswith(p))
            if any(m.get("tool_calls", [{}])[0].get("function", {}).get("name") == "finish"
                   for m in messages if m["role"] == "assistant"):
                return {"content": "", "tool_calls": [{"name": "finish", "arguments": {"summary": "checked"},
                                                       "raw": "{}"}],
                        "usage": {"prompt_tokens": 500, "completion_tokens": 50}, "seconds": 1.0,
                        "finish_reason": "tool_calls"}
            # The fake model sees the request alone and uses two messages to identify a new session.
            tail = messages[-1:] if placement == "tail" and len(messages) > 2 and messages[-1]["role"] == "user" else []
            messages = [messages[0], {"role": "user", "content": prompt}, *messages[2:len(messages) - len(tail)]]
        return inner(messages, tools, **kwargs)

    out = tmp_path / "out"
    run.run_experiment(TASK, [spec], 1, 4096, 700, out, chat, final_repeats=2)
    result = rows(out)
    assert [(r["design"], r["store"], r["placement"], r["gate"]) for r in result] == [
        (spec, store, placement, gate)] * 2
    assert all(r["passed"] for r in result)
    wakes, finals, practices = seen[:2], [], []
    for workspace, prompt, kwargs, session in seen[2:]:
        (finals if workspace.name == "ws_final" else practices).append((workspace, kwargs, session))
        assert kwargs["memory_at"] == placement and kwargs["finish_gate"] is gate
        assert session["gated"]
    assert len(finals) == 2
    assert all(kwargs["memory"] == "" and not kwargs.get("finish_gate", False) and not session["gated"]
               for _, _, kwargs, session in wakes)
    if store in ("replay", "dreaming"):
        assert practices
        assert all(workspace.parent.name in (f"practice_{label}_s1", f"practice_{label}_s2")
                   for workspace, _, _ in practices)
    else:
        assert practices == []
    for suffix in ("", ".r1"):
        assert (out / "seed0" / f"final_{label}{suffix}.jsonl").is_file()
        assert (out / "seed0" / f"final_{label}_ws{suffix}").is_dir()
    if store == "requests":
        prompts = [s["prompt"] for s in tasks.load_task(TASK)["sessions"][:2]]
        assert result[0]["memory"] == designs.inject("requests", prompts)
        assert result[0]["sleep_cost"]["calls"] == 0


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
    assert all(set(r) == {"exercises", "refined", "accepted", "items_before", "candidate"}
               and len(r["exercises"]) == 2 for r in replay + dreaming)
    entries = [(r, e) for r in replay + dreaming for e in r["exercises"]]
    assert all(set(e) == {"request", "hypothetical", "must_add", "must_not_add", "score", "score2", "check_passed",
                          "check_passed2", "missing", "present", "missing2", "present2"} for _, e in entries)
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
    run.run_experiment(TASK, ["workspace", "handoff"], 1, 4096, 700, out, chat, wake_retries=1, final_repeats=3)
    result = rows(out)
    assert [r["design"] for r in result] == ["workspace", "handoff"]
    assert all(not r["valid"] and r["wake_attempts"] == 2 for r in result)
    assert all("repeat" not in r for r in result)
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
    assert table.splitlines()[-1] == "final repeats: 1"


def test_summary_repeats(tmp_path):
    write_results(tmp_path, [summary_row(d, passed=r == 0, repeat=r, steps=2 + 4 * r)
                             for d in ("workspace", "handoff") for r in range(3)]
                  + [summary_row(d, seed=1, valid=False) for d in ("workspace", "handoff")])
    table = run.summarize(tmp_path)
    assert "| n valid | 1/2 | 1/2 |" in table
    assert "| completion rate | 0.33 | 0.33 |" in table
    assert "| mean steps | 6.0 | 6.0 |" in table
    assert "| wake validity | 0.75 | 0.75 |" in table  # metrics still average rows
    assert table.splitlines()[-1] == "final repeats: 3"


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
    monkeypatch.setattr(run, "run_experiment", lambda *args, **kwargs: seen.update(args=args, kwargs=kwargs))
    monkeypatch.setattr(run, "summarize", lambda out: "table")
    run.main(["--out", "somewhere"])
    task, designs_, seeds, window, reserve, out, chat = seen["args"]
    assert (task, designs_, seeds, window, reserve, str(out)) == (
        "settings_gen", ["workspace", "handoff"], 3, 4096, 1024, "somewhere")
    assert seen["kwargs"] == {"final_repeats": 1}


@pytest.mark.parametrize("count", [0, -1, 10])
def test_cli_rejects_repeats(monkeypatch, capsys, count):
    monkeypatch.setenv("DW_MODEL", "fake-model")
    with pytest.raises(SystemExit) as caught:
        run.main(["--final-repeats", str(count)])
    assert caught.value.code == 2
    assert "final-repeats must be between 1 and 9" in capsys.readouterr().err


@pytest.mark.parametrize("count", [1, 9])
def test_cli_forwards_repeats(monkeypatch, count):
    seen = {}
    monkeypatch.setenv("DW_MODEL", "fake-model")
    monkeypatch.setattr(run, "run_experiment", lambda *a, **kw: seen.update(kw))
    monkeypatch.setattr(run, "summarize", lambda out: "table")
    run.main(["--final-repeats", str(count), "--out", "somewhere"])
    assert seen == {"final_repeats": count}


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


def test_final_repeats(tmp_path, monkeypatch):
    out = tmp_path / "out"
    real, starts = agent.run_session, []

    def spy(task_id, workspace, prompt, **kwargs):
        if SESSIONS[prompt] == "final":
            starts.append((Path(workspace), kwargs["seed"], tree(workspace)))
        return real(task_id, workspace, prompt, **kwargs)

    monkeypatch.setattr(run.agent, "run_session", spy)
    trap = [w for w in writes("trap") if w[1]["path"] == "src/settings.py"]
    chat = scripted_agent({"s1": perfect("s1"), "s2": perfect("s2"),
                           "final": trap + [("write_file", {"path": "repeat.txt", "content": "changed"}),
                                            ("finish", {"summary": "done"})]})
    run.run_experiment(TASK, ["workspace", "handoff"], 2, 4096, 700, out, chat, final_repeats=3)
    result = rows(out)
    assert [(r["seed"], r["design"], r["repeat"]) for r in result] == [
        (k, d, r) for k in range(2) for d in ("workspace", "handoff") for r in range(3)]
    assert len(starts) == 12 and len(chat.sleep_seeds) == 4  # two sleeps per seed, only for handoff
    assert json.loads((out / "config.json").read_text(encoding="utf-8"))["args"]["final_repeats"] == 3
    for row, (workspace, seed, before) in zip(result, starts):
        k, design, r = row["seed"], row["design"], row["repeat"]
        seed_dir = out / f"seed{k}"
        assert workspace == seed_dir / "ws_final"
        assert seed == k * run.SEED_STRIDE + run.FINAL_OFFSET + r * run.SESSION_STRIDE
        assert before == tree(seed_dir / "ws_wake") and "repeat.txt" not in before
        suffix = f".r{r}" if r else ""
        log = seed_dir / f"final_{design}{suffix}.jsonl"
        assert log.read_text(encoding="utf-8").strip()
        archive = seed_dir / f"final_{design}_ws{suffix}"
        assert (archive / "repeat.txt").read_text(encoding="utf-8") == "changed"
        assert row["repeated_mistake"] and not row["passed"] and not row["checks"]["test_trap_avoided"]
        first = next(x for x in result if x["seed"] == k and x["design"] == design)
        for key in ("sleep_cost", "memory_chars", "memory_tokens", "memory", "practice_log", "contamination"):
            assert row[key] == first[key]
        assert not workspace.exists()


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
        run.run_experiment(TASK, ["workspace", "handoff"], 2, 4096, 700, out, chat, final_repeats=2)
    assert [(r["seed"], r["design"], r["repeat"]) for r in rows(out)] == [
        (0, d, r) for d in ("workspace", "handoff") for r in range(2)]


def test_an_existing_results_file_is_never_appended_to(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / "results.jsonl").write_text("old\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="results.jsonl"):
        run.run_experiment(TASK, ["workspace"], 1, 4096, 700, out, scripted_agent({}))
    assert (out / "results.jsonl").read_text(encoding="utf-8") == "old\n" and not (out / "config.json").exists()


def test_an_evaluator_error_during_the_wake_aborts_instead_of_failing_the_wake(tmp_path, monkeypatch):
    monkeypatch.setattr(run.tasks, "evaluate", lambda *a, **k: {"passed": False, "checks": {}, "error": "gen boom"})
    sessions = tasks.load_task(TASK)["sessions"]
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
