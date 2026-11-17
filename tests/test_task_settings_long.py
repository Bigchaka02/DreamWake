"""Validate the long policy-retention task without a model."""
import json
import shutil

import pytest

from dreamwake import run, tasks

TASK = "settings_long"
S3_CHECKS = {"test_is_secure", "test_no_regression", "test_audit_ok"}
S4_CHECKS = {"test_service_name", "test_service_doc", "test_audit_ok"}
S5_CHECKS = {"test_describe", "test_timeout", "test_honors_revision", "test_is_secure",
             "test_service_name", "test_service_doc", "test_audit_ok", "test_no_regression"}
FINAL_CHECKS = {"test_feature_works", "test_honors_revision", "test_trap_avoided",
                "test_convention_followed", "test_no_regression", "test_honors_second_policy"}


@pytest.fixture
def ws(tmp_path):
    return tasks.make_workspace(TASK, tmp_path / "ws")


def template(ws):
    return json.loads((ws / "settings.tmpl.json").read_text(encoding="utf-8"))


def write_template(ws, value):
    (ws / "settings.tmpl.json").write_text(json.dumps(value, indent=2), encoding="utf-8")


def assert_checks(result, names, failed=()):
    assert result == {"passed": not bool(failed),
                      "checks": {name: name not in failed for name in names}}


def test_long_loads_inheritance_and_exact_sessions():
    task, base = tasks.load_task(TASK), tasks.load_task("settings_gen")
    assert task["id"] == TASK and task["base"] == "settings_gen"
    for key in ("check_cmd", "trap_paths", "protected", "eval_setup", "workspace", "oracle_memory"):
        assert task[key] == base[key]
    assert task["reference"] == task["dir"] / "reference"
    assert [s["id"] for s in task["sessions"]] == ["s1", "s2", "s3", "s4", "s5", "final"]
    assert all(s["max_steps"] == 20 and set(s) == {"id", "prompt", "max_steps"}
               for s in task["sessions"])
    for child_index, base_index in ((0, 0), (1, 1), (5, 2)):
        assert task["sessions"][child_index] == base["sessions"][base_index]
    assert task["sessions"][2]["prompt"] == (
        'Add `client.is_secure()` that returns True when the configured base URL starts with "https://" '
        'and False otherwise. Make the checks pass.')
    assert task["sessions"][3]["prompt"] == (
        'Add a setting named SERVICE_NAME with the value "dreamwake", exposed through '
        '`client.service_name()` returning that string. Team decision, from now on: every new entry '
        'in settings.tmpl.json also carries a "doc" field with a one-sentence description of the setting. '
        'Existing entries stay as they are. Make the checks pass.')
    assert task["sessions"][4]["prompt"] == (
        'Add `retry.describe()` that returns the string "retries: N" where N is the configured number '
        'of attempts. Make the checks pass.')
    assert {p.name for p in task["hidden_tests"].iterdir()} == {"s3", "s4", "s5", "final"}


@pytest.mark.parametrize("override_parent, local_folder", [(False, False), (False, True),
                                                          (True, False), (True, True)])
def test_reference_parent_and_folder_resolve_independently(tmp_path, monkeypatch,
                                                         override_parent, local_folder):
    root = tmp_path / "tasks"
    base, child = root / "base", root / "child"
    for directory, metadata in ((base, {"id": "base"}), (child, {"id": "child", "base": "base"})):
        (directory / "reference").mkdir(parents=True)
        (directory / "task.json").write_text(json.dumps(metadata), encoding="utf-8")
    reference = base / "reference"
    (reference / "parents.json").write_text('{"seed": null, "ready": "seed"}', encoding="utf-8")
    for name in ("seed", "ready"):
        (reference / name).mkdir()
    (reference / "seed" / "seed.txt").write_text("seed", encoding="utf-8")
    (reference / "ready" / "state.txt").write_text("base", encoding="utf-8")
    if override_parent:
        (child / "reference" / "parents.json").write_text('{"ready": null}', encoding="utf-8")
    if local_folder:
        (child / "reference" / "ready").mkdir()
        (child / "reference" / "ready" / "state.txt").write_text("child", encoding="utf-8")
    monkeypatch.setattr(tasks, "TASKS_DIR", root)
    ws = tmp_path / "ws"
    tasks.apply_reference("child", "ready", ws)
    assert (ws / "state.txt").read_text(encoding="utf-8") == ("child" if local_folder else "base")
    assert (ws / "seed.txt").exists() is not override_parent


def test_gen_accepts_doc_without_changing_the_tool(tmp_path):
    ws = tasks.make_workspace("settings_gen", tmp_path / "ws")
    tasks.apply_reference("settings_gen", "s1", ws)
    value = template(ws)
    value["TIMEOUT_S"]["doc"] = "Request timeout in seconds."
    write_template(ws, value)
    code, out = tasks.run_check("settings_gen", ws)
    assert code == 0, out
    assert "TIMEOUT_S = 30\n" in (ws / "src" / "settings.py").read_text(encoding="utf-8")


@pytest.mark.parametrize("state, stage, names, failed", [
    ("s1", "s1", {"test_feature_works", "test_audit_ok"}, set()),
    ("s2", "s2", {"test_feature_works", "test_honors_revision", "test_audit_ok"}, set()),
    ("s2", "s3", S3_CHECKS, {"test_is_secure"}),
    ("s3", "s3", S3_CHECKS, set()),
    ("s3", "s4", S4_CHECKS, {"test_service_name", "test_service_doc"}),
    ("s4", "s4", S4_CHECKS, set()),
    ("s4", "s5", S5_CHECKS, {"test_describe"}),
    ("s5", "s5", S5_CHECKS, set()),
])
def test_stage_transitions_fail_only_missing_work(ws, state, stage, names, failed):
    tasks.apply_reference(TASK, state, ws)
    assert_checks(tasks.evaluate(TASK, stage, ws), names, failed)


@pytest.mark.parametrize("mutation, failed", [
    ("seconds", {"test_honors_revision"}),
    ("no_doc", {"test_service_doc"}),
    ("no_secure", {"test_is_secure"}),
    ("constant_secure", {"test_is_secure"}),
    ("wrong_timeout", {"test_honors_revision"}),
])
def test_s5_rejects_each_undone_decision(ws, mutation, failed):
    tasks.apply_reference(TASK, "s5", ws)
    value = template(ws)
    client = ws / "src" / "client.py"
    source = client.read_text(encoding="utf-8")
    if mutation == "seconds":
        del value["TIMEOUT_MS"]
        value["TIMEOUT_S"] = {"type": "int", "value": 30}
        source = source.replace("settings.TIMEOUT_MS / 1000", "float(settings.TIMEOUT_S)")
        changes = ws / "CHANGES.md"
        changes.write_text(changes.read_text(encoding="utf-8").replace("TIMEOUT_MS", "TIMEOUT_S"),
                           encoding="utf-8")
    elif mutation == "no_doc":
        del value["SERVICE_NAME"]["doc"]
    elif mutation == "no_secure":
        source = source.replace("def is_secure():", "def removed_is_secure():")
    elif mutation == "constant_secure":
        source = source.replace('settings.BASE_URL.startswith("https://")', "False")
    else:
        value["TIMEOUT_MS"]["value"] = 3000
        source = source.replace("settings.TIMEOUT_MS / 1000", "30.0")
    write_template(ws, value)
    client.write_text(source, encoding="utf-8")
    assert_checks(tasks.evaluate(TASK, "s5", ws), S5_CHECKS, failed)


def test_s5_compares_timeout_numerically(ws):
    tasks.apply_reference(TASK, "s5", ws)
    value = template(ws)
    value["TIMEOUT_MS"] = {"type": "str", "value": "30000"}
    write_template(ws, value)
    client = ws / "src" / "client.py"
    client.write_text(client.read_text(encoding="utf-8").replace(
        "settings.TIMEOUT_MS / 1000", "float(settings.TIMEOUT_MS) / 1000"), encoding="utf-8")
    assert_checks(tasks.evaluate(TASK, "s5", ws), S5_CHECKS)


@pytest.mark.parametrize("state, failed", [
    ("final", set()),
    ("stale_ms", {"test_honors_revision"}),
    ("no_doc", {"test_honors_second_policy"}),
])
def test_final_variants_fail_exactly_the_ignored_policy(ws, state, failed):
    tasks.apply_reference(TASK, "s5", ws)
    baseline = shutil.copytree(ws, ws.parent / "baseline")
    tasks.apply_reference(TASK, state, ws)
    assert_checks(tasks.evaluate(TASK, "final", ws, baseline_dir=baseline), FINAL_CHECKS, failed)
    code, out = tasks.run_check(TASK, ws)
    assert code == 0, out


def test_starting_workspace_fails_all_final_checks(ws):
    baseline = shutil.copytree(ws, ws.parent / "baseline")
    assert_checks(tasks.evaluate(TASK, "final", ws, baseline_dir=baseline), FINAL_CHECKS, FINAL_CHECKS)


def test_references_contain_only_changes_from_parent(tmp_path):
    task = tasks.load_task(TASK)
    parents = json.loads((task["reference"] / "parents.json").read_text(encoding="utf-8"))
    assert parents == {"s3": "s2", "s4": "s3", "s5": "s4", "final": "s5",
                       "stale_ms": "s5", "no_doc": "s5"}
    for state, parent in parents.items():
        ws = tasks.make_workspace(TASK, tmp_path / state)
        tasks.apply_reference(TASK, parent, ws)
        paths = [p for p in (task["reference"] / state).rglob("*") if p.is_file()]
        assert paths
        for path in paths:
            inherited = ws / path.relative_to(task["reference"] / state)
            assert not inherited.exists() or path.read_bytes() != inherited.read_bytes(), path


def test_scripted_run_executes_six_sessions_and_five_sleeps(tmp_path):
    task = tasks.load_task(TASK)
    sessions = task["sessions"]
    scripts = {}
    for session in sessions:
        root = tasks.load_task("settings_gen")["reference"] if session["id"] in ("s1", "s2") else task["reference"]
        calls = []
        for path in sorted((root / session["id"]).rglob("*")):
            if path.is_file():
                relative = path.relative_to(root / session["id"]).as_posix()
                calls += [("read_file", {"path": relative}),
                          ("write_file", {"path": relative, "content": path.read_text(encoding="utf-8")})]
        scripts[session["prompt"]] = calls + [("run_tests", {}), ("finish", {"summary": "done"})]
    queue, starts, sleeps = [], [], []

    def chat(messages, tools=None, *, seed=None, **kwargs):
        if tools is None:
            sleeps.append((seed, messages))
            return {"content": "Keep both team policies.", "tool_calls": [],
                    "usage": {"prompt_tokens": 80, "completion_tokens": 30},
                    "seconds": 1.0, "finish_reason": "stop"}
        request = next(m["content"] for m in messages if m["role"] == "user")
        if not queue:
            starts.append(request)
            queue[:] = scripts[request]
        name, args = queue.pop(0)
        return {"content": "", "tool_calls": [{"name": name, "arguments": args, "raw": json.dumps(args)}],
                "usage": {"prompt_tokens": 500, "completion_tokens": 50}, "seconds": 1.0,
                "finish_reason": "tool_calls"}

    out = tmp_path / "out"
    run.run_experiment(TASK, ["handoff"], 1, 4096, 700, out, chat, wake_retries=0)
    assert starts == [s["prompt"] for s in sessions]
    assert [seed for seed, _ in sleeps] == [run.SLEEP_OFFSET + i * run.SESSION_STRIDE for i in range(5)]
    for (_, messages), session in zip(sleeps, sessions[:-1]):
        assert session["prompt"] in "\n".join(m["content"] for m in messages)
    [row] = [json.loads(line) for line in (out / "results.jsonl").read_text(encoding="utf-8").splitlines()]
    assert row["valid"] and row["passed"] and row["wake_attempts"] == 1
    assert row["wake"] == {**{f"s{i}_passed": True for i in range(1, 6)}, "trap_experienced": False}
    assert row["checks"] == dict.fromkeys(FINAL_CHECKS, True)
    assert row["sleep_cost"]["calls"] == 5
    for session in sessions[:-1]:
        snapshot = out / "seed0" / f"snap_{session['id']}_after"
        assert tasks.evaluate(TASK, session["id"], snapshot)["passed"]
