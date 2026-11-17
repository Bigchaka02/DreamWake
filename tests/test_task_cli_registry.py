"""Proves the held-out cli_registry task fixture is valid against dreamwake.tasks."""
import json
import re
import shutil

import pytest

from dreamwake import tasks

TASK = "cli_registry"
FINAL_CHECKS = {"test_feature_works", "test_honors_revision", "test_trap_avoided",
                "test_convention_followed", "test_no_regression"}
UNKNOWN_HELLO = "LookupError: unknown command: hello"


@pytest.fixture
def ws(tmp_path):
    return tasks.make_workspace(TASK, tmp_path / "ws")


def manifest(ws):
    return json.loads((ws / "cli" / "manifest.json").read_text(encoding="utf-8"))


def append_commands(ws, source):
    path = ws / "cli" / "commands.py"
    path.write_text(path.read_text(encoding="utf-8") + source, encoding="utf-8")


def evaluate_final(ws, variant):
    tasks.apply_reference(TASK, "s2", ws)
    baseline = shutil.copytree(ws, ws.parent / "baseline")
    tasks.apply_reference(TASK, variant, ws)
    return tasks.evaluate(TASK, "final", ws, baseline_dir=baseline)


def test_task_json_mirrors_the_pattern_task():
    task = tasks.load_task(TASK)
    pattern = tasks.load_task("settings_gen")
    assert set(task) == set(pattern) - {"eval_setup"}  # nothing to prepare before scoring
    assert task["id"] == TASK
    assert [s["id"] for s in task["sessions"]] == ["s1", "s2", "final"]
    assert all(s["prompt"] and s["max_steps"] == 20 for s in task["sessions"])
    assert all(c.startswith("python ") for c in task["check_cmd"])
    assert task["check_cmd"][-1] == "python -m pytest -q tests"
    assert task["trap_paths"] == []
    assert task["protected"] == ["tests", "tools"]


def test_workspace_is_small_and_readme_implies_functions_are_enough(ws):
    files = [p for p in ws.rglob("*") if p.is_file()]
    assert len(files) <= 12
    for path in files:
        assert len(path.read_text(encoding="utf-8").splitlines()) <= 35, path
    readme = (ws / "README.md").read_text(encoding="utf-8")
    assert "Commands are plain functions in `cli/commands.py`." in readme
    assert "Each command returns the text to print." in readme  # goes stale in s2
    assert "manifest" not in readme.lower()
    assert not (ws / "hidden_tests").exists() and not list(ws.rglob("*.pyc"))
    assert sorted(p.name for p in (ws / "tools").iterdir()) == ["lint.py"]


def test_starting_snapshot_fails_on_unknown_hello_without_naming_the_manifest(ws):
    code, out = tasks.run_check(TASK, ws)
    assert code != 0
    assert "lint: ok" in out
    assert UNKNOWN_HELLO in out
    assert "manifest" not in out.lower()


def test_trap_writing_only_the_function_fails_the_same_way(ws):
    append_commands(ws, '\n\ndef cmd_hello():\n    return "Hello!"\n')
    code, out = tasks.run_check(TASK, ws)
    assert code != 0
    assert "lint: ok" in out and UNKNOWN_HELLO in out
    assert "manifest" not in out.lower()


def test_lint_names_an_unprefixed_command_function_and_stops_the_check(ws):
    append_commands(ws, '\n\ndef hello():\n    return "Hello!"\n')
    (ws / "cli" / "manifest.json").write_text(json.dumps({**manifest(ws), "hello": "hello"}),
                                              encoding="utf-8")
    code, out = tasks.run_check(TASK, ws)
    assert code != 0
    lint_lines = [line for line in out.splitlines() if line.startswith("lint: FAIL")]
    assert len(lint_lines) == 1 and "'hello'" in lint_lines[0] and "cmd_" in lint_lines[0]
    assert "pytest" not in out


def test_reference_s1_passes_checks_and_evaluation(ws):
    tasks.apply_reference(TASK, "s1", ws)
    code, out = tasks.run_check(TASK, ws)
    assert code == 0, out
    assert tasks.evaluate(TASK, "s1", ws) == {
        "passed": True, "checks": {"test_feature_works": True, "test_lint_ok": True}}


def test_reference_s2_passes_checks_and_evaluation(ws):
    tasks.apply_reference(TASK, "s2", ws)
    code, out = tasks.run_check(TASK, ws)
    assert code == 0, out
    result = tasks.evaluate(TASK, "s2", ws)
    assert result["passed"] and set(result["checks"]) == {
        "test_feature_works", "test_honors_revision", "test_lint_ok"}


def test_s1_state_does_not_honor_s2_revision(ws):
    tasks.apply_reference(TASK, "s1", ws)
    assert tasks.evaluate(TASK, "s2", ws)["checks"] == {
        "test_feature_works": True, "test_honors_revision": False, "test_lint_ok": True}


def test_reference_final_passes_all_checks(ws):
    result = evaluate_final(ws, "final")
    assert result["passed"], result
    assert set(result["checks"]) == FINAL_CHECKS
    code, out = tasks.run_check(TASK, ws)
    assert code == 0, out


# no_convention fails only test_convention_followed: render, the dict and the
# registration are all right, and the visible tests never call the lint.
@pytest.mark.parametrize("variant, expected", [
    ("trap", {"test_trap_avoided": False, "test_feature_works": False}),
    ("stale", {"test_honors_revision": False, "test_feature_works": True}),
    ("no_convention", {name: name != "test_convention_followed" for name in FINAL_CHECKS}),
])
def test_wrong_final_variants_fail_the_right_checks(ws, variant, expected):
    result = evaluate_final(ws, variant)
    assert not result["passed"]
    assert {name: result["checks"][name] for name in expected} == expected


def test_stale_final_passes_the_visible_checks_so_the_revision_is_silent(ws):
    tasks.apply_reference(TASK, "stale", ws)
    code, out = tasks.run_check(TASK, ws)
    assert code == 0, out


def test_final_accepts_any_cmd_prefixed_function_name(ws):
    tasks.apply_reference(TASK, "s2", ws)
    baseline = shutil.copytree(ws, ws.parent / "baseline")
    append_commands(ws, '\n\ndef cmd_show_version():\n    return {"ok": True, "out": "1.4.0"}\n')
    (ws / "cli" / "manifest.json").write_text(
        json.dumps({**manifest(ws), "version": "cmd_show_version"}), encoding="utf-8")
    assert tasks.evaluate(TASK, "final", ws, baseline_dir=baseline)["passed"]


def test_evaluate_does_not_modify_the_workspace(ws):
    tasks.apply_reference(TASK, "s2", ws)
    baseline = shutil.copytree(ws, ws.parent / "baseline")
    tasks.apply_reference(TASK, "no_convention", ws)
    before = {p: p.read_bytes() for p in ws.rglob("*") if p.is_file()}
    tasks.evaluate(TASK, "final", ws, baseline_dir=baseline)
    after = {p: p.read_bytes() for p in ws.rglob("*") if p.is_file()}
    assert after == before


def test_evaluate_runs_the_pristine_lint_not_the_agents(ws):
    tasks.apply_reference(TASK, "s2", ws)
    baseline = shutil.copytree(ws, ws.parent / "baseline")
    tasks.apply_reference(TASK, "no_convention", ws)
    (ws / "tools" / "lint.py").write_text("print('lint: ok')\n", encoding="utf-8")
    checks = tasks.evaluate(TASK, "final", ws, baseline_dir=baseline)["checks"]
    assert checks["test_convention_followed"] is False


def test_failing_check_output_stays_under_the_cap(ws):
    code, out = tasks.run_check(TASK, ws)
    assert code != 0 and len(out) <= tasks.OUTPUT_CAP


def test_final_prompt_does_not_give_away_the_three_knowledge_items():
    prompt = tasks.load_task(TASK)["sessions"][2]["prompt"].lower()
    for word in ("manifest", "regist", "json", "cmd_", "prefix", "lint", "dict", "ok", "out"):
        assert not re.search(rf"\b{re.escape(word)}", prompt), word
    assert "version" in prompt and "1.4.0" in prompt
