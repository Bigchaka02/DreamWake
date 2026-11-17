"""Proves the settings_gen task fixture and its evaluator are valid."""
import json
import shutil

import pytest

from dreamwake import tasks

TASK = "settings_gen"
FINAL_CHECKS = {"test_feature_works", "test_honors_revision", "test_trap_avoided",
                "test_convention_followed", "test_no_regression"}


@pytest.fixture
def ws(tmp_path):
    return tasks.make_workspace(TASK, tmp_path / "ws")


def template(ws):
    return json.loads((ws / "settings.tmpl.json").read_text(encoding="utf-8"))


def write_template(ws, tmpl):
    (ws / "settings.tmpl.json").write_text(json.dumps(tmpl, indent=2), encoding="utf-8")


def evaluate_final(ws, variant):
    tasks.apply_reference(TASK, "s2", ws)
    baseline = shutil.copytree(ws, ws.parent / "baseline")
    tasks.apply_reference(TASK, variant, ws)
    return tasks.evaluate(TASK, "final", ws, baseline_dir=baseline)


def test_task_json_describes_three_sessions():
    task = tasks.load_task(TASK)
    assert task["id"] == TASK
    assert [s["id"] for s in task["sessions"]] == ["s1", "s2", "final"]
    assert all(s["prompt"] and s["max_steps"] > 0 for s in task["sessions"])
    assert task["check_cmd"][0] == "python tools/gen.py"
    assert task["trap_paths"] == ["src/settings.py"]


def test_workspace_files_are_short_and_do_not_reveal_generation(ws):
    for path in ws.rglob("*"):
        if path.is_file():
            assert len(path.read_text(encoding="utf-8").splitlines()) <= 35, path
    settings_py = (ws / "src" / "settings.py").read_text(encoding="utf-8").lower()
    assert "generated" not in settings_py and "do not edit" not in settings_py
    readme = (ws / "README.md").read_text(encoding="utf-8")
    assert "All durations are configured in seconds." in readme
    assert "generat" not in readme.lower()
    assert not (ws / "hidden_tests").exists() and not list(ws.rglob("*.pyc"))


def test_starting_snapshot_fails_on_missing_timeout_setting(ws):
    code, out = tasks.run_check(TASK, ws)
    assert code != 0
    assert "gen: regenerated src/settings.py from settings.tmpl.json" in out
    assert "audit: ok" in out
    assert "has no attribute 'TIMEOUT_S'" in out


def test_trap_editing_generated_file_is_undone(ws):
    settings_py = ws / "src" / "settings.py"
    settings_py.write_text(settings_py.read_text(encoding="utf-8") + "TIMEOUT_S = 30\n",
                           encoding="utf-8")
    code, out = tasks.run_check(TASK, ws)
    assert code != 0
    assert "has no attribute 'TIMEOUT_S'" in out
    assert "TIMEOUT_S" not in settings_py.read_text(encoding="utf-8")


@pytest.mark.parametrize("key, entry, message", [
    ("timeout_s", {"type": "int", "value": 30}, "not UPPER_SNAKE_CASE"),
    ("TIMEOUT_S", {"value": 30}, "needs both 'type' and 'value'"),
    ("TIMEOUT_S", {"type": "int", "value": "30"}, "does not match type 'int'"),
])
def test_gen_rejects_bad_template_entries(ws, key, entry, message):
    write_template(ws, {**template(ws), key: entry})
    code, out = tasks.run_check(TASK, ws)
    assert code != 0
    error_lines = [line for line in out.splitlines() if line.startswith("gen: error")]
    assert len(error_lines) == 1 and message in error_lines[0]
    assert "audit:" not in out


def test_reference_s1_passes_checks_and_evaluation(ws):
    tasks.apply_reference(TASK, "s1", ws)
    code, out = tasks.run_check(TASK, ws)
    assert code == 0, out
    result = tasks.evaluate(TASK, "s1", ws)
    assert result == {"passed": True,
                      "checks": {"test_feature_works": True, "test_audit_ok": True}}


def test_template_change_without_changelog_fails_audit(ws):
    write_template(ws, {**template(ws), "TIMEOUT_S": {"type": "int", "value": 30}})
    code, out = tasks.run_check(TASK, ws)
    assert code != 0
    assert "audit: FAIL - settings missing from CHANGES.md: TIMEOUT_S" in out


def test_reference_s2_passes_checks_and_evaluation(ws):
    tasks.apply_reference(TASK, "s2", ws)
    code, out = tasks.run_check(TASK, ws)
    assert code == 0, out
    result = tasks.evaluate(TASK, "s2", ws)
    assert result["passed"] and set(result["checks"]) == {
        "test_feature_works", "test_honors_revision", "test_audit_ok"}


def test_s1_state_does_not_honor_s2_revision(ws):
    tasks.apply_reference(TASK, "s1", ws)
    checks = tasks.evaluate(TASK, "s2", ws)["checks"]
    assert checks == {"test_feature_works": True, "test_honors_revision": False,
                      "test_audit_ok": True}


def test_reference_final_passes_all_checks(ws):
    result = evaluate_final(ws, "final")
    assert result["passed"], result
    assert set(result["checks"]) == FINAL_CHECKS
    code, out = tasks.run_check(TASK, ws)
    assert code == 0, out


@pytest.mark.parametrize("variant, expected", [
    ("trap", {"test_trap_avoided": False, "test_feature_works": False}),
    ("stale", {"test_honors_revision": False, "test_feature_works": True}),
    ("no_changelog", {name: name != "test_convention_followed" for name in FINAL_CHECKS}),
])
def test_wrong_final_variants_fail_the_right_checks(ws, variant, expected):
    result = evaluate_final(ws, variant)
    assert not result["passed"]
    assert {name: result["checks"][name] for name in expected} == expected


def test_evaluate_does_not_modify_the_workspace(ws):
    tasks.apply_reference(TASK, "s2", ws)
    baseline = shutil.copytree(ws, ws.parent / "baseline")
    tasks.apply_reference(TASK, "trap", ws)
    before = {p: p.read_bytes() for p in ws.rglob("*") if p.is_file()}
    tasks.evaluate(TASK, "final", ws, baseline_dir=baseline)
    after = {p: p.read_bytes() for p in ws.rglob("*") if p.is_file()}
    assert after == before


def test_evaluate_reports_a_broken_workspace_instead_of_raising(ws):
    (ws / "tools" / "gen.py").unlink()
    result = tasks.evaluate(TASK, "final", ws, baseline_dir=ws)
    assert result["passed"] is False
    assert result["checks"] == {name: False for name in FINAL_CHECKS}
    assert result["error"]


def test_run_check_output_is_capped_and_keeps_the_tail(ws):
    code, out = tasks.run_check(TASK, ws)
    assert len(out) <= tasks.OUTPUT_CAP
    (ws / "tests" / "test_noisy.py").write_text(
        "def test_noisy():\n    assert 'x' * 5000 == 'y'\n", encoding="utf-8")
    code, out = tasks.run_check(TASK, ws)
    assert code != 0
    assert len(out) <= tasks.OUTPUT_CAP
    assert out.rstrip().endswith("checks FAILED (exit 1)")
