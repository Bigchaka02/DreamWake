"""Proves the held-out catalog_legacy task fixture and its evaluator are valid."""
import json
import re
import shutil

import pytest

from dreamwake import tasks

TASK = "catalog_legacy"
FINAL_CHECKS = {"test_feature_works", "test_honors_revision", "test_trap_avoided",
                "test_convention_followed", "test_no_regression"}


@pytest.fixture
def ws(tmp_path):
    return tasks.make_workspace(TASK, tmp_path / "ws")


def write_catalog(ws, catalog):
    (ws / "locale" / "en.json").write_text(json.dumps(catalog, indent=2), encoding="utf-8")


def evaluate_final(ws, variant):
    tasks.apply_reference(TASK, "s2", ws)
    baseline = shutil.copytree(ws, ws.parent / "baseline")
    tasks.apply_reference(TASK, variant, ws)
    return tasks.evaluate(TASK, "final", ws, baseline_dir=baseline)


def test_task_json_describes_three_sessions():
    task = tasks.load_task(TASK)
    assert task["id"] == TASK
    assert [s["id"] for s in task["sessions"]] == ["s1", "s2", "final"]
    assert all(s["prompt"] and s["max_steps"] == 20 for s in task["sessions"])
    assert task["check_cmd"] == ["python tools/lint.py", "python -m pytest -q tests"]
    assert task["trap_paths"] == ["app/messages.py"]
    assert task["protected"] == ["tests", "tools"]
    assert "eval_setup" not in task and not (task["workspace"] / "tools" / "gen.py").exists()


def test_workspace_files_are_short_and_do_not_reveal_the_dead_module(ws):
    files = [p for p in ws.rglob("*") if p.is_file()]
    assert len(files) <= 12
    for path in files:
        assert len(path.read_text(encoding="utf-8").splitlines()) <= 35, path
    messages_py = (ws / "app" / "messages.py").read_text(encoding="utf-8").lower()
    for word in ("legacy", "unused", "dead", "deprecated", "not used", "locale", "json"):
        assert word not in messages_py, word
    readme = (ws / "README.md").read_text(encoding="utf-8")
    assert "Messages are defined in `app/messages.py`." in readme
    assert "Message keys are single lowercase words." in readme
    assert "locale" not in readme and "legacy" not in readme.lower()
    assert not (ws / "hidden_tests").exists() and not list(ws.rglob("*.pyc"))


def test_starting_snapshot_fails_on_missing_greeting(ws):
    code, out = tasks.run_check(TASK, ws)
    assert code != 0
    assert "lint: ok" in out
    assert "KeyError: 'greeting'" in out


def test_trap_editing_the_legacy_module_changes_nothing(ws):
    messages_py = ws / "app" / "messages.py"
    messages_py.write_text(messages_py.read_text(encoding="utf-8").replace(
        '    "title": "Notes",\n', '    "greeting": "Hello!",\n    "title": "Notes",\n'), encoding="utf-8")
    assert '"greeting"' in messages_py.read_text(encoding="utf-8")
    code, out = tasks.run_check(TASK, ws)
    assert code != 0
    assert "KeyError: 'greeting'" in out


def test_lint_names_the_first_out_of_order_pair_and_stops_the_check(ws):
    write_catalog(ws, {"title": "Notes", "greeting": "Hello!", "about": "x"})
    code, out = tasks.run_check(TASK, ws)
    assert code != 0
    error_lines = [line for line in out.splitlines() if line.startswith("lint: FAIL")]
    assert error_lines == ["lint: FAIL - locale/en.json keys not sorted: 'greeting' must come before 'title'"]
    assert "pytest" not in out


def test_reference_s1_passes_checks_and_evaluation(ws):
    tasks.apply_reference(TASK, "s1", ws)
    code, out = tasks.run_check(TASK, ws)
    assert code == 0, out
    result = tasks.evaluate(TASK, "s1", ws)
    assert result == {"passed": True, "checks": {"test_feature_works": True, "test_lint_ok": True}}


def test_s1_fix_in_a_hardcoded_string_does_not_count(ws):
    ui = ws / "app" / "ui.py"
    ui.write_text(ui.read_text(encoding="utf-8").replace('t("greeting")', '"Hello!"'), encoding="utf-8")
    assert tasks.run_check(TASK, ws)[0] == 0
    assert tasks.evaluate(TASK, "s1", ws)["checks"]["test_feature_works"] is False


def test_reference_s2_passes_checks_and_evaluation(ws):
    tasks.apply_reference(TASK, "s2", ws)
    code, out = tasks.run_check(TASK, ws)
    assert code == 0, out
    result = tasks.evaluate(TASK, "s2", ws)
    assert result == {"passed": True, "checks": {"test_feature_works": True, "test_honors_revision": True,
                                                 "test_lint_ok": True}}


def test_s1_state_does_not_honor_s2_revision(ws):
    tasks.apply_reference(TASK, "s1", ws)
    checks = tasks.evaluate(TASK, "s2", ws)["checks"]
    assert checks == {"test_feature_works": True, "test_honors_revision": False, "test_lint_ok": True}


def test_reference_final_passes_all_checks(ws):
    result = evaluate_final(ws, "final")
    assert result["passed"], result
    assert set(result["checks"]) == FINAL_CHECKS
    code, out = tasks.run_check(TASK, ws)
    assert code == 0, out


def test_final_accepts_any_ui_prefixed_key_name(ws):
    tasks.apply_reference(TASK, "s2", ws)
    baseline = shutil.copytree(ws, ws.parent / "baseline")
    tasks.apply_reference(TASK, "final", ws)
    write_catalog(ws, {"ui.bye": "Goodbye!", "ui.greeting": "Hello!", "ui.title": "Notes"})
    ui = ws / "app" / "ui.py"
    ui.write_text(ui.read_text(encoding="utf-8").replace('t("ui.farewell")', 't("ui.bye")'), encoding="utf-8")
    assert tasks.evaluate(TASK, "final", ws, baseline_dir=baseline)["passed"]


@pytest.mark.parametrize("variant, expected", [
    ("trap", {"test_trap_avoided": False, "test_feature_works": False}),
    ("stale", {"test_honors_revision": False, "test_feature_works": True}),
    ("no_convention", {name: name != "test_convention_followed" for name in FINAL_CHECKS}),
])
def test_wrong_final_variants_fail_the_right_checks(ws, variant, expected):
    result = evaluate_final(ws, variant)
    assert not result["passed"]
    assert {name: result["checks"][name] for name in expected} == expected


def test_reviving_the_legacy_module_fails_trap_avoided(ws):
    tasks.apply_reference(TASK, "s2", ws)
    baseline = shutil.copytree(ws, ws.parent / "baseline")
    tasks.apply_reference(TASK, "final", ws)
    write_catalog(ws, {"ui.greeting": "Hello!", "ui.title": "Notes"})
    (ws / "app" / "messages.py").write_text('MESSAGES = {"title": "Notes", "ui.farewell": "Goodbye!"}\n',
                                            encoding="utf-8")
    ui = ws / "app" / "ui.py"
    ui.write_text(ui.read_text(encoding="utf-8").replace(
        'return t("ui.farewell")', 'from .messages import MESSAGES\n    return MESSAGES["ui.farewell"]'),
        encoding="utf-8")
    checks = tasks.evaluate(TASK, "final", ws, baseline_dir=baseline)["checks"]
    assert checks["test_feature_works"] is True and checks["test_trap_avoided"] is False


def test_evaluate_does_not_modify_the_workspace(ws):
    tasks.apply_reference(TASK, "s2", ws)
    baseline = shutil.copytree(ws, ws.parent / "baseline")
    tasks.apply_reference(TASK, "trap", ws)
    before = {p: p.read_bytes() for p in ws.rglob("*") if p.is_file()}
    tasks.evaluate(TASK, "final", ws, baseline_dir=baseline)
    after = {p: p.read_bytes() for p in ws.rglob("*") if p.is_file()}
    assert after == before


def test_run_check_output_is_capped_and_keeps_the_tail(ws):
    code, out = tasks.run_check(TASK, ws)
    assert code != 0 and len(out) <= tasks.OUTPUT_CAP
    assert out.rstrip().endswith("checks FAILED (exit 1)")


def test_evaluate_runs_pristine_lint_not_the_agents(ws):
    tasks.apply_reference(TASK, "s1", ws)
    write_catalog(ws, {"title": "Notes", "greeting": "Hello!"})
    (ws / "tools" / "lint.py").write_text("print('lint: ok')\n", encoding="utf-8")
    assert tasks.run_check(TASK, ws)[0] == 0
    assert tasks.evaluate(TASK, "s1", ws)["checks"]["test_lint_ok"] is False


def test_final_prompt_gives_away_none_of_the_three_knowledge_items():
    prompt = tasks.load_task(TASK)["sessions"][2]["prompt"].lower()
    for word in ("locale", "en.json", "json", "catalog", "messages", "legacy", "i18n", "sort", "order",
                 "alphabet", "lint", "prefix", "namespace", "dotted", "key", "`ui.farewell`", "'ui."):
        assert not re.search(rf"(?<![\w.]){re.escape(word)}", prompt), word
    assert "goodbye!" in prompt and "app.ui.farewell()" in prompt
