"""Hidden checks after session 1. Run with cwd = workspace copy."""
import subprocess
import sys


def test_feature_works():
    from app import i18n, ui
    assert ui.greeting() == "Hello!"
    assert i18n.t("greeting") == "Hello!"


def test_lint_ok():
    assert subprocess.run([sys.executable, "tools/lint.py"]).returncode == 0
