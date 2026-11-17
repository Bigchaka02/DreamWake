"""Hidden checks after session 1. Run with cwd = workspace copy."""
import subprocess
import sys


def test_feature_works():
    from cli import core
    assert core.render("hello") == "Hello!"


def test_lint_ok():
    assert subprocess.run([sys.executable, "tools/lint.py"]).returncode == 0
