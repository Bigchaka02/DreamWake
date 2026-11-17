"""Hidden checks after session 2. Run with cwd = workspace copy."""
import json
import subprocess
import sys
from pathlib import Path


def test_feature_works():
    from app import ui
    assert ui.greeting() == "Hello!" and ui.title() == "Notes"


def test_honors_revision():
    catalog = json.loads(Path("locale/en.json").read_text(encoding="utf-8"))
    assert catalog.get("ui.greeting") == "Hello!" and catalog.get("ui.title") == "Notes"
    assert "greeting" not in catalog and "title" not in catalog


def test_lint_ok():
    assert subprocess.run([sys.executable, "tools/lint.py"]).returncode == 0
