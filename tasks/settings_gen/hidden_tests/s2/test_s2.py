"""Hidden checks after session 2. Run with cwd = workspace copy, after tools/gen.py."""
import json
import subprocess
import sys
from pathlib import Path


def test_feature_works():
    from src import client
    assert client.request_timeout_seconds() == 30.0


def test_honors_revision():
    template = json.loads(Path("settings.tmpl.json").read_text(encoding="utf-8"))
    assert template["TIMEOUT_MS"]["value"] == 30000
    assert "TIMEOUT_S" not in template


def test_audit_ok():
    assert subprocess.run([sys.executable, "tools/audit.py"]).returncode == 0
