"""Hidden checks after session 4. Run with cwd = workspace copy, after tools/gen.py."""
import json
import subprocess
import sys
from pathlib import Path


def test_service_name():
    from src import client
    assert client.service_name() == "dreamwake"


def test_service_doc():
    template = json.loads(Path("settings.tmpl.json").read_text(encoding="utf-8"))
    doc = template["SERVICE_NAME"].get("doc")
    assert isinstance(doc, str) and doc.strip()


def test_audit_ok():
    assert subprocess.run([sys.executable, "tools/audit.py"]).returncode == 0
