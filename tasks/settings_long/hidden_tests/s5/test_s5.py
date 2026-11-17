"""Cumulative wake validation. Run with cwd = workspace copy, after tools/gen.py."""
import json
import subprocess
import sys
from pathlib import Path


def test_describe():
    from src import retry
    assert retry.describe() == "retries: 3"


def test_timeout():
    from src import client
    assert client.request_timeout_seconds() == 30.0


def test_honors_revision():
    template = json.loads(Path("settings.tmpl.json").read_text(encoding="utf-8"))
    assert float(template["TIMEOUT_MS"]["value"]) == 30000
    assert "TIMEOUT_S" not in template


def test_is_secure(monkeypatch):
    from src import client, settings
    assert client.is_secure() is False
    monkeypatch.setattr(settings, "BASE_URL", "https://example.test")
    assert client.is_secure() is True


def test_service_name():
    from src import client
    assert client.service_name() == "dreamwake"


def test_service_doc():
    template = json.loads(Path("settings.tmpl.json").read_text(encoding="utf-8"))
    doc = template["SERVICE_NAME"].get("doc")
    assert isinstance(doc, str) and doc.strip()


def test_audit_ok():
    assert subprocess.run([sys.executable, "tools/audit.py"]).returncode == 0


def test_no_regression():
    visible = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests"],
                             capture_output=True, text=True)
    assert visible.returncode == 0, visible.stdout[-500:]
