"""Hidden checks after session 3. Run with cwd = workspace copy, after tools/gen.py."""
import subprocess
import sys


def test_is_secure(monkeypatch):
    from src import client, settings
    assert client.is_secure() is False
    monkeypatch.setattr(settings, "BASE_URL", "https://example.test")
    assert client.is_secure() is True


def test_no_regression():
    visible = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests"],
                             capture_output=True, text=True)
    assert visible.returncode == 0, visible.stdout[-500:]


def test_audit_ok():
    assert subprocess.run([sys.executable, "tools/audit.py"]).returncode == 0
