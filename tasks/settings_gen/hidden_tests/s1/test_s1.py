"""Hidden checks after session 1. Run with cwd = workspace copy, after tools/gen.py."""
import subprocess
import sys


def test_feature_works():
    from src import client, settings
    assert client.request_timeout_seconds() == 30.0
    assert settings.TIMEOUT_S == 30


def test_audit_ok():
    assert subprocess.run([sys.executable, "tools/audit.py"]).returncode == 0
