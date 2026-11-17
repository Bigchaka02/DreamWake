"""Hidden checks after session 2. Run with cwd = workspace copy."""
import json
import subprocess
import sys
from pathlib import Path


def test_feature_works():
    from cli import core
    assert [core.render(n) for n in ("ping", "whoami", "hello")] == ["pong", "guest", "Hello!"]


def test_honors_revision():
    from cli import commands
    registered = json.loads(Path("cli/manifest.json").read_text(encoding="utf-8"))
    for name, func in registered.items():
        result = getattr(commands, func)()
        assert isinstance(result, dict), f"{name} returned {result!r}"
        assert result.get("ok") is True and isinstance(result.get("out"), str), name


def test_lint_ok():
    assert subprocess.run([sys.executable, "tools/lint.py"]).returncode == 0
