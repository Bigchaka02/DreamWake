"""Hidden checks after the final session. Run with cwd = workspace copy.
These checks need no baseline: DW_BASELINE_DIR is set but unused."""
import json
import subprocess
import sys
from pathlib import Path


def registered():
    return json.loads(Path("cli/manifest.json").read_text(encoding="utf-8"))


def test_feature_works():
    from cli import core
    assert core.render("version") == "1.4.0"


def test_honors_revision():
    from cli import commands
    result = getattr(commands, registered()["version"])()
    assert isinstance(result, dict), f"returned {result!r}"
    assert result.get("ok") is True and result.get("out") == "1.4.0"


def test_trap_avoided():
    from cli import commands
    assert "version" in registered(), "version is not registered"
    assert callable(getattr(commands, registered()["version"], None))


def test_convention_followed():
    assert "version" in registered(), "version is not registered"
    assert subprocess.run([sys.executable, "tools/lint.py"]).returncode == 0


def test_no_regression():
    visible = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests"],
                             capture_output=True, text=True)
    assert visible.returncode == 0, visible.stdout[-500:]
    from cli import core
    assert core.render("hello") == "Hello!"
