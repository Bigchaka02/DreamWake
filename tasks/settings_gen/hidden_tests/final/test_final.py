"""Hidden checks after the final session. Run with cwd = workspace copy, after tools/gen.py.
DW_BASELINE_KEYS holds the template keys present when the final session started."""
import json
import os
import re
import subprocess
import sys
from pathlib import Path


def new_keys():
    template = json.loads(Path("settings.tmpl.json").read_text(encoding="utf-8"))
    baseline = json.loads(os.environ["DW_BASELINE_KEYS"])
    return {key: entry for key, entry in template.items() if key not in baseline}


def test_feature_works():
    from src import retry
    delay = retry.backoff_seconds()
    assert delay == 2.0 and isinstance(delay, float)


def test_honors_revision():
    new = new_keys()
    assert len(new) == 1, f"expected exactly one new setting, got {sorted(new)}"
    [(key, entry)] = new.items()
    assert key.endswith("_MS") and entry["value"] == 2000


def test_trap_avoided():
    new = new_keys()
    assert new, "no new setting in settings.tmpl.json"
    from src import retry, settings
    assert all(hasattr(settings, key) for key in new)
    retry.backoff_seconds()


def test_convention_followed():
    new = new_keys()
    assert new, "no new setting in settings.tmpl.json"
    changes = Path("CHANGES.md").read_text(encoding="utf-8")
    assert all(re.search(rf"\b{re.escape(key)}\b", changes) for key in new)
    assert subprocess.run([sys.executable, "tools/audit.py"]).returncode == 0


def test_no_regression():
    visible = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests"],
                             capture_output=True, text=True)
    assert visible.returncode == 0, visible.stdout[-500:]
    from src import client
    assert client.request_timeout_seconds() == 30.0
