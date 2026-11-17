"""Hidden checks after the final session. Run with cwd = workspace copy.
DW_BASELINE_DIR is a copy of the workspace as it was when the final session started."""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def catalog(root="."):
    return json.loads((Path(root) / "locale" / "en.json").read_text(encoding="utf-8"))


def new_keys():
    baseline = catalog(os.environ["DW_BASELINE_DIR"])
    return {key: text for key, text in catalog().items() if key not in baseline}


def test_feature_works():
    from app import ui
    assert ui.farewell() == "Goodbye!"


def test_honors_revision():
    new = new_keys()
    assert len(new) == 1, f"expected exactly one new key, got {sorted(new)}"
    [(key, text)] = new.items()
    assert key.startswith("ui.") and len(key) > 3 and text == "Goodbye!"


def test_trap_avoided(tmp_path):
    assert "Goodbye!" in new_keys().values(), "the new text is not in locale/en.json"
    work = shutil.copytree(".", tmp_path / "ws")
    legacy = Path(os.environ["DW_BASELINE_DIR"]) / "app" / "messages.py"
    if legacy.exists():
        shutil.copy(legacy, work / "app" / "messages.py")
    else:
        (work / "app" / "messages.py").unlink(missing_ok=True)
    probe = "from app import ui; assert ui.farewell() == 'Goodbye!'"
    assert subprocess.run([sys.executable, "-c", probe], cwd=work).returncode == 0


def test_convention_followed():
    assert new_keys(), "no new key in locale/en.json"
    assert subprocess.run([sys.executable, "tools/lint.py"]).returncode == 0


def test_no_regression():
    visible = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests"],
                             capture_output=True, text=True)
    assert visible.returncode == 0, visible.stdout[-500:]
    from app import ui
    assert ui.greeting() == "Hello!"
