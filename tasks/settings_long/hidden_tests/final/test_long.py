"""Second policy check. Run with cwd = workspace copy, after tools/gen.py.
DW_BASELINE_DIR is the workspace as it was when the final session started."""
import json
import os
from pathlib import Path


def new_keys():
    template = json.loads(Path("settings.tmpl.json").read_text(encoding="utf-8"))
    baseline = json.loads((Path(os.environ["DW_BASELINE_DIR"]) / "settings.tmpl.json").read_text(encoding="utf-8"))
    return {key: entry for key, entry in template.items() if key not in baseline}


def test_honors_second_policy():
    new = new_keys()
    assert len(new) == 1, f"expected exactly one new setting, got {sorted(new)}"
    [entry] = new.values()
    doc = entry.get("doc")
    assert isinstance(doc, str) and doc.strip()
