"""Check that every setting in settings.tmpl.json is listed in CHANGES.md."""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    template = json.loads((ROOT / "settings.tmpl.json").read_text(encoding="utf-8"))
    changes = (ROOT / "CHANGES.md").read_text(encoding="utf-8")
    missing = [key for key in template if not re.search(rf"\b{re.escape(key)}\b", changes)]
    if missing:
        sys.exit("audit: FAIL - settings missing from CHANGES.md: " + ", ".join(missing))
    print("audit: ok")


if __name__ == "__main__":
    main()
