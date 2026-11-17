"""Regenerate src/settings.py from settings.tmpl.json."""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TYPES = {"int": int, "float": float, "str": str, "bool": bool}


def main():
    template = json.loads((ROOT / "settings.tmpl.json").read_text(encoding="utf-8"))
    lines = []
    for key, entry in template.items():
        if not re.fullmatch(r"[A-Z][A-Z0-9]*(_[A-Z0-9]+)*", key):
            sys.exit(f"gen: error - key {key!r} is not UPPER_SNAKE_CASE")
        if not isinstance(entry, dict) or not {"type", "value"} <= set(entry):
            sys.exit(f"gen: error - {key} needs both 'type' and 'value'")
        if type(entry["value"]) is not TYPES.get(entry["type"]):
            sys.exit(f"gen: error - {key} value does not match type {entry['type']!r}")
        lines.append(f"{key} = {entry['value']!r}\n")
    (ROOT / "src" / "settings.py").write_text("".join(lines), encoding="utf-8")
    print("gen: regenerated src/settings.py from settings.tmpl.json")


if __name__ == "__main__":
    main()
