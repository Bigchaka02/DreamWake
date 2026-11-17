"""Check that every registered command function is named cmd_<something>."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    registered = json.loads((ROOT / "cli" / "manifest.json").read_text(encoding="utf-8"))
    for name, func in registered.items():
        if not str(func).startswith("cmd_"):
            sys.exit(f"lint: FAIL - command {name!r}: function {func!r} must start with cmd_")
    print("lint: ok")


if __name__ == "__main__":
    main()
