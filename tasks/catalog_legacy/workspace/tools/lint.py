"""Check that the keys in locale/en.json are in sorted order."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    keys = list(json.loads((ROOT / "locale" / "en.json").read_text(encoding="utf-8")))
    for before, after in zip(keys, keys[1:]):
        if after < before:
            sys.exit(f"lint: FAIL - locale/en.json keys not sorted: {after!r} must come before {before!r}")
    print("lint: ok")


if __name__ == "__main__":
    main()
