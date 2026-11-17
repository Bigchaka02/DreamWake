"""Look up user-facing text by key."""
import json
from pathlib import Path

CATALOG = Path(__file__).resolve().parent.parent / "locale" / "en.json"


def t(key):
    return json.loads(CATALOG.read_text(encoding="utf-8"))[key]
