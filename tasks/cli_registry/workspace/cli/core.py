"""Look up a command by name and return the text it prints."""
import json
from pathlib import Path

from . import commands

MANIFEST = Path(__file__).with_name("manifest.json")


def render(name):
    registered = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if name not in registered:
        raise LookupError(f"unknown command: {name}")
    result = getattr(commands, registered[name])()
    if isinstance(result, dict):
        return result["out"]
    return result
