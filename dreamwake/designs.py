"""Memory designs: what survives the context reset. Each design is a sleep() branch plus an inject() branch."""
import json

from .agent import render_chunks, render_events
from .llm import CHARS_PER_TOKEN, estimate_tokens

HANDOFF_INSTRUCTION = (
    "Your conversation memory is about to be erased. Write a handoff note to yourself, at most 120 words; it is the "
    "only thing you will remember next time. Cover: 1) the goal and the current state; 2) decisions and project "
    "rules that still hold - when a newer decision replaced an older one, state the newer one and say what it "
    "replaces; 3) what you tried that failed, and why; 4) the next step. Plain text only.")
EXTRACT_INSTRUCTION = (
    "This is part of the log of your own work session on a software project. Your memory of it will be erased. "
    "List what a future session on this project must know, as JSON lines, one object per line: "
    "{\"kind\": \"rule\" or \"lesson\", \"text\": \"...\"}.\n"
    "- rule: a project rule or a user decision that future changes must follow. Write it as a general instruction "
    "for any future change, not as a description of what you did. Name the exact files, suffixes or formats "
    "involved.\n"
    "- lesson: something you tried that failed, why it failed, and what works instead.\n"
    "If the user changed an earlier decision, write the new rule and say what it replaces. Include only what this "
    "log supports. At most 5 items, each at most 35 words. Output only the JSON lines.")
MERGE_INSTRUCTION = (
    "You keep a short memory of rules and lessons for a software project. Merge the NEW items into the CURRENT "
    "memory and output the complete updated memory as JSON lines, one object per line: "
    "{\"kind\": \"rule\" or \"lesson\", \"text\": \"...\"}.\n"
    "- Keep every current item that is still true.\n"
    "- If a new item changes or contradicts a current item, keep only the newer one and end it with "
    "'(replaces: <the old rule in a few words>)'.\n"
    "- Merge duplicates. Keep each item a general instruction, at most 35 words. At most 8 items. "
    "Output only the JSON lines.")
NOTE_CAP = 900  # characters of injected memory, for every design
ITEM_CAP, TEXT_CAP = 8, 240
FRAMING_TOKENS = 50  # chat-template tokens around the single message, by estimate


def _loads(text):
    try:
        return json.loads(text)
    except ValueError:
        return None


def parse_items(text):
    """Lenient: JSON lines or a JSON array, either may be fenced; anything malformed is skipped."""
    lines = [line for line in text.splitlines() if not line.strip().startswith("```")]
    values = _loads("\n".join(lines))
    if values is None:
        values = [_loads(line) for line in lines]
    items = []
    for value in values if isinstance(values, list) else [values]:
        if isinstance(value, dict) and isinstance(value.get("text"), str) and value["text"].strip():
            if value.get("kind", "rule") in ("rule", "lesson"):
                items.append({"kind": value.get("kind", "rule"), "text": value["text"].strip()[:TEXT_CAP]})
    return items


def _json_lines(items):
    return "\n".join(json.dumps(item) for item in items)


def _merge_prompt(current, candidates):
    return (f"CURRENT memory:\n{_json_lines(current) or '(empty)'}\n\nNEW items:\n{_json_lines(candidates)}\n\n"
            + MERGE_INSTRUCTION)


def _consolidate(artifact, events, chat, window, reserve, seed):
    cost = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "seconds": 0.0}

    def ask(prompt):
        reply = chat([{"role": "user", "content": prompt}], max_tokens=reserve, seed=seed)
        cost["calls"] += 1
        cost["seconds"] += reply["seconds"]
        for key in ("prompt_tokens", "completion_tokens"):
            cost[key] += reply["usage"][key]
        return parse_items(reply["content"] or "")

    current = artifact["items"] if artifact else []
    budget = window - reserve - FRAMING_TOKENS
    max_chars = max(0, int(CHARS_PER_TOKEN * (budget - estimate_tokens("\n\n" + EXTRACT_INSTRUCTION))))
    candidates = [item for chunk in render_chunks(events, max_chars)
                  for item in ask(chunk + "\n\n" + EXTRACT_INSTRUCTION)]
    while candidates and estimate_tokens(_merge_prompt(current, candidates)) > budget:
        candidates.pop(0)  # shortcut: the oldest candidates go unmerged; merge per chunk if this happens live
    items = current
    if candidates:
        items = ask(_merge_prompt(current, candidates))
        if not items:  # one bad generation must not lose the memory
            for item in current + candidates:
                if item["text"] not in [kept["text"] for kept in items]:
                    items.append(item)
    items = items[:ITEM_CAP]
    while len(inject("consolidated", {"items": items})) > NOTE_CAP:
        items = items[:-1]
    return {"items": items}, cost


def sleep(design, artifact, events, *, chat, window, reserve, seed=None):
    """Turn one session's events into the design's artifact. Returns (artifact, cost)."""
    cost = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "seconds": 0.0}
    if design == "workspace":
        return None, cost
    if design == "handoff":
        request, rest = events[0]["text"], events[1:]  # the session's user request, pinned like in the live agent
        head = f"Your previous handoff note:\n{artifact}\n\n" if artifact else ""
        head += f"The request for this session:\n{request}\n\nThe end of the session, as you can still see it:\n"
        tail = "\n\n" + HANDOFF_INSTRUCTION
        max_chars = max(0, int(CHARS_PER_TOKEN * (window - reserve - estimate_tokens(head + tail) - FRAMING_TOKENS)))
        prompt = head + render_events(rest, max_chars) + tail
        reply = chat([{"role": "user", "content": prompt}], max_tokens=reserve, seed=seed)
        cost = {"calls": 1, **reply["usage"], "seconds": reply["seconds"]}
        return (reply["content"] or "").strip()[:NOTE_CAP], cost
    if design == "consolidated":
        return _consolidate(artifact, events, chat, window, reserve, seed)
    raise ValueError(f"unknown design: {design!r}")


def inject(design, artifact):
    """The memory text the next session starts with ("" for none)."""
    if design == "workspace":
        return ""
    if design == "handoff":
        return artifact or ""
    if design == "consolidated":
        items = artifact["items"] if artifact else []
        return "\n".join(f"- {item['kind'].upper()}: {item['text']}" for item in items)
    raise ValueError(f"unknown design: {design!r}")
