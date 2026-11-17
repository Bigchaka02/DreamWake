"""Memory designs: what survives the context reset. Each design is a sleep() branch plus an inject() branch."""
from .agent import render_events
from .llm import CHARS_PER_TOKEN, estimate_tokens

HANDOFF_INSTRUCTION = (
    "Your conversation memory is about to be erased. Write a handoff note to yourself, at most 120 words; it is the "
    "only thing you will remember next time. Cover: 1) the goal and the current state; 2) decisions and project "
    "rules that still hold - when a newer decision replaced an older one, state the newer one and say what it "
    "replaces; 3) what you tried that failed, and why; 4) the next step. Plain text only.")
NOTE_CAP = 900
FRAMING_TOKENS = 50  # chat-template tokens around the single message, by estimate


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
    raise ValueError(f"unknown design: {design!r}")


def inject(design, artifact):
    """The memory text the next session starts with ("" for none)."""
    if design == "workspace":
        return ""
    if design == "handoff":
        return artifact or ""
    raise ValueError(f"unknown design: {design!r}")
