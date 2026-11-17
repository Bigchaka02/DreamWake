"""Memory designs: what survives the context reset. Each design is a sleep() branch plus an inject() branch."""
import json
import re
import shutil
from pathlib import Path

from . import agent, tasks
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
DREAM_INSTRUCTION = (
    "Invent 2 new, different change requests of the same kind and size that a teammate could plausibly ask next on "
    "this project. They are hypothetical practice exercises, not real requests. Phrase them the way a user who does "
    "NOT know the project's rules would. Output JSON lines, one object per line: {\"request\": \"...\"}")
REVIEW_INSTRUCTION = (
    "You are checking whether an agent followed the project's rules in a practice exercise. For each rule decide "
    "whether the changes follow it. Output JSON lines, one per broken rule: {\"rule\": <number>, \"problem\": "
    "\"<what was done wrong, concretely>\"}. If the check failed for a reason no rule covers, add {\"rule\": 0, "
    "\"problem\": \"...\"}. If nothing is wrong, output the single word NONE.")
REFINE_INSTRUCTION = (
    "An agent that was given these rules still made the mistakes above in a practice exercise. Rewrite the rules so "
    "that an agent reading them would not make these mistakes: make each broken rule more specific and actionable - "
    "say exactly what to do and where, and how to name or convert things, with one short example. Keep every other "
    "rule as it is. Do not add facts about the practice exercise itself: it was hypothetical. Output the complete "
    "list as JSON lines, one object per line: {\"kind\": \"rule\" or \"lesson\", \"text\": \"...\"}")
NOTE_CAP = 900  # characters of injected memory, for every design
ITEM_CAP, TEXT_CAP = 8, 240
FRAMING_TOKENS = 50  # chat-template tokens around the single message, by estimate
PRACTICE_ROUNDS = 2
FILE_CAP, CHANGES_CAP, CHECK_TAIL = 500, 1500, 400  # characters shown to the reviewer


def _loads(text):
    try:
        return json.loads(text)
    except ValueError:
        return None


def _objects(text, key):
    """Lenient: the JSON objects with a non-empty string `key`, from JSON lines or a JSON array, either may be
    fenced; anything malformed is skipped."""
    lines = [line for line in text.splitlines() if not line.strip().startswith("```")]
    values = _loads("\n".join(lines))
    if values is None:
        values = [_loads(line) for line in lines]
    return [value for value in (values if isinstance(values, list) else [values])
            if isinstance(value, dict) and isinstance(value.get(key), str) and value[key].strip()]


def parse_items(text):
    return [{"kind": value.get("kind", "rule"), "text": value["text"].strip()[:TEXT_CAP]}
            for value in _objects(text, "text") if value.get("kind", "rule") in ("rule", "lesson")]


def _json_lines(items):
    return "\n".join(json.dumps(item) for item in items)


def _merge_prompt(current, candidates):
    return (f"CURRENT memory:\n{_json_lines(current) or '(empty)'}\n\nNEW items:\n{_json_lines(candidates)}\n\n"
            + MERGE_INSTRUCTION)


def _ask(chat, prompt, reserve, seed, cost):
    """One single-message chat call, added to cost. Returns the reply text."""
    reply = chat([{"role": "user", "content": prompt}], max_tokens=reserve, seed=seed)
    cost["calls"] += 1
    cost["seconds"] += reply["seconds"]
    for key in ("prompt_tokens", "completion_tokens"):
        cost[key] += reply["usage"][key]
    return reply["content"] or ""


def _capped(items):
    """Whole items dropped from the end until the memory fits ITEM_CAP and NOTE_CAP."""
    items = items[:ITEM_CAP]
    while len(inject("consolidated", {"items": items})) > NOTE_CAP:
        items = items[:-1]
    return items


def _consolidate(artifact, events, chat, window, reserve, seed):
    cost = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "seconds": 0.0}

    def ask(prompt):
        return parse_items(_ask(chat, prompt, reserve, seed, cost))

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
    return {"items": _capped(items)}, cost


def _numbered(items):
    return "\n".join(f"{i}. {item['kind'].upper()}: {item['text']}" for i, item in enumerate(items, 1)) or "(none)"


def _changes(events):
    """What a session wrote: the last successful write per path, capped for the reviewer."""
    failed = {e["step"] for e in events if e["type"] == "result" and e["text"].startswith("error")}
    written = {}
    for e in events:
        if e["type"] == "call" and e["tool"] == "write_file" and e["step"] not in failed:
            written[e["args"]["path"]] = e["args"]["content"][:FILE_CAP]
    return "\n\n".join(f"FILE {path}:\n{content}" for path, content in written.items())[:CHANGES_CAP]


def _review_prompt(rules, request, changes, passed, output, budget):
    """Trimmed to the budget: the changes first, then the check output."""
    output = output[-CHECK_TAIL:]

    def prompt():
        return (f"RULES:\n{_numbered(rules)}\n\nREQUEST:\n{request}\n\nCHANGES:\n{changes or '(none)'}\n\n"
                f"CHECK RESULT: {'passed' if passed else 'failed'}\n{output}\n\n" + REVIEW_INSTRUCTION)
    while (excess := estimate_tokens(prompt()) - budget) > 0 and (changes or output):
        cut = int(excess * CHARS_PER_TOKEN) + 1
        changes, output = (changes[:-cut], output) if changes else (changes, output[cut:])
    return prompt()


def _refine_prompt(rules, problems, budget):
    """Trimmed to the budget by dropping the last mistakes (one is always kept)."""
    mistakes = [f"- rule {p['rule']}: {p['problem']}" for p in problems]
    while True:
        prompt = f"RULES:\n{_numbered(rules)}\n\nMISTAKES:\n" + "\n".join(mistakes) + "\n\n" + REFINE_INSTRUCTION
        if estimate_tokens(prompt) <= budget or len(mistakes) <= 1:
            return prompt
        mistakes.pop()


def _practise(design, items, practice, chat, window, reserve, seed, cost):
    """PRACTICE_ROUNDS rounds of attempt -> review, and on any problem refine -> validate; a rewrite is kept only
    if it lowers the score without losing items. Returns (items, practice log); adds to cost."""
    cost["practice_sessions"] = 0
    budget, used = window - reserve - FRAMING_TOKENS, [0]

    def fresh(span=1):  # shortcut: offsets (at most 4 * max_steps + 11) stay under run.py's stride of 100 up to 22
        used[0] += span
        return None if seed is None else seed + used[0] - span + 1

    def ask(prompt):
        return _ask(chat, prompt, reserve, fresh(), cost)

    def attempt(request, snapshot, memory_items, rules):
        sandbox = Path(practice["scratch"]) / f"attempt{cost['practice_sessions']}"
        cost["practice_sessions"] += 1
        shutil.rmtree(sandbox, ignore_errors=True)
        tasks._copy(snapshot, sandbox)  # never run in the snapshot itself
        result = agent.run_session(practice["task_id"], sandbox, request,
                                   memory=inject("consolidated", {"items": memory_items}),
                                   max_steps=practice["max_steps"], window=window, reserve=reserve, chat=chat,
                                   seed=fresh(practice["max_steps"] + 1), log_path=sandbox.with_suffix(".jsonl"))
        for key in ("prompt_tokens", "completion_tokens", "seconds"):
            cost[key] += result[key]
        code, output = tasks.run_check(practice["task_id"], sandbox)
        reply = ask(_review_prompt(rules, request, _changes(result["events"]), code == 0, output, budget))
        problems = [{"rule": v.get("rule"), "problem": v["problem"].strip()[:TEXT_CAP]}
                    for v in _objects(reply, "problem")]
        return (code != 0) + len(problems), problems

    real = practice["request"]
    if design == "replay":
        exercises = [(real, practice["ws_before"], False)] * PRACTICE_ROUNDS
    else:
        reply = ask(f"Here is a change request you handled on this project:\n{real}\n\n" + DREAM_INSTRUCTION)
        dreams = [v["request"].strip() for v in _objects(reply, "request")][:PRACTICE_ROUNDS]
        exercises = [(dream, practice["ws_after"], True) for dream in dreams]
        exercises += [(real, practice["ws_after"], False)] * (PRACTICE_ROUNDS - len(dreams))
    log = []
    for request, snapshot, hypothetical in exercises:
        score, problems = attempt(request, snapshot, items, items)
        entry = {"request": request, "hypothetical": hypothetical, "score": score, "problems": problems,
                 "refined": score > 0, "score2": None, "accepted": False}
        if score:
            candidate = _capped(parse_items(ask(_refine_prompt(items, problems, budget))))
            entry["score2"], _ = attempt(request, snapshot, candidate, items)  # reviewed against the ORIGINAL rules
            if entry["score2"] < score and len(candidate) >= len(items):
                items, entry["accepted"] = candidate, True
        log.append(entry)
    return items, log


def _identifiers(text):
    return {s for s in re.findall(r"`([^`\n]+)`", text) + re.findall(r"\b[A-Z][A-Z0-9_]{4,}\b", text) if len(s) >= 5}


def contamination(memory, practice_log, wake_events):
    """Identifiers (backticked or UPPER_SNAKE, 5+ characters) of hypothetical practice requests that reached the
    memory but appear nowhere in the real sessions. Sorted."""
    real = json.dumps(wake_events, ensure_ascii=False)
    dreamt = _identifiers("\n".join(entry["request"] for entry in practice_log if entry["hypothetical"]))
    return sorted(name for name in dreamt if name in memory and name not in real)


def sleep(design, artifact, events, *, chat, window, reserve, seed=None, practice=None):
    """Turn one session's events into the design's artifact. Returns (artifact, cost).
    practice: the session's task_id, request, workspace snapshots, scratch dir and max_steps (dreaming, replay)."""
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
    if design in ("dreaming", "replay"):
        artifact, cost = _consolidate(artifact, events, chat, window, reserve, seed)
        items, log = _practise(design, artifact["items"], practice, chat, window, reserve, seed, cost)
        return {"items": items, "practice": log}, cost
    raise ValueError(f"unknown design: {design!r}")


def inject(design, artifact):
    """The memory text the next session starts with ("" for none)."""
    if design == "workspace":
        return ""
    if design == "handoff":
        return artifact or ""
    if design in ("consolidated", "dreaming", "replay"):  # practice requests are never injected
        items = artifact["items"] if artifact else []
        return "\n".join(f"- {item['kind'].upper()}: {item['text']}" for item in items)
    raise ValueError(f"unknown design: {design!r}")
