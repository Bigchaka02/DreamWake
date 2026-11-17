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
    "- If a new item contradicts a current item, keep only the newer one and end it with "
    "'(replaces: <the old rule in at most 5 words>)'. Never copy an old item's text into a new one.\n"
    "- Merge duplicates. Keep each item a general instruction, at most 35 words. At most 8 items. "
    "Output only the JSON lines.")
DREAM_INSTRUCTION = (
    "Write 2 practice exercises that test whether an agent applies the rules above without being told. Each "
    "exercise is a small new feature request of the same kind as the example - not a policy change, and not "
    "something already done. Prefer rules that nothing in the project checks automatically (a user decision, a "
    "naming, format or unit policy): breaking those gives no error. Write each request the way a teammate who does "
    "not know the rules would: say only what they need, in everyday terms, without naming files, setting names, "
    "suffixes or conventions. Then give short exact strings for checking the result: \"must_add\" = strings that "
    "will appear in the text added to the project when the rules are applied; \"must_not_add\" = strings that would "
    "appear only if a rule were ignored. Keep each string short (a suffix, a converted value), never a whole line. "
    "The exercises are hypothetical. Output JSON lines, one object per line: "
    "{\"request\": \"...\", \"must_add\": [\"...\"], \"must_not_add\": [\"...\"]}")
REFINE_INSTRUCTION = (
    "An agent that was given the rules above got this request and made the mistakes listed. Rewrite the rules so "
    "that an agent reading them gets it right: make the rule that was not applied more specific and actionable - "
    "say exactly what to do when a request is phrased like this one, including how to name or convert things, with "
    "one short example. Never reverse or weaken a rule. Keep every other rule as it is. Do not record the practice "
    "request or its feature as a fact: it was hypothetical. Output the complete list as JSON lines, one object per "
    "line: {\"kind\": \"rule\" or \"lesson\", \"text\": \"...\"}")
REVIEW_INSTRUCTION = (
    "Above are the user's requests on this project, word for word, oldest first, and the RULES the agent will "
    "carry into its next session. Check that every standing decision of the user (a naming, format, unit or "
    "process policy meant for all future work, not a one-off task) is stated in the rules as a general "
    "instruction that keeps the user's exact terms. If one is missing, too narrow or out of date, fix it by "
    "editing exactly ONE rule: give its number, or 0 to add a new rule, and the full new text of that rule (at "
    "most 35 words). If the rules already cover every standing decision, answer with number -1 and an empty "
    "text.")
NOTE_CAP = 900  # characters of injected memory, for every design
ITEM_CAP, TEXT_CAP = 8, 240
REPLACES, REPLACES_CAP = "(replaces:", 60  # a longer or unterminated "(replaces: ...)" copies an old rule
FRAMING_TOKENS = 50  # chat-template tokens around the single message, by estimate
EXERCISES, EXPECT_CAP = 2, 4  # practice exercises per sleep; strings per must_add / must_not_add list
ADDED_CAP, CHECK_TAIL = 800, 300  # characters of added text and of check output shown to the refine call


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


def _clean(text):
    """The text cut before its "(replaces: ...)" when that is unterminated or longer than REPLACES_CAP."""
    start = text.find(REPLACES)
    end = text.find(")", start)
    if start >= 0 and (end < 0 or len(text[start + len(REPLACES):end].strip()) > REPLACES_CAP):
        return text[:start].strip()
    return text


def _item_text(text):
    return _clean(re.sub(r"^(?:RULE|LESSON)\s*:\s*", "", text.strip(), flags=re.IGNORECASE)[:TEXT_CAP])


def parse_items(text):
    items = [{"kind": value.get("kind", "rule"), "text": _item_text(value["text"])}
             for value in _objects(text, "text") if value.get("kind", "rule") in ("rule", "lesson")]
    return [item for item in items if item["text"]]


def _json_lines(items):
    return "\n".join(json.dumps(item) for item in items)


def _merge_prompt(current, candidates):
    return (f"CURRENT memory:\n{_json_lines(current) or '(empty)'}\n\nNEW items:\n{_json_lines(candidates)}\n\n"
            + MERGE_INSTRUCTION)


def _schema(name, properties, required):
    """An OpenAI-style strict JSON-schema response format for one object."""
    return {"type": "json_schema", "json_schema": {"name": name, "strict": True, "schema": {
        "type": "object", "properties": properties, "required": required}}}


def _ask(chat, prompt, reserve, seed, cost, response_format=None):
    """One single-message chat call, added to cost. Returns the reply text."""
    options = {"response_format": response_format} if response_format is not None else {}
    reply = chat([{"role": "user", "content": prompt}], max_tokens=reserve, seed=seed, **options)
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


def _edit(items, number, text):
    """items with exactly one rule changed: number 1..len(items) replaces that item's text (kind kept);
    0 adds a new rule at the front. Returns (candidate, reason); candidate is None when the edit is refused."""
    text = _item_text(text)
    if not text:
        return None, "empty text"
    if type(number) is not int or not 0 <= number <= len(items):
        return None, "bad number"
    candidate = items[:]
    index = number - 1 if number else 0
    if number:
        candidate[index] = {**items[index], "text": text}
    else:
        candidate.insert(0, {"kind": "rule", "text": text})
    candidate = _capped(candidate)
    if index >= len(candidate):
        return None, "does not fit"
    memory = inject("consolidated", {"items": candidate})
    if number and any(name not in memory for name in _identifiers(items[index]["text"])):
        return None, "edit drops identifiers"
    return candidate, ""


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


def _archive(requests):
    return "\n".join(f"{i}. {text}" for i, text in enumerate(
        (text.replace("\r\n", " ").replace("\r", " ").replace("\n", " ") for text in requests), 1))


def _fit(build, budget, *texts):
    """build(*texts), cutting the end off texts[0], then texts[1], ... until the prompt fits the budget."""
    texts = list(texts)
    for i in range(len(texts)):
        while (excess := estimate_tokens(build(*texts)) - budget) > 0 and texts[i]:
            texts[i] = texts[i][:-(int(excess * CHARS_PER_TOKEN) + 1)]
    return build(*texts)


def _strings(value):
    """A dream's must_add / must_not_add: stripped, non-empty strings, at most EXPECT_CAP (one string is a list)."""
    values = [value] if isinstance(value, str) else value if isinstance(value, list) else []
    return [s.strip() for s in values if isinstance(s, str) and s.strip()][:EXPECT_CAP]


def _dreams(reply):
    """The usable dreamed exercises: a request and at least one must_add string."""
    dreams = [{"request": v["request"].strip(), "hypothetical": True, "must_add": _strings(v.get("must_add")),
               "must_not_add": _strings(v.get("must_not_add"))} for v in _objects(reply, "request")]
    return [dream for dream in dreams if dream["must_add"]]


def _added(sandbox, snapshot):
    """The text a practice attempt added: per file, the lines not in the snapshot's version (all of a new file)."""
    added = []
    for path in sorted(Path(sandbox).rglob("*")):
        relative = path.relative_to(sandbox)
        if not path.is_file() or agent.SKIP_DIRS & set(relative.parts):
            continue
        old = Path(snapshot) / relative
        before = set(old.read_text(encoding="utf-8", errors="ignore").splitlines()) if old.is_file() else set()
        added += [line for line in path.read_text(encoding="utf-8", errors="ignore").splitlines()
                  if line not in before]
    return "\n".join(added)


def _refine_prompt(items, exercise, result, budget):
    """Trimmed to the budget: the added text first, then the check output, then the request."""
    faults = ([f"The added text should contain '{s}' but does not" for s in result["missing"]]
              + [f"The added text contains '{s}', which must not appear" for s in result["present"]])

    def build(added, output, request):
        problems = ([] if result["passed"] else [f"The project check failed: {output}"]) + faults
        return (f"RULES:\n{_numbered(items)}\n\nREQUEST: {request}\n\nWHAT THE AGENT ADDED:\n{added or '(nothing)'}"
                "\n\nPROBLEMS:\n" + "\n".join(problems) + "\n\n" + REFINE_INSTRUCTION)
    return _fit(build, budget, result["added"][:ADDED_CAP], result["output"][-CHECK_TAIL:], exercise["request"])


def _practise(design, items, practice, chat, window, reserve, seed, cost):
    """Attempt EXERCISES exercises with the items, scored by code; if any fails, one refine call rewrites the
    items and both are attempted again. The rewrite is kept only if no score rises, the total falls and no item
    is lost. Returns (items, this sleep's practice record); adds to cost."""
    cost["practice_sessions"] = 0
    budget, used = window - reserve - FRAMING_TOKENS, [0]

    def fresh(span=1):  # shortcut: offsets (at most 4 * max_steps + 6) fit run.py's stride of 1,000 up to 248
        used[0] += span
        return None if seed is None else seed + used[0] - span + 1

    def ask(prompt):
        return _ask(chat, prompt, reserve, fresh(), cost)

    def attempt(exercise, memory_items):
        sandbox = Path(practice["scratch"]) / f"attempt{cost['practice_sessions']}"
        cost["practice_sessions"] += 1
        shutil.rmtree(sandbox, ignore_errors=True)
        tasks._copy(snapshot, sandbox)  # never run in the snapshot itself
        result = agent.run_session(practice["task_id"], sandbox, exercise["request"],
                                   memory=inject("consolidated", {"items": memory_items}),
                                   memory_at=practice.get("memory_at", "system"),
                                   finish_gate=practice.get("finish_gate", False),
                                   max_steps=practice["max_steps"], window=window, reserve=reserve, chat=chat,
                                   seed=fresh(practice["max_steps"] + 1), log_path=sandbox.with_suffix(".jsonl"))
        for key in ("prompt_tokens", "completion_tokens", "seconds"):
            cost[key] += result[key]
        code, output = tasks.run_check(practice["task_id"], sandbox)
        added = _added(sandbox, snapshot)
        missing = [s for s in exercise["must_add"] if s not in added]
        present = [s for s in exercise["must_not_add"] if s in added]
        return {"score": (code != 0) + len(missing) + len(present), "passed": code == 0, "output": output,
                "added": added, "missing": missing, "present": present}

    real = {"request": practice["request"], "hypothetical": False, "must_add": [], "must_not_add": []}
    if design == "replay":
        snapshot, exercises = practice["ws_before"], [real] * EXERCISES
    else:
        snapshot = practice["ws_after"]
        prompt = _fit(lambda example: (f"RULES:\n{_numbered(items)}\n\nAn example of what users ask on this project:"
                                       f"\n{example}\n\n" + DREAM_INSTRUCTION), budget, real["request"])
        exercises = _dreams(ask(prompt))[:EXERCISES]
        exercises += [real] * (EXERCISES - len(exercises))
    first = [attempt(exercise, items) for exercise in exercises]
    record = {"exercises": [{**exercise, "score": result["score"], "score2": None, "check_passed": result["passed"],
                             "check_passed2": None, "missing": result["missing"], "present": result["present"],
                             "missing2": None, "present2": None} for exercise, result in zip(exercises, first)],
              "items_before": [item.copy() for item in items], "candidate": None, "refined": False, "accepted": False}
    failed = [(exercise, result) for exercise, result in zip(exercises, first) if result["score"]]
    if not failed:
        return items, record
    candidate = _capped(parse_items(ask(_refine_prompt(items, *failed[0], budget))))
    record["candidate"] = candidate
    second = [attempt(exercise, candidate) for exercise in exercises]
    for entry, result in zip(record["exercises"], second):
        entry["score2"], entry["check_passed2"] = result["score"], result["passed"]
        entry["missing2"], entry["present2"] = result["missing"], result["present"]
    before, after = [r["score"] for r in first], [r["score"] for r in second]
    record["refined"] = True
    record["accepted"] = (all(a <= b for a, b in zip(after, before)) and sum(after) < sum(before)
                          and len(candidate) >= len(items))
    return (candidate if record["accepted"] else items), record


def _identifiers(text):
    return {s for s in re.findall(r"`([^`\n]+)`", text) + re.findall(r"\b[A-Z][A-Z0-9_]{4,}\b", text) if len(s) >= 5}


def contamination(memory, practice_log, wake_events):
    """Identifiers (backticked or UPPER_SNAKE, 5+ characters) of hypothetical practice requests that reached the
    memory but appear nowhere in the real sessions. Sorted."""
    real = json.dumps(wake_events, ensure_ascii=False)
    dreamt = _identifiers("\n".join(entry["request"] for record in practice_log for entry in record["exercises"]
                                    if entry["hypothetical"]))
    return sorted(name for name in dreamt if name in memory and name not in real)


def sleep(design, artifact, events, *, chat, window, reserve, seed=None, practice=None):
    """Turn one session's events into the design's artifact. Returns (artifact, cost).
    practice: requests archive (review); task_id, request, workspace snapshots, scratch dir and max_steps
    (dreaming, replay)."""
    cost = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "seconds": 0.0}
    if design == "workspace":
        return None, cost
    if design == "requests":
        return (artifact or []) + [events[0]["text"]], cost
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
    if design == "review":
        artifact, cost = _consolidate(artifact, events, chat, window, reserve, seed)
        items = artifact["items"]
        requests = practice.get("requests", []) if practice else []
        if not requests:
            return {"items": items, "review": []}, cost
        prompt = _fit(lambda archive, rules: (f"USER REQUESTS:\n{archive}\n\nRULES:\n{rules}\n\n"
                                              + REVIEW_INSTRUCTION),
                      window - reserve - FRAMING_TOKENS, _archive(requests), _numbered(items))
        reply = _loads(_ask(chat, prompt, reserve, None if seed is None else seed + 1, cost,
                            _schema("edit", {"number": {"type": "integer"}, "text": {"type": "string"}},
                                    ["number", "text"])))
        record = {"number": None, "text": None, "applied": False, "reason": "invalid reply",
                  "items_before": [item.copy() for item in items]}
        if isinstance(reply, dict) and type(reply.get("number")) is int:
            number, text = reply["number"], reply.get("text")
            record.update(number=number, text=text, reason="")
            if number != -1:
                candidate, record["reason"] = _edit(items, number, text if isinstance(text, str) else "")
                if candidate is not None:
                    items, record["applied"] = candidate, True
        return {"items": items, "review": [record]}, cost
    if design in ("dreaming", "replay"):
        artifact, cost = _consolidate(artifact, events, chat, window, reserve, seed)
        items, record = _practise(design, artifact["items"], practice, chat, window, reserve, seed, cost)
        return {"items": items, "practice": [record]}, cost  # run.py appends each sleep's records, s1 first
    raise ValueError(f"unknown design: {design!r}")


def inject(design, artifact):
    """The memory text the next session starts with ("" for none)."""
    if design == "workspace":
        return ""
    if design == "handoff":
        return artifact or ""
    if design == "requests":
        requests = artifact or []
        kept, remaining = [], NOTE_CAP
        for i in reversed(range(len(requests))):
            prefix = f"- USER REQUEST {i + 1}: "
            text = requests[i].replace("\r\n", " ").replace("\r", " ").replace("\n", " ")
            space = remaining - (1 if kept else 0) - len(prefix)
            if len(text) > space:
                marker = " [...]"
                if space - len(marker) >= 80:
                    kept.append(prefix + text[:space - len(marker)] + marker)
                break
            kept.append(prefix + text)
            remaining -= len(prefix + text) + (1 if len(kept) > 1 else 0)
        return "\n".join(reversed(kept))
    if design in ("consolidated", "dreaming", "replay", "review"):  # offline requests are never injected
        items = artifact["items"] if artifact else []
        return "\n".join(f"- {item['kind'].upper()}: {item['text']}" for item in items)
    raise ValueError(f"unknown design: {design!r}")
