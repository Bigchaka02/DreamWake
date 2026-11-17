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
    "Above are the user's requests on this project, word for word, oldest first. Some of them contain standing "
    "decisions: rules the user wants followed in all future work (a naming, format, unit or process policy), not "
    "just in that one task. Find the standing decisions that no automatic check would enforce.\n"
    "Write up to 4 practice exercises. Each exercise tests ONE standing decision:\n"
    "- \"decision\": the user's sentence that states it, copied character for character from the requests above.\n"
    "- \"request\": a small new feature request of the kind this user makes, for something that none of the "
    "requests above already asked for, to which the decision clearly applies. Write it the way a teammate who "
    "never saw the decision would: plain everyday words, in the old or everyday terms, nothing about how to store "
    "or name things, no file names, no setting names, no suffixes.\n"
    "- \"must_add\": 1 or 2 short strings copied character for character from the decision (for example a "
    "required suffix) that will appear in the text added to the project if the decision is followed.\n"
    "- \"must_not_add\": short strings that would appear only if the decision were ignored. Never a whole "
    "setting name, never a number.\n"
    "If there is no standing decision, return an empty list. The exercises are hypothetical.")
REPAIR_INSTRUCTION = (
    "An agent that was given the RULES above got the REQUEST and broke a standing decision of the user. The "
    "decision, in the user's own words, is quoted above. Change the rules so that an agent reading them follows "
    "the decision in requests like this one. Edit exactly ONE rule: give its number, or 0 to add a new rule, and "
    "the full new text of that rule (at most 35 words). State the decision as a general instruction for all "
    "future work, keep the user's exact terms (names, suffixes, units), and say what to do when a request is "
    "phrased in other terms. Do not mention the practice request: it was hypothetical.")
REVIEW_INSTRUCTION = (
    "Above are the user's requests on this project, word for word, oldest first, and the RULES the agent will "
    "carry into its next session. Check that every standing decision of the user (a naming, format, unit or "
    "process policy meant for all future work, not a one-off task) is stated in the rules as a general "
    "instruction that keeps the user's exact terms. If one is missing, too narrow or out of date, fix it by "
    "editing exactly ONE rule: give its number, or 0 to add a new rule, and the full new text of that rule (at "
    "most 35 words). If the rules already cover every standing decision, answer with number -1 and an empty "
    "text.")
NOTE_CAP = 900  # characters of injected memory, for every design
STATE_DONE_CAP, STATE_OPEN_CAP = 160, 400
ITEM_CAP, TEXT_CAP = 8, 240
REPLACES, REPLACES_CAP = "(replaces:", 60  # a longer or unterminated "(replaces: ...)" copies an old rule
FRAMING_TOKENS = 50  # chat-template tokens around the single message, by estimate
EXERCISES, EXPECT_CAP = 2, 4  # practice exercises per sleep; strings per must_add / must_not_add list
CONTROL_CAP, CONFIRM, REPLAY_ATTEMPTS = 3, 2, 5
REPLY_CAP, DECISION_MIN = 2000, 20
DREAM_SUFFIX = " This is a new, separate feature: add it even if something similar already exists."
ADDED_CAP, CHECK_TAIL = 800, 300  # characters of added text and of check output shown to the repair call


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
    """Non-numeric strings, preserving exact check text (one string is a list)."""
    values = [value] if isinstance(value, str) else value if isinstance(value, list) else []
    return [s for s in values if isinstance(s, str) and not re.fullmatch(r"[\d.,\s]+", s)]


def _dreams(reply, requests):
    """Offered exercises with grounded expectations and the first rejection reason, if any."""
    value = _loads(reply)
    if not isinstance(value, dict) or not isinstance(value.get("exercises"), list):
        return []
    archive = [" ".join(request.split()) for request in requests]
    dreams = []
    for entry in value["exercises"]:
        if not isinstance(entry, dict) or not isinstance(entry.get("request"), str) or not entry["request"].strip():
            continue
        request = entry["request"].strip()
        decision = entry.get("decision", "")
        decision = decision.strip() if isinstance(decision, str) else ""
        must_add = [s for s in _strings(entry.get("must_add")) if len(s) >= 2 and s in decision][:EXPECT_CAP]
        must_not_add = [s for s in _strings(entry.get("must_not_add"))
                        if len(s) >= 3 and not any(s in kept for kept in must_add)][:EXPECT_CAP]
        normal = " ".join(decision.split())
        reason = ""
        if len(normal) < DECISION_MIN or not any(normal in text for text in archive):
            reason = "decision is not the user's words"
        elif not must_add:
            reason = "no grounded must_add"
        elif any(s.lower() in request.lower() for s in must_add + must_not_add):
            reason = "request contains a check string"
        elif re.search(r"`|\w+_\w*|\b[\w/\\-]+\.(?:py|json|md|txt|toml|yaml|yml|cfg|ini)\b", request):
            reason = "request names an identifier or file"
        dreams.append({"request": request, "hypothetical": True, "decision": decision, "must_add": must_add,
                       "must_not_add": must_not_add, "rejected": reason})
    return dreams


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


def _repair_prompt(items, exercise, result, budget):
    """Trimmed to the budget: the added text first, then the check output, then the request."""
    faults = ([f"The added text should contain '{s}' but does not" for s in result["missing"]]
              + [f"The added text contains '{s}', which must not appear" for s in result["present"]])

    def build(added, output, request):
        problems = ([] if result["passed"] else [f"The project check failed: {output}"]) + faults
        return (f"RULES:\n{_numbered(items)}\n\nUSER DECISION: {exercise['decision']}\n\n"
                f"REQUEST: {request}\n\nWHAT THE AGENT ADDED:\n{added or '(nothing)'}"
                "\n\nPROBLEMS:\n" + "\n".join(problems) + "\n\n" + REPAIR_INSTRUCTION)
    return _fit(build, budget, result["added"][:ADDED_CAP], result["output"][-CHECK_TAIL:], exercise["request"])


def _practise(design, items, practice, chat, window, reserve, seed, cost):
    """Verify grounded dreams (or replay real work), then confirm a single-rule repair against failures."""
    cost["practice_sessions"] = 0
    budget, used = window - reserve - FRAMING_TOKENS, 0
    snapshot = practice["ws_before"] if design == "replay" else practice["ws_after"]
    protected = tasks.load_task(practice["task_id"]).get("protected", [])

    def fresh(span=1):
        # At most 8 * (max_steps + 1) + 2 offsets; run.py's 20 steps fit its stride of 1,000.
        nonlocal used
        used += span
        return None if seed is None else seed + used - span + 1

    def attempt(exercise, memory_text, memory_at, finish_gate):
        sandbox = Path(practice["scratch"]) / f"attempt{cost['practice_sessions']}"
        cost["practice_sessions"] += 1
        shutil.rmtree(sandbox, ignore_errors=True)
        tasks._copy(snapshot, sandbox)  # never run in the snapshot itself
        # the dream writer cannot see the project: without the suffix the agent answers "already there"
        request = exercise["request"] + (DREAM_SUFFIX if exercise["hypothetical"] else "")
        result = agent.run_session(practice["task_id"], sandbox, request,
                                   memory=memory_text, memory_at=memory_at, finish_gate=finish_gate,
                                   max_steps=practice["max_steps"], window=window, reserve=reserve, chat=chat,
                                   seed=fresh(practice["max_steps"] + 1), log_path=sandbox.with_suffix(".jsonl"))
        for key in ("prompt_tokens", "completion_tokens", "seconds"):
            cost[key] += result[key]
        for folder in protected:
            shutil.rmtree(sandbox / folder, ignore_errors=True)
            source = Path(snapshot) / folder
            if source.exists():
                tasks._copy(source, sandbox / folder)
        code, output = tasks.run_check(practice["task_id"], sandbox)
        added = _added(sandbox, snapshot)
        missing = [s for s in exercise["must_add"] if s not in added]
        present = [s for s in exercise["must_not_add"] if s in added]
        return {"score": (code != 0) + len(missing) + len(present), "passed": code == 0, "output": output,
                "added": added, "missing": missing, "present": present}

    record = {"exercises": [], "dreams_offered": [], "dream_replies": [],
              "items_before": [item.copy() for item in items], "edit": None, "candidate": None,
              "refined": False, "accepted": False, "reason": ""}
    if design == "replay":
        exercises = [{"request": practice["request"], "hypothetical": False, "decision": practice["request"],
                      "must_add": [], "must_not_add": []} for _ in range(REPLAY_ATTEMPTS)]
    else:
        requests = practice.get("requests", [])
        prompt = _fit(lambda archive: f"USER REQUESTS:\n{archive}\n\n" + DREAM_INSTRUCTION,
                      budget, _archive(requests))
        reply = _ask(chat, prompt, reserve, fresh(), cost, _schema("exercises", {"exercises": {
            "type": "array", "maxItems": 4, "items": {"type": "object", "properties": {
                "decision": {"type": "string"}, "request": {"type": "string"},
                "must_add": {"type": "array", "items": {"type": "string"}},
                "must_not_add": {"type": "array", "items": {"type": "string"}}},
                "required": ["decision", "request", "must_add", "must_not_add"]}}}, ["exercises"]))
        record["dream_replies"] = [reply[:REPLY_CAP]]
        dreams = _dreams(reply, requests)
        record["dreams_offered"] = [{k: v for k, v in dream.items() if k != "hypothetical"}
                                    | {"control_score": None} for dream in dreams]
        exercises, controls = [], 0
        for dream, offered in zip(dreams, record["dreams_offered"]):
            if dream["rejected"]:
                continue
            if len(exercises) == EXERCISES or controls == CONTROL_CAP:
                break
            control = attempt(dream, "- USER DECISION: " + dream["decision"], "request", False)
            controls += 1
            offered["control_score"] = control["score"]
            if control["score"] == 0:
                exercises.append({k: v for k, v in dream.items() if k != "rejected"})
        if not exercises:
            record["reason"] = "no verified dreams"
            return items, record

    placement, gate = practice.get("memory_at", "system"), practice.get("finish_gate", False)
    memory = inject("consolidated", {"items": items})
    first = [attempt(exercise, memory, placement, gate) for exercise in exercises]
    record["exercises"] = [{**exercise, "score": result["score"], "check_passed": result["passed"],
                            "missing": result["missing"], "present": result["present"], "scores2": None}
                           for exercise, result in zip(exercises, first)]
    failed = next((i for i, result in enumerate(first) if result["score"]), None)
    if failed is None:
        record["reason"] = "all exercises passed"
        return items, record
    reply = _loads(_ask(chat, _repair_prompt(items, exercises[failed], first[failed], budget), reserve,
                        fresh(), cost, _schema("edit", {"number": {"type": "integer"}, "text": {"type": "string"}},
                                               ["number", "text"])))
    record["reason"] = "no edit"
    if not isinstance(reply, dict) or type(reply.get("number")) is not int:
        return items, record
    number, text = reply["number"], reply.get("text")
    record["edit"] = {"number": number, "text": text}
    if number == -1:
        return items, record
    candidate, record["reason"] = _edit(items, number, text if isinstance(text, str) else "")
    if candidate is None:
        return items, record
    record["candidate"], record["refined"] = candidate, True
    memory = inject("consolidated", {"items": candidate})
    scores = [attempt(exercises[failed], memory, placement, gate)["score"] for _ in range(CONFIRM)]
    record["exercises"][failed]["scores2"] = scores
    accepted = all(score < first[failed]["score"] for score in scores)
    if failed + 1 < len(exercises):
        other = failed + 1
        score = attempt(exercises[other], memory, placement, gate)["score"]
        record["exercises"][other]["scores2"] = [score]
        accepted = accepted and score <= first[other]["score"]
    record["accepted"] = accepted
    record["reason"] = "" if accepted else "not confirmed"
    return (candidate if accepted else items), record


def _identifiers(text):
    return {s for s in re.findall(r"`([^`\n]+)`", text) + re.findall(r"\b[A-Z][A-Z0-9_]{4,}\b", text) if len(s) >= 5}


def contamination(memory, practice_log, wake_events):
    """Identifiers (backticked or UPPER_SNAKE, 5+ characters) of hypothetical practice requests that reached the
    memory but appear nowhere in the real sessions. Sorted."""
    real = json.dumps(wake_events, ensure_ascii=False)
    dreamt = _identifiers("\n".join(
        [entry["request"] for record in practice_log for entry in record["exercises"] if entry["hypothetical"]]
        + [entry["request"] for record in practice_log for entry in record.get("dreams_offered", [])]))
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
    if design == "hybrid":
        artifact, cost = _consolidate(artifact, events, chat, window, reserve, seed)
        # first line only: a multi-call finish carries the ignored-calls note after it
        finished = any(e["type"] == "result" and e["text"].split("\n")[0] == "finished" for e in events)
        return {"items": artifact["items"], "state": {"request": events[0]["text"], "finished": finished}}, cost
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
    if design in ("consolidated", "dreaming", "replay", "review", "hybrid"):  # offline requests are never injected
        items = artifact["items"] if artifact else []
        lines = [f"- {item['kind'].upper()}: {item['text']}" for item in items]
        if design == "hybrid" and artifact:
            state = artifact["state"]
            request = state["request"].replace("\r\n", " ").replace("\r", " ").replace("\n", " ")
            cap = STATE_DONE_CAP if state["finished"] else STATE_OPEN_CAP
            if len(request) > cap:
                request = request[:cap - len(" [...]")] + " [...]"
            status = "finished" if state["finished"] else "NOT finished - continue this work"
            lines.insert(0, f"- LAST SESSION ({status}): {request}")
            while len("\n".join(lines)) > NOTE_CAP:
                lines.pop()
        return "\n".join(lines)
    raise ValueError(f"unknown design: {design!r}")
