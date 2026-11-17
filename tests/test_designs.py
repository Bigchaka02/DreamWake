"""Memory designs: what each one keeps across the reset, and what its sleep step sends."""
import json
from pathlib import Path

import pytest

from dreamwake import agent, designs, llm, tasks


def session_events(n, request="SESSION PROMPT"):
    events = [{"type": "user", "text": request}]
    for step in range(1, n + 1):
        events += [{"type": "call", "step": step, "tool": "read_file", "args": {"path": f"file{step}.py"}},
                   {"type": "result", "step": step, "text": f"result {step} " + "x" * 300}]
    return events


class FakeChat:
    def __init__(self, note):
        self.note, self.calls = note, []

    def __call__(self, messages, tools=None, **kwargs):
        self.calls.append({"messages": messages, "tools": tools, **kwargs})
        content = self.note(messages[0]["content"]) if callable(self.note) else self.note
        return {"content": content, "tool_calls": [], "usage": {"prompt_tokens": 50, "completion_tokens": 20},
                "seconds": 1.5, "finish_reason": "stop"}


def test_schema_and_ask_forward_response_format_only_when_supplied():
    properties, required = {"number": {"type": "integer"}}, ["number"]
    response_format = designs._schema("edit", properties, required)
    assert response_format == {"type": "json_schema", "json_schema": {"name": "edit", "strict": True,
                               "schema": {"type": "object", "properties": properties, "required": required}}}
    fake = FakeChat("{}")
    cost = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "seconds": 0.0}
    assert designs._ask(fake, "prompt", 100, 42, cost, response_format=response_format) == "{}"
    assert fake.calls[0]["response_format"] is response_format

    def legacy_chat(messages, *, max_tokens, seed):
        return fake(messages, max_tokens=max_tokens, seed=seed)
    designs._ask(legacy_chat, "prompt", 100, 42, cost)
    assert "response_format" not in fake.calls[1]
    assert cost == {"calls": 2, "prompt_tokens": 100, "completion_tokens": 40, "seconds": 3.0}


def test_archive_numbers_verbatim_requests_and_flattens_only_line_breaks():
    assert designs._archive(["  first\nline  ", "second\r\nline\rlast", ""]) == (
        "1.   first line  \n2. second line last\n3. ")
    assert designs._archive([]) == ""


def test_edit_replaces_one_item_keeps_kind_and_does_not_mutate_items():
    items = [{"kind": "rule", "text": "Keep the policy."}, {"kind": "lesson", "text": "Use TIMEOUT_MS."}]
    before = json.loads(json.dumps(items))
    candidate, reason = designs._edit(items, 2, "  lEsSoN : Use TIMEOUT_MS for every delay.  ")
    assert candidate == [items[0], {"kind": "lesson", "text": "Use TIMEOUT_MS for every delay."}]
    assert reason == "" and items == before
    assert candidate is not items and candidate[1] is not items[1]


def test_edit_adds_a_rule_at_front_and_caps_whole_items_from_end():
    items = [{"kind": "lesson", "text": f"item {i}"} for i in range(8)]
    before = json.loads(json.dumps(items))
    candidate, reason = designs._edit(items, 0, " RULE: New policy. ")
    assert candidate == [{"kind": "rule", "text": "New policy."}] + items[:7]
    assert reason == "" and items == before
    assert designs._edit([], 0, "first") == ([{"kind": "rule", "text": "first"}], "")
    long = [{"kind": "rule", "text": str(i) * 210} for i in range(4)]
    candidate, reason = designs._edit(long, 0, "n" * 240)
    assert candidate == [{"kind": "rule", "text": "n" * 240}] + long[:2] and reason == ""
    assert len(designs.inject("consolidated", {"items": candidate})) <= designs.NOTE_CAP


@pytest.mark.parametrize("number", [-1, 2, 1.0, "1", None, True])
def test_edit_refuses_bad_numbers_without_mutation(number):
    items = [{"kind": "rule", "text": "old"}]
    assert designs._edit(items, number, "new") == (None, "bad number")
    assert items == [{"kind": "rule", "text": "old"}]


@pytest.mark.parametrize("text", ["", "  ", " RULE :  ", "lesson:", "(replaces: unfinished"])
def test_edit_refuses_empty_cleaned_text_without_mutation(text):
    items = [{"kind": "rule", "text": "old"}]
    assert designs._edit(items, 1, text) == (None, "empty text")
    assert items == [{"kind": "rule", "text": "old"}]


def test_edit_preserves_every_old_identifier_anywhere_in_the_capped_memory():
    items = [{"kind": "rule", "text": "Use `settings.json` and TIMEOUT_MS."}]
    assert designs._edit(items, 1, "Use TIMEOUT_MS.") == (None, "edit drops identifiers")
    assert designs._edit(items, 1, "Use settings.json.") == (None, "edit drops identifiers")
    other = {"kind": "rule", "text": "Keep settings.json and TIMEOUT_MS."}
    assert designs._edit(items + [other], 1, "General policy.") == (
        [{"kind": "rule", "text": "General policy."}, other], "")
    full = items + [{"kind": "rule", "text": "x" * 210}] * 3 + [other]
    assert designs._edit(full, 1, "n" * 240) == (None, "edit drops identifiers")
    assert items == [{"kind": "rule", "text": "Use `settings.json` and TIMEOUT_MS."}]


def test_edit_refuses_when_capping_drops_the_edited_item_without_mutation():
    items = [{"kind": "rule", "text": "x" * 210} for _ in range(4)]
    before = json.loads(json.dumps(items))
    assert designs._edit(items, 4, "n" * 240) == (None, "does not fit")
    assert items == before
    many = [{"kind": "rule", "text": "short"} for _ in range(9)]
    assert designs._edit(many, 9, "new") == (None, "does not fit")


def test_edit_removes_labels_before_text_cap_then_cleans_replaces_tail():
    assert designs._edit([], 0, "  RULE : " + "x" * 300) == ([{"kind": "rule", "text": "x" * 240}], "")
    assert designs._edit([], 0, "lesson: Keep policy. (replaces: unfinished") == (
        [{"kind": "rule", "text": "Keep policy."}], "")


@pytest.mark.parametrize("label", ["RULE:", " rule : ", "LeSsOn: ", "LESSON :"])
def test_parse_items_removes_leading_labels_before_capping_and_cleaning(label):
    assert designs.parse_items(lines(("lesson", label + "x" * 300))) == [
        {"kind": "lesson", "text": "x" * 240}]
    assert designs.parse_items(lines(("rule", label + " (replaces: unfinished"))) == []


def test_workspace_design_keeps_nothing_and_costs_nothing():
    fake = FakeChat("unused")
    artifact, cost = designs.sleep("workspace", None, session_events(3), chat=fake, window=4096, reserve=700)
    assert artifact is None and fake.calls == []
    assert cost == {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "seconds": 0.0}
    assert designs.inject("workspace", artifact) == ""


def test_requests_sleep_keeps_every_verbatim_request_without_model_calls():
    fake = FakeChat("unused")
    first = "  First request\nwith two lines.  "
    second = "Second request"
    artifact, cost = designs.sleep("requests", None, session_events(1, first), chat=fake, window=4096, reserve=700)
    assert artifact == [first]
    saved = artifact
    artifact, cost2 = designs.sleep("requests", artifact, session_events(2, second),
                                   chat=fake, window=4096, reserve=700)
    assert artifact == [first, second] and saved == [first] and fake.calls == []
    assert cost == cost2 == {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "seconds": 0.0}


@pytest.mark.parametrize("artifact", [None, []])
def test_requests_empty_artifact_renders_empty(artifact):
    assert designs.inject("requests", artifact) == ""


def test_requests_render_in_original_order_with_original_numbers_and_flattened_line_breaks():
    assert designs.inject("requests", ["first\nline", "second\r\nline\r", ""]) == (
        "- USER REQUEST 1: first line\n- USER REQUEST 2: second line \n- USER REQUEST 3: ")


def test_requests_whole_request_at_the_cap_is_not_cut():
    prefix = "- USER REQUEST 2: "
    text = "n" * (designs.NOTE_CAP - len(prefix))
    assert designs.inject("requests", ["older", text]) == prefix + text


@pytest.mark.parametrize("kept", [79, 80, 140])
def test_requests_cut_requires_eighty_text_characters_and_stops_before_older_requests(kept):
    prefix2, prefix3, marker = "- USER REQUEST 2: ", "- USER REQUEST 3: ", " [...]"
    newest = "n" * (designs.NOTE_CAP - len(prefix2 + prefix3 + marker) - 1 - kept)
    artifact = ["ancient", "o" * 1000, newest]
    expected = (prefix2 + "o" * kept + marker + "\n" if kept >= 80 else "") + prefix3 + newest
    rendered = designs.inject("requests", artifact)
    assert rendered == expected and len(rendered) <= designs.NOTE_CAP
    assert "ancient" not in rendered


def test_requests_newest_oversize_request_is_cut_to_the_cap():
    prefix = "- USER REQUEST 10: "
    marker = " [...]"
    artifact = ["old"] * 9 + ["x" * 2000]
    expected = prefix + "x" * (designs.NOTE_CAP - len(prefix + marker)) + marker
    assert designs.inject("requests", artifact) == expected


def test_handoff_sends_previous_note_and_the_transcript_tail_that_fits():
    fake = FakeChat("  new note  ")
    window, reserve = 1000, 300
    events = session_events(30)
    artifact, cost = designs.sleep("handoff", "OLD NOTE", events, chat=fake, window=window, reserve=reserve)
    assert artifact == "new note" and designs.inject("handoff", artifact) == "new note"
    assert cost == {"calls": 1, "prompt_tokens": 50, "completion_tokens": 20, "seconds": 1.5}
    [call] = fake.calls
    assert call["tools"] is None and call["max_tokens"] == reserve
    [message] = call["messages"]
    prompt = message["content"]
    assert message["role"] == "user" and llm.estimate_tokens(prompt) <= window - reserve
    assert "OLD NOTE" in prompt and designs.HANDOFF_INSTRUCTION in prompt
    assert "STEP 30 read_file(path='file30.py')" in prompt  # the most recent step is kept
    assert "STEP 1 " not in prompt  # the oldest steps are not
    assert prompt.index("OLD NOTE") < prompt.index("STEP 30") < prompt.index(designs.HANDOFF_INSTRUCTION)


def test_handoff_keeps_the_session_request_in_full_between_the_note_and_the_tail():
    fake = FakeChat("note")
    window, reserve = 1000, 300
    request = "REQUEST START " + "r" * 1000 + " REQUEST END"  # longer than the transcript's per-event cap
    designs.sleep("handoff", "OLD NOTE", session_events(30, request), chat=fake, window=window, reserve=reserve)
    prompt = fake.calls[0]["messages"][0]["content"]
    assert llm.estimate_tokens(prompt) <= window - reserve
    assert prompt.count(request) == 1 and "STEP 30 " in prompt and "STEP 1 " not in prompt
    assert (prompt.index("OLD NOTE") < prompt.index(request) < prompt.index("STEP 30")
            < prompt.index(designs.HANDOFF_INSTRUCTION))


def test_handoff_without_previous_note_sends_the_request_and_the_whole_short_transcript():
    fake = FakeChat("note")
    events = session_events(2)
    designs.sleep("handoff", None, events, chat=fake, window=4096, reserve=700)
    prompt = fake.calls[0]["messages"][0]["content"]
    assert prompt.count("SESSION PROMPT") == 1 and "previous" not in prompt.lower()
    assert prompt.index("SESSION PROMPT") < prompt.index(agent.render_events(events[1:]))


def test_handoff_note_is_capped():
    artifact, _ = designs.sleep("handoff", None, session_events(1), chat=FakeChat("w" * 2000),
                                window=4096, reserve=700)
    assert artifact == "w" * 900


def test_handoff_instruction_text():
    assert designs.HANDOFF_INSTRUCTION.startswith("Your conversation memory is about to be erased.")
    assert designs.HANDOFF_INSTRUCTION.endswith("4) the next step. Plain text only.")


def test_unknown_design_is_rejected():
    with pytest.raises(ValueError):
        designs.inject("telepathy", None)
    with pytest.raises(ValueError):
        designs.sleep("telepathy", None, [], chat=FakeChat(""), window=4096, reserve=700)


def test_handoff_forwards_the_seed_to_chat():
    fake = FakeChat("note")
    designs.sleep("handoff", None, session_events(1), chat=fake, window=4096, reserve=700, seed=42)
    assert fake.calls[0]["seed"] == 42
    designs.sleep("handoff", None, session_events(1), chat=fake, window=4096, reserve=700)
    assert fake.calls[1]["seed"] is None


# --- consolidated ---------------------------------------------------------------------------------

def lines(*items):
    return "\n".join(json.dumps({"kind": kind, "text": text}) for kind, text in items)


def memory_chat(extract, merge):
    """Answers extract prompts with `extract` and the merge prompt with `merge` (either may be a function)."""
    def note(prompt):
        reply = merge if prompt.endswith(designs.MERGE_INSTRUCTION) else extract
        return reply(prompt) if callable(reply) else reply
    return FakeChat(note)


def prompts(fake):
    return [call["messages"][0]["content"] for call in fake.calls]


SECONDS = ("rule", "Store durations in seconds with the _S suffix.")
MILLIS = ("rule", "Store durations in milliseconds with the _MS suffix. (replaces: seconds)")
LESSON = ("lesson", "Editing src/settings.py fails: it is generated; edit settings.tmpl.json instead.")


def test_consolidated_extracts_once_per_chunk_then_merges_once_within_the_window():
    fake = memory_chat(lines(MILLIS), lines(MILLIS))
    window, reserve = 1000, 300
    designs.sleep("consolidated", None, session_events(30), chat=fake, window=window, reserve=reserve, seed=42)
    *extracts, merge = prompts(fake)
    assert len(extracts) > 1 and all(p.endswith("\n\n" + designs.EXTRACT_INSTRUCTION) for p in extracts)
    assert all(p.startswith("USER: SESSION PROMPT\n") for p in extracts)
    for step in range(1, 31):  # every step reaches exactly one extract call
        assert sum(f"STEP {step} read_file(" in p for p in extracts) == 1
    assert merge.startswith("CURRENT memory:\n(empty)\n\nNEW items:\n")
    assert all(llm.estimate_tokens(p) <= window - reserve for p in prompts(fake))
    assert all(c["tools"] is None and c["max_tokens"] == reserve and c["seed"] == 42 for c in fake.calls)


def test_consolidated_merge_prompt_fits_the_window_with_many_long_candidates_keeping_the_newest():
    count = iter(range(1000))
    fake = memory_chat(lambda p: lines(*[("rule", f"item {next(count)} " + "w" * 220)] * 5), "")
    window, reserve = 1000, 300
    designs.sleep("consolidated", None, session_events(30), chat=fake, window=window, reserve=reserve)
    merge = prompts(fake)[-1]
    assert llm.estimate_tokens(merge) <= window - reserve
    assert f"item {next(count) - 1} " in merge and "item 0 " not in merge


@pytest.mark.parametrize("reply", [
    lines(SECONDS, LESSON),
    json.dumps([{"kind": "rule", "text": SECONDS[1]}, {"kind": "lesson", "text": LESSON[1]}]),
    "```json\n" + json.dumps([{"kind": "rule", "text": SECONDS[1]}, {"kind": "lesson", "text": LESSON[1]}]) + "\n```",
    "Here you go:\n```json\n" + lines(SECONDS, LESSON) + "\n```\nDone.",
    "junk\n" + json.dumps({"text": SECONDS[1]}) + "\n{broken\n" + lines(LESSON) + "\n"
    + lines(("fact", "bad kind"), ("rule", "   ")) + '\n[1, 2]\n"text"',
])
def test_parse_items_accepts_lines_arrays_and_fences_and_skips_junk(reply):
    assert designs.parse_items(reply) == [{"kind": "rule", "text": SECONDS[1]}, {"kind": "lesson", "text": LESSON[1]}]


def test_parse_items_trims_text_and_returns_nothing_for_garbage():
    assert designs.parse_items(lines(("rule", "  " + "a" * 300))) == [{"kind": "rule", "text": "a" * 240}]
    assert designs.parse_items("") == [] and designs.parse_items("no json here") == []


def test_consolidated_merge_reply_supersedes_the_old_rule():
    fake = memory_chat(lines(MILLIS), lines(MILLIS, LESSON))
    old = {"items": [{"kind": "rule", "text": SECONDS[1]}, {"kind": "lesson", "text": LESSON[1]}]}
    artifact, _ = designs.sleep("consolidated", old, session_events(2), chat=fake, window=4096, reserve=700)
    assert artifact == {"items": [{"kind": "rule", "text": MILLIS[1]}, {"kind": "lesson", "text": LESSON[1]}]}
    merge = prompts(fake)[-1]
    assert merge == (f"CURRENT memory:\n{lines(SECONDS, LESSON)}\n\nNEW items:\n{lines(MILLIS)}\n\n"
                     + designs.MERGE_INSTRUCTION)


@pytest.mark.parametrize("bad", ["", "I could not merge these.", "```json\n```"])
def test_consolidated_bad_merge_reply_falls_back_to_current_plus_new_without_duplicates(bad):
    fake = memory_chat(lines(SECONDS, MILLIS), bad)
    old = {"items": [{"kind": "rule", "text": SECONDS[1]}, {"kind": "lesson", "text": LESSON[1]}]}
    artifact, _ = designs.sleep("consolidated", old, session_events(2), chat=fake, window=4096, reserve=700)
    assert [item["text"] for item in artifact["items"]] == [SECONDS[1], LESSON[1], MILLIS[1]]


def test_consolidated_without_candidates_skips_the_merge_and_keeps_memory():
    fake = memory_chat("nothing useful", lines(MILLIS))
    old = {"items": [{"kind": "rule", "text": SECONDS[1]}]}
    artifact, cost = designs.sleep("consolidated", old, session_events(2), chat=fake, window=4096, reserve=700)
    assert artifact == old and len(fake.calls) == 1 and cost["calls"] == 1
    assert not prompts(fake)[0].startswith("CURRENT memory")


def test_consolidated_cap_drops_whole_items_from_the_end():
    many = [("rule", f"rule {i}") for i in range(10)]
    artifact, _ = designs.sleep("consolidated", None, session_events(2), chat=memory_chat(lines(MILLIS), lines(*many)),
                                window=4096, reserve=700)
    assert [item["text"] for item in artifact["items"]] == [f"rule {i}" for i in range(8)]
    long = [("rule", f"rule {i}. " + "z" * 200) for i in range(8)]
    artifact, _ = designs.sleep("consolidated", None, session_events(2), chat=memory_chat(lines(MILLIS), lines(*long)),
                                window=4096, reserve=700)
    assert [item["text"] for item in artifact["items"]] == [text for _, text in long[:len(artifact["items"])]]
    assert len(designs.inject("consolidated", artifact)) <= designs.NOTE_CAP
    assert len(artifact["items"]) == 4  # five "- RULE: ..." lines of 216 characters exceed 900


def test_consolidated_inject_formats_rules_and_lessons_in_order():
    artifact = {"items": [{"kind": "rule", "text": MILLIS[1]}, {"kind": "lesson", "text": LESSON[1]}]}
    assert designs.inject("consolidated", artifact) == f"- RULE: {MILLIS[1]}\n- LESSON: {LESSON[1]}"
    assert designs.inject("consolidated", None) == "" and designs.inject("consolidated", {"items": []}) == ""


def test_nearby_store_and_at_request_constant_are_removed():
    assert not hasattr(designs, "AT_REQUEST")
    with pytest.raises(ValueError, match="unknown design"):
        designs.inject("nearby", None)
    with pytest.raises(ValueError, match="unknown design"):
        designs.sleep("nearby", None, session_events(1), chat=FakeChat(""), window=4096, reserve=700)


def test_consolidated_cost_sums_every_call():
    fake = memory_chat(lines(MILLIS), lines(MILLIS))
    _, cost = designs.sleep("consolidated", None, session_events(30), chat=fake, window=1000, reserve=300)
    n = len(fake.calls)
    assert n > 2 and cost == {"calls": n, "prompt_tokens": 50 * n, "completion_tokens": 20 * n, "seconds": 1.5 * n}


def test_consolidated_second_sleep_merges_into_the_first_sleeps_memory():
    first = memory_chat(lines(SECONDS), lines(SECONDS))
    artifact, _ = designs.sleep("consolidated", None, session_events(2), chat=first, window=4096, reserve=700)
    second = memory_chat(lines(MILLIS), lines(MILLIS))
    artifact, _ = designs.sleep("consolidated", artifact, session_events(2), chat=second, window=4096, reserve=700)
    assert prompts(second)[-1].startswith(f"CURRENT memory:\n{lines(SECONDS)}\n\nNEW items:\n{lines(MILLIS)}\n\n")
    assert artifact == {"items": [{"kind": "rule", "text": MILLIS[1]}]}


def test_consolidated_instruction_texts():
    assert designs.EXTRACT_INSTRUCTION.startswith("This is part of the log of your own work session")
    assert designs.EXTRACT_INSTRUCTION.endswith("each at most 35 words. Output only the JSON lines.")
    assert designs.MERGE_INSTRUCTION.startswith("You keep a short memory of rules and lessons")
    assert designs.MERGE_INSTRUCTION.endswith("At most 8 items. Output only the JSON lines.")


# --- dreaming and replay: practise with the memory, keep a rewrite only if practice improves -------

TASK = "settings_gen"
REAL = "Rename TIMEOUT_S to TIMEOUT_MS with value 30000. Make the checks pass."
DREAMS = [{"request": "Add a cache lifetime of 5 seconds.", "must_add": ["_MS"], "must_not_add": ["_SECONDS"]},
          {"request": "Add a poll interval of 9 seconds.", "must_add": ["_MS"], "must_not_add": ["_SECONDS"]}]
SPECIFIC = ("rule", "Store every duration in milliseconds: name it <NAME>_MS, e.g. RETRY_DELAY_MS = 2000.")
ITEMS = [{"kind": k, "text": t} for k, t in (MILLIS, LESSON)]
REFINED = [{"kind": k, "text": t} for k, t in (SPECIFIC, LESSON)]
MEMORY = designs.inject("consolidated", {"items": ITEMS})
CANDIDATE = designs.inject("consolidated", {"items": REFINED})
RULES = f"RULES:\n1. RULE: {MILLIS[1]}\n2. LESSON: {LESSON[1]}\n\n"
WRITTEN = {0: "POLL_MS = 1", 1: "nothing", 2: "POLL_SECONDS = 1"}  # what scores 0, 1, 2 against a DREAMS entry


def dream_lines(*dreams):
    return "\n".join(json.dumps(d) for d in dreams)


def snapshots(tmp_path, passing=True):
    """practice= for sleep: two task workspaces marked 'before' / 'after' (checks pass unless passing=False)."""
    for name in ("before", "after"):
        tasks.make_workspace(TASK, tmp_path / name)
        if passing:
            tasks.apply_reference(TASK, "s1", tmp_path / name)
        (tmp_path / name / "MARK.txt").write_text(name, encoding="utf-8")
    return {"task_id": TASK, "request": REAL, "ws_before": tmp_path / "before", "ws_after": tmp_path / "after",
            "scratch": tmp_path / "scratch", "max_steps": 12}


def tree(root):
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in Path(root).rglob("*") if p.is_file()}


def practice_chat(refine=lines(SPECIFIC, LESSON), dream=dream_lines(*DREAMS), writes=(("notes.txt", "practice"),)):
    """Sleep calls answered by instruction (any other tool-less prompt fails the test). A practice session writes
    `writes` one file per step, then finishes; `writes` may be a function of (request, refined), where refined
    means the session's memory holds the SPECIFIC rule."""
    state = {}

    def chat(messages, tools=None, *, max_tokens, seed=None, **kwargs):
        chat.calls.append({"messages": messages, "tools": tools, "max_tokens": max_tokens, "seed": seed})
        usage = {"prompt_tokens": 50, "completion_tokens": 20}
        if tools:
            if not any(m["role"] == "assistant" for m in messages[2:]):
                request = messages[1]["content"].partition(agent.MEMORY.format(memory=""))[0]
                plan = writes(request, any(SPECIFIC[1] in m["content"] for m in messages)) if callable(writes) \
                    else writes
                state["queue"] = [("write_file", {"path": p, "content": c}) for p, c in plan]
            name, args = state["queue"].pop(0) if state["queue"] else ("finish", {"summary": "done"})
            return {"content": "", "tool_calls": [{"name": name, "arguments": args, "raw": json.dumps(args)}],
                    "usage": usage, "seconds": 1.0, "finish_reason": "tool_calls"}
        prompt = messages[0]["content"]
        replies = {designs.EXTRACT_INSTRUCTION: lines(MILLIS), designs.MERGE_INSTRUCTION: lines(MILLIS, LESSON),
                   designs.DREAM_INSTRUCTION: dream, designs.REFINE_INSTRUCTION: refine}
        [content] = [reply for ending, reply in replies.items() if prompt.endswith(ending)]
        return {"content": content, "tool_calls": [], "usage": usage, "seconds": 1.5, "finish_reason": "stop"}
    chat.calls = []
    return chat


def scored(table):
    """writes= for practice_chat: per (DREAMS index, refined) the score that session's added text should get."""
    requests = [d["request"] for d in DREAMS]
    return lambda request, refined: [("notes.txt", WRITTEN[table[requests.index(request), refined]])]


@pytest.fixture
def spy(monkeypatch):
    """What every practice session received: its sandbox's MARK, request, memory, and the sandbox itself."""
    real, seen = agent.run_session, []

    def run_session(task_id, workspace, prompt, **kwargs):
        mark = (Path(workspace) / "MARK.txt").read_text(encoding="utf-8")
        seen.append({"mark": mark, "request": prompt, "memory": kwargs["memory"], "workspace": Path(workspace),
                     "max_steps": kwargs["max_steps"], "seed": kwargs["seed"],
                     "memory_at": kwargs.get("memory_at", "system"), "finish_gate": kwargs.get("finish_gate", False)})
        return real(task_id, workspace, prompt, **kwargs)
    monkeypatch.setattr(designs.agent, "run_session", run_session)
    return seen


def sleep(design, fake, practice, window=4096, reserve=700, seed=None):
    return designs.sleep(design, None, session_events(2), chat=fake, window=window, reserve=reserve, seed=seed,
                         practice=practice)


@pytest.mark.parametrize("design", ["replay", "dreaming"])
@pytest.mark.parametrize("options", [{}, {"memory_at": "request", "finish_gate": True},
                                    {"memory_at": "tail", "finish_gate": True}])
def test_practice_attempts_and_candidates_receive_placement_and_gate(tmp_path, spy, design, options):
    practice = {**snapshots(tmp_path, passing=False), **options}
    fake = practice_chat()
    artifact, cost = sleep(design, fake, practice)
    assert artifact["practice"][0]["refined"] and cost["practice_sessions"] == 4
    placement, gate = options.get("memory_at", "system"), options.get("finish_gate", False)
    assert [(s["memory_at"], s["finish_gate"]) for s in spy] == [(placement, gate)] * 4
    for session in spy:
        lines = [json.loads(line) for line in session["workspace"].with_suffix(".jsonl").read_text(
            encoding="utf-8").splitlines()]
        gated = [e for e in lines if e["type"] == "result" and e["text"] == agent.GATE.format(memory=session["memory"])]
        assert len(gated) == int(gate)


def instruction_prompts(fake, instruction):
    return [p for p in prompts(fake) if p.endswith(instruction)]


def sleep_calls(fake):
    return [c for c in fake.calls if c["tools"] is None]


def test_added_text_is_the_new_lines_of_changed_files_and_all_lines_of_new_files(tmp_path):
    snapshot, sandbox = tmp_path / "snap", tmp_path / "box"
    for root in (snapshot, sandbox):
        (root / "src").mkdir(parents=True)
        (root / "same.txt").write_text("unchanged\n", encoding="utf-8")
        (root / "gone.txt").write_text("only here\n", encoding="utf-8")
    (snapshot / "src" / "a.py").write_text("X = 1\nY = 2\n", encoding="utf-8")
    (sandbox / "src" / "a.py").write_text("X = 1\nNEW_MS = 5\nY = 2\n", encoding="utf-8")
    (sandbox / "gone.txt").unlink()
    (sandbox / "b.txt").write_bytes(b"first\nbad \xff byte\n")
    for skipped in ("__pycache__", ".pytest_cache", "src/__pycache__"):
        (sandbox / skipped).mkdir(parents=True, exist_ok=True)
        (sandbox / skipped / "junk.txt").write_text("SKIPPED_MS\n", encoding="utf-8")
    assert designs._added(sandbox, snapshot) == "first\nbad  byte\nNEW_MS = 5"
    assert designs._added(snapshot, snapshot) == ""


def test_replay_practises_the_real_request_twice_on_copies_of_the_snapshot_before_without_expectations(
        tmp_path, spy):
    practice = snapshots(tmp_path)
    fake = practice_chat()
    artifact, _ = sleep("replay", fake, practice)
    assert [(s["mark"], s["request"]) for s in spy] == [("before", REAL)] * 2
    assert all(s["memory"] == MEMORY and s["max_steps"] == 12 for s in spy)
    sandboxes = [s["workspace"] for s in spy]
    assert len(set(sandboxes)) == 2 and all(p.is_relative_to(practice["scratch"]) for p in sandboxes)
    [record] = artifact["practice"]
    assert [(e["request"], e["hypothetical"], e["must_add"], e["must_not_add"]) for e in record["exercises"]] == [
        (REAL, False, [], [])] * 2
    assert len(sleep_calls(fake)) == 2  # extract + merge: no dream, no refine, no review


def test_dreaming_asks_once_for_exercises_with_the_rules_and_the_real_request_and_practises_them_after(
        tmp_path, spy):
    fake = practice_chat()
    artifact, _ = sleep("dreaming", fake, snapshots(tmp_path))
    assert [(s["mark"], s["request"]) for s in spy] == [("after", d["request"]) for d in DREAMS] * 2  # + refined
    assert [{k: e[k] for k in ("request", "must_add", "must_not_add")} for e in artifact["practice"][0]["exercises"]
            ] == DREAMS
    assert all(e["hypothetical"] for e in artifact["practice"][0]["exercises"])
    [dream] = instruction_prompts(fake, designs.DREAM_INSTRUCTION)
    assert dream == (RULES + f"An example of what users ask on this project:\n{REAL}\n\n"
                     + designs.DREAM_INSTRUCTION)


GOOD = {"request": "Add a poll interval of 9 seconds.", "must_add": ["_MS"], "must_not_add": []}


@pytest.mark.parametrize("reply, expected", [
    ("Sure!\n" + dream_lines(GOOD) + "\n{broken\n" + json.dumps({"request": "  ", "must_add": ["_MS"]}),
     [GOOD, None]),
    ("I cannot invent requests.", [None, None]),
    (dream_lines({"request": "no checks"}, {"request": "blank checks", "must_add": ["  ", ""]},
                 {"request": "not a list", "must_add": 7}), [None, None]),
    ("```json\n" + json.dumps([GOOD, {**GOOD, "request": "second"}, {**GOOD, "request": "third"}]) + "\n```",
     [GOOD, {**GOOD, "request": "second"}]),
    (dream_lines({"request": "  lenient  ", "must_add": " _MS ", "must_not_add": [" a", "", 3, "b", "c", "d", "e"]}),
     [{"request": "lenient", "must_add": ["_MS"], "must_not_add": ["a", "b", "c", "d"]}, None]),
])
def test_dreams_without_a_request_and_a_must_add_string_fall_back_to_the_real_request(tmp_path, spy, reply,
                                                                                       expected):
    artifact, _ = sleep("dreaming", practice_chat(dream=reply), snapshots(tmp_path))
    plain = {"request": REAL, "must_add": [], "must_not_add": []}
    got = [{k: e[k] for k in ("request", "hypothetical", "must_add", "must_not_add")}
           for e in artifact["practice"][0]["exercises"]]
    assert got == [{**plain, "hypothetical": False} if d is None else {**d, "hypothetical": True} for d in expected]
    assert [(s["mark"], s["request"]) for s in spy[:2]] == [("after", e["request"]) for e in got]


def test_score_counts_a_failed_check_missing_must_add_and_present_must_not_add_strings(tmp_path, spy):
    dream = {"request": "Add a poll interval.", "must_add": ["_MS", "5000", "POLL"], "must_not_add": ["_SECONDS", "x!"]}
    fake = practice_chat(dream=dream_lines(dream, GOOD), writes=[("notes.txt", "POLL_SECONDS = 5")])
    artifact, _ = sleep("dreaming", fake, snapshots(tmp_path, passing=False))
    first, second = artifact["practice"][0]["exercises"]
    assert (first["score"], first["check_passed"]) == (1 + 2 + 1, False)  # check, _MS and 5000, _SECONDS
    assert (second["score"], second["check_passed"]) == (1 + 1, False)  # check, _MS


def test_scores_of_zero_make_no_refine_call_and_run_exactly_two_sessions_with_this_log(tmp_path, spy):
    fake = practice_chat()
    artifact, cost = sleep("replay", fake, snapshots(tmp_path))
    assert not instruction_prompts(fake, designs.REFINE_INSTRUCTION) and len(spy) == 2
    assert cost["practice_sessions"] == 2 and cost["calls"] == 2
    entry = {"request": REAL, "hypothetical": False, "must_add": [], "must_not_add": [], "score": 0, "score2": None,
             "check_passed": True, "check_passed2": None, "missing": [], "present": [], "missing2": None,
             "present2": None}
    assert artifact == {"items": ITEMS, "practice": [{"exercises": [entry, entry], "refined": False,
                                                      "accepted": False, "items_before": ITEMS, "candidate": None}]}


@pytest.mark.parametrize("design", ["replay", "dreaming"])
def test_practice_record_without_failures_keeps_a_copy_and_no_candidate(tmp_path, design):
    items = [item.copy() for item in ITEMS]
    cost = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "seconds": 0.0}
    fake = practice_chat(writes=[("notes.txt", "POLL_MS = 1")])
    kept, record = designs._practise(design, items, snapshots(tmp_path), fake, 4096, 700, None, cost)
    assert kept is items and record["items_before"] == ITEMS
    assert record["items_before"] is not items
    assert all(before is not original for before, original in zip(record["items_before"], items))
    assert record["candidate"] is None and not record["refined"] and not record["accepted"]
    assert [(e["missing"], e["present"], e["missing2"], e["present2"]) for e in record["exercises"]] == [
        ([], [], None, None)] * 2
    items[0]["text"] = "changed after practice"
    items.append({"kind": "rule", "text": "added after practice"})
    assert record["items_before"] == ITEMS


def test_practice_record_keeps_the_capped_parsed_accepted_candidate_and_both_attempts(tmp_path, spy):
    extra = [("rule", f"extra {i}") for i in range(8)]
    fake = practice_chat(refine="junk\n" + lines(SPECIFIC, LESSON, *extra),
                         writes=scored({(0, False): 2, (1, False): 0, (0, True): 0, (1, True): 0}))
    artifact, _ = sleep("dreaming", fake, snapshots(tmp_path))
    [record] = artifact["practice"]
    expected = REFINED + [{"kind": kind, "text": text} for kind, text in extra[:6]]
    assert record["items_before"] == ITEMS and record["candidate"] == expected
    assert record["refined"] and record["accepted"] and artifact["items"] == expected
    assert [s["memory"] for s in spy[2:]] == [designs.inject("consolidated", {"items": expected})] * 2
    assert [(e["missing"], e["present"], e["missing2"], e["present2"]) for e in record["exercises"]] == [
        (["_MS"], ["_SECONDS"], [], []), ([], [], [], [])]


@pytest.mark.parametrize("candidate", [REFINED, REFINED[:1], []])
def test_practice_record_keeps_rejected_candidates_including_empty_and_both_attempts(tmp_path, spy, candidate):
    fake = practice_chat(refine=json.dumps(candidate), writes=[("notes.txt", "POLL_SECONDS = 1")])
    artifact, _ = sleep("dreaming", fake, snapshots(tmp_path))
    [record] = artifact["practice"]
    assert record["items_before"] == ITEMS and record["candidate"] == candidate
    assert record["refined"] and not record["accepted"] and artifact["items"] == ITEMS
    assert [s["memory"] for s in spy[2:]] == [designs.inject("consolidated", {"items": candidate})] * 2
    assert [(e["missing"], e["present"], e["missing2"], e["present2"]) for e in record["exercises"]] == [
        (["_MS"], ["_SECONDS"], ["_MS"], ["_SECONDS"])] * 2


def test_a_failing_exercise_triggers_one_refine_and_two_more_sessions_with_the_candidate(tmp_path, spy):
    fake = practice_chat(writes=scored({(0, False): 2, (1, False): 0, (0, True): 0, (1, True): 0}))
    artifact, cost = sleep("dreaming", fake, snapshots(tmp_path))
    assert len(instruction_prompts(fake, designs.REFINE_INSTRUCTION)) == 1 and cost["practice_sessions"] == 4
    assert [(s["request"], s["memory"]) for s in spy] == [(d["request"], m) for m in (MEMORY, CANDIDATE)
                                                          for d in DREAMS]
    assert len({s["workspace"] for s in spy}) == 4
    record = artifact["practice"][0]
    assert [(e["score"], e["score2"], e["check_passed"], e["check_passed2"]) for e in record["exercises"]] == [
        (2, 0, True, True), (0, 0, True, True)]
    assert (record["refined"], record["accepted"]) == (True, True) and artifact["items"] == REFINED


@pytest.mark.parametrize("table, accepted", [
    ({(0, False): 2, (1, False): 0, (0, True): 0, (1, True): 0}, True),   # failed one improves, other holds
    ({(0, False): 2, (1, False): 1, (0, True): 1, (1, True): 0}, True),   # both improve
    ({(0, False): 2, (1, False): 0, (0, True): 0, (1, True): 1}, False),  # the other one gets worse
    ({(0, False): 1, (1, False): 1, (0, True): 1, (1, True): 1}, False),  # the sum stays equal
    ({(0, False): 0, (1, False): 2, (0, True): 1, (1, True): 0}, False),  # the first gets worse
])
def test_a_candidate_is_accepted_only_if_no_exercise_gets_worse_and_the_sum_drops(tmp_path, spy, table, accepted):
    artifact, _ = sleep("dreaming", practice_chat(writes=scored(table)), snapshots(tmp_path))
    record = artifact["practice"][0]
    assert [(e["score"], e["score2"]) for e in record["exercises"]] == [(table[0, False], table[0, True]),
                                                                        (table[1, False], table[1, True])]
    assert record["refined"] and record["accepted"] is accepted
    assert artifact["items"] == (REFINED if accepted else ITEMS)


def test_a_candidate_with_fewer_items_is_rejected_even_if_it_scores_better(tmp_path, spy):
    fake = practice_chat(refine=lines(SPECIFIC), writes=scored({(0, False): 2, (1, False): 2,
                                                                (0, True): 0, (1, True): 0}))
    artifact, _ = sleep("dreaming", fake, snapshots(tmp_path))
    record = artifact["practice"][0]
    assert [e["score2"] for e in record["exercises"]] == [0, 0] and not record["accepted"]
    assert artifact["items"] == ITEMS


def test_refine_prompt_is_built_from_the_first_failing_exercise_with_code_generated_problems(tmp_path, spy):
    fake = practice_chat(writes=scored({(0, False): 0, (1, False): 2, (0, True): 0, (1, True): 0}))
    sleep("dreaming", fake, snapshots(tmp_path))
    [refine] = instruction_prompts(fake, designs.REFINE_INSTRUCTION)
    assert refine.startswith(RULES + f"REQUEST: {DREAMS[1]['request']}\n\nWHAT THE AGENT ADDED:\nPOLL_SECONDS = 1\n")
    assert refine.endswith("\n\nPROBLEMS:\nThe added text should contain '_MS' but does not\n"
                           "The added text contains '_SECONDS', which must not appear\n\n" + designs.REFINE_INSTRUCTION)
    assert len(sleep_calls(fake)) == 2 + 1 + 1  # consolidation, dream, refine: nothing else asks the model
    assert not any(p.startswith("USER REQUESTS:\n") for p in prompts(fake))


def test_refine_prompt_reports_a_failed_check_with_the_end_of_its_output(tmp_path, spy):
    fake = practice_chat()
    sleep("replay", fake, snapshots(tmp_path, passing=False))
    [refine] = instruction_prompts(fake, designs.REFINE_INSTRUCTION)
    assert refine.startswith(RULES + f"REQUEST: {REAL}\n\nWHAT THE AGENT ADDED:\npractice\n")
    problems = refine.split("\n\nPROBLEMS:\n")[1].removesuffix("\n\n" + designs.REFINE_INSTRUCTION)
    assert problems.startswith("The project check failed: ") and problems.endswith("checks FAILED (exit 1)")
    assert len(problems) == len("The project check failed: ") + 300


def test_added_text_in_the_refine_prompt_is_capped(tmp_path, spy):
    fake = practice_chat(writes=[(f"f{i}.txt", str(i) * 300) for i in range(4)])
    sleep("replay", fake, snapshots(tmp_path, passing=False))
    [refine] = instruction_prompts(fake, designs.REFINE_INSTRUCTION)
    added = refine.split("WHAT THE AGENT ADDED:\n")[1].split("\n\nPROBLEMS:\n")[0]
    assert added == ("0" * 300 + "\n" + "1" * 300 + "\n" + "2" * 300)[:800]


@pytest.mark.parametrize("text, expected", [
    (MILLIS[1], MILLIS[1]),  # short and terminated: kept
    ("Store durations in _MS. (replaces: Always add new settings to settings.tmpl.json first, then run tools.gen",
     "Store durations in _MS."),
    ("Use _MS. (replaces: " + "x" * 61 + ") Name it X_MS.", "Use _MS."),
    ("Use _MS. (replaces: " + "x" * 60 + ")", "Use _MS. (replaces: " + "x" * 60 + ")"),
    ("r" * 225 + " (replaces: seconds)", "r" * 225),  # the length cap leaves it unterminated
])
def test_parse_items_cuts_replaces_tails_that_are_unterminated_or_too_long(text, expected):
    assert designs.parse_items(lines(("rule", text))) == [{"kind": "rule", "text": expected}]


def test_parse_items_drops_an_item_that_is_only_a_replaces_tail():
    assert designs.parse_items(lines(("rule", "(replaces: everything that came before it"), MILLIS)) == [
        {"kind": "rule", "text": MILLIS[1]}]


def test_merged_items_lose_their_long_replaces_tails(tmp_path):
    tail = " (replaces: Always add new settings to settings.tmpl.json first, then run tools.gen"
    artifact, _ = designs.sleep("consolidated", None, session_events(2), window=4096, reserve=700,
                                chat=memory_chat(lines(MILLIS), lines(("rule", "Store durations in _MS." + tail))))
    assert artifact == {"items": [{"kind": "rule", "text": "Store durations in _MS."}]}


def test_practice_never_touches_the_snapshots(tmp_path, spy):
    practice = snapshots(tmp_path)
    before, after = tree(practice["ws_before"]), tree(practice["ws_after"])
    for design in ("replay", "dreaming"):
        sleep(design, practice_chat(writes=[("notes.txt", "x"), ("src/new.py", "y = 1\n")]), practice)
    assert tree(practice["ws_before"]) == before and tree(practice["ws_after"]) == after
    assert len(spy) == 2 + 4 and all((s["workspace"] / "src" / "new.py").exists() for s in spy)


@pytest.mark.parametrize("design, dream_calls", [("replay", 0), ("dreaming", 1)])
def test_practice_cost_sums_every_call_and_practice_session(tmp_path, spy, design, dream_calls):
    fake = practice_chat()
    _, cost = sleep(design, fake, snapshots(tmp_path, passing=False))
    calls = sleep_calls(fake)
    assert len(calls) == 2 + dream_calls + 1 and cost["practice_sessions"] == 4  # consolidation, dream, refine
    assert cost["calls"] == len(calls) and len(fake.calls) - len(calls) == 4 * 2  # write + finish per session
    assert cost["prompt_tokens"] == 50 * len(fake.calls) and cost["completion_tokens"] == 20 * len(fake.calls)
    assert cost["seconds"] >= 1.5 * len(calls)


def test_every_practice_prompt_fits_the_window(tmp_path, spy):
    window, reserve = 1300, 300
    practice = snapshots(tmp_path, passing=False)
    practice["request"] = "Add a setting. " + "r" * 600
    dreams = [{"request": c * 600, "must_add": [c * 100] * 4, "must_not_add": ["_S" + c * 100] * 4} for c in "de"]
    fake = practice_chat(writes=[(f"f{i}.txt", str(i) * 600) for i in range(4)], dream=dream_lines(*dreams),
                         refine=lines(*[("rule", f"rule {i} " + "z" * 100) for i in range(8)]))
    for design in ("replay", "dreaming"):
        sleep(design, fake, practice, window=window, reserve=reserve)
    refines = instruction_prompts(fake, designs.REFINE_INSTRUCTION)
    assert len(refines) == 2 and len(instruction_prompts(fake, designs.DREAM_INSTRUCTION)) == 1
    shown = [p.split("WHAT THE AGENT ADDED:\n")[1].split("\n\nPROBLEMS:\n")[0] for p in refines]
    assert all(s.startswith("000") for s in shown) and min(map(len, shown)) < designs.ADDED_CAP  # trimmed to fit
    starts = [c for c in fake.calls if c["tools"] and len(c["messages"]) == 2]  # later steps: the agent's own ratio
    assert len(starts) == 8
    for call in sleep_calls(fake) + starts:
        size = (agent.estimate_prompt(call["messages"]) if call["tools"]
                else llm.estimate_tokens(call["messages"][0]["content"]))
        assert size <= window - reserve and call["max_tokens"] == reserve


def test_practice_seeds_are_distinct_and_stay_within_this_sleeps_hundred(tmp_path, spy):
    fake = practice_chat()
    sleep("dreaming", fake, snapshots(tmp_path, passing=False), seed=1000)
    consolidation = 2
    seeds = [c["seed"] for c in sleep_calls(fake)[consolidation:]]
    seeds += [s["seed"] + step for s in spy for step in range(1, s["max_steps"] + 1)]
    assert len(spy) == 4 and len(seeds) == 2 + 4 * 12
    assert len(set(seeds)) == len(seeds) and all(1000 < s < 1100 for s in seeds)
    fake = practice_chat()
    sleep("dreaming", fake, snapshots(tmp_path / "again"))
    assert {c["seed"] for c in fake.calls} == {None}


def test_dreaming_and_replay_inject_only_their_items():
    artifact = {"items": [{"kind": "rule", "text": MILLIS[1]}],
                "practice": [{"exercises": [{"request": "SECRET_DREAM_NAME"}], "refined": False, "accepted": False}]}
    for design in ("dreaming", "replay"):
        assert designs.inject(design, artifact) == f"- RULE: {MILLIS[1]}" and designs.inject(design, None) == ""


def test_second_practice_sleep_consolidates_into_the_first_items(tmp_path, spy):
    fake = practice_chat()
    first, _ = sleep("replay", fake, snapshots(tmp_path))
    second, _ = designs.sleep("replay", first, session_events(2), chat=fake, window=4096, reserve=700,
                              practice=snapshots(tmp_path / "s2"))
    merges = instruction_prompts(fake, designs.MERGE_INSTRUCTION)
    assert len(merges) == 2 and merges[1].startswith(f"CURRENT memory:\n{lines(MILLIS, LESSON)}\n\nNEW items:\n")
    assert len(second["practice"]) == 1  # each sleep returns only its own record


def test_contamination_lists_identifiers_that_only_a_dream_put_into_memory():
    memory = ("- RULE: name delays like `RETRY_DELAY_MS`; TIMEOUT_MS stays.\n- RULE: CACHE_TTL_MS = 5000; "
              "BACKOFF_MS too; LOG_LEVEL_NAME; `abcd`")

    def record(*exercises):
        return {"exercises": [{"request": r, "hypothetical": h} for r, h in exercises], "refined": False,
                "accepted": False}
    log = [record(("Add CACHE_TTL_MS, `RETRY_DELAY_MS` and SHORT.", True), ("Set LOG_LEVEL_NAME.", False)),
           record(("Add TIMEOUT_MS, BACKOFF_MS and `abcd`.", True))]
    wake = [[{"type": "user", "text": "Rename TIMEOUT_S to TIMEOUT_MS."}],
            [{"type": "user", "text": "s2"}, {"type": "result", "step": 1, "text": "BACKOFF_MS = 1"}]]
    assert designs.contamination(memory, log, wake) == ["CACHE_TTL_MS", "RETRY_DELAY_MS"]
    assert designs.contamination(memory, [], wake) == []


def test_practice_instruction_texts():
    assert designs.EXERCISES == 2
    assert designs.DREAM_INSTRUCTION.startswith("Write 2 practice exercises that test whether an agent applies")
    assert "without naming files, setting names, suffixes or conventions" in designs.DREAM_INSTRUCTION
    assert designs.DREAM_INSTRUCTION.endswith(
        'one object per line: {"request": "...", "must_add": ["..."], "must_not_add": ["..."]}')
    assert designs.REFINE_INSTRUCTION.startswith("An agent that was given the rules above got this request")
    assert "Never reverse or weaken a rule." in designs.REFINE_INSTRUCTION
    assert designs.REFINE_INSTRUCTION.endswith('{"kind": "rule" or "lesson", "text": "..."}')


# --- review ---------------------------------------------------------------------------------------

def review_chat(reply):
    return FakeChat(lambda prompt: reply if prompt.startswith("USER REQUESTS:\n") else lines(MILLIS, LESSON))


@pytest.mark.parametrize("number, text, expected", [
    (1, "  RULE: Store every duration in milliseconds with the _MS suffix.  ",
     [{"kind": "rule", "text": "Store every duration in milliseconds with the _MS suffix."}, ITEMS[1]]),
    (0, "LESSON: List every setting in CHANGES.md.",
     [{"kind": "rule", "text": "List every setting in CHANGES.md."}] + ITEMS),
])
def test_review_applies_one_edit_with_schema_prompt_record_and_exact_extra_cost(number, text, expected):
    requests = ["Use milliseconds.\nFor all durations.", "Edit the template."]
    fake = review_chat(json.dumps({"number": number, "text": text}))
    artifact, cost = sleep("review", fake, {"requests": requests}, seed=42)
    consolidated, base_cost = sleep("consolidated", review_chat("unused"), None, seed=42)
    assert artifact == {"items": expected, "review": [{"number": number, "text": text, "applied": True,
                                                    "reason": "", "items_before": ITEMS}]}
    assert len(fake.calls) == base_cost["calls"] + 1
    assert cost == {"calls": base_cost["calls"] + 1, "prompt_tokens": base_cost["prompt_tokens"] + 50,
                    "completion_tokens": base_cost["completion_tokens"] + 20, "seconds": base_cost["seconds"] + 1.5}
    assert "practice" not in artifact and consolidated == {"items": ITEMS}
    *ordinary, review = fake.calls
    assert all("response_format" not in call for call in ordinary)
    assert review["response_format"] == designs._schema(
        "edit", {"number": {"type": "integer"}, "text": {"type": "string"}}, ["number", "text"])
    assert review["messages"] == [{"role": "user", "content": (
        "USER REQUESTS:\n1. Use milliseconds. For all durations.\n2. Edit the template.\n\n"
        + RULES + designs.REVIEW_INSTRUCTION)}]
    assert review["max_tokens"] == 700 and review["seed"] == 43
    before = artifact["review"][0]["items_before"]
    assert all(a is not b for a, b in zip(before, artifact["items"][-2:]))
    artifact["items"][-1]["text"] = "changed later"
    assert before == ITEMS


@pytest.mark.parametrize("reply, number, text, reason", [
    ('{"number": -1, "text": ""}', -1, "", ""),
    ("not JSON", None, None, "invalid reply"),
    ('[{"number": 0, "text": "new"}]', None, None, "invalid reply"),
    ('{"number": "0", "text": "new"}', None, None, "invalid reply"),
    ('{"number": true, "text": "new"}', None, None, "invalid reply"),
    ('{"number": 1.0, "text": "new"}', None, None, "invalid reply"),
    ('{"text": "new"}', None, None, "invalid reply"),
    ('```json\n{"number": 0, "text": "new"}\n```', None, None, "invalid reply"),
    ('{"number": 0, "text": "a"}\n{"number": 0, "text": "b"}', None, None, "invalid reply"),
    ('{"number": 3, "text": "new"}', 3, "new", "bad number"),
    ('{"number": 0, "text": "RULE: "}', 0, "RULE: ", "empty text"),
    ('{"number": 0}', 0, None, "empty text"),
    ('{"number": 0, "text": 7}', 0, 7, "empty text"),
])
def test_review_records_no_edit_malformed_reply_and_refusals(reply, number, text, reason):
    fake = review_chat(reply)
    artifact, cost = sleep("review", fake, {"requests": [REAL]})
    assert artifact == {"items": ITEMS, "review": [{"number": number, "text": text, "applied": False,
                                                 "reason": reason, "items_before": ITEMS}]}
    assert len(fake.calls) == cost["calls"] == 3
    assert cost == {"calls": 3, "prompt_tokens": 150, "completion_tokens": 60, "seconds": 4.5}
    assert fake.calls[-1]["seed"] is None


def test_review_records_identifier_refusal_and_retains_consolidated_items():
    items = [{"kind": "rule", "text": "Use TIMEOUT_MS."}]
    fake = FakeChat(lambda prompt: '{"number": 1, "text": "Use seconds."}'
                    if prompt.startswith("USER REQUESTS:\n") else json.dumps(items))
    artifact, _ = sleep("review", fake, {"requests": [REAL]})
    assert artifact == {"items": items, "review": [{"number": 1, "text": "Use seconds.", "applied": False,
                                                 "reason": "edit drops identifiers", "items_before": items}]}


@pytest.mark.parametrize("practice", [None, {}, {"requests": []}])
def test_review_skips_extra_call_without_an_archive(practice):
    fake = review_chat("unused")
    artifact, cost = sleep("review", fake, practice)
    assert artifact == {"items": ITEMS, "review": []} and len(fake.calls) == cost["calls"] == 2


@pytest.mark.parametrize("room", [100, -50])
def test_review_prompt_fits_by_cutting_archive_before_rules(room):
    items = [{"kind": "rule", "text": "x" * 200} for _ in range(4)]
    rules = designs._numbered(items)
    base = f"USER REQUESTS:\n\n\nRULES:\n{rules}\n\n" + designs.REVIEW_INSTRUCTION
    reserve = 100
    budget = llm.estimate_tokens(base) + room
    fake = FakeChat(lambda prompt: '{"number": -1, "text": ""}'
                    if prompt.startswith("USER REQUESTS:\n") else "no candidates")
    artifact, _ = designs.sleep("review", {"items": items}, session_events(1), chat=fake,
                               window=budget + reserve + designs.FRAMING_TOKENS, reserve=reserve,
                               practice={"requests": ["r" * 5000]})
    prompt = prompts(fake)[-1]
    assert llm.estimate_tokens(prompt) <= budget
    archive, shown_rules = prompt.removeprefix("USER REQUESTS:\n").split("\n\nRULES:\n")
    shown_rules = shown_rules.removesuffix("\n\n" + designs.REVIEW_INSTRUCTION)
    if room > 0:
        assert archive.startswith("1. rrr") and len(archive) < 5000 and shown_rules == rules
    else:
        assert archive == "" and rules.startswith(shown_rules) and len(shown_rules) < len(rules)
    assert artifact["items"] == items


def test_review_injects_only_items_exactly_like_consolidated():
    artifact = {"items": ITEMS, "review": [{"text": "SECRET_ARCHIVE"}], "requests": ["SECRET_ARCHIVE"]}
    assert designs.inject("review", artifact) == designs.inject("consolidated", artifact)
    assert designs.inject("review", None) == designs.inject("review", {"items": []}) == ""


def test_review_instruction_text_is_pinned():
    assert designs.REVIEW_INSTRUCTION == (
        "Above are the user's requests on this project, word for word, oldest first, and the RULES the agent will "
        "carry into its next session. Check that every standing decision of the user (a naming, format, unit or "
        "process policy meant for all future work, not a one-off task) is stated in the rules as a general "
        "instruction that keeps the user's exact terms. If one is missing, too narrow or out of date, fix it by "
        "editing exactly ONE rule: give its number, or 0 to add a new rule, and the full new text of that rule (at "
        "most 35 words). If the rules already cover every standing decision, answer with number -1 and an empty "
        "text.")
