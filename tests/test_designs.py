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


# --- hybrid ---------------------------------------------------------------------------------------

@pytest.mark.parametrize("tail, finished", [
    ([], False),
    ([{"type": "result", "step": 31, "text": "finished"}], True),
    ([{"type": "result", "step": 31, "text": "finished"},
      {"type": "result", "step": 32, "text": "later"}], True),
    ([{"type": "say", "step": 31, "text": "finished"}], False),
    ([{"type": "result", "step": 31, "text": "finished "}], False),
    ([{"type": "result", "step": 31, "text": "finished\n" + agent.IGNORED.format(n=1)}], True),
])
def test_hybrid_sleep(tail, finished):
    request = "  Exact request\r\nwith lines\nand spaces.  "
    events = session_events(30, request) + tail
    old = {"items": [{"kind": "rule", "text": SECONDS[1]}],
           "state": {"request": "OLD STATE", "finished": True}}
    hybrid_chat = memory_chat(lines(MILLIS), lines(MILLIS, LESSON))
    base_chat = memory_chat(lines(MILLIS), lines(MILLIS, LESSON))
    artifact, cost = designs.sleep("hybrid", old, events, chat=hybrid_chat, window=1000, reserve=300, seed=42,
                                   practice={"requests": [request]})
    base, base_cost = designs.sleep("consolidated", old, events, chat=base_chat, window=1000, reserve=300, seed=42)
    assert artifact == {"items": base["items"], "state": {"request": request, "finished": finished}}
    assert hybrid_chat.calls == base_chat.calls and len(hybrid_chat.calls) > 2 and cost == base_cost
    assert prompts(hybrid_chat)[-1].startswith(f"CURRENT memory:\n{lines(SECONDS)}\n\nNEW items:")
    assert "OLD STATE" not in "\n".join(prompts(hybrid_chat))


@pytest.mark.parametrize("first_finished", [False, True])
def test_hybrid_latest_state(first_finished):
    first_events = session_events(1, "  first\nrequest  ")
    second_events = session_events(1, "second\r\nrequest")
    done = {"type": "result", "step": 2, "text": "finished"}
    (first_events if first_finished else second_events).append(done)
    first, _ = designs.sleep("hybrid", None, first_events, chat=memory_chat(lines(SECONDS), lines(SECONDS)),
                             window=4096, reserve=700)
    saved = json.loads(json.dumps(first))
    fake = memory_chat(lines(MILLIS), lines(MILLIS, LESSON))
    second, _ = designs.sleep("hybrid", first, second_events, chat=fake, window=4096, reserve=700)
    assert first == saved
    assert first["state"] == {"request": first_events[0]["text"], "finished": first_finished}
    assert second == {"items": [{"kind": k, "text": t} for k, t in (MILLIS, LESSON)],
                      "state": {"request": second_events[0]["text"], "finished": not first_finished}}
    assert prompts(fake)[-1].startswith(f"CURRENT memory:\n{lines(SECONDS)}\n\nNEW items:")
    rendered = designs.inject("hybrid", second)
    prefix = "- LAST SESSION (NOT finished - continue this work): " if first_finished else "- LAST SESSION (finished): "
    assert rendered == prefix + "second request\n" + designs.inject("consolidated", second)


@pytest.mark.parametrize("finished, cap", [(True, 160), (False, 400)])
@pytest.mark.parametrize("extra", [-1, 0, 1, 1000])
def test_hybrid_state_caps(finished, cap, extra):
    assert (designs.STATE_DONE_CAP, designs.STATE_OPEN_CAP) == (160, 400)
    start = "  first\r\nsecond\rthird\n  "
    flat_start = "  first second third   "
    request = start + "x" * (cap + extra - len(flat_start))
    flat = flat_start + "x" * (cap + extra - len(flat_start))
    artifact = {"items": [], "state": {"request": request, "finished": finished}}
    prefix = "- LAST SESSION (finished): " if finished else "- LAST SESSION (NOT finished - continue this work): "
    expected = flat[:cap - len(" [...]")] + " [...]" if extra > 0 else flat
    assert designs.inject("hybrid", artifact) == prefix + expected
    assert len(expected) == min(cap, len(flat))
    assert artifact["state"]["request"] == request


@pytest.mark.parametrize("finished, cap, count", [(True, 160, 3), (False, 400, 2)])
@pytest.mark.parametrize("extra", [-1, 0, 1])
def test_hybrid_note_cap(finished, cap, count, extra):
    prefix = "- LAST SESSION (finished): " if finished else "- LAST SESSION (NOT finished - continue this work): "
    state_line = prefix + "s" * cap
    items = [{"kind": "rule", "text": str(i) * 240} for i in range(count - 1)]
    before = state_line + "\n" + designs.inject("consolidated", {"items": items})
    remaining = designs.NOTE_CAP - len(before + "\n- LESSON: ")
    items += [{"kind": "lesson", "text": "last " + "z" * (remaining + extra - len("last "))},
              {"kind": "rule", "text": "drop this rule"}]
    artifact = {"items": items, "state": {"request": "s" * cap, "finished": finished}}
    saved = json.loads(json.dumps(artifact))
    kept = count if extra <= 0 else count - 1
    expected = state_line + "\n" + designs.inject("consolidated", {"items": items[:kept]})
    rendered = designs.inject("hybrid", artifact)
    assert rendered == expected and len(rendered) <= designs.NOTE_CAP
    assert len(rendered + "\n" + designs.inject("consolidated", {"items": [items[kept]]})) > designs.NOTE_CAP
    if extra <= 0:
        assert len(rendered) == designs.NOTE_CAP + extra
    assert artifact == saved


@pytest.mark.parametrize("artifact", [None, {}, "", []])
def test_hybrid_empty(artifact):
    assert designs.inject("hybrid", artifact) == ""


# --- dreaming and replay --------------------------------------------------------------------------

TASK = "practice_tiny"
REAL = "Rename TIMEOUT_S to TIMEOUT_MS with value 30000. Make the checks pass."
DECISION = "Store every duration in milliseconds with the _MS suffix."
DREAMS = [{"decision": DECISION, "request": request, "must_add": ["_MS"], "must_not_add": ["_SECONDS"]}
          for request in ("Add a cache lifetime of 5 seconds.", "Add a poll interval of 9 seconds.")]
SPECIFIC = ("rule", "Store every duration in milliseconds with _MS; convert requests in seconds.")
ITEMS = [{"kind": k, "text": t} for k, t in (MILLIS, LESSON)]
REFINED = [{"kind": k, "text": t} for k, t in (SPECIFIC, LESSON)]
MEMORY = designs.inject("consolidated", {"items": ITEMS})
CANDIDATE = designs.inject("consolidated", {"items": REFINED})
RULES = f"RULES:\n1. RULE: {MILLIS[1]}\n2. LESSON: {LESSON[1]}\n\n"
WRITTEN = {0: "POLL_MS = 1", 1: "nothing", 2: "POLL_SECONDS = 1"}


def dream_reply(*dreams):
    return json.dumps({"exercises": list(dreams)})


def sleep(design, fake, practice, window=4096, reserve=700, seed=None):
    return designs.sleep(design, None, session_events(2), chat=fake, window=window, reserve=reserve, seed=seed,
                         practice=practice)


def test_dreams_clean_grounded_strings_and_keep_rejected_entries():
    decision = "Always use aa bb cc dd ee and _MS for every future duration."
    dream = {"decision": "  " + decision + "  ", "request": "  Make a timer.  ",
             "must_add": [None, "", "x", "12", " 1., 2 ", "zz", "AA", "aa", "bb", "cc", "dd", "ee"],
             "must_not_add": [7, "", "ab", "123", " 3., 4 ", "aa", "AAA", "BBB", "CCC", "DDD", "EEE"]}
    [got] = designs._dreams(dream_reply(dream, {}, {"request": " "}, {"request": 3}), [decision])
    assert got == {"request": "Make a timer.", "hypothetical": True, "decision": decision,
                   "must_add": ["aa", "bb", "cc", "dd"], "must_not_add": ["AAA", "BBB", "CCC", "DDD"],
                   "rejected": ""}
    [got] = designs._dreams(dream_reply({**DREAMS[0], "must_add": "_MS", "must_not_add": "_SECONDS"}), [DECISION])
    assert got["must_add"] == ["_MS"] and got["must_not_add"] == ["_SECONDS"]


@pytest.mark.parametrize("reply", ["junk", "[]", "{}", '{"exercises": {}}',
                                   json.dumps(DREAMS), json.dumps(DREAMS[0]),
                                   "```json\n" + dream_reply(*DREAMS) + "\n```",
                                   dream_reply(*DREAMS) + "\n{}"])
def test_dreams_require_one_object_with_an_exercises_list(reply):
    assert designs._dreams(reply, [DECISION]) == []


@pytest.mark.parametrize("changes, reason", [
    ({"decision": "Use _MS."}, "decision is not the user's words"),
    ({"decision": "Always store future durations using _MS."}, "decision is not the user's words"),
    ({"decision": None}, "decision is not the user's words"),
    ({"must_add": ["not quoted", "1", "x"]}, "no grounded must_add"),
    ({"request": "Add a timer using _ms."}, "request contains a check string"),
    ({"request": "Add a timer using _seconds."}, "request contains a check string"),
    ({"request": "Add `timer`."}, "request names an identifier or file"),
    ({"request": "Add timer_value."}, "request names an identifier or file"),
    ({"request": "Add timer_."}, "request names an identifier or file"),
    *[({"request": "Edit folder/file." + ext}, "request names an identifier or file")
      for ext in ("py", "json", "md", "txt", "toml", "yaml", "yml", "cfg", "ini")],
    ({"decision": "bad", "must_add": [], "request": "Edit x.py with _MS."}, "decision is not the user's words"),
    ({"must_add": [], "request": "Edit x.py."}, "no grounded must_add"),
    ({"request": "Edit x.py with _MS."}, "request contains a check string"),
])
def test_dreams_reject_first_grounding_or_request_problem(changes, reason):
    [got] = designs._dreams(dream_reply({**DREAMS[0], **changes}), [DECISION])
    assert got["rejected"] == reason


def test_dreams_match_decisions_across_odd_whitespace_but_ground_strings_exactly():
    decision = "Store every\tduration in milliseconds\nwith the _MS suffix."
    [got] = designs._dreams(dream_reply({**DREAMS[0], "decision": decision}),
                          ["Please do this: Store  every\r\nduration in milliseconds with the _MS suffix. Thanks."])
    assert got["decision"] == decision and got["rejected"] == ""
    [got] = designs._dreams(dream_reply({**DREAMS[0], "decision": decision, "must_add": ["every duration"]}),
                          [DECISION])
    assert got["rejected"] == "no grounded must_add"


@pytest.fixture
def practice(tmp_path, monkeypatch):
    task_dir = tmp_path / "tasks" / TASK
    task_dir.mkdir(parents=True)
    (task_dir / "task.json").write_text(json.dumps({
        "check_cmd": ["python checks/check.py"], "protected": ["checks", "absent"]}), encoding="utf-8")
    monkeypatch.setattr(tasks, "TASKS_DIR", task_dir.parent)
    for mark in ("before", "after"):
        root = tmp_path / mark
        (root / "checks").mkdir(parents=True)
        (root / "checks" / "check.py").write_text(
            "from pathlib import Path\nassert Path('ok.txt').read_text() == 'yes'\n", encoding="utf-8")
        (root / "ok.txt").write_text("yes", encoding="utf-8")
        (root / "MARK.txt").write_text(mark, encoding="utf-8")
    return {"task_id": TASK, "request": REAL, "requests": [DECISION, REAL],
            "ws_before": tmp_path / "before", "ws_after": tmp_path / "after",
            "scratch": tmp_path / "scratch", "max_steps": 20}


@pytest.fixture
def spy(monkeypatch):
    real, seen = agent.run_session, []

    def run_session(task_id, workspace, prompt, **kwargs):
        result = real(task_id, workspace, prompt, **kwargs)
        seen.append({"workspace": Path(workspace), "request": prompt, **kwargs, "result": result})
        return result
    monkeypatch.setattr(agent, "run_session", run_session)
    return seen


def tree(root):
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in Path(root).rglob("*") if p.is_file()
            and not agent.SKIP_DIRS & set(p.relative_to(root).parts)}


def practice_chat(dream=None, repair=None, writes=None):
    state = {"queue": [], "attempt": 0}
    dream = dream_reply(*DREAMS) if dream is None else dream
    repair = json.dumps({"number": 1, "text": SPECIFIC[1]}) if repair is None else repair

    def chat(messages, tools=None, **kwargs):
        chat.calls.append({"messages": messages, "tools": tools, **kwargs})
        content, calls = "", []
        if tools is not None:
            if not any(m["role"] == "assistant" for m in messages):
                request = messages[1]["content"].partition(agent.MEMORY.format(memory=""))[0]
                memory = "\n".join(m["content"] for m in messages)
                plan = writes(request, memory, state["attempt"]) if callable(writes) else writes
                plan = [("notes.txt", "POLL_MS = 1")] if plan is None else plan
                state["attempt"] += 1
                state["queue"] = []
                for path, text in plan:
                    state["queue"] += [("read_file", {"path": path}),
                                       ("write_file", {"path": path, "content": text})]
            name, args = state["queue"].pop(0) if state["queue"] else ("finish", {"summary": "done"})
            calls = [{"name": name, "arguments": args, "raw": json.dumps(args)}]
        else:
            prompt = messages[0]["content"]
            if prompt.endswith(designs.EXTRACT_INSTRUCTION):
                content = lines(MILLIS)
            elif prompt.endswith(designs.MERGE_INSTRUCTION):
                content = lines(MILLIS, LESSON)
            elif prompt.endswith(designs.DREAM_INSTRUCTION):
                content = dream
            else:
                content = repair
        return {"content": content, "tool_calls": calls, "usage": {"prompt_tokens": 50, "completion_tokens": 20},
                "seconds": 1.5, "finish_reason": "tool_calls" if calls else "stop"}
    chat.calls = []
    return chat


def instruction_calls(fake, instruction):
    return [c for c in fake.calls if c["tools"] is None and c["messages"][0]["content"].endswith(instruction)]


def assert_record(record):
    assert set(record) == {"exercises", "dreams_offered", "dream_replies", "items_before", "edit", "candidate",
                           "refined", "accepted", "reason"}
    assert all(set(e) == {"request", "hypothetical", "decision", "must_add", "must_not_add", "score",
                          "check_passed", "missing", "present", "scores2"} for e in record["exercises"])
    assert all(set(d) == {"decision", "request", "must_add", "must_not_add", "rejected", "control_score"}
               for d in record["dreams_offered"])


def test_attempt_restores_protected_folders_before_check_and_added_text(practice, spy):
    (practice["ws_before"] / "ok.txt").write_text("no", encoding="utf-8")
    fake = practice_chat(repair='{"number": -1, "text": ""}', writes=[
        ("checks/check.py", "# CHEAT_MS\n"), ("absent/cheat.txt", "CHEAT_MS")])
    before = tree(practice["ws_before"])
    artifact, cost = sleep("replay", fake, practice)
    [record] = artifact["practice"]
    assert_record(record)
    assert all(e["score"] == 1 and not e["check_passed"] for e in record["exercises"])
    assert tree(practice["ws_before"]) == before
    assert len({s["workspace"] for s in spy}) == cost["practice_sessions"] == 5
    for session in spy:
        sandbox = session["workspace"]
        assert sandbox != practice["ws_before"] and sandbox.is_relative_to(practice["scratch"])
        assert tree(sandbox)["checks/check.py"] == before["checks/check.py"]
        assert not (sandbox / "absent").exists()
        assert designs._added(sandbox, practice["ws_before"]) == ""
    prompt = instruction_calls(fake, designs.REPAIR_INSTRUCTION)[0]["messages"][0]["content"]
    assert "CHEAT_MS" not in prompt and "WHAT THE AGENT ADDED:\n(nothing)" in prompt


@pytest.mark.parametrize("options", [{}, {"memory_at": "tail", "finish_gate": True},
                                    {"memory_at": "request", "finish_gate": True}])
def test_controls_use_only_decision_and_practice_uses_design_placement(practice, spy, options):
    practice.update(options)
    artifact, cost = sleep("dreaming", practice_chat(), practice)
    [record] = artifact["practice"]
    assert_record(record)
    assert record["reason"] == "all exercises passed" and not record["refined"]
    assert cost["practice_sessions"] == 4 and len(spy) == 4
    assert [s["memory"] for s in spy[:2]] == ["- USER DECISION: " + DECISION] * 2
    assert [(s["memory_at"], s["finish_gate"]) for s in spy[:2]] == [("request", False)] * 2
    assert [(s["memory_at"], s["finish_gate"]) for s in spy[2:]] == [
        (options.get("memory_at", "system"), options.get("finish_gate", False))] * 2
    assert all(s["memory"] == MEMORY for s in spy[2:])
    assert [s["result"]["gated"] for s in spy] == [False, False] + [options.get("finish_gate", False)] * 2
    assert all((s["workspace"] / "MARK.txt").read_text() == "after" for s in spy)
    assert all(e["scores2"] is None for e in record["exercises"])


@pytest.mark.parametrize("scores, controls, verified", [
    ([0, 0, 0, 0], [0, 0, None, None], [0, 1]),
    ([1, 0, 0, 0], [1, 0, 0, None], [1, 2]),
    ([1, 1, 1, 0], [1, 1, 1, None], []),
    ([1, 1, 0, 0], [1, 1, 0, None], [2]),
])
def test_controls_verify_in_order_stop_at_two_or_three_and_never_fallback(practice, spy, scores, controls, verified):
    dreams = [{**DREAMS[0], "request": f"Make timer number {i}."} for i in range(4)]
    rejected = {**DREAMS[0], "request": "Edit bad.py."}
    def writes(request, memory, attempt):
        index = [d["request"] for d in dreams].index(request)
        score = scores[index] if "USER DECISION:" in memory else 0
        return [("notes.txt", WRITTEN[score])]
    artifact, cost = sleep("dreaming", practice_chat(dream=dream_reply(rejected, *dreams), writes=writes), practice)
    [record] = artifact["practice"]
    assert_record(record)
    assert [d["control_score"] for d in record["dreams_offered"]] == [None] + controls
    assert [e["request"] for e in record["exercises"]] == [dreams[i]["request"] for i in verified]
    assert cost["practice_sessions"] == sum(s is not None for s in controls) + len(verified)
    assert record["reason"] == ("all exercises passed" if verified else "no verified dreams")
    assert artifact["items"] == ITEMS and record["candidate"] is None and record["edit"] is None
    assert not record["accepted"] and not record["refined"]
    assert all(s["request"] != REAL for s in spy)


@pytest.mark.parametrize("reply", ["not JSON", '{"exercises": []}', dream_reply({**DREAMS[0], "decision": "invented"})])
def test_no_usable_dream_ends_without_attempt_or_fallback(practice, spy, reply):
    artifact, cost = sleep("dreaming", practice_chat(dream=reply), practice)
    [record] = artifact["practice"]
    assert_record(record)
    assert record["reason"] == "no verified dreams" and record["exercises"] == []
    assert record["dream_replies"] == [reply] and record["candidate"] is None and record["edit"] is None
    assert artifact["items"] == ITEMS and cost["practice_sessions"] == 0 and spy == []


def test_generation_uses_archive_schema_budget_and_capped_raw_reply(practice):
    practice["requests"] = ["old\nline " + "x" * 6000, DECISION]
    fake = practice_chat(dream="z" * 3000)
    artifact, _ = sleep("dreaming", fake, practice, window=1600, reserve=300)
    [call] = instruction_calls(fake, designs.DREAM_INSTRUCTION)
    prompt = call["messages"][0]["content"]
    assert prompt.startswith("USER REQUESTS:\n1. old line xxx") and "RULES:\n" not in prompt
    assert prompt.endswith("\n\n" + designs.DREAM_INSTRUCTION)
    assert llm.estimate_tokens(prompt) <= 1600 - 300 - designs.FRAMING_TOKENS
    assert len(prompt) < len(designs._archive(practice["requests"])) + len(designs.DREAM_INSTRUCTION)
    assert call["response_format"] == designs._schema("exercises", {"exercises": {
        "type": "array", "maxItems": 4, "items": {"type": "object", "properties": {
            "decision": {"type": "string"}, "request": {"type": "string"},
            "must_add": {"type": "array", "items": {"type": "string"}},
            "must_not_add": {"type": "array", "items": {"type": "string"}}},
            "required": ["decision", "request", "must_add", "must_not_add"]}}}, ["exercises"])
    assert artifact["practice"][0]["dream_replies"] == ["z" * 2000]


@pytest.mark.parametrize("second, accepted", [([0, 0, 0], True), ([1, 0, 0], True),
    ([0, 2, 0], False), ([2, 0, 0], False), ([0, 0, 1], False)])
def test_repair_edits_one_rule_and_requires_both_confirmations_and_other_not_worse(practice, spy, second, accepted):
    def writes(request, memory, attempt):
        if "USER DECISION:" in memory:
            score = 0
        elif SPECIFIC[1] in memory:
            score = second[attempt - 4]
        else:
            score = 2 if request == DREAMS[0]["request"] else 0
        return [("notes.txt", WRITTEN[score])]
    fake = practice_chat(writes=writes)
    practice.update(memory_at="tail", finish_gate=True)
    artifact, cost = sleep("dreaming", fake, practice)
    [record] = artifact["practice"]
    assert_record(record)
    assert record["items_before"] == ITEMS and record["candidate"] == REFINED
    assert record["edit"] == {"number": 1, "text": SPECIFIC[1]}
    assert record["refined"] and record["accepted"] is accepted
    assert record["reason"] == ("" if accepted else "not confirmed")
    assert [e["score"] for e in record["exercises"]] == [2, 0]
    assert [e["scores2"] for e in record["exercises"]] == [second[:2], second[2:]]
    assert record["exercises"][0]["missing"] == ["_MS"] and record["exercises"][0]["present"] == ["_SECONDS"]
    assert artifact["items"] == (REFINED if accepted else ITEMS) and cost["practice_sessions"] == 7
    assert [s["request"] for s in spy[4:]] == [DREAMS[0]["request"]] * 2 + [DREAMS[1]["request"]]
    assert all(s["memory"] == CANDIDATE and s["memory_at"] == "tail" and s["finish_gate"] for s in spy[4:])
    [call] = instruction_calls(fake, designs.REPAIR_INSTRUCTION)
    assert call["messages"][0]["content"] == (RULES + f"USER DECISION: {DECISION}\n\n"
        f"REQUEST: {DREAMS[0]['request']}\n\nWHAT THE AGENT ADDED:\nPOLL_SECONDS = 1\n\nPROBLEMS:\n"
        "The added text should contain '_MS' but does not\n"
        "The added text contains '_SECONDS', which must not appear\n\n" + designs.REPAIR_INSTRUCTION)
    assert call["response_format"] == designs._schema(
        "edit", {"number": {"type": "integer"}, "text": {"type": "string"}}, ["number", "text"])
    assert all(a is not b for a, b in zip(record["items_before"], artifact["items"]))


@pytest.mark.parametrize("reply, edit, reason", [
    ('{"number": -1, "text": ""}', {"number": -1, "text": ""}, "no edit"),
    ("bad JSON", None, "no edit"), ('[]', None, "no edit"),
    ('{"number": true, "text": "new"}', None, "no edit"),
    ('{"number": "1", "text": "new"}', None, "no edit"),
    ('{"number": 5, "text": "new"}', {"number": 5, "text": "new"}, "bad number"),
    ('{"number": 1}', {"number": 1, "text": None}, "empty text"),
    ('{"number": 1, "text": 7}', {"number": 1, "text": 7}, "empty text"),
    ('{"number": 2, "text": "Use seconds."}', {"number": 2, "text": "Use seconds."}, "edit drops identifiers"),
])
def test_repair_no_edit_invalid_reply_and_refusal_leave_rules_unchanged(practice, spy, reply, edit, reason):
    fake = practice_chat(repair=reply, writes=lambda r, m, i: [("notes.txt", WRITTEN[0 if i < 2 else 1])])
    expected = ITEMS
    if reason == "edit drops identifiers":
        expected = [ITEMS[0], {"kind": "rule", "text": "Keep TIMEOUT_MS."}]
        cost = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "seconds": 0.0}
        kept, record = designs._practise("dreaming", expected, practice, fake, 4096, 700, None, cost)
        artifact = {"items": kept, "practice": [record]}
    else:
        artifact, cost = sleep("dreaming", fake, practice)
    [record] = artifact["practice"]
    assert_record(record)
    assert record["edit"] == edit and record["reason"] == reason
    assert record["candidate"] is None and not record["refined"] and not record["accepted"]
    assert all(e["scores2"] is None for e in record["exercises"])
    assert artifact["items"] == expected and len(spy) == cost["practice_sessions"] == 4


@pytest.mark.parametrize("repair", [False, True])
def test_replay_runs_five_real_requests_before_snapshot_and_repairs_once(practice, spy, repair):
    if repair:
        (practice["ws_before"] / "ok.txt").write_text("no", encoding="utf-8")
    def writes(request, memory, attempt):
        return [("ok.txt", "yes")] if SPECIFIC[1] in memory else [("notes.txt", "practice")]
    fake = practice_chat(writes=writes)
    artifact, cost = sleep("replay", fake, practice)
    [record] = artifact["practice"]
    assert_record(record)
    assert record["dreams_offered"] == record["dream_replies"] == []
    assert len(record["exercises"]) == 5 and cost["practice_sessions"] == (8 if repair else 5)
    assert all(s["request"] == REAL and (s["workspace"] / "MARK.txt").read_text() == "before" for s in spy)
    assert all(e["decision"] == REAL and not e["hypothetical"] and e["must_add"] == e["must_not_add"] == []
               for e in record["exercises"])
    assert [e["scores2"] for e in record["exercises"]] == ([[0, 0], [0], None, None, None] if repair else [None] * 5)
    assert record["refined"] is repair and record["accepted"] is repair
    assert record["reason"] == ("" if repair else "all exercises passed")
    assert artifact["items"] == (REFINED if repair else ITEMS)
    assert not instruction_calls(fake, designs.DREAM_INSTRUCTION)
    assert len(instruction_calls(fake, designs.REPAIR_INSTRUCTION)) == int(repair)


def test_last_failed_exercise_is_confirmed_twice_without_a_next_exercise(practice, spy):
    def writes(request, memory, attempt):
        score = int(attempt == 3)
        return [("notes.txt", WRITTEN[score])]
    artifact, cost = sleep("dreaming", practice_chat(writes=writes), practice)
    [record] = artifact["practice"]
    assert_record(record)
    assert record["accepted"] and [e["scores2"] for e in record["exercises"]] == [None, [0, 0]]
    assert cost["practice_sessions"] == 6 and len(spy) == 6


@pytest.mark.parametrize("design", ["dreaming", "replay"])
def test_practice_cost_seeds_snapshot_and_record_copy(practice, spy, design):
    if design == "replay":
        (practice["ws_before"] / "ok.txt").write_text("no", encoding="utf-8")
    before, after = tree(practice["ws_before"]), tree(practice["ws_after"])
    def writes(request, memory, attempt):
        if design == "replay":
            return [("ok.txt", "yes")] if SPECIFIC[1] in memory else []
        score = 0 if "USER DECISION:" in memory or SPECIFIC[1] in memory else 1
        return [("notes.txt", WRITTEN[score])]
    fake = practice_chat(writes=writes)
    artifact, cost = sleep(design, fake, practice, seed=1000)
    [record] = artifact["practice"]
    assert_record(record)
    assert cost["practice_sessions"] == len(spy) == (7 if design == "dreaming" else 8)
    offline = [c for c in fake.calls if c["tools"] is None]
    assert cost["calls"] == len(offline) == (4 if design == "dreaming" else 3)
    assert cost["prompt_tokens"] == 50 * len(fake.calls) and cost["completion_tokens"] == 20 * len(fake.calls)
    assert cost["seconds"] == 1.5 * len(offline) + sum(s["result"]["seconds"] for s in spy)
    seeds = [c["seed"] for c in offline[2:]]
    seeds += [s["seed"] + i for s in spy for i in range(1, s["max_steps"] + 1)]
    assert len(seeds) == len(set(seeds)) and all(1000 < seed < 2000 for seed in seeds)
    assert tree(practice["ws_before"]) == before and tree(practice["ws_after"]) == after
    artifact["items"][1]["text"] = "changed later"
    assert record["items_before"] == ITEMS
    assert "USER DECISION" not in designs.inject(design, artifact)


def test_contamination_includes_offered_unverified_requests_and_only_hypothetical_exercises():
    log = [{"exercises": [{"request": "Use REAL_NAME", "hypothetical": False},
                           {"request": "Use DREAM_NAME and WAKE_NAME", "hypothetical": True}],
            "dreams_offered": [{"request": "Edit `offered.py` and OFFERED_NAME"}]}]
    memory = "REAL_NAME DREAM_NAME WAKE_NAME offered.py OFFERED_NAME"
    assert designs.contamination(memory, log, [[{"type": "user", "text": "WAKE_NAME"}]]) == [
        "DREAM_NAME", "OFFERED_NAME", "offered.py"]
    assert designs.contamination(memory, [{"exercises": []}], []) == []


def test_dreams_preserve_check_whitespace_lengths_and_substring_exclusions():
    decision = "Use aa and _MS and keep every future duration consistent."
    [got] = designs._dreams(dream_reply({**DREAMS[0], "decision": decision,
        "must_add": [" _MS ", "aa", "_M", "\t_MS\t"],
        "must_not_add": ["_MS", " _MS ", "aa", "bad", "  odd  "]}), [decision])
    assert got["must_add"] == [" _MS ", "aa", "_M"]
    assert got["must_not_add"] == ["bad", "  odd  "] and got["rejected"] == ""
    [got] = designs._dreams(dream_reply({**DREAMS[0], "decision": "x" * 20, "must_add": ["xx"]}), ["x" * 20])
    assert got["rejected"] == ""
    [got] = designs._dreams(dream_reply({**DREAMS[0], "decision": "x" * 19, "must_add": ["xx"]}), ["x" * 19])
    assert got["rejected"] == "decision is not the user's words"


def test_attempt_score_sums_check_failure_missing_and_forbidden_strings(practice):
    dream = {**DREAMS[0], "must_add": ["_MS", "milliseconds"]}
    def writes(request, memory, attempt):
        return [("ok.txt", "yes" if "USER DECISION:" in memory else "no"),
                ("notes.txt", "POLL_MS milliseconds" if "USER DECISION:" in memory else "POLL_SECONDS")]
    artifact, _ = sleep("dreaming", practice_chat(dream=dream_reply(dream), writes=writes,
                                                repair='{"number": -1, "text": ""}'), practice)
    [record] = artifact["practice"]
    assert_record(record)
    assert record["dreams_offered"][0]["control_score"] == 0
    [exercise] = record["exercises"]
    assert (exercise["score"], exercise["check_passed"], exercise["missing"], exercise["present"]) == (
        4, False, ["_MS", "milliseconds"], ["_SECONDS"])


@pytest.mark.parametrize("room, added_kept, output_kept", [(1000, True, True), (400, False, False),
                                                         (40, False, False), (0, False, False)])
def test_repair_prompt_caps_and_fits_added_then_output_then_request(room, added_kept, output_kept):
    exercise = {**DREAMS[0], "request": "request " * 150}
    result = {"added": "a" * 1200, "output": "HEAD" + "o" * 1000 + "TAIL", "passed": False,
              "missing": ["_MS"], "present": ["_SECONDS"]}
    empty = {**result, "added": "", "output": ""}
    minimum = designs._repair_prompt(ITEMS, {**exercise, "request": ""}, empty, 10000)
    budget = llm.estimate_tokens(minimum) + room
    prompt = designs._repair_prompt(ITEMS, exercise, result, budget)
    assert llm.estimate_tokens(prompt) <= budget
    assert f"USER DECISION: {DECISION}" in prompt and prompt.endswith(designs.REPAIR_INSTRUCTION)
    added = prompt.split("WHAT THE AGENT ADDED:\n")[1].split("\n\nPROBLEMS:")[0]
    output = prompt.split("The project check failed: ")[1].split("\nThe added text")[0]
    request = prompt.split("\n\nREQUEST: ")[1].split("\n\nWHAT THE AGENT ADDED:")[0]
    if added_kept:
        assert added == "a" * designs.ADDED_CAP and output == result["output"][-designs.CHECK_TAIL:]
    else:
        assert added == "(nothing)"
    if output_kept:
        assert output == result["output"][-designs.CHECK_TAIL:]
    else:
        assert len(output) < designs.CHECK_TAIL
    if output != result["output"][-designs.CHECK_TAIL:]:
        assert added == "(nothing)" and result["output"][-designs.CHECK_TAIL:].startswith(output)
    if request != exercise["request"]:
        assert added == "(nothing)" and output == "" and exercise["request"].startswith(request)


def test_practice_without_seed_and_second_sleep_keep_only_one_record(practice, spy):
    fake = practice_chat()
    first, _ = sleep("replay", fake, practice)
    second, _ = designs.sleep("replay", first, session_events(2), chat=fake, window=4096, reserve=700,
                              practice=practice)
    assert len(second["practice"]) == 1 and second["practice"][0]["items_before"] == ITEMS
    merges = instruction_calls(fake, designs.MERGE_INSTRUCTION)
    assert merges[1]["messages"][0]["content"].startswith(f"CURRENT memory:\n{lines(MILLIS, LESSON)}")
    assert all(call["seed"] is None for call in fake.calls) and all(s["seed"] is None for s in spy)
    artifact = {"items": ITEMS, "practice": [{"request": "SECRET_DREAM"}], "requests": ["SECRET_ARCHIVE"]}
    assert designs.inject("dreaming", artifact) == designs.inject("replay", artifact) == MEMORY


def test_practice_constants_and_instruction_texts_are_pinned():
    assert (designs.EXERCISES, designs.EXPECT_CAP, designs.CONTROL_CAP, designs.CONFIRM,
            designs.REPLAY_ATTEMPTS, designs.REPLY_CAP, designs.DECISION_MIN) == (2, 4, 3, 2, 5, 2000, 20)
    assert not hasattr(designs, "REFINE_INSTRUCTION")
    assert designs.DREAM_INSTRUCTION == (
        "Above are the user's requests on this project, word for word, oldest first. Some of them contain standing "
        "decisions: rules the user wants followed in all future work (a naming, format, unit or process policy), not "
        "just in that one task. Find the standing decisions that no automatic check would enforce.\n"
        "Write up to 4 practice exercises. Each exercise tests ONE standing decision:\n"
        "- \"decision\": the user's sentence that states it, copied character for character from the requests above.\n"
        "- \"request\": a small new feature request of the kind this user makes, to which the decision clearly "
        "applies. Write it the way a teammate who never saw the decision would: plain everyday words, in the old or "
        "everyday terms, nothing about how to store or name things, no file names, no setting names, no suffixes.\n"
        "- \"must_add\": 1 or 2 short strings copied character for character from the decision (for example a "
        "required suffix) that will appear in the text added to the project if the decision is followed.\n"
        "- \"must_not_add\": short strings that would appear only if the decision were ignored. Never a whole "
        "setting name, never a number.\n"
        "If there is no standing decision, return an empty list. The exercises are hypothetical.")
    assert designs.REPAIR_INSTRUCTION == (
        "An agent that was given the RULES above got the REQUEST and broke a standing decision of the user. The "
        "decision, in the user's own words, is quoted above. Change the rules so that an agent reading them follows "
        "the decision in requests like this one. Edit exactly ONE rule: give its number, or 0 to add a new rule, and "
        "the full new text of that rule (at most 35 words). State the decision as a general instruction for all "
        "future work, keep the user's exact terms (names, suffixes, units), and say what to do when a request is "
        "phrased in other terms. Do not mention the practice request: it was hypothetical.")


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
