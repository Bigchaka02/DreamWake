"""Memory designs: what each one keeps across the reset, and what its sleep step sends."""
import json

import pytest

from dreamwake import agent, designs, llm


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


def test_workspace_design_keeps_nothing_and_costs_nothing():
    fake = FakeChat("unused")
    artifact, cost = designs.sleep("workspace", None, session_events(3), chat=fake, window=4096, reserve=700)
    assert artifact is None and fake.calls == []
    assert cost == {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "seconds": 0.0}
    assert designs.inject("workspace", artifact) == ""


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
