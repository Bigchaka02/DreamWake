"""Memory designs: what each one keeps across the reset, and what its sleep step sends."""
import pytest

from dreamwake import agent, designs, llm


def session_events(n):
    events = [{"type": "user", "text": "SESSION PROMPT"}]
    for step in range(1, n + 1):
        events += [{"type": "call", "step": step, "tool": "read_file", "args": {"path": f"file{step}.py"}},
                   {"type": "result", "step": step, "text": f"result {step} " + "x" * 300}]
    return events


class FakeChat:
    def __init__(self, note):
        self.note, self.calls = note, []

    def __call__(self, messages, tools=None, **kwargs):
        self.calls.append({"messages": messages, "tools": tools, **kwargs})
        return {"content": self.note, "tool_calls": [], "usage": {"prompt_tokens": 50, "completion_tokens": 20},
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
    assert "STEP 1 " not in prompt and "SESSION PROMPT" not in prompt  # the oldest part is not
    assert prompt.index("OLD NOTE") < prompt.index("STEP 30") < prompt.index(designs.HANDOFF_INSTRUCTION)


def test_handoff_without_previous_note_sends_the_whole_short_transcript():
    fake = FakeChat("note")
    designs.sleep("handoff", None, session_events(2), chat=fake, window=4096, reserve=700)
    prompt = fake.calls[0]["messages"][0]["content"]
    assert agent.render_events(session_events(2)) in prompt and "previous" not in prompt.lower()


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
