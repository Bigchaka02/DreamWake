"""The LLM client (network mocked), the agent loop, its tools, trimming and transcript rendering."""
import io
import json
import urllib.error

import pytest

from dreamwake import agent, llm, tasks

TASK = "settings_gen"
PROMPT = "do the thing"


def reply(*calls, content=""):
    """A chat() reply; each call is (name, arguments) where arguments None means unparseable JSON."""
    return {"content": content,
            "tool_calls": [{"name": name, "arguments": args, "raw": "{bad" if args is None else json.dumps(args)}
                           for name, args in calls],
            "usage": {"prompt_tokens": 100, "completion_tokens": 10}, "seconds": 0.5,
            "finish_reason": "tool_calls" if calls else "stop"}


class FakeChat:
    def __init__(self, replies):
        self.replies, self.calls = list(replies), []

    def __call__(self, messages, tools=None, **kwargs):
        self.calls.append({"messages": json.loads(json.dumps(messages)), "tools": tools, **kwargs})
        return self.replies.pop(0)


FINISH = reply(("finish", {"summary": "done"}))


def with_prompt_tokens(a_reply, prompt_tokens):
    return {**a_reply, "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": 10}}


@pytest.fixture
def ws(tmp_path):
    return tasks.make_workspace(TASK, tmp_path / "ws")


def run(ws, replies, **kwargs):
    fake = FakeChat(replies)
    kwargs = {"max_steps": 10, "window": 100_000, "reserve": 500, **kwargs}
    return agent.run_session(TASK, ws, PROMPT, chat=fake, **kwargs), fake


def results(result):
    return [e["text"] for e in result["events"] if e["type"] == "result"]


# --- llm -----------------------------------------------------------------------------------------

def fake_urlopen(monkeypatch, response=None, error=None):
    sent = {}

    def urlopen(request, timeout=None):
        sent.update(url=request.full_url, body=json.loads(request.data))
        if error:
            raise error
        return io.BytesIO(json.dumps(response).encode())
    monkeypatch.setattr(llm.urllib.request, "urlopen", urlopen)
    return sent


def test_chat_posts_to_the_configured_endpoint_and_parses_the_reply(monkeypatch):
    monkeypatch.setenv("DW_BASE_URL", "http://example.invalid/v1")
    monkeypatch.setenv("DW_MODEL", "tiny-model")
    sent = fake_urlopen(monkeypatch, {
        "choices": [{"finish_reason": "tool_calls", "message": {"content": None, "tool_calls": [
            {"id": "a", "type": "function", "function": {"name": "read_file", "arguments": '{"path": "x"}'}},
            {"id": "b", "type": "function", "function": {"name": "finish", "arguments": "{oops"}}]}}],
        "usage": {"prompt_tokens": 42, "completion_tokens": 7}})
    out = llm.chat([{"role": "user", "content": "hi"}], agent.TOOLS, max_tokens=99, seed=5)
    assert sent["url"] == "http://example.invalid/v1/chat/completions"
    assert sent["body"]["model"] == "tiny-model" and sent["body"]["max_tokens"] == 99
    assert sent["body"]["seed"] == 5 and sent["body"]["tools"] == agent.TOOLS
    assert sent["body"]["reasoning_effort"] == "none"
    assert out["content"] == "" and out["finish_reason"] == "tool_calls"
    assert out["tool_calls"] == [{"name": "read_file", "arguments": {"path": "x"}, "raw": '{"path": "x"}'},
                                 {"name": "finish", "arguments": None, "raw": "{oops"}]
    assert out["usage"] == {"prompt_tokens": 42, "completion_tokens": 7}
    assert out["seconds"] >= 0


def test_chat_defaults_to_the_ipv4_loopback_endpoint(monkeypatch):
    monkeypatch.delenv("DW_BASE_URL", raising=False)
    monkeypatch.setenv("DW_MODEL", "tiny-model")
    sent = fake_urlopen(monkeypatch, {"choices": [{"message": {"content": "hi"}}]})
    out = llm.chat([], max_tokens=10)
    assert sent["url"] == "http://127.0.0.1:1234/v1/chat/completions"
    assert "tools" not in sent["body"] and "seed" not in sent["body"]
    assert out["content"] == "hi" and out["tool_calls"] == [] and out["usage"]["prompt_tokens"] == 0


def test_chat_without_dw_model_raises(monkeypatch):
    monkeypatch.delenv("DW_MODEL", raising=False)
    with pytest.raises(RuntimeError, match="DW_MODEL"):
        llm.chat([], max_tokens=10)


def test_chat_http_error_raises_with_the_response_body(monkeypatch):
    monkeypatch.setenv("DW_MODEL", "tiny-model")
    error = urllib.error.HTTPError("u", 400, "Bad Request", {}, io.BytesIO(b"context length exceeded"))
    fake_urlopen(monkeypatch, error=error)
    with pytest.raises(RuntimeError, match="context length exceeded") as caught:
        llm.chat([], max_tokens=10)
    assert not isinstance(caught.value, llm.ContextOverflow)


def test_chat_over_length_prompt_raises_context_overflow(monkeypatch):
    monkeypatch.setenv("DW_MODEL", "tiny-model")
    body = b'{"error": {"type": "exceed_context_size_error", "message": "too long"}}'
    fake_urlopen(monkeypatch, error=urllib.error.HTTPError("u", 400, "Bad Request", {}, io.BytesIO(body)))
    with pytest.raises(llm.ContextOverflow, match="exceed_context_size_error"):
        llm.chat([], max_tokens=10)


def test_estimate_tokens_is_chars_over_3_2_plus_one():
    assert llm.estimate_tokens("") == 1 and llm.estimate_tokens("x" * 32) == 11
    assert llm.estimate_tokens("x" * 31) == 10


# --- session loop --------------------------------------------------------------------------------

def test_finish_ends_the_session(ws):
    result, fake = run(ws, [reply(("list_files", {})), FINISH])
    assert result["finished"] and not result["ran_out_of_steps"] and result["steps"] == 2
    assert result["overflow"] is False
    assert len(fake.calls) == 2
    assert all(c["tools"] == agent.TOOLS and c["max_tokens"] == 500 for c in fake.calls)
    assert [e["type"] for e in result["events"]] == ["user", "call", "result", "call", "result"]
    assert result["events"][0] == {"type": "user", "text": PROMPT}


def test_step_cap_ends_the_session_with_ran_out_of_steps(ws):
    result, fake = run(ws, [reply(("list_files", {}))] * 5, max_steps=3)
    assert result["steps"] == 3 and len(fake.calls) == 3
    assert not result["finished"] and result["ran_out_of_steps"]


def test_usage_is_totalled(ws):
    replies = [reply(("list_files", {})), FINISH]
    replies[0]["usage"] = {"prompt_tokens": 300, "completion_tokens": 20}
    result, _ = run(ws, replies)
    assert result["prompt_tokens"] == 400 and result["completion_tokens"] == 30
    assert result["max_prompt_tokens"] == 300 and result["seconds"] > 0


def test_system_prompt_states_the_step_budget_and_appends_memory(ws):
    _, fake = run(ws, [FINISH], max_steps=7)
    assert fake.calls[0]["messages"][0] == {"role": "system", "content": agent.SYSTEM.format(max_steps=7)}
    assert "at most 7 steps" in agent.SYSTEM.format(max_steps=7)
    _, fake = run(ws, [FINISH], memory="durations use _MS")
    system = fake.calls[0]["messages"][0]["content"]
    assert system.endswith("(your conversation memory was wiped since then):\ndurations use _MS")
    assert fake.calls[0]["messages"][1] == {"role": "user", "content": PROMPT}


@pytest.mark.parametrize("memory_at", ["system", "request", "tail"])
@pytest.mark.parametrize("memory", ["", "durations use _MS"])
def test_memory_placement(ws, tmp_path, memory_at, memory):
    log = tmp_path / "session.jsonl"
    result, fake = run(ws, [reply(("list_files", {})), FINISH], max_steps=7, memory=memory,
                       memory_at=memory_at, log_path=log)
    note = agent.MEMORY.format(memory=memory) if memory else ""
    pinned = [{"role": "system", "content": agent.SYSTEM.format(max_steps=7)
               + (note if memory_at == "system" else "")},
              {"role": "user", "content": PROMPT + (note if memory_at == "request" else "")}]
    assert all(call["messages"][:2] == pinned for call in fake.calls)
    assert result["events"][0] == {"type": "user", "text": PROMPT}
    assert json.loads(log.read_text(encoding="utf-8").splitlines()[0]) == result["events"][0]


@pytest.mark.parametrize("memory_at", ["", "user", None, []])
def test_invalid_memory_placement(ws, memory_at):
    with pytest.raises(ValueError, match="memory_at"):
        run(ws, [FINISH], memory_at=memory_at)


def test_tail_follows_history_and_overflow_marker_without_entering_events(ws):
    memory = "durations use _MS"
    tail = {"role": "user", "content": agent.MEMORY.format(memory=memory).strip()}
    fake = OverflowOnce([reply(("list_files", {})), reply(("list_files", {})), FINISH], overflow_on=[3])
    result = agent.run_session(TASK, ws, PROMPT, chat=fake, max_steps=10, window=100_000, reserve=500,
                               memory=memory, memory_at="tail")
    assert fake.calls[0]["messages"] == [
        {"role": "system", "content": agent.SYSTEM.format(max_steps=10)},
        {"role": "user", "content": PROMPT}, tail]
    assert fake.calls[1]["messages"][2:-1][0]["role"] == "assistant"
    assert fake.calls[1]["messages"][-2]["role"] == "tool"
    retried = fake.calls[-1]["messages"]
    assert retried[2] == {"role": "user", "content": agent.MARKER}
    assert retried[3]["role"] == "assistant" and retried[-2]["role"] == "tool"
    assert all(c["messages"][-1] == tail and c["messages"].count(tail) == 1 for c in fake.calls)
    assert result["dropped_units"] == 1 and len(result["events"]) == 7
    assert memory not in json.dumps(result["events"])


def test_tail_survives_trimming_and_counts_toward_the_window(ws):
    memory = "m" * 500
    _, probe = run(ws, [FINISH], memory=memory, memory_at="tail")
    first = probe.calls[0]["messages"]
    reserve = 200
    window = agent.estimate_prompt(first) + reserve + 600
    calls = [with_prompt_tokens(reply(("read_file", {"path": "README.md"})), 100_000)] * 8
    result, fake = run(ws, calls + [FINISH], memory=memory, memory_at="tail", window=window, reserve=reserve)
    assert result["dropped_units"] > 0
    assert all(c["messages"][-1] == first[-1] for c in fake.calls)
    assert all(agent.estimate_prompt(c["messages"]) <= window - reserve for c in fake.calls)
    with pytest.raises(ValueError, match="window"):
        run(ws, [FINISH], memory=memory, memory_at="tail", reserve=reserve,
            window=agent.estimate_prompt(first) + reserve - 1)


def test_empty_memory_sends_identical_messages_across_all_placements(ws):
    messages = []
    for placement in ("system", "request", "tail"):
        _, fake = run(ws, [reply(("list_files", {})), FINISH], memory_at=placement)
        messages.append([c["messages"] for c in fake.calls])
    assert messages[0] == messages[1] == messages[2]


@pytest.mark.parametrize("memory_at", ["system", "request", "tail"])
def test_finish_gate_fires_once_on_the_first_valid_finish_and_records_the_result(ws, memory_at):
    memory = "- RULE: use _MS"
    expected = ("Not finished yet. First check your changes against each of these notes, one by one:\n"
                + memory + "\nIf a change breaks a note, fix it and run the checks again. Then call finish.")
    result, fake = run(ws, [reply(("finish", {})), FINISH, reply(("list_files", {})), FINISH],
                       memory=memory, memory_at=memory_at, finish_gate=True)
    assert agent.GATE.format(memory=memory) == expected
    assert result["gated"] and result["finished"] and result["steps"] == 4
    assert results(result)[0].startswith("error")
    assert results(result)[1] == expected and results(result)[-1] == "finished"
    assert [e for e in result["events"] if e["type"] == "result" and e["text"] == expected] == [
        {"type": "result", "step": 2, "text": expected}]
    assert any(m.get("role") == "tool" and m["content"] == expected for m in fake.calls[2]["messages"])


@pytest.mark.parametrize("memory, finish_gate", [("", True), ("", False), ("note", False)])
def test_finish_gate_is_inactive_without_memory_or_when_disabled(ws, memory, finish_gate):
    result, fake = run(ws, [FINISH], memory=memory, finish_gate=finish_gate)
    assert not result["gated"] and result["finished"] and len(fake.calls) == 1
    assert results(result) == ["finished"]


def test_step_cap_after_finish_gate_is_reported_as_max_steps(ws):
    result, _ = run(ws, [FINISH], memory="note", finish_gate=True, max_steps=1)
    assert result["gated"] and not result["finished"] and result["steps"] == 1
    assert result["ended_by"] == "max_steps" and result["ran_out_of_steps"]


def test_seed_differs_per_step(ws):
    _, fake = run(ws, [reply(("list_files", {}))] * 2 + [FINISH], seed=1000)
    assert [c["seed"] for c in fake.calls] == [1001, 1002, 1003]


def test_ended_by_says_why_the_session_stopped(ws):
    result, _ = run(ws, [FINISH])
    assert result["ended_by"] == "finish"
    result, _ = run(ws, [reply(("list_files", {}))] * 3, max_steps=3)
    assert result["ended_by"] == "max_steps" and result["ran_out_of_steps"]
    result, _ = run(ws, [reply(content="a"), reply(content="b")], max_steps=2)  # two silent replies on the last step
    assert result["ended_by"] == "no_tool_calls" and not result["ran_out_of_steps"]
    result, _ = run(ws, [reply(content="a"), reply(content="b"), FINISH])
    assert result["ended_by"] == "no_tool_calls" and result["steps"] == 2
    fake = OverflowOnce([FINISH], overflow_on=[1])
    result = agent.run_session(TASK, ws, PROMPT, chat=fake, max_steps=10, window=100_000, reserve=500)
    assert result["ended_by"] == "overflow" and not result["ran_out_of_steps"]


def test_tools_work_inside_the_workspace(ws):
    (ws / "src" / "__pycache__").mkdir()
    (ws / "src" / "__pycache__" / "x.pyc").write_bytes(b"")
    result, _ = run(ws, [reply(("write_file", {"path": "notes/a.txt", "content": "hi\n"})),
                         reply(("read_file", {"path": "notes/a.txt"})),
                         reply(("list_files", {})), reply(("run_tests", {})), FINISH])
    assert (ws / "notes" / "a.txt").read_bytes() == b"hi\n"
    wrote, read, listing, checks, done = results(result)
    assert "notes/a.txt" in wrote and read == "hi\n"
    assert {"notes/a.txt", "src/settings.py", "tests/test_client.py"} <= set(listing.splitlines())
    assert "__pycache__" not in listing
    assert "has no attribute 'TIMEOUT_S'" in checks and "checks FAILED" in checks


def test_read_file_output_is_capped(ws):
    (ws / "big.txt").write_text("x" * 5000, encoding="utf-8")
    result, _ = run(ws, [reply(("read_file", {"path": "big.txt"})), FINISH])
    text = results(result)[0]
    assert text.startswith("x" * 3000) and text.count("x") == 3000 and len(text) < 3100


@pytest.mark.parametrize("call", [
    ("read_file", {"path": "../outside.txt"}),
    ("read_file", {"path": "OUTSIDE_ABS"}),
    ("write_file", {"path": "../evil.txt", "content": "x"}),
    ("write_file", {"path": "src/../../evil.txt", "content": "x"}),
])
def test_paths_outside_the_workspace_are_refused(ws, call):
    outside = ws.parent / "outside.txt"
    outside.write_text("secret", encoding="utf-8")
    name, args = call
    args = {k: str(outside) if v == "OUTSIDE_ABS" else v for k, v in args.items()}
    result, _ = run(ws, [reply((name, args)), FINISH])
    text = results(result)[0]
    assert text.startswith("error") and "outside" in text and "secret" not in text
    assert not (ws.parent / "evil.txt").exists()


def test_tool_errors_give_the_relative_path_and_the_plain_reason_only(ws):
    result, _ = run(ws, [reply(("read_file", {"path": "nope.txt"})), reply(("write_file", {"path": "src", "content": "x"})),
                         reply(("read_file", {"path": "src"})), FINISH])
    missing, *others = results(result)[:3]
    assert missing == "error: No such file or directory: nope.txt"
    for text in [missing, *others]:
        assert text.startswith("error:") and str(ws.resolve()) not in text and str(ws.parent) not in text, text


def test_writing_an_existing_file_that_was_not_read_is_refused_and_leaves_it_alone(ws):
    original = (ws / "CHANGES.md").read_bytes()
    result, _ = run(ws, [reply(("write_file", {"path": "CHANGES.md", "content": "new"})), FINISH])
    assert results(result)[0] == ("error: CHANGES.md already exists - read_file it first, "
                                  "then write its complete new content")
    assert (ws / "CHANGES.md").read_bytes() == original


def test_an_existing_file_can_be_written_after_it_was_read_however_the_path_is_spelled(ws):
    result, _ = run(ws, [reply(("read_file", {"path": "./CHANGES.md"})),
                         reply(("write_file", {"path": "CHANGES.md", "content": "new"})), FINISH])
    assert results(result)[1].startswith("wrote") and (ws / "CHANGES.md").read_text(encoding="utf-8") == "new"


def test_a_file_the_agent_created_can_be_rewritten_without_reading_it(ws):
    result, _ = run(ws, [reply(("write_file", {"path": "new.txt", "content": "1"})),
                         reply(("write_file", {"path": "new.txt", "content": "2"})), FINISH])
    assert all(text.startswith("wrote") for text in results(result)[:2])
    assert (ws / "new.txt").read_text(encoding="utf-8") == "2"


def test_each_session_starts_with_nothing_read(ws):
    result, _ = run(ws, [reply(("read_file", {"path": "CHANGES.md"})), FINISH])
    result, _ = run(ws, [reply(("write_file", {"path": "CHANGES.md", "content": "new"})), FINISH])
    assert results(result)[0].startswith("error") and (ws / "CHANGES.md").read_text(encoding="utf-8") != "new"


def test_bad_arguments_and_unknown_tools_give_error_results(ws):
    result, _ = run(ws, [reply(("read_file", None)), reply(("delete_everything", {})),
                         reply(("read_file", {})), reply(("write_file", {"path": "a", "content": 5})),
                         reply(("read_file", {"path": "no/such.txt"})), FINISH])
    assert result["finished"] and result["steps"] == 6
    errors = results(result)[:5]
    assert all(e.startswith("error") for e in errors), errors
    assert "delete_everything" in errors[1] and "'path'" in errors[2] and "'content'" in errors[3]
    assert result["events"][1]["args"] == {"raw": "{bad"}


class StrictServerChat(FakeChat):
    """Like LM Studio: a replayed tool call whose arguments are "" or invalid JSON gets HTTP 500, and a
    non-object (e.g. '"a.txt"') silently garbles the prompt, so anything but a JSON object string fails here."""
    def __call__(self, messages, tools=None, **kwargs):
        for message in messages:
            for call in message.get("tool_calls") or []:
                try:
                    ok = isinstance(json.loads(call["function"]["arguments"]), dict)
                except ValueError:
                    ok = False
                if not ok:
                    raise RuntimeError("HTTP 500 from server: Internal Server Error")
        return super().__call__(messages, tools, **kwargs)


@pytest.mark.parametrize("raw, replayed", [("{bad", {"raw": "{bad"}), ("", {}), ('"a.txt"', {"raw": '"a.txt"'})])
def test_replayed_tool_calls_always_carry_a_json_object(ws, raw, replayed):
    bad = reply(("read_file", {}))
    bad["tool_calls"][0].update(raw=raw, arguments=llm._parse_arguments(raw))
    fake = StrictServerChat([bad, FINISH])
    result = agent.run_session(TASK, ws, PROMPT, chat=fake, max_steps=10, window=100_000, reserve=500)
    assert result["finished"] and results(result)[0].startswith("error")
    assert json.loads(fake.calls[1]["messages"][2]["tool_calls"][0]["function"]["arguments"]) == replayed


def test_only_the_first_tool_call_of_a_reply_is_executed(ws):
    result, fake = run(ws, [reply(("write_file", {"path": "one.txt", "content": "1"}),
                                  ("write_file", {"path": "two.txt", "content": "2"})), FINISH])
    assert (ws / "one.txt").exists() and not (ws / "two.txt").exists()
    assistant, tool = fake.calls[1]["messages"][2:]
    assert assistant["role"] == "assistant" and len(assistant["tool_calls"]) == 1
    assert assistant["tool_calls"][0]["function"]["name"] == "write_file"
    assert tool["role"] == "tool" and tool["tool_call_id"] == assistant["tool_calls"][0]["id"]
    assert [e["type"] for e in result["events"]].count("call") == 2


def test_reply_without_a_tool_call_is_nudged_and_two_in_a_row_end_the_session(ws):
    result, fake = run(ws, [reply(content="thinking"), reply(("list_files", {})),
                            reply(content="hmm"), reply(content="still hmm"), FINISH])
    assert result["steps"] == 4 and not result["finished"] and not result["ran_out_of_steps"]
    assert fake.calls[1]["messages"][2:] == [{"role": "assistant", "content": "thinking"},
                                             {"role": "user", "content": agent.NUDGE}]
    assert {"type": "say", "step": 1, "text": "thinking"} in result["events"]
    assert agent.NUDGE == "Use a tool. Call finish when the work is done."


def test_assistant_text_alongside_a_call_is_recorded(ws):
    result, _ = run(ws, [reply(("list_files", {}), content="let me look"), FINISH])
    assert result["events"][1] == {"type": "call", "step": 1, "tool": "list_files", "args": {},
                                   "content": "let me look"}


def test_trimming_keeps_pinned_messages_never_splits_units_and_fits_the_window(ws):
    _, probe = run(ws, [FINISH])
    pinned = probe.calls[0]["messages"]
    reserve = 200
    window = agent.estimate_prompt(pinned) + reserve + 900
    replies = [with_prompt_tokens(reply(("read_file", {"path": "README.md"})), 100_000) for _ in range(8)]
    result, fake = run(ws, replies + [with_prompt_tokens(FINISH, 100_000)], window=window, reserve=reserve)
    assert result["finished"] and result["dropped_units"] > 0
    assert len([e for e in result["events"] if e["type"] == "call"]) == 9  # events stay untrimmed
    marked = 0
    for call in fake.calls:
        messages = call["messages"]
        assert agent.estimate_prompt(messages) <= window - reserve
        assert messages[:2] == pinned
        rest = messages[2:]
        if rest and rest[0] == {"role": "user", "content": agent.MARKER}:
            marked += 1
            rest = rest[1:]
        for i, message in enumerate(rest):  # every tool result directly follows the call it answers
            if message["role"] == "tool":
                assert rest[i - 1]["role"] == "assistant" and i > 0
                assert rest[i - 1]["tool_calls"][0]["id"] == message["tool_call_id"]
        assert not rest or rest[0]["role"] == "assistant"
    assert marked > 0 and agent.MARKER == "[earlier steps omitted to fit the context window]"


class OverflowOnce(FakeChat):
    """Raises ContextOverflow on the given call numbers (1-based), like the server on an over-length prompt."""
    def __init__(self, replies, overflow_on):
        super().__init__(replies)
        self.overflow_on = set(overflow_on)

    def __call__(self, messages, tools=None, **kwargs):
        if len(self.calls) + 1 in self.overflow_on:
            self.overflow_on.discard(len(self.calls) + 1)
            self.calls.append({"messages": json.loads(json.dumps(messages)), "overflow": True})
            raise llm.ContextOverflow("exceed_context_size_error")
        return super().__call__(messages, tools, **kwargs)


def test_context_overflow_drops_the_oldest_unit_and_retries_the_same_step(ws):
    fake = OverflowOnce([reply(("list_files", {})), reply(("list_files", {})), FINISH], overflow_on=[3])
    result = agent.run_session(TASK, ws, PROMPT, chat=fake, max_steps=10, window=100_000, reserve=500)
    assert result["finished"] and not result["overflow"] and result["steps"] == 3
    assert result["dropped_units"] == 1
    overflowed, retried = fake.calls[2], fake.calls[3]
    assert len(overflowed["messages"]) == 6  # pinned + two units
    assert retried["messages"][2] == {"role": "user", "content": agent.MARKER}
    assert retried["messages"][3:] == overflowed["messages"][4:]  # only the oldest unit went
    assert retried["seed"] is None and len(fake.calls) == 4


def test_context_overflow_with_nothing_left_to_drop_ends_the_session(ws):
    fake = OverflowOnce([FINISH], overflow_on=[1])
    result = agent.run_session(TASK, ws, PROMPT, chat=fake, max_steps=10, window=100_000, reserve=500)
    assert result["overflow"] and not result["finished"] and not result["ran_out_of_steps"]
    assert result["steps"] == 1 and len(fake.calls) == 1


def test_pinned_messages_larger_than_the_window_raise(ws):
    with pytest.raises(ValueError, match="window"):
        run(ws, [FINISH], window=300, reserve=200)


def test_token_estimate_is_calibrated_to_the_servers_prompt_count(ws, tmp_path):
    _, probe = run(ws, [FINISH])
    reserve = 200
    window = agent.estimate_prompt(probe.calls[0]["messages"]) + reserve + 900  # too small for the raw estimate
    log = tmp_path / "log.jsonl"
    replies = [with_prompt_tokens(reply(("read_file", {"path": "README.md"})), 50) for _ in range(8)]
    result, fake = run(ws, replies + [with_prompt_tokens(FINISH, 50)], window=window, reserve=reserve, log_path=log)
    assert result["finished"] and result["dropped_units"] == 0
    ratio = 1.1 * 50 / agent.estimate_prompt(fake.calls[0]["messages"])
    steps = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    assert [s["ratio"] for s in steps if s["type"] == "step"][0] == pytest.approx(ratio)


def test_ratio_is_capped_at_one_and_ignores_replies_without_a_prompt_count(ws, tmp_path):
    log = tmp_path / "log.jsonl"
    list_files = reply(("list_files", {}))
    replies = [with_prompt_tokens(list_files, n) for n in (50, 0, 100_000)] + [FINISH]
    run(ws, replies, log_path=log)
    ratios = [json.loads(line)["ratio"] for line in log.read_text(encoding="utf-8").splitlines()
              if json.loads(line)["type"] == "step"]
    assert ratios[0] < 1.0 and ratios[1] == ratios[0] and ratios[2] == 1.0


def test_log_path_gets_events_and_one_line_per_step(ws, tmp_path):
    log = tmp_path / "logs" / "s1.jsonl"
    result, _ = run(ws, [reply(("list_files", {})), FINISH], log_path=log)
    lines = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    assert [line for line in lines if line["type"] != "step"] == result["events"]
    steps = [line for line in lines if line["type"] == "step"]
    assert [s["step"] for s in steps] == [1, 2]
    assert steps[0]["prompt_tokens"] == 100 and steps[0]["seconds"] == 0.5 and steps[0]["dropped_units"] == 0


# --- render_events -------------------------------------------------------------------------------

EVENTS = [{"type": "user", "text": "fix it"},
          {"type": "call", "step": 1, "tool": "write_file", "args": {"path": "a.py", "content": "y" * 2000}},
          {"type": "result", "step": 1, "text": "wrote 2000 characters to a.py"},
          {"type": "say", "step": 2, "text": "done?"},
          {"type": "user", "text": agent.NUDGE},
          {"type": "call", "step": 3, "tool": "run_tests", "args": {}, "content": "checking"},
          {"type": "result", "step": 3, "text": "z" * 2000}]


def test_render_events_shows_calls_and_results_with_long_text_capped():
    text = render = agent.render_events(EVENTS)
    assert render.startswith("USER: fix it\nSTEP 1 write_file(path='a.py')")
    assert "-> wrote 2000 characters to a.py" in text and "STEP 2" in text and "checking" in text
    assert "STEP 3 run_tests()" in text and agent.NUDGE in text
    assert 600 <= text.count("y") < 700 and 600 <= text.count("z") < 700


def test_render_events_tail_respects_max_chars_and_cuts_at_step_boundaries():
    full = agent.render_events(EVENTS)
    assert agent.render_events(EVENTS, max_chars=len(full)) == full
    step3 = full[full.index("STEP 3"):]
    tail = agent.render_events(EVENTS, max_chars=len(step3) + 5)
    assert tail == step3
    two = agent.render_events(EVENTS, max_chars=len(full) - 1)
    assert two.startswith("STEP 1") and len(two) <= len(full) - 1
    assert agent.render_events(EVENTS, max_chars=0) == ""


def test_render_chunks_cover_every_step_once_in_order_within_max_chars_each_starting_with_the_request():
    events = [{"type": "user", "text": "fix it"}]
    for step in range(1, 21):
        events += [{"type": "call", "step": step, "tool": "read_file", "args": {"path": f"f{step}.py"}},
                   {"type": "result", "step": step, "text": "r" * 100}]
    head = "USER: fix it\n"
    chunks = agent.render_chunks(events, 700)
    assert len(chunks) > 1
    assert all(chunk.startswith(head) and len(chunk) <= 700 for chunk in chunks)
    assert "\n".join(chunk[len(head):] for chunk in chunks) == agent.render_events(events)[len(head):]


def test_render_chunks_truncate_a_step_too_long_for_one_chunk_instead_of_dropping_it():
    chunks = agent.render_chunks(EVENTS, 300)
    assert len(chunks) == 3 and all(len(chunk) <= 300 for chunk in chunks)
    assert chunks[0].startswith("USER: fix it\nSTEP 1 write_file(path='a.py')")
    assert chunks[1] == "USER: fix it\nSTEP 2 said: done?\nUSER: " + agent.NUDGE
    assert chunks[2].startswith("USER: fix it\nSTEP 3 said: checking\nSTEP 3 run_tests()")
