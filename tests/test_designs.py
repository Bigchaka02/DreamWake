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


# --- dreaming and replay: practise with the memory, keep a rewrite only if practice improves -------

TASK = "settings_gen"
REAL = "Rename TIMEOUT_S to TIMEOUT_MS with value 30000. Make the checks pass."
DREAMS = ["Add a cache lifetime of 5 seconds as a setting.", "Add a poll interval of 9 seconds as a setting."]
SPECIFIC = ("rule", "Store every duration in milliseconds: name it <NAME>_MS, e.g. RETRY_DELAY_MS = 2000.")
MEMORY = designs.inject("consolidated", {"items": [{"kind": k, "text": t} for k, t in (MILLIS, LESSON)]})
PROBLEM = json.dumps({"rule": 1, "problem": "named the delay RETRY_DELAY_SECONDS"})


def dream_lines(*requests):
    return "\n".join(json.dumps({"request": r}) for r in requests)


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


def practice_chat(reviews=(), refine=lines(SPECIFIC, LESSON), dream=dream_lines(*DREAMS),
                  writes=(("notes.txt", "practice"),)):
    """Sleep calls answered by instruction; reviews are used in order (then NONE). A practice session
    writes `writes` one per step, then finishes."""
    reviews, state = list(reviews), {}

    def chat(messages, tools=None, *, max_tokens, seed=None, **kwargs):
        chat.calls.append({"messages": messages, "tools": tools, "max_tokens": max_tokens, "seed": seed})
        usage = {"prompt_tokens": 50, "completion_tokens": 20}
        if tools:
            if len(messages) == 2:
                state["queue"] = [("write_file", {"path": p, "content": c}) for p, c in writes]
            name, args = state["queue"].pop(0) if state["queue"] else ("finish", {"summary": "done"})
            return {"content": "", "tool_calls": [{"name": name, "arguments": args, "raw": json.dumps(args)}],
                    "usage": usage, "seconds": 1.0, "finish_reason": "tool_calls"}
        prompt = messages[0]["content"]
        replies = {designs.EXTRACT_INSTRUCTION: lines(MILLIS), designs.MERGE_INSTRUCTION: lines(MILLIS, LESSON),
                   designs.DREAM_INSTRUCTION: dream, designs.REFINE_INSTRUCTION: refine,
                   designs.REVIEW_INSTRUCTION: reviews.pop(0) if reviews and prompt.endswith(
                       designs.REVIEW_INSTRUCTION) else "NONE"}
        [content] = [reply for ending, reply in replies.items() if prompt.endswith(ending)]
        return {"content": content, "tool_calls": [], "usage": usage, "seconds": 1.5, "finish_reason": "stop"}
    chat.calls = []
    return chat


@pytest.fixture
def spy(monkeypatch):
    """What every practice session received: its sandbox's MARK, request, memory, and the sandbox itself."""
    real, seen = agent.run_session, []

    def run_session(task_id, workspace, prompt, **kwargs):
        mark = (Path(workspace) / "MARK.txt").read_text(encoding="utf-8")
        seen.append({"mark": mark, "request": prompt, "memory": kwargs["memory"], "workspace": Path(workspace),
                     "max_steps": kwargs["max_steps"], "seed": kwargs["seed"]})
        return real(task_id, workspace, prompt, **kwargs)
    monkeypatch.setattr(designs.agent, "run_session", run_session)
    return seen


def sleep(design, fake, practice, window=4096, reserve=700, seed=None):
    return designs.sleep(design, None, session_events(2), chat=fake, window=window, reserve=reserve, seed=seed,
                         practice=practice)


def review_prompts(fake):
    return [p for p in prompts(fake) if p.endswith(designs.REVIEW_INSTRUCTION)]


def test_replay_practises_the_real_request_on_fresh_copies_of_the_snapshot_before_the_session(tmp_path, spy):
    practice = snapshots(tmp_path)
    artifact, _ = sleep("replay", practice_chat(), practice)
    assert [(s["mark"], s["request"]) for s in spy] == [("before", REAL)] * designs.PRACTICE_ROUNDS
    assert all(s["memory"] == MEMORY and s["max_steps"] == 12 for s in spy)
    sandboxes = [s["workspace"] for s in spy]
    assert len(set(sandboxes)) == 2 and all(p.is_relative_to(practice["scratch"]) for p in sandboxes)
    assert [entry["hypothetical"] for entry in artifact["practice"]] == [False, False]


def test_dreaming_practises_invented_requests_on_copies_of_the_snapshot_after_the_session(tmp_path, spy):
    fake = practice_chat()
    artifact, _ = sleep("dreaming", fake, snapshots(tmp_path))
    assert [(s["mark"], s["request"]) for s in spy] == [("after", DREAMS[0]), ("after", DREAMS[1])]
    assert [(e["request"], e["hypothetical"]) for e in artifact["practice"]] == [(d, True) for d in DREAMS]
    [dream] = [p for p in prompts(fake) if p.endswith(designs.DREAM_INSTRUCTION)]
    assert dream == f"Here is a change request you handled on this project:\n{REAL}\n\n" + designs.DREAM_INSTRUCTION


@pytest.mark.parametrize("reply, expected", [
    ("Sure!\n" + dream_lines(DREAMS[0]) + "\n{broken\n" + json.dumps({"request": "  "}), [(DREAMS[0], True), (REAL, False)]),
    ("I cannot invent requests.", [(REAL, False), (REAL, False)]),
    ("```json\n" + json.dumps([{"request": r} for r in DREAMS + ["third"]]) + "\n```", [(d, True) for d in DREAMS]),
])
def test_dreaming_falls_back_to_the_real_request_on_the_after_snapshot_for_missing_dreams(tmp_path, spy, reply,
                                                                                          expected):
    artifact, _ = sleep("dreaming", practice_chat(dream=reply), snapshots(tmp_path))
    assert [(e["request"], e["hypothetical"]) for e in artifact["practice"]] == expected
    assert [(s["mark"], s["request"]) for s in spy] == [("after", request) for request, _ in expected]


def test_a_clean_practice_attempt_makes_no_refine_call_and_logs_score_zero(tmp_path, spy):
    fake = practice_chat()
    artifact, _ = sleep("replay", fake, snapshots(tmp_path))
    assert not any(p.endswith(designs.REFINE_INSTRUCTION) for p in prompts(fake)) and len(spy) == 2
    assert artifact["practice"] == [{"request": REAL, "hypothetical": False, "score": 0, "problems": [],
                                     "refined": False, "score2": None, "accepted": False}] * 2
    assert artifact["items"] == [{"kind": k, "text": t} for k, t in (MILLIS, LESSON)]


def test_review_prompt_shows_numbered_rules_request_changes_and_the_failed_check(tmp_path, spy):
    fake = practice_chat(reviews=[PROBLEM, PROBLEM, PROBLEM, PROBLEM])
    artifact, _ = sleep("replay", fake, snapshots(tmp_path, passing=False))
    review = review_prompts(fake)[0]
    assert review.startswith(f"RULES:\n1. RULE: {MILLIS[1]}\n2. LESSON: {LESSON[1]}\n\nREQUEST:\n{REAL}\n\n"
                             "CHANGES:\nFILE notes.txt:\npractice\n\nCHECK RESULT: failed\n")
    check_tail = review.split("CHECK RESULT: failed\n")[1].removesuffix("\n\n" + designs.REVIEW_INSTRUCTION)
    assert len(check_tail) == 400 and check_tail.endswith("checks FAILED (exit 1)")  # the last 400 characters
    assert artifact["practice"][0]["score"] == 2  # the failed check plus one problem
    assert artifact["practice"][0]["problems"] == [{"rule": 1, "problem": "named the delay RETRY_DELAY_SECONDS"}]


def test_changes_keep_the_last_successful_write_per_path_and_are_capped(tmp_path, spy):
    writes = [("a.txt", "first"), ("a.txt", "second " + "a" * 600), ("b.txt", "b" * 600), ("c.txt", "c" * 600),
              ("d.txt", "d" * 600), ("../outside.txt", "refused")]
    fake = practice_chat(writes=writes)
    sleep("replay", fake, snapshots(tmp_path))
    changes = review_prompts(fake)[0].split("CHANGES:\n")[1].split("\n\nCHECK RESULT:")[0]
    assert changes.startswith("FILE a.txt:\nsecond " + "a" * 493 + "\n\nFILE b.txt:\n" + "b" * 500)
    assert "first" not in changes and "outside" not in changes and len(changes) == 1500


def test_a_failing_attempt_is_refined_validated_and_accepted_when_the_score_drops(tmp_path, spy):
    fake = practice_chat(reviews=[PROBLEM + "\n" + PROBLEM, "NONE", "NONE"])
    artifact, _ = sleep("replay", fake, snapshots(tmp_path))
    refined = [{"kind": k, "text": t} for k, t in (SPECIFIC, LESSON)]
    assert artifact["items"] == refined
    first, second = artifact["practice"]
    problem = {"rule": 1, "problem": "named the delay RETRY_DELAY_SECONDS"}
    assert first == {"request": REAL, "hypothetical": False, "score": 2, "problems": [problem] * 2,
                     "refined": True, "score2": 0, "accepted": True}
    assert second["score"] == 0 and not second["refined"]
    candidate = designs.inject("consolidated", {"items": refined})
    assert [s["memory"] for s in spy] == [MEMORY, candidate, candidate]  # attempt, validation, round 2
    [refine] = [p for p in prompts(fake) if p.endswith(designs.REFINE_INSTRUCTION)]
    assert refine.startswith(f"RULES:\n1. RULE: {MILLIS[1]}\n2. LESSON: {LESSON[1]}\n\nMISTAKES:\n"
                             "- rule 1: named the delay RETRY_DELAY_SECONDS\n")
    attempt, validation, round2 = review_prompts(fake)
    assert validation.startswith(f"RULES:\n1. RULE: {MILLIS[1]}\n2. LESSON: {LESSON[1]}\n")  # the ORIGINAL rules
    assert round2.startswith(f"RULES:\n1. RULE: {SPECIFIC[1]}\n")
    assert len({s["workspace"] for s in spy}) == 3


@pytest.mark.parametrize("validation", [PROBLEM, PROBLEM + "\n" + PROBLEM])
def test_a_candidate_that_does_not_lower_the_score_is_rejected(tmp_path, spy, validation):
    fake = practice_chat(reviews=[PROBLEM, validation, "NONE"])
    artifact, _ = sleep("replay", fake, snapshots(tmp_path))
    assert artifact["items"] == [{"kind": k, "text": t} for k, t in (MILLIS, LESSON)]
    assert artifact["practice"][0]["refined"] and not artifact["practice"][0]["accepted"]
    assert artifact["practice"][0]["score2"] == validation.count("\n") + 1
    assert spy[2]["memory"] == MEMORY  # round 2 practises with the kept rules


def test_a_candidate_with_fewer_items_is_rejected_even_if_it_scores_better(tmp_path, spy):
    fake = practice_chat(reviews=[PROBLEM, "NONE"], refine=lines(SPECIFIC))
    artifact, _ = sleep("replay", fake, snapshots(tmp_path))
    assert artifact["practice"][0]["score2"] == 0 and not artifact["practice"][0]["accepted"]
    assert artifact["items"] == [{"kind": k, "text": t} for k, t in (MILLIS, LESSON)]


def test_practice_never_touches_the_snapshots(tmp_path, spy):
    practice = snapshots(tmp_path)
    before, after = tree(practice["ws_before"]), tree(practice["ws_after"])
    for design in ("replay", "dreaming"):
        sleep(design, practice_chat(reviews=[PROBLEM], writes=[("notes.txt", "x"), ("src/new.py", "y = 1\n")]),
              practice)
    assert tree(practice["ws_before"]) == before and tree(practice["ws_after"]) == after
    assert any((s["workspace"] / "src" / "new.py").exists() for s in spy)


@pytest.mark.parametrize("design, dream_calls", [("replay", 0), ("dreaming", 1)])
def test_practice_cost_sums_every_call_and_practice_session(tmp_path, spy, design, dream_calls):
    fake = practice_chat(reviews=[PROBLEM, "NONE"])
    _, cost = sleep(design, fake, snapshots(tmp_path))
    sleep_calls = [c for c in fake.calls if c["tools"] is None]
    session_steps = len(fake.calls) - len(sleep_calls)
    # 1 extract + 1 merge, dream, 3 reviews (round 1 attempt + validation, round 2), 1 refine
    assert len(sleep_calls) == 2 + dream_calls + 3 + 1 and cost["practice_sessions"] == 3
    assert cost["calls"] == len(sleep_calls)
    assert cost["prompt_tokens"] == 50 * len(fake.calls) and cost["completion_tokens"] == 20 * len(fake.calls)
    assert cost["seconds"] >= 1.5 * len(sleep_calls) and session_steps == 3 * 2


def test_every_practice_prompt_fits_the_window(tmp_path, spy):
    window, reserve = 1300, 300
    practice = snapshots(tmp_path, passing=False)
    practice["request"] = "Add a setting. " + "r" * 600
    many = "\n".join(json.dumps({"rule": 1, "problem": "p" * 230}) for _ in range(30))
    long = [(f"f{i}.txt", str(i) * 600) for i in range(4)]
    fake = practice_chat(reviews=[many], writes=long, dream=dream_lines("d" * 600, "e" * 600),
                         refine=lines(*[("rule", f"rule {i} " + "z" * 100) for i in range(8)]))
    for design in ("replay", "dreaming"):
        sleep(design, fake, practice, window=window, reserve=reserve)
    assert any(p.endswith(designs.REFINE_INSTRUCTION) for p in prompts(fake))
    shown = [p.split("CHANGES:\n")[1].split("\n\nCHECK RESULT:")[0] for p in review_prompts(fake)]
    assert all(s.startswith("FILE f0.txt:\n000") for s in shown)
    assert min(map(len, shown)) < designs.CHANGES_CAP  # the changes were trimmed to fit
    starts = [c for c in fake.calls if c["tools"] and len(c["messages"]) == 2]  # later steps: the agent's own ratio
    assert len(starts) == 8
    for call in [c for c in fake.calls if c["tools"] is None] + starts:
        size = (agent.estimate_prompt(call["messages"]) if call["tools"]
                else llm.estimate_tokens(call["messages"][0]["content"]))
        assert size <= window - reserve and call["max_tokens"] == reserve


def test_practice_seeds_are_distinct_and_stay_within_this_sleeps_hundred(tmp_path, spy):
    fake = practice_chat(reviews=[PROBLEM, PROBLEM, PROBLEM])
    sleep("dreaming", fake, snapshots(tmp_path), seed=1000)
    consolidation = 2
    seeds = [c["seed"] for c in fake.calls[consolidation:] if c["tools"] is None]
    seeds += [s["seed"] + step for s in spy for step in range(1, s["max_steps"] + 1)]
    assert len(set(seeds)) == len(seeds) and all(1000 < s < 1100 for s in seeds)
    assert len(spy) == 4  # two rounds, each refined and validated
    fake = practice_chat()
    sleep("dreaming", fake, snapshots(tmp_path / "again"))
    assert {c["seed"] for c in fake.calls} == {None}


def test_dreaming_and_replay_inject_only_their_items():
    artifact = {"items": [{"kind": "rule", "text": MILLIS[1]}], "practice": [{"request": "SECRET_DREAM_NAME"}]}
    for design in ("dreaming", "replay"):
        assert designs.inject(design, artifact) == f"- RULE: {MILLIS[1]}" and designs.inject(design, None) == ""


def test_second_practice_sleep_consolidates_into_the_first_items(tmp_path, spy):
    fake = practice_chat()
    first, _ = sleep("replay", fake, snapshots(tmp_path))
    designs.sleep("replay", first, session_events(2), chat=fake, window=4096, reserve=700,
                  practice=snapshots(tmp_path / "s2"))
    merges = [p for p in prompts(fake) if p.endswith(designs.MERGE_INSTRUCTION)]
    assert len(merges) == 2 and merges[1].startswith(f"CURRENT memory:\n{lines(MILLIS, LESSON)}\n\nNEW items:\n")


def test_contamination_lists_identifiers_that_only_a_dream_put_into_memory():
    memory = ("- RULE: name delays like `RETRY_DELAY_MS`; TIMEOUT_MS stays.\n- RULE: CACHE_TTL_MS = 5000; "
              "BACKOFF_MS too; LOG_LEVEL_NAME; `abcd`")
    log = [{"request": "Add CACHE_TTL_MS, `RETRY_DELAY_MS`, TIMEOUT_MS, BACKOFF_MS, SHORT and `abcd`.",
            "hypothetical": True},
           {"request": "Set LOG_LEVEL_NAME.", "hypothetical": False}]
    wake = [[{"type": "user", "text": "Rename TIMEOUT_S to TIMEOUT_MS."}],
            [{"type": "user", "text": "s2"}, {"type": "result", "step": 1, "text": "BACKOFF_MS = 1"}]]
    assert designs.contamination(memory, log, wake) == ["CACHE_TTL_MS", "RETRY_DELAY_MS"]
    assert designs.contamination(memory, [], wake) == []


def test_practice_instruction_texts():
    assert designs.PRACTICE_ROUNDS == 2
    assert designs.DREAM_INSTRUCTION.startswith("Invent 2 new, different change requests of the same kind")
    assert designs.DREAM_INSTRUCTION.endswith('Output JSON lines, one object per line: {"request": "..."}')
    assert designs.REVIEW_INSTRUCTION.startswith("You are checking whether an agent followed the project's rules")
    assert designs.REVIEW_INSTRUCTION.endswith("If nothing is wrong, output the single word NONE.")
    assert designs.REFINE_INSTRUCTION.startswith("An agent that was given these rules still made the mistakes above")
    assert designs.REFINE_INSTRUCTION.endswith('{"kind": "rule" or "lesson", "text": "..."}')
