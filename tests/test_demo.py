"""The interruptible CLI, using real tiny projects and fake model replies only."""
import importlib
import json

import pytest

from dreamwake import agent, designs, tasks
from test_agent import FINISH, reply

NOTE = "Keep durations in milliseconds."


class DemoChat:
    def __init__(self, *replies):
        self.replies = iter(replies or [FINISH])
        self.work, self.sleeps = [], []

    def __call__(self, messages, tools=None, **kwargs):
        record = {"messages": json.loads(json.dumps(messages)), **kwargs}
        if tools is None:
            self.sleeps.append(record)
            prompt = messages[0]["content"]
            if designs.HANDOFF_INSTRUCTION in prompt:
                content = NOTE
            elif designs.REVIEW_INSTRUCTION in prompt:
                content = json.dumps({"number": -1, "text": ""})
            else:
                content = json.dumps({"kind": "rule", "text": NOTE})
            return reply(content=content)
        self.work.append(record)
        item = next(self.replies)
        if isinstance(item, BaseException):
            raise item
        return item


@pytest.fixture
def demo():
    return importlib.import_module("dreamwake.demo")


@pytest.fixture
def project(tmp_path, monkeypatch):
    root = tmp_path / "tasks" / "settings_resume"
    workspace = root / "workspace"
    workspace.mkdir(parents=True)
    (workspace / "source.txt").write_text("original", encoding="utf-8")
    (workspace / "check.py").write_text("raise SystemExit(0)\n", encoding="utf-8")
    prompts = ["First task.", "Second task.",
               "Continue the unfinished work from your previous session and finish it. Make the checks pass."]
    task = {"id": "settings_resume", "sessions": [{"prompt": p} for p in prompts],
            "check_cmd": ["python check.py"]}
    (root / "task.json").write_text(json.dumps(task), encoding="utf-8")
    monkeypatch.setattr(tasks, "TASKS_DIR", root.parent)
    monkeypatch.delenv("DW_MODEL", raising=False)
    return tmp_path / "demo", prompts


def state_at(path):
    return json.loads((path / "state.json").read_text(encoding="utf-8"))


def test_fresh_demo_creates_layout_and_prints_headings(demo, project, capsys):
    state, prompts = project
    chat = DemoChat(reply(("write_file", {"path": "new.txt", "content": "work"})), FINISH)
    assert demo.main(["--state", str(state)], chat=chat) == 0
    saved = state_at(state)
    assert set(saved) == {"task", "design", "artifact", "requests", "sessions_done", "last_ended_by"}
    assert saved["task"] == "settings_resume" and saved["design"] == "hybrid"
    assert saved["sessions_done"] == 1 and saved["last_ended_by"] == "finish"
    assert saved["requests"] == prompts[:1] and isinstance(saved["artifact"], dict)
    assert (state / "workspace" / "source.txt").read_text(encoding="utf-8") == "original"
    assert (state / "workspace" / "new.txt").read_text(encoding="utf-8") == "work"
    assert (state / "session1.jsonl").exists()
    assert len(chat.sleeps) == 2  # one sleep, extracting and merging its rule
    output = capsys.readouterr().out
    headings = ["== Session 1 (conversation memory: empty) ==",
                "Remembered from earlier sessions:\n(nothing)", "Request:\n" + prompts[0],
                "Session ended: finish after 2 steps.", "Stored for next time (",
                "Project checks: passed", "Run the same command again to continue with an empty conversation."]
    assert [output.index(h) for h in headings] == sorted(output.index(h) for h in headings)
    assert "  1 write_file(" in output and "    -> wrote 4 characters to new.txt" in output
    assert "at most 20 steps" in chat.work[0]["messages"][0]["content"]
    assert chat.work[0]["max_tokens"] == 1024
    memory = designs.inject("hybrid", saved["artifact"])
    assert f"Stored for next time ({len(memory)} characters):\n{memory}" in output


def test_missing_state_starts_with_a_clean_workspace(demo, project):
    state, _ = project
    workspace = state / "workspace"
    workspace.mkdir(parents=True)
    (workspace / "leftover.txt").write_text("stale", encoding="utf-8")
    (workspace / "source.txt").write_text("modified", encoding="utf-8")
    demo.main(["--state", str(state), "--design", "workspace"], chat=DemoChat())
    assert not (workspace / "leftover.txt").exists()
    assert (workspace / "source.txt").read_text(encoding="utf-8") == "original"


@pytest.mark.parametrize("cutoff", ["steps", "ctrl_c"])
def test_unfinished_demo_remembers_and_continues_without_consuming_a_prompt(demo, project, capsys, cutoff):
    state, prompts = project
    first = DemoChat(reply(("write_file", {"path": "progress.txt", "content": "partial"})), KeyboardInterrupt())
    flags = ["--interrupt-after", "1"] if cutoff == "steps" else []
    demo.main(["--state", str(state), *flags], chat=first)
    saved = state_at(state)
    assert saved["last_ended_by"] == ("max_steps" if cutoff == "steps" else "interrupted")
    assert first.sleeps and saved["sessions_done"] == 1
    remembered = designs.inject("hybrid", saved["artifact"])
    capsys.readouterr()
    second = DemoChat()
    demo.main(["--state", str(state)], chat=second)
    output = capsys.readouterr().out
    assert "== Session 2 (conversation memory: empty) ==" in output
    assert "Remembered from earlier sessions:\n" + remembered in output
    assert "Request:\n" + demo.CONTINUE in output
    assert second.work[0]["messages"] == [
        {"role": "system", "content": agent.SYSTEM.format(max_steps=20) + agent.MEMORY.format(memory=remembered)},
        {"role": "user", "content": demo.CONTINUE}]
    assert (state / "workspace" / "progress.txt").exists() and (state / "session2.jsonl").exists()
    assert state_at(state)["requests"] == [prompts[0], demo.CONTINUE]
    third = DemoChat()
    demo.main(["--state", str(state)], chat=third)
    assert third.work[0]["messages"][1]["content"] == prompts[1]


def test_finished_demo_uses_next_prompt_and_saved_task_and_design(demo, project, capsys):
    state, prompts = project
    demo.main(["--state", str(state), "--design", "handoff"], chat=DemoChat())
    chat = DemoChat()
    demo.main(["--state", str(state)], chat=chat)
    assert chat.work[0]["messages"][1]["content"] == prompts[1]
    assert agent.MEMORY.format(memory=NOTE) in chat.work[0]["messages"][0]["content"]
    assert "Remembered from earlier sessions:\n" + NOTE in capsys.readouterr().out
    assert state_at(state)["design"] == "handoff"


def test_saved_nondefault_task_is_used_when_task_is_omitted(demo, project):
    state, prompts = project
    (tasks.TASKS_DIR / "settings_resume").rename(tasks.TASKS_DIR / "custom")
    demo.main(["--state", str(state), "--task", "custom"], chat=DemoChat())
    chat = DemoChat()
    demo.main(["--state", str(state)], chat=chat)
    assert state_at(state)["task"] == "custom"
    assert chat.work[0]["messages"][1]["content"] == prompts[1]


@pytest.mark.parametrize("text", ["My custom request.", "", "First task."])
def test_request_override_does_not_consume_task_prompts(demo, project, text):
    state, prompts = project
    demo.main(["--state", str(state), "--design", "workspace", "--request", text], chat=DemoChat())
    assert state_at(state)["requests"] == [text]
    chat = DemoChat()
    demo.main(["--state", str(state)], chat=chat)
    assert chat.work[0]["messages"][1]["content"] == prompts[0]


def test_request_override_takes_precedence_over_continuation(demo, project):
    state, _ = project
    demo.main(["--state", str(state), "--interrupt-after", "1"], chat=DemoChat(reply(("list_files", {}))))
    chat = DemoChat()
    demo.main(["--state", str(state), "--request", "Instead do this."], chat=chat)
    assert chat.work[0]["messages"][1]["content"] == "Instead do this."


@pytest.mark.parametrize("flag,value", [("--task", "other"), ("--design", "requests")])
def test_saved_task_or_design_mismatch_requires_reset(demo, project, capsys, flag, value):
    state, _ = project
    demo.main(["--state", str(state)], chat=DemoChat())
    before = (state / "state.json").read_bytes()
    with pytest.raises(SystemExit) as error:
        demo.main(["--state", str(state), flag, value], chat=DemoChat())
    assert error.value.code == 2 and "--reset" in capsys.readouterr().err
    assert (state / "state.json").read_bytes() == before and not (state / "session2.jsonl").exists()


def test_reset_removes_old_workspace_logs_and_state(demo, project):
    state, prompts = project
    demo.main(["--state", str(state)], chat=DemoChat())
    demo.main(["--state", str(state)], chat=DemoChat())
    (state / "workspace" / "old.txt").write_text("old", encoding="utf-8")
    demo.main(["--state", str(state), "--reset", "--design", "requests"], chat=DemoChat())
    saved = state_at(state)
    assert saved["sessions_done"] == 1 and saved["requests"] == prompts[:1]
    assert saved["artifact"] == prompts[:1] and saved["design"] == "requests"
    assert not (state / "session2.jsonl").exists() and not (state / "workspace" / "old.txt").exists()


@pytest.mark.parametrize("spec", ["oracle", "replay@tail+gate", "dreaming@request", "workspace@tail",
                                      "workspace+gate", "unknown", ""])
def test_unsupported_design_is_refused_before_work(demo, project, capsys, spec):
    state, _ = project
    chat = DemoChat()
    with pytest.raises(SystemExit) as error:
        demo.main(["--state", str(state), "--design", spec], chat=chat)
    assert error.value.code == 2 and spec.split("@")[0].split("+")[0] in capsys.readouterr().err
    assert not chat.work and not state.exists()


@pytest.mark.parametrize("store", ["handoff", "consolidated", "requests", "review", "hybrid"])
@pytest.mark.parametrize("placement", ["system", "request", "tail"])
def test_supported_design_placement_and_gate_survive_reload(demo, project, store, placement):
    state, _ = project
    spec = f"{store}@{placement}+gate"
    flags = ["--state", str(state), "--design", spec, "--window", "6000", "--reserve", "700"]
    demo.main(flags, chat=DemoChat())
    memory = designs.inject(store, state_at(state)["artifact"])
    chat = DemoChat(FINISH, FINISH)
    demo.main(["--state", str(state), "--window", "6000", "--reserve", "700"], chat=chat)
    first = chat.work[0]["messages"]
    assert memory in first[{"system": 0, "request": 1, "tail": 2}[placement]]["content"]
    assert len(first) == (3 if placement == "tail" else 2)
    assert len(chat.work) == 2
    assert any(m["role"] == "tool" and m["content"] == agent.GATE.format(memory=memory)
               for m in chat.work[1]["messages"])
    assert chat.work[0]["max_tokens"] == 700 and state_at(state)["design"] == spec
    if store == "review":
        assert "1. First task.\n2. Second task." in chat.sleeps[-1]["messages"][0]["content"]


def test_exhausted_prompts_exit_without_work_or_state_changes(demo, project, capsys):
    state, prompts = project
    for _ in prompts:
        demo.main(["--state", str(state), "--design", "workspace"], chat=DemoChat())
    before = (state / "state.json").read_bytes()
    capsys.readouterr()
    chat = DemoChat()
    assert demo.main(["--state", str(state)], chat=chat) == 0
    assert "task's requests are all done" in capsys.readouterr().out.lower()
    assert not chat.work and not chat.sleeps and (state / "state.json").read_bytes() == before
    assert not (state / "session4.jsonl").exists()


def test_event_output_is_one_line_and_truncated_and_checks_can_fail(demo, project, capsys):
    state, _ = project
    long = "x" * 105 + "\nsecond line"
    chat = DemoChat(reply(content=long), reply(("read_file", {"path": "check.py"})),
                    reply(("write_file", {"path": "check.py", "content": "raise SystemExit(1)\n"})), FINISH)
    demo.main(["--state", str(state), "--design", "workspace"], chat=chat)
    output = capsys.readouterr().out
    assert "  1 said: " + "x" * 100 + "\n" in output and "x" * 101 not in output
    assert "    -> raise SystemExit(0) " in output
    assert "Project checks: FAILED" in output


def test_missing_model_is_a_clear_cli_error(demo, project, capsys):
    state, _ = project
    with pytest.raises(SystemExit) as error:
        demo.main(["--state", str(state)])
    assert error.value.code == 2 and "DW_MODEL" in capsys.readouterr().err
    assert not state.exists()
