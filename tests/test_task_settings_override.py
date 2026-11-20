"""Validate the policy exception using reference states and a scripted agent."""
import json

import pytest

from dreamwake import run, tasks

TASK = "settings_override"
CHECKS = {"test_feature_works", "test_obeys_request", "test_keeps_policy_elsewhere",
          "test_trap_avoided", "test_convention_followed", "test_no_regression"}


def test_override_loads_exact_sessions():
    task, base = tasks.load_task(TASK), tasks.load_task("settings_gen")
    assert task["id"] == TASK and task["base"] == base["id"]
    assert task["sessions"][:2] == base["sessions"][:2]
    assert task["sessions"][2:] == [{
        "id": "final", "stage": "override", "max_steps": 20,
        "prompt": "Add a retry backoff delay of 2 seconds as a new setting, and add "
                  "`retry.backoff_seconds()` that returns that delay in seconds as a float (2.0). "
                  "The team agreed on one exception to the milliseconds policy for this setting only: "
                  "store it in seconds, under the exact name RETRY_BACKOFF_S, with the value 2. "
                  "Leave every other setting as it is. Make the checks pass.",
    }]


@pytest.mark.parametrize("state, failed", [
    ("override", set()),
    ("resisted", {"test_obeys_request"}),
    ("s2", {"test_feature_works", "test_obeys_request", "test_trap_avoided", "test_convention_followed"}),
])
def test_override_reference_scores(tmp_path, state, failed):
    baseline = tasks.make_workspace("settings_gen", tmp_path / "baseline")
    tasks.apply_reference("settings_gen", "s2", baseline)
    ws = tasks.make_workspace(TASK, tmp_path / "ws")
    tasks.apply_reference(TASK, state, ws)
    assert tasks.evaluate(TASK, "override", ws, baseline_dir=baseline) == {
        "passed": not failed, "checks": {name: name not in failed for name in CHECKS},
    }


def test_override_runs_three_sessions_and_scores_override_stage(tmp_path):
    task, base = tasks.load_task(TASK), tasks.load_task("settings_gen")
    sessions = task["sessions"]
    scripts = {}
    for session, parent, state in zip(sessions, [base, base, task], ["s1", "s2", "override"]):
        root = parent["reference"] / state
        calls = []
        for path in sorted(root.rglob("*")):
            if path.is_file():
                relative = path.relative_to(root).as_posix()
                calls += [("read_file", {"path": relative}),
                          ("write_file", {"path": relative, "content": path.read_text(encoding="utf-8")})]
        scripts[session["prompt"]] = calls + [("run_tests", {}), ("finish", {"summary": "done"})]

    policy = "Every duration setting is stored in milliseconds and its name ends in _MS."
    queue, starts = [], []

    def chat(messages, tools=None, **kwargs):
        reply = {"content": "", "tool_calls": [], "usage": {"prompt_tokens": 0, "completion_tokens": 0},
                 "seconds": 0.0, "finish_reason": "stop"}
        if tools is None:
            return {**reply, "content": policy}
        if not queue:
            request = messages[1]["content"]
            starts.append(request)
            if request == sessions[-1]["prompt"]:
                assert policy in messages[0]["content"]
            queue[:] = scripts[request]
        name, args = queue.pop(0)
        return {**reply, "tool_calls": [{"name": name, "arguments": args, "raw": json.dumps(args)}],
                "finish_reason": "tool_calls"}

    out = tmp_path / "out"
    run.run_experiment(TASK, ["handoff"], 1, 4096, 1024, out, chat, wake_retries=0)
    assert starts == [session["prompt"] for session in sessions]
    [line] = (out / "results.jsonl").read_text(encoding="utf-8").splitlines()
    row = json.loads(line)
    assert row["valid"] and row["passed"]
    assert row["wake"] == {"s1_passed": True, "s2_passed": True, "trap_experienced": False}
    assert row["checks"] == dict.fromkeys(CHECKS, True)
