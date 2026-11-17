"""One usable work session, with durable memory and an empty conversation on each run."""
import argparse
import json
import os
import shutil
from pathlib import Path

from . import agent, designs, llm, run, tasks

CONTINUE = "Continue the unfinished work from your previous session and finish it. Make the checks pass."


def _show_event(event):
    kind = event["type"]
    if kind == "call":
        args = ", ".join(f"{key}={value!r}" for key, value in event["args"].items())[:100]
        print(f"  {event['step']} {event['tool']}({args})", flush=True)
    elif kind in ("result", "say"):
        text = event["text"][:100].replace("\r", " ").replace("\n", " ")
        prefix = "    -> " if kind == "result" else f"  {event['step']} said: "
        print(prefix + text, flush=True)


def main(argv=None, chat=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", help="task to start (default: settings_resume)")
    parser.add_argument("--design", help="memory design (default: hybrid)")
    parser.add_argument("--state", default="runs/demo")
    parser.add_argument("--interrupt-after", type=int)
    parser.add_argument("--request")
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--window", type=int, default=4096)
    parser.add_argument("--reserve", type=int, default=1024)
    args = parser.parse_args(argv)
    if chat is None:
        if not os.environ.get("DW_MODEL"):
            parser.error("set DW_MODEL to the model id served by the endpoint")
        chat = llm.chat
    state_dir = Path(args.state).resolve()
    state_path = state_dir / "state.json"
    if args.reset and state_dir.exists():
        shutil.rmtree(state_dir)
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        for key in ("task", "design"):
            given = getattr(args, key)
            if given is not None and given != state[key]:
                parser.error(f"--{key} does not match saved {key}; use --reset to start over")
    else:
        state = {"task": "settings_resume" if args.task is None else args.task,
                 "design": "hybrid" if args.design is None else args.design,
                 "artifact": None, "requests": [], "sessions_done": 0, "last_ended_by": None}
    try:
        store, placement, gate = run.parse_design(state["design"])
    except ValueError as error:
        parser.error(str(error))
    if store in ("oracle", "replay", "dreaming"):
        parser.error(f"the demo does not support the {store} store")
    task = tasks.load_task(state["task"])
    workspace = state_dir / "workspace"
    if not state_path.exists():
        if workspace.exists():
            if not workspace.resolve().is_relative_to(state_dir):
                parser.error("workspace must be inside --state")
            shutil.rmtree(workspace)
        tasks.make_workspace(state["task"], workspace)
    used_task_prompt = False
    if args.request is not None:
        request = args.request
    elif state["sessions_done"] and state["last_ended_by"] != "finish":
        request = CONTINUE
    else:
        # Requests alone cannot distinguish an override or continuation from a task prompt.
        used = sum(event.get("task_prompt", False)
                   for k in range(1, state["sessions_done"] + 1)
                   for line in (state_dir / f"session{k}.jsonl").read_text(encoding="utf-8").splitlines()
                   if (event := json.loads(line))["type"] == "demo")
        if used >= len(task["sessions"]):
            print("The task's requests are all done.")
            return 0
        request = task["sessions"][used]["prompt"]
        used_task_prompt = True
    k = state["sessions_done"] + 1
    memory = designs.inject(store, state["artifact"])
    print(f"== Session {k} (conversation memory: empty) ==")
    print("Remembered from earlier sessions:")
    print(memory or "(nothing)")
    print("Request:")
    print(request)
    log_path = state_dir / f"session{k}.jsonl"
    result = agent.run_session(state["task"], workspace, request,
                               max_steps=20 if args.interrupt_after is None else args.interrupt_after,
                               window=args.window, reserve=args.reserve, chat=chat, memory=memory,
                               memory_at=placement, finish_gate=gate, log_path=log_path, on_event=_show_event)
    print(f"Session ended: {result['ended_by']} after {result['steps']} steps.")
    with log_path.open("a", encoding="utf-8") as log:
        log.write(json.dumps({"type": "demo", "task_prompt": used_task_prompt}) + "\n")
    requests = state["requests"] + [request]
    artifact, _ = designs.sleep(store, state["artifact"], result["events"], chat=chat,
                               window=args.window, reserve=args.reserve, practice={"requests": requests})
    state.update(artifact=artifact, requests=requests, sessions_done=k, last_ended_by=result["ended_by"])
    state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    memory = designs.inject(store, artifact)
    print(f"Stored for next time ({len(memory)} characters):")
    print(memory or "(nothing)")
    code, _ = tasks.run_check(state["task"], workspace)
    print("Project checks: " + ("passed" if code == 0 else "FAILED"))
    print("Run the same command again to continue with an empty conversation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
