"""Run the experiment: a shared wake (s1, s2), then per design sleep -> reset -> final session -> hidden tests.

python -m dreamwake.run --task settings_gen --designs workspace,handoff --seeds 3 --window 4096 --reserve 1024
python -m dreamwake.run --summarize runs/<name>
"""
import argparse
import json
import os
import posixpath
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from . import agent, designs, llm, tasks

REPO = Path(__file__).resolve().parent.parent
SEED_STRIDE, ATTEMPT_STRIDE, SESSION_STRIDE = 100_000, 1_000, 100  # chat seed = sum of these + step
FINAL_OFFSET = 99_000  # same final-session seeds for every design (common random numbers)


def _git_commit():
    try:
        proc = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True)
    except OSError:
        return "unknown"
    return proc.stdout.strip() or "unknown"


def _wrote_trap(events, trap_paths):
    return any(e["type"] == "call" and e["tool"] == "write_file"
               and posixpath.normpath(str(e["args"].get("path", "")).replace("\\", "/")) in trap_paths
               for e in events)


def _session(task_id, session, workspace, chat, window, reserve, seed, log_path, memory=""):
    return agent.run_session(task_id, workspace, session["prompt"], max_steps=session["max_steps"], window=window,
                             reserve=reserve, chat=chat, memory=memory, seed=seed, log_path=log_path)


def _wake(task_id, sessions, seed_dir, k, chat, window, reserve, wake_retries):
    """Run s1 then s2 on one workspace until s2's hidden tests pass. Returns (workspace or None, wake)."""
    for attempt in range(1, wake_retries + 2):
        workspace = seed_dir / "ws_wake"
        shutil.rmtree(workspace, ignore_errors=True)
        tasks.make_workspace(task_id, workspace)
        suffix = "" if attempt == 1 else f".attempt{attempt}"
        wake = {"attempts": attempt, "events": []}
        for i, sid in enumerate(("s1", "s2")):
            seed = k * SEED_STRIDE + attempt * ATTEMPT_STRIDE + i * SESSION_STRIDE
            result = _session(task_id, sessions[sid], workspace, chat, window, reserve, seed,
                              seed_dir / f"wake_{sid}{suffix}.jsonl")
            wake[f"{sid}_passed"] = tasks.evaluate(task_id, sid, workspace)["passed"]
            wake["events"].append(result["events"])
        if wake["s2_passed"]:
            return workspace, wake
    return None, wake


def _final(task_id, design, wake, wake_ws, baseline, sessions, seed_dir, k, chat, window, reserve, trap_paths):
    artifact, sleep_cost = None, {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "seconds": 0.0}
    for events in wake["events"]:
        artifact, cost = designs.sleep(design, artifact, events, chat=chat, window=window, reserve=reserve)
        sleep_cost = {key: sleep_cost[key] + cost[key] for key in sleep_cost}
    memory = designs.inject(design, artifact)
    workspace = seed_dir / f"ws_final_{design}"
    shutil.rmtree(workspace, ignore_errors=True)
    tasks._copy(wake_ws, workspace)
    result = _session(task_id, sessions["final"], workspace, chat, window, reserve, k * SEED_STRIDE + FINAL_OFFSET,
                      seed_dir / f"final_{design}.jsonl", memory)
    evaluation = tasks.evaluate(task_id, "final", workspace, baseline_dir=baseline)
    return {"checks": evaluation["checks"], "passed": evaluation["passed"], "eval_error": evaluation.get("error"),
            "steps": result["steps"], "ran_out_of_steps": result["ran_out_of_steps"], "overflow": result["overflow"],
            "repeated_mistake": _wrote_trap(result["events"], trap_paths),
            "prompt_tokens": result["prompt_tokens"], "completion_tokens": result["completion_tokens"],
            "max_prompt_tokens": result["max_prompt_tokens"], "seconds": result["seconds"],
            "sleep_cost": sleep_cost, "memory_chars": len(memory),
            "memory_tokens": llm.estimate_tokens(memory) if memory else 0, "memory": memory}


def run_experiment(task_id, designs_, seeds, window, reserve, out_dir, chat, wake_retries=3):
    for design in designs_:
        designs.inject(design, None)  # fail fast on an unknown design name, before any LLM call
    task = tasks.load_task(task_id)
    sessions = {s["id"]: s for s in task["sessions"]}
    trap_paths = set(task.get("trap_paths", []))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model = os.environ.get("DW_MODEL")
    config = {"args": {"task": task_id, "designs": list(designs_), "seeds": seeds, "window": window,
                       "reserve": reserve, "out": str(out), "wake_retries": wake_retries},
              "model": model, "git_commit": _git_commit(), "started": datetime.now(timezone.utc).isoformat()}
    (out / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    for k in range(seeds):
        seed_dir = out / f"seed{k}"
        wake_ws, wake = _wake(task_id, sessions, seed_dir, k, chat, window, reserve, wake_retries)
        flags = {"s1_passed": wake["s1_passed"], "s2_passed": wake["s2_passed"],
                 "trap_experienced": any(_wrote_trap(events, trap_paths) for events in wake["events"])}
        baseline = None
        if wake_ws:
            baseline = seed_dir / "baseline"
            shutil.rmtree(baseline, ignore_errors=True)
            tasks._copy(wake_ws, baseline)
        for design in designs_:
            row = {"task": task_id, "seed": k, "design": design, "model": model, "window": window,
                   "reserve": reserve, "wake_attempts": wake["attempts"], "wake": flags, "valid": bool(wake_ws)}
            if wake_ws:
                row.update(_final(task_id, design, wake, wake_ws, baseline, sessions, seed_dir, k, chat, window,
                                  reserve, trap_paths))
            with open(out / "results.jsonl", "a", encoding="utf-8") as results:
                results.write(json.dumps(row) + "\n")


def _mean(values, fmt="{:.1f}"):
    values = list(values)
    return fmt.format(sum(values) / len(values)) if values else "-"


def summarize(out_dir):
    """A Markdown table: one row per metric, one column per design."""
    lines = (Path(out_dir) / "results.jsonl").read_text(encoding="utf-8").splitlines()
    rows = [json.loads(line) for line in lines if line.strip()]
    names = list(dict.fromkeys(r["design"] for r in rows))
    checks = list(dict.fromkeys(c for r in rows if r["valid"] for c in r["checks"]))
    rate = "{:.2f}"
    metrics = [("completion rate", lambda r: r["passed"], rate)]
    metrics += [(f"{c} pass rate", lambda r, c=c: r["checks"].get(c, False), rate) for c in checks]
    metrics += [("repeated-mistake rate", lambda r: r["repeated_mistake"], rate),
                ("mean steps", lambda r: r["steps"], "{:.1f}"),
                ("mean final tokens", lambda r: r["prompt_tokens"] + r["completion_tokens"], "{:.1f}"),
                ("mean final seconds", lambda r: r["seconds"], "{:.1f}"),
                ("mean sleep seconds", lambda r: r["sleep_cost"]["seconds"], "{:.1f}"),
                ("mean injected tokens", lambda r: r["memory_tokens"], "{:.1f}")]
    by_design = {d: [r for r in rows if r["design"] == d] for d in names}
    valid = {d: [r for r in by_design[d] if r["valid"]] for d in names}
    table = ["| metric | " + " | ".join(names) + " |", "|---|" + "---|" * len(names),
             "| n valid | " + " | ".join(f"{len(valid[d])}/{len(by_design[d])}" for d in names) + " |"]
    for label, value, fmt in metrics:
        table.append(f"| {label} | " + " | ".join(_mean(map(value, valid[d]), fmt) for d in names) + " |")
    return "\n".join(table)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m dreamwake.run", description=__doc__.splitlines()[0])
    parser.add_argument("--task", default="settings_gen")
    parser.add_argument("--designs", default="workspace,handoff")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--window", type=int, default=4096)
    parser.add_argument("--reserve", type=int, default=1024)
    parser.add_argument("--out", help="output directory (default runs/<UTC timestamp>)")
    parser.add_argument("--summarize", metavar="DIR", help="print the summary of an existing run and exit")
    args = parser.parse_args(argv)
    if args.summarize:
        print(summarize(args.summarize))
        return
    if not os.environ.get("DW_MODEL"):
        parser.error("set DW_MODEL to the model id served by the endpoint")
    if not 0 < args.reserve < args.window or args.seeds < 1:
        parser.error("need 0 < reserve < window and seeds >= 1")
    out = args.out or Path("runs") / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_experiment(args.task, args.designs.split(","), args.seeds, args.window, args.reserve, out, llm.chat)
    print(summarize(out))


if __name__ == "__main__":
    main()
