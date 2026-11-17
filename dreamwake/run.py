"""Run the experiment: a shared wake (s1, s2), then per design sleep -> reset -> final session -> hidden tests.

python -m dreamwake.run --task settings_gen --designs workspace,handoff --seeds 3 --window 4096 --reserve 1024
python -m dreamwake.run --summarize runs/<name>
"""
import argparse
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from . import agent, designs, llm, tasks

REPO = Path(__file__).resolve().parent.parent
SEED_STRIDE, ATTEMPT_STRIDE, SESSION_STRIDE = 100_000, 1_000, 100  # chat seed = sum of these + step
FINAL_OFFSET = 99_000  # same final-session seeds for every design (common random numbers)
SLEEP_OFFSET = 98_000  # likewise for the sleep calls; stays clear of the wake and final seeds


def _git(*args):
    try:
        proc = subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True)
    except OSError:
        return None
    return proc.stdout.strip() if proc.returncode == 0 else None


def _git_commit():
    return _git("rev-parse", "HEAD") or "unknown"


def _git_dirty():
    status = _git("status", "--porcelain")
    return None if status is None else bool(status)


def _wrote_trap(events, trap_paths, workspace):
    """True when a write_file call that succeeded targeted a trap path, however the model spelled it."""
    workspace = Path(workspace).resolve()
    results = {e["step"]: e["text"] for e in events if e["type"] == "result"}
    wanted = {path.casefold() for path in trap_paths}
    for e in events:
        if e["type"] == "call" and e["tool"] == "write_file" and not results.get(e["step"], "").startswith("error"):
            try:
                target = agent._inside(workspace, str(e["args"].get("path", "")))  # as the tool resolves it
            except (PermissionError, OSError, ValueError):
                continue
            if target.relative_to(workspace).as_posix().casefold() in wanted:
                return True
    return False


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
            evaluation = tasks.evaluate(task_id, sid, workspace)
            if evaluation.get("error"):  # the evaluator broke; that says nothing about the agent
                raise RuntimeError(evaluation["error"])
            wake[f"{sid}_passed"] = evaluation["passed"]
            wake["events"].append(result["events"])
        if wake["s2_passed"]:
            return workspace, wake
    return None, wake


def _final(task_id, design, wake, wake_ws, baseline, sessions, seed_dir, k, chat, window, reserve, trap_paths):
    artifact, sleep_cost = None, {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "seconds": 0.0}
    for i, events in enumerate(wake["events"]):
        artifact, cost = designs.sleep(design, artifact, events, chat=chat, window=window, reserve=reserve,
                                       seed=k * SEED_STRIDE + SLEEP_OFFSET + i * SESSION_STRIDE)
        sleep_cost = {key: sleep_cost[key] + cost[key] for key in sleep_cost}
    memory = designs.inject(design, artifact)
    workspace = seed_dir / "ws_final"  # the same path for every design: the path must not name the design
    shutil.rmtree(workspace, ignore_errors=True)
    tasks._copy(wake_ws, workspace)
    result = _session(task_id, sessions["final"], workspace, chat, window, reserve, k * SEED_STRIDE + FINAL_OFFSET,
                      seed_dir / f"final_{design}.jsonl", memory)
    evaluation = tasks.evaluate(task_id, "final", workspace, baseline_dir=baseline)
    repeated = _wrote_trap(result["events"], trap_paths, workspace)
    archive = seed_dir / f"final_{design}_ws"
    shutil.rmtree(archive, ignore_errors=True)
    shutil.move(workspace, archive)
    return {"checks": evaluation["checks"], "passed": evaluation["passed"], "eval_error": evaluation.get("error"),
            "steps": result["steps"], "ended_by": result["ended_by"], "ran_out_of_steps": result["ran_out_of_steps"],
            "overflow": result["overflow"], "repeated_mistake": repeated,
            "prompt_tokens": result["prompt_tokens"], "completion_tokens": result["completion_tokens"],
            "max_prompt_tokens": result["max_prompt_tokens"], "seconds": result["seconds"],
            "sleep_cost": sleep_cost, "memory_chars": len(memory),
            "memory_tokens": llm.estimate_tokens(agent.MEMORY.format(memory=memory)) if memory else 0,
            "memory": memory}


def run_experiment(task_id, designs_, seeds, window, reserve, out_dir, chat, wake_retries=3):
    for design in designs_:
        designs.inject(design, None)  # fail fast on an unknown design name, before any LLM call
    task = tasks.load_task(task_id)
    sessions = {s["id"]: s for s in task["sessions"]}
    trap_paths = set(task.get("trap_paths", []))
    out = Path(out_dir)
    if (out / "results.jsonl").exists():
        raise FileExistsError(f"{out / 'results.jsonl'} already exists: use a new --out directory")
    out.mkdir(parents=True, exist_ok=True)
    model = os.environ.get("DW_MODEL")
    config = {"args": {"task": task_id, "designs": list(designs_), "seeds": seeds, "window": window,
                       "reserve": reserve, "out": str(out), "wake_retries": wake_retries},
              "model": model, "git_commit": _git_commit(), "git_dirty": _git_dirty(),
              "started": datetime.now(timezone.utc).isoformat()}
    (out / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    for k in range(seeds):
        seed_dir = out / f"seed{k}"
        rows = []  # written only once every design of this seed is done, so seeds stay paired
        try:
            wake_ws, wake = _wake(task_id, sessions, seed_dir, k, chat, window, reserve, wake_retries)
            flags = {"s1_passed": wake["s1_passed"], "s2_passed": wake["s2_passed"],
                     "trap_experienced": any(_wrote_trap(events, trap_paths, seed_dir / "ws_wake")
                                             for events in wake["events"])}
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
                rows.append(row)
        except (RuntimeError, OSError) as error:  # a transport or evaluator fault loses this seed, not the run
            with open(out / "errors.jsonl", "a", encoding="utf-8") as errors:
                errors.write(json.dumps({"seed": k, "error": f"{type(error).__name__}: {error}"}) + "\n")
            continue
        with open(out / "results.jsonl", "a", encoding="utf-8") as results:
            results.write("".join(json.dumps(row) + "\n" for row in rows))


def _mean(values, fmt="{:.1f}"):
    values = list(values)
    return fmt.format(sum(values) / len(values)) if values else "-"


def summarize(out_dir):
    """A Markdown table: one row per metric, one column per design."""
    out = Path(out_dir)
    lines = (out / "results.jsonl").read_text(encoding="utf-8").splitlines()
    rows = [json.loads(line) for line in lines if line.strip()]
    names = list(dict.fromkeys(r["design"] for r in rows))
    seeds = {r["seed"]: {x["design"] for x in rows if x["seed"] == r["seed"]} for r in rows}
    rows = [r for r in rows if seeds[r["seed"]] >= set(names)]  # a seed counts only if every design has its row
    errors = out / "errors.jsonl"
    errored = len(errors.read_text(encoding="utf-8").splitlines()) if errors.exists() else 0
    scored = [r for r in rows if r["valid"] and not r.get("eval_error")]  # an evaluator fault is not a failure
    checks = list(dict.fromkeys(c for r in scored for c in r["checks"]))
    rate = "{:.2f}"
    metrics = [("completion rate", lambda r: r["passed"], rate),
               ("ran-out-of-steps rate", lambda r: r["ran_out_of_steps"], rate)]
    metrics += [(f"{c} pass rate", lambda r, c=c: r["checks"].get(c, False), rate) for c in checks]
    metrics += [("repeated-mistake rate", lambda r: r["repeated_mistake"], rate),
                ("mean steps", lambda r: r["steps"], "{:.1f}"),
                ("mean final tokens", lambda r: r["prompt_tokens"] + r["completion_tokens"], "{:.1f}"),
                ("mean final seconds", lambda r: r["seconds"], "{:.1f}"),
                ("mean sleep seconds", lambda r: r["sleep_cost"]["seconds"], "{:.1f}"),
                ("mean injected tokens", lambda r: r["memory_tokens"], "{:.1f}")]
    by_design = {d: [r for r in rows if r["design"] == d] for d in names}
    valid = {d: [r for r in by_design[d] if r["valid"]] for d in names}
    final = {d: [r for r in scored if r["design"] == d] for d in names}
    wake_rows = [("wake validity", lambda r: r["valid"], rate),
                 ("mean wake attempts", lambda r: r["wake_attempts"], "{:.1f}"),
                 ("trap-experienced rate", lambda r: r["wake"]["trap_experienced"], rate)]
    table = ["| metric | " + " | ".join(names) + " |", "|---|" + "---|" * len(names),
             "| n valid | " + " | ".join(f"{len(valid[d])}/{len(by_design[d])}" for d in names) + " |"]
    for label, value, fmt in wake_rows:
        table.append(f"| {label} | " + " | ".join(_mean(map(value, by_design[d]), fmt) for d in names) + " |")
    for label, value, fmt in metrics:
        table.append(f"| {label} | " + " | ".join(_mean(map(value, final[d]), fmt) for d in names) + " |")
    table += [f"{d} eval errors: {sum(1 for r in valid[d] if r.get('eval_error'))}" for d in names]
    return "\n".join(table + [f"errored seeds: {errored}"])


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
