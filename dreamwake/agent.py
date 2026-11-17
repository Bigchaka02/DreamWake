"""The plain agent loop: five tools, a rolling context window, and a full untrimmed event record."""
import json
import time
from pathlib import Path

from . import tasks
from .llm import ContextOverflow, estimate_tokens

SYSTEM = ("You are a coding agent working in a small Python project. Use the tools to inspect and change files, "
          "and run_tests to check your work. When the request is complete and the checks pass, call finish. "
          "You have at most {max_steps} steps, so do not waste them.")
MEMORY = ("\n\nNotes carried over from your earlier sessions on this project "
          "(your conversation memory was wiped since then):\n{memory}")
MARKER = "[earlier steps omitted to fit the context window]"
NUDGE = "Use a tool. Call finish when the work is done."
READ_CAP = 3000
RENDER_CAP = 600
SKIP_DIRS = {"__pycache__", ".pytest_cache"}


def _tool(name, description, *params):
    return {"type": "function", "function": {"name": name, "description": description, "parameters": {
        "type": "object", "properties": {p: {"type": "string"} for p in params}, "required": list(params)}}}


TOOLS = [
    _tool("list_files", "List all project files."),
    _tool("read_file", "Read a file.", "path"),
    _tool("write_file", "Overwrite a whole file (creates folders).", "path", "content"),
    _tool("run_tests", "Run the project checks."),
    _tool("finish", "End the session when the request is done.", "summary"),
]
REQUIRED = {t["function"]["name"]: t["function"]["parameters"]["required"] for t in TOOLS}


def estimate_prompt(messages):
    """Estimated tokens of everything a chat call sends: tool schemas plus all messages."""
    return estimate_tokens(json.dumps(TOOLS) + json.dumps(messages))


def _inside(workspace, path):
    target = (workspace / path).resolve()
    if not target.is_relative_to(workspace):
        raise PermissionError(f"path '{path}' is outside the project")
    return target


def _execute(task_id, workspace, name, args):
    """Run one tool call and return its result text; bad model input becomes an error string."""
    if name not in REQUIRED:
        return f"error: unknown tool '{name}'; the tools are {', '.join(REQUIRED)}"
    if not isinstance(args, dict):
        return f"error: the arguments for {name} are not a valid JSON object"
    for param in REQUIRED[name]:
        if not isinstance(args.get(param), str):
            return f"error: {name} needs a string argument '{param}'"
    try:
        if name == "list_files":
            files = sorted(p.relative_to(workspace).as_posix() for p in workspace.rglob("*")
                           if p.is_file() and not SKIP_DIRS & set(p.relative_to(workspace).parts))
            return "\n".join(files) or "(no files)"
        if name == "read_file":
            text = _inside(workspace, args["path"]).read_text(encoding="utf-8", errors="replace")
            return text if len(text) <= READ_CAP else text[:READ_CAP] + f"\n[... cut at {READ_CAP} characters ...]"
        if name == "write_file":
            target = _inside(workspace, args["path"])
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(args["content"], encoding="utf-8", newline="")
            return f"wrote {len(args['content'])} characters to {args['path']}"
        if name == "run_tests":
            return tasks.run_check(task_id, workspace)[1]
        return "finished"
    except (OSError, ValueError) as error:
        return f"error: {error}"


def run_session(task_id, workspace, prompt, *, max_steps, window, reserve, chat, memory="", seed=None,
                log_path=None):
    workspace = Path(workspace).resolve()
    system = SYSTEM.format(max_steps=max_steps) + (MEMORY.format(memory=memory) if memory else "")
    pinned = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
    units, dropped = [], 0  # a unit = an assistant message plus the message(s) answering it
    events = [{"type": "user", "text": prompt}]
    log = list(events)
    totals = {"prompt_tokens": 0, "completion_tokens": 0, "max_prompt_tokens": 0}
    steps, finished, silent, overflow, start = 0, False, 0, False, time.perf_counter()
    try:
        while steps < max_steps and not finished and silent < 2:
            steps += 1
            reply = None
            while reply is None:
                messages = (pinned + ([{"role": "user", "content": MARKER}] if dropped else [])
                            + [m for unit in units[dropped:] for m in unit])
                if estimate_prompt(messages) > window - reserve:
                    if dropped == len(units):
                        raise ValueError(f"system prompt, tools and request alone exceed the window "
                                         f"({window} - {reserve})")
                    dropped += 1
                    continue
                try:
                    reply = chat(messages, TOOLS, max_tokens=reserve, seed=None if seed is None else seed + steps)
                except ContextOverflow:  # the estimate was too low: drop one more unit and retry this step
                    log.append({"type": "overflow", "step": steps, "dropped_units": dropped})
                    if dropped == len(units):
                        overflow = True
                        break
                    dropped += 1
            if overflow:
                break
            usage = reply["usage"]
            totals["prompt_tokens"] += usage["prompt_tokens"]
            totals["completion_tokens"] += usage["completion_tokens"]
            totals["max_prompt_tokens"] = max(totals["max_prompt_tokens"], usage["prompt_tokens"])
            log.append({"type": "step", "step": steps, **usage, "seconds": reply["seconds"],
                        "finish_reason": reply["finish_reason"], "dropped_units": dropped})
            content = reply["content"] or ""
            if reply["tool_calls"]:
                silent = 0
                call = reply["tool_calls"][0]
                call_id = f"call_{steps}"
                event = {"type": "call", "step": steps, "tool": call["name"],
                         "args": call["arguments"] if isinstance(call["arguments"], dict) else {"raw": call["raw"]}}
                if content:
                    event["content"] = content
                text = _execute(task_id, workspace, call["name"], call["arguments"])
                finished = call["name"] == "finish" and not text.startswith("error")
                new = [event, {"type": "result", "step": steps, "text": text}]
                # LM Studio rejects (HTTP 500) or garbles a replayed call whose arguments are not a JSON object
                units.append([{"role": "assistant", "content": content, "tool_calls": [
                                  {"id": call_id, "type": "function", "function": {
                                      "name": call["name"],
                                      "arguments": json.dumps(event["args"], ensure_ascii=False)}}]},
                              {"role": "tool", "tool_call_id": call_id, "content": text}])
            else:
                silent += 1
                new = [{"type": "say", "step": steps, "text": content}, {"type": "user", "text": NUDGE}]
                units.append([{"role": "assistant", "content": content}, {"role": "user", "content": NUDGE}])
            events += new
            log += new
    finally:
        if log_path:
            Path(log_path).parent.mkdir(parents=True, exist_ok=True)
            Path(log_path).write_text("".join(json.dumps(line) + "\n" for line in log), encoding="utf-8")
    return {"finished": finished, "steps": steps,
            "ran_out_of_steps": not finished and not overflow and steps >= max_steps, "overflow": overflow,
            "events": events, **totals, "seconds": time.perf_counter() - start, "dropped_units": dropped}


def _cap(text, limit=RENDER_CAP):
    return text if len(text) <= limit else text[:limit] + " [...]"


def _render(event):
    if event["type"] == "user":
        return "USER: " + _cap(event["text"])
    if event["type"] == "say":
        return f"STEP {event['step']} said: {_cap(event['text'])}"
    if event["type"] == "result":
        return "-> " + _cap(event["text"])
    args = event["args"]
    shown = ", ".join(f"{k}={_cap(str(v), 200)!r}" for k, v in args.items() if k != "content")
    lines = [f"STEP {event['step']} said: {_cap(event['content'])}"] if event.get("content") else []
    lines.append(f"STEP {event['step']} {event['tool']}({shown})")
    if "content" in args:
        lines.append(_cap(str(args["content"])))
    return "\n".join(lines)


def render_events(events, max_chars=None):
    """Compact transcript; with max_chars, the most recent whole steps that fit."""
    blocks, step = [], object()
    for event in events:
        if not blocks or event.get("step", step) != step:  # a stepless user event joins the current block
            blocks.append([])
            step = event.get("step", step)
        blocks[-1].append(_render(event))
    texts = ["\n".join(block) for block in blocks]
    if max_chars is None:
        return "\n".join(texts)
    kept, size = [], -1
    for text in reversed(texts):
        size += len(text) + 1
        if size > max_chars:
            break
        kept.insert(0, text)
    if not kept and texts and max_chars > 0:
        return texts[-1][-max_chars:]  # even the last step is too long: keep its end
    return "\n".join(kept)
