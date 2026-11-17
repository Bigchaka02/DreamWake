"""Probe local LM Studio models: load cost, VRAM, speed, tool use, reasoning, context limits.

Stdlib only. Raw JSON goes to runs/probe/. Examples:
    python scripts/probe_models.py                                   # all 4 candidates, ctx 4096, all parts
    python scripts/probe_models.py --models qwen/qwen3.5-9b --ctx 8192 --parts speed
    python scripts/probe_models.py --models google/gemma-4-e4b --parts embed --keep
Assumes `lms server start` has been run. Unloads everything before each load (one LLM at a time).
"""
import argparse, json, os, random, re, subprocess, time, urllib.error, urllib.request
import inspect, json.decoder, textwrap
from pydoc_data.topics import topics

LMS = os.path.expanduser(r"~\.lmstudio\bin\lms.exe")
BASE = "http://127.0.0.1:1234"  # NOT "localhost": on this Windows box it tries ::1 first and adds ~2.0 s per request
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs", "probe")
CANDIDATES = ["google/gemma-4-e4b", "qwen/qwen3.5-9b", "qwen3.8-flash-next", "qwen3.8-27b-gsq-rco@iq2_xs"]
EMBED = "text-embedding-nomic-embed-text-v1.5"
OFF = {"reasoning_effort": "none"}  # measured to disable reasoning; verified per model in the 'reasoning' part
PROSE = "\n\n".join(topics[k] for k in ("naming", "exceptions", "execmodel", "assignment", "import"))
CODE = inspect.getsource(json.decoder) + inspect.getsource(textwrap)


def lms(*args, timeout=180):
    t = time.time()
    try:
        p = subprocess.run([LMS, *args], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout)
        out, rc = p.stdout + p.stderr, p.returncode
    except subprocess.TimeoutExpired:
        out, rc = "TIMEOUT", -1
    return {"rc": rc, "secs": round(time.time() - t, 2), "out": re.sub(r"\x1b\[[0-9;?]*[A-Za-z]|Loading [^\n]*?%\s*\S", "", out)[-1500:]}


def api(path, body=None, timeout=300):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(BASE + path, data=data, method="POST" if data else "GET", headers={"Content-Type": "application/json"})
    t = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as f:
            return f.status, json.loads(f.read().decode()), round(time.time() - t, 3)
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode(errors="replace")[:2000], round(time.time() - t, 3)
    except Exception as e:  # timeouts, connection errors
        return -1, repr(e), round(time.time() - t, 3)


def chat(model, messages, max_tokens=256, temperature=0.7, extra=None, tools=None):
    body = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": temperature, **(extra or {})}
    if tools:
        body["tools"] = tools
    return api("/api/v0/chat/completions", body)  # same as /v1 but adds a `stats` block (ttft, tok/s)


def vram_mib():
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
    return int(out.strip().splitlines()[0])


def load(model, ctx):
    lms("unload", "--all")
    time.sleep(2)
    r = {"vram_before_mib": vram_mib()}
    r["load"] = lms("load", model, "--gpu", "max", "-c", str(ctx), "--parallel", "1", "-y")
    m = re.search(r"loaded successfully in ([\d.]+)s", r["load"]["out"])
    r["load_secs_reported"] = float(m.group(1)) if m else None
    r["ok"] = m is not None
    time.sleep(2)
    r["vram_after_mib"] = vram_mib()
    r["vram_model_mib"] = r["vram_after_mib"] - r["vram_before_mib"]
    ps = lms("ps", "--json")["out"]
    r["ps"] = ps[:600]
    return r


def msg_of(resp):
    return resp["choices"][0]["message"] if isinstance(resp, dict) and resp.get("choices") else {}


def summary(status, resp, wall):
    if status != 200 or not isinstance(resp, dict):
        return {"status": status, "error": str(resp)[:600], "wall": wall}
    m, u = msg_of(resp), resp.get("usage", {})
    content = m.get("content") or ""
    return {"status": status, "wall": wall, "usage": u, "stats": resp.get("stats"), "finish": resp["choices"][0].get("finish_reason"),
            "reasoning_chars": len(m.get("reasoning_content") or ""), "think_tag_in_content": "<think>" in content,
            "reasoning_tokens": (u.get("completion_tokens_details") or {}).get("reasoning_tokens"), "content_head": content[:300]}


def part_speed(model, ctx):
    chat(model, [{"role": "user", "content": "Say hi."}], 8, extra=OFF)  # warm-up
    r = {}
    gen = summary(*chat(model, [{"role": "user", "content": "Write about 200 words describing how a lighthouse works. No headings."}], 260, extra=OFF))
    r["gen_200"] = gen
    sizes = [3000] + ([int(ctx * 0.8)] if ctx > 4096 else [])
    for n in sizes:
        text = f"[{random.random()}] " + (PROSE * 4)[: n * 4]
        s = summary(*chat(model, [{"role": "user", "content": text + "\n\nReply with the single word OK."}], 120, extra=OFF))
        pt, ttft = (s.get("usage") or {}).get("prompt_tokens"), (s.get("stats") or {}).get("time_to_first_token")
        s["pp_tok_per_s"] = round(pt / ttft) if pt and ttft else None
        r[f"pp_{n}"] = s
    return r


QUESTION = "A farmer has 17 sheep. All but 9 run away. He then buys 3 times as many as remain and sells 5. How many sheep now?"
TOOLS = [{"type": "function", "function": {"name": "read_file", "description": "Read a text file and return its contents.",
          "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
         {"type": "function", "function": {"name": "write_file", "description": "Write content to a file, replacing it.",
          "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}}}]
TASK = "Read notes.txt, then write its second line, upper-cased, to out.txt. Use the tools. Reply DONE when finished."


def part_reasoning(model):
    r = {}
    variants = {"default": ({}, ""), "reasoning_effort_none": (OFF, ""), "reasoning_effort_low": ({"reasoning_effort": "low"}, ""),
                "chat_template_kwargs_off": ({"chat_template_kwargs": {"enable_thinking": False}}, ""), "slash_no_think": ({}, " /no_think")}
    for name, (extra, suffix) in variants.items():
        r[name] = summary(*chat(model, [{"role": "user", "content": QUESTION + suffix}], 2000, extra=extra))
    for name, extra in (("tool_turn1_default", {}), ("tool_turn1_off", OFF)):
        r[name] = summary(*chat(model, [{"role": "user", "content": TASK}], 2000, extra=extra, tools=TOOLS))
    return r


def tool_trial(model, extra, max_turns=6):
    line2 = f"build {random.randint(1000, 9999)} passed on {random.choice(['monday', 'friday', 'sunday'])}"
    fs = {"notes.txt": f"status report\n{line2}\nend of notes\n"}
    msgs = [{"role": "system", "content": "You are a coding agent. Act only through the provided tools."}, {"role": "user", "content": TASK}]
    t = {"turns": 0, "well_formed": True, "lat": [], "calls": [], "error": None}
    for _ in range(max_turns):
        status, resp, wall = chat(model, msgs, 1024, 0.7, extra, TOOLS)
        t["turns"] += 1
        t["lat"].append(wall)
        if status != 200:
            t["error"] = str(resp)[:300]
            break
        m = msg_of(resp)
        calls = m.get("tool_calls") or []
        if not calls:
            t["final"] = (m.get("content") or "")[:200]
            if "tool_call" in t["final"] or '"name"' in t["final"]:
                t["well_formed"] = False  # tool call leaked into text instead of being parsed
            break
        msgs.append({"role": "assistant", "content": m.get("content") or "", "tool_calls": calls})
        for c in calls:
            fn = c.get("function", {})
            try:
                args = json.loads(fn.get("arguments") or "")
                ok = fn.get("name") in ("read_file", "write_file") and isinstance(args, dict) and "path" in args
            except ValueError:
                args, ok = None, False
            t["well_formed"] &= ok
            t["calls"].append({"name": fn.get("name"), "args": fn.get("arguments"), "ok": ok})
            if not ok:
                result = "ERROR: malformed call"
            elif fn["name"] == "read_file":
                result = fs.get(args["path"].lstrip("./"), "ERROR: file not found")
            else:
                fs[args["path"].lstrip("./")] = str(args.get("content", ""))
                result = "OK"
            msgs.append({"role": "tool", "tool_call_id": c.get("id", ""), "content": result})
    t["out"] = fs.get("out.txt")
    t["correct"] = (t["out"] or "").strip() == line2.upper() and "final" in t
    return t


def part_tools(model, n=10):
    trials = [tool_trial(model, OFF) for _ in range(n)]
    lat = [x for t in trials for x in t["lat"]]
    return {"n": n, "well_formed_frac": sum(t["well_formed"] for t in trials) / n, "correct_frac": sum(t["correct"] for t in trials) / n,
            "mean_turn_latency_s": round(sum(lat) / len(lat), 3), "mean_turns": sum(t["turns"] for t in trials) / n, "trials": trials}


def part_overflow(model, ctx):
    r = {}
    filler = (PROSE * 6)[: int(ctx * 1.25 * 4)]
    text = f"START-CODE is 4417.\n{filler}\nEND-CODE is 9932.\nWhat are START-CODE and END-CODE? Write MISSING for any you cannot see."
    r["over_by_25pct"] = summary(*chat(model, [{"role": "user", "content": text}], 60, extra=OFF))
    fit = (PROSE * 3)[: int(ctx * 0.85 * 4)]  # prompt fits, but prompt + max_tokens exceeds ctx
    r["prompt_fits_gen_overflows"] = summary(*chat(model, [{"role": "user", "content": fit + "\n\nNow write a 2000-word essay on rivers."}], 2000, extra=OFF))
    return r


def part_count(model):
    r = {}
    def ptok(text):
        s, resp, _ = chat(model, [{"role": "user", "content": text}], 1, extra=OFF)
        return resp["usage"]["prompt_tokens"] if s == 200 else None
    empty = ptok("")
    r["template_overhead_tokens"] = empty
    for name, src in (("prose", PROSE[:8000]), ("python", CODE[:8000])):
        n = ptok(src)
        r[name] = {"chars": len(src), "tokens": n - empty if n and empty is not None else None}
        r[name]["chars_per_token"] = round(len(src) / r[name]["tokens"], 3) if r[name]["tokens"] else None
    a, b = ptok("Count:"), ptok("Count:" + " apple" * 200)
    r["linearity_200_x_apple"] = (b - a) if a and b else None  # expect ~200
    s = summary(*chat(model, [{"role": "user", "content": "Write a long story about a fox."}], 50, extra=OFF))
    r["max_tokens_50"] = {"completion_tokens": (s.get("usage") or {}).get("completion_tokens"), "finish": s.get("finish")}
    return r


def part_embed(model):
    r = {"load": lms("load", EMBED, "-y"), "vram_with_both_mib": vram_mib()}
    texts = [f"memory note {i}: the agent edited file_{i}.py to fix bug #{i * 7}" for i in range(20)]
    for k in ("cold", "warm"):
        s, resp, wall = api("/v1/embeddings", {"model": EMBED, "input": texts})
        r[k] = {"status": s, "wall": wall, "n": len(resp["data"]) if s == 200 else None,
                "dim": len(resp["data"][0]["embedding"]) if s == 200 else None, "err": None if s == 200 else str(resp)[:300]}
    r["llm_still_ok"] = summary(*chat(model, [{"role": "user", "content": "Say OK."}], 5, extra=OFF))["status"]
    r["unload"] = lms("unload", EMBED)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(CANDIDATES))
    ap.add_argument("--ctx", type=int, default=4096)
    ap.add_argument("--parts", default="speed,reasoning,tools,overflow,count")
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--keep", action="store_true", help="leave the last model loaded")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    parts = a.parts.split(",")
    for model in a.models.split(","):
        res = {"model": model, "ctx": a.ctx, "when": time.strftime("%Y-%m-%d %H:%M:%S"), "load": load(model, a.ctx)}
        print(model, "load:", res["load"]["load_secs_reported"], "s, vram", res["load"]["vram_model_mib"], "MiB", flush=True)
        if res["load"]["ok"]:
            for p in parts:
                t = time.time()
                try:
                    res[p] = part_speed(model, a.ctx) if p == "speed" else part_tools(model, a.trials) if p == "tools" else \
                        part_overflow(model, a.ctx) if p == "overflow" else part_embed(model) if p == "embed" else \
                        part_count(model) if p == "count" else part_reasoning(model)
                except Exception as e:  # record and continue with the next part
                    res[p] = {"exception": repr(e)}
                print(f"  {p} done in {time.time() - t:.0f}s", flush=True)
        name = re.sub(r"[^A-Za-z0-9.-]+", "_", model) + f"_ctx{a.ctx}_{'-'.join(parts)}_{time.strftime('%Y%m%d-%H%M%S')}.json"
        with open(os.path.join(OUT, name), "w", encoding="utf-8") as f:
            json.dump(res, f, indent=1)
        print("  wrote", name, flush=True)
    if not a.keep:
        lms("unload", "--all")


if __name__ == "__main__":
    main()
