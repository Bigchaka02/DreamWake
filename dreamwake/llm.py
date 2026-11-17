"""The only module that talks to the network: one call to an OpenAI-compatible chat endpoint."""
import json
import os
import time
import urllib.error
import urllib.request

# 127.0.0.1, not localhost: on this machine urllib tries IPv6 first for "localhost" (~2 s per request).
DEFAULT_BASE_URL = "http://127.0.0.1:1234/v1"
CHARS_PER_TOKEN = 3.2  # measured 3.4-3.6 for code, 4.5 for prose (docs/notes/000-machine-inventory.md); kept low


class ContextOverflow(RuntimeError):
    """The server refused the prompt as longer than the loaded context."""


def estimate_tokens(text):
    return int(len(text) / CHARS_PER_TOKEN) + 1


def _parse_arguments(raw):
    if not raw.strip():
        return {}  # some servers send "" for a call without arguments
    try:
        value = json.loads(raw)
    except ValueError:
        return None
    return value if isinstance(value, dict) else None


def chat(messages, tools=None, *, max_tokens, temperature=0.7, seed=None, response_format=None):
    model = os.environ.get("DW_MODEL")
    if not model:
        raise RuntimeError("set DW_MODEL to the model id served by the endpoint")
    base_url = os.environ.get("DW_BASE_URL", DEFAULT_BASE_URL).rstrip("/")
    body = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": temperature,
            "reasoning_effort": "none"}  # the only switch that turns reasoning off on the probed models
    if tools:
        body["tools"] = tools
    if seed is not None:
        body["seed"] = seed
    if response_format is not None:
        body["response_format"] = response_format
    request = urllib.request.Request(base_url + "/chat/completions", data=json.dumps(body).encode("utf-8"),
                                     headers={"Content-Type": "application/json"})
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", "replace")
        kind = ContextOverflow if error.code == 400 and "exceed_context_size_error" in detail else RuntimeError
        raise kind(f"HTTP {error.code} from {base_url}: {detail}") from error
    seconds = time.perf_counter() - start
    choice = data["choices"][0]
    message = choice["message"]
    calls = []
    for call in message.get("tool_calls") or []:
        raw = call["function"].get("arguments") or ""
        calls.append({"name": call["function"]["name"], "arguments": _parse_arguments(raw), "raw": raw})
    usage = data.get("usage") or {}
    return {"content": message.get("content") or "", "tool_calls": calls,
            "usage": {"prompt_tokens": usage.get("prompt_tokens", 0),
                      "completion_tokens": usage.get("completion_tokens", 0)},
            "seconds": seconds, "finish_reason": choice.get("finish_reason") or ""}
