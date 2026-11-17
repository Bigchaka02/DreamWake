# 000 Machine inventory: local models for DreamWake

Date 2026-11-16. Machine: Ryzen 7 9800X3D, 31 GB RAM, RTX 5080 16 GB (driver 610.88, ~0.7 GB VRAM used by the desktop at idle).
LM Studio runtime `llama.cpp-win-x86_64-nvidia-cuda12-avx2@2.73.0`. Raw data: `runs/probe/*.json`, from `scripts/probe_models.py`.
**[M]** = measured here. **[I]** = inferred or from general knowledge, not measured.

## Measurements per model [M]
Every model was loaded alone with `--gpu max -c 4096 --parallel 1`. VRAM is the nvidia-smi delta after load.
Gen is a ~200-token completion (server `stats.tokens_per_second`). PP is a ~2750-token prompt: prompt_tokens / time_to_first_token.
Tools = 10 trials of read notes.txt -> write line 2 upper-cased to out.txt, at temperature 0.7, reasoning off, max 6 turns.

| model | quant | load s (warm) | VRAM | gen tok/s | PP tok/s | well-formed calls | task correct | mean s/turn | turns |
|---|---|---|---|---|---|---|---|---|---|
| google/gemma-4-e4b | Q4_K_M | 3.6 (cold first load: 11.0) | 4.44 GB | 153 | 15,400 | 10/10 | 10/10 | 0.16 | 3.0 |
| qwen/qwen3.5-9b | Q8_0 | 8.5 | 9.74 GB | 86 | 7,550 | 10/10 | 10/10 | 0.49 | 3.0 |
| qwen3.8-27b-gsq-rco@iq2_xs | IQ2_XS | 8.0 | 10.68 GB | 75 (MTP speculative, see below) | 1,900 | 10/10 | 10/10 | 0.33 | 3.0 |
| qwen3.8-flash-next | Q8_0 (arch qwen4exp) | **load fails** | - | - | - | - | - | - | - |
| qwen/qwen3.8-27b | Q4_K_M, 17.74 GB | not attempted: larger than 16 GB VRAM | | | | | | | |

- flash-next: llama-server exits with code 3221225477 (0xC0000005, access violation) before it becomes healthy. This happened twice, once with `--gpu max` and once with default offload. `--estimate-only` claims 5.0 GiB fits. Other runtimes are installed but I did not switch them, because that is a persistent setting.
- The tool task is too easy to separate the models: all three scored 30/30 and always used exactly read -> write -> "DONE". It proves the plumbing works, not that the tool use is robust.
- The 27B iq2_xs build turns on MTP speculative decoding by default. `usage` then gains `*_draft_tokens_count` fields; about 32% of drafts were accepted on prose.
- `lms load --estimate-only` is not a good guide: for gemma it estimated 6.29 GiB and the measured load was 4.44 GB.

## Q3 Reasoning tokens [M]
- All three models reason by default. The reasoning comes back in `message.reasoning_content` and is counted in `usage.completion_tokens_details.reasoning_tokens`. `<think>` never showed up in `content`.
- Cost with reasoning left on, on an arithmetic word problem (reasoning tokens): gemma 261, 27B 264, 9B 841.
- Cost on tool-turn 1 with reasoning on: gemma 137, 9B 57, 27B 33 reasoning tokens. With it off, that turn takes 17-26 tokens in total.
- **What reliably turns it off:** `"reasoning_effort": "none"` on `/v1/chat/completions` gave 0 reasoning tokens on all three models in every call, including all 30 tool trials. The native `/api/v1/chat` with `"reasoning": "off"` also worked (tested on gemma only).
- **What did NOT work:** `chat_template_kwargs.enable_thinking=false`, `/no_think` in the prompt, `"reasoning": "off"` on the OpenAI endpoint, and `reasoning_effort: "low"`, which caps nothing: on qwen3.5-9b it ran 1,921 reasoning tokens into the 2,000 max_tokens limit and returned an empty answer. `/no_think` and the template kwarg did the same on that model.
- `/api/v1/models` lists `capabilities.reasoning.allowed_options: ["off","on"], default "on"` for every model. I found no budget or cap, only on or off.

## Q4 Context enforcement and token counting [M]
- **Prompt longer than the context (4,634 tokens at ctx 4096):** HTTP 400 `exceed_context_size_error`. The body includes `n_prompt_tokens` and `n_ctx`. There is no truncation and no rolling window. All three models behaved the same.
- **Prompt fits but prompt + max_tokens does not:** generation stops when the total reaches exactly 4096, with `finish_reason: "length"` (`stats.stop_reason: contextLengthReached`). The client has to cap max_tokens.
- **`usage` is accurate:** 200 repetitions of " apple" added exactly 200 prompt_tokens, and `max_tokens: 50` returned completion_tokens = 50.
- **Template overhead:** an empty user message costs 9 tokens (gemma) or 12 (qwen). Tools add about 100 (gemma) or about 320 (qwen) prompt tokens.
- **No tokenize endpoint:** I probed `/api/v0/tokenize`, `/v1/tokenize`, `/api/v1/tokenize` and `/v1/count_tokens`; all were rejected. `max_tokens: 0` is rejected (the minimum is 1).
- **Cheapest exact count:** send the message with `max_tokens: 1` and read `usage.prompt_tokens`. That takes about 0.2 s for 2.7k tokens on gemma. An over-length request also reports `n_prompt_tokens` without generating anything.
- [I] The `lmstudio` Python/JS SDK has tokenize/countTokens, but it is not installed and is not stdlib.
- **Characters per token [M]:** English prose (pydoc text) 4.48 on all three. Python code (stdlib json.decoder + textwrap) 3.42 on gemma, 3.63 on both qwens. Conservative estimate: tokens ≈ chars/4.0 for prose, chars/3.2 for code.
- [M] With the default `--parallel 4`, gemma at ctx 4096 still accepted a 3,241-token prompt, so the context is not split per slot. Even so, we load with `--parallel 1` to keep it simple.

## Q5 Larger windows, gemma-4-e4b [M]

| ctx | VRAM | gen tok/s | PP tok/s |
|---|---|---|---|
| 4096 | 4.44 GB | 153 | 15.4k (2.7k-token prompt) |
| 8192 | 4.51 GB | 151 | 16.8k (6.0k-token prompt) |
| 16384 | 4.65 GB | 153 | 16.8k (11.8k-token prompt, 0.70 s) |

All three fit fully on the GPU, and the KV cache grows very little. I did not measure generation speed at a near-full context.

## Q6 Embeddings [M]
- Nomic embed (80 MiB) loaded in 1.0 s next to gemma (total 5.6 GB VRAM). `POST /v1/embeddings` with 20 short strings: 20 vectors of dim 768 in 0.105 s cold, 0.059 s warm. The LLM kept answering normally. Unloaded again after the test.

## Q7 Training feasibility (report only, nothing installed)
- [I] The GGUF files are quantized (2-8 bit) llama.cpp weights served by LM Studio. For our purposes they are **inference-only**: PyTorch/PEFT tooling does not train on them.
- [M] `pip index versions torch` from the venv (Python 3.14.2, win_amd64): PyPI up to 2.15.0 ([I] CPU-only on Windows); `download.pytorch.org/whl/cu128` 2.9.0-2.11.0; `.../whl/cu130` 2.9.0-2.14.1. So cp314 Windows CUDA wheels do exist. [I] An RTX 50-series card (sm_120) needs a CUDA ≥ 12.8 build, so use cu128 or cu130 (about 2-3 GB per wheel).
- [I] LoRA training would also need: a separate download of the base model's non-GGUF safetensors weights (about 8-16 GB in bf16 for a 4-9B model); transformers + peft + accelerate (+ bitsandbytes for QLoRA so a 9B fits in 16 GB); then a llama.cpp conversion (adapter-to-GGUF, or merge and re-quantize) to serve it back through LM Studio. All of that needs the manager's approval under the engineering rules.

## Other findings [M]
- **On Windows, `http://localhost:1234` costs +2.0 s on every request** from Python urllib (it tries IPv6 ::1 first), against 0.002 s for `http://127.0.0.1:1234`. Always use 127.0.0.1.
- `/api/v0/chat/completions` is the OpenAI-shaped endpoint plus `stats` (ttft, tok/s, stop_reason). `/v1/...` returns an empty `stats`.
- [I] LM Studio may JIT-load any model named in a request, so always check `lms ps` first.

## Decision
**(a) Model.**
- Primary: **google/gemma-4-e4b** (Q4_K_M). It is the smallest and fastest (4.4 GB, 153 tok/s, 15k tok/s PP, 0.16 s/turn), with 10/10 tool use and a reasoning switch that reliably turns off.
- Runner-up: **qwen3.8-27b-gsq-rco@iq2_xs** (10.7 GB, 75 tok/s, 10/10).
  - [I] It is likely the stronger coder; use it if gemma floors on harder tasks.
  - qwen3.5-9b is a close third. Its reasoning runs away unless `reasoning_effort: none` is set.
- [I] The tool test did not separate the models. Re-check the primary on the first real DreamWake task before trusting it.

**(b) Window.**
- Load at **4096**. That is the project's premise, and the server enforces it hard with a 400 error rather than silently truncating.
- Reserve **1024 tokens for output** (`max_tokens: 1024`), which leaves a prompt budget of ≤ 3072. That budget includes the tool schemas (about 100 tokens for gemma).
- When estimating, budget by chars/3.2 and verify against `usage.prompt_tokens`.
- [M] 8192 costs nothing measurable here (+0.07 GB, same speed), so it can be a second experimental condition.

**(c) Commands.**
`lms` is LM Studio's CLI (`%USERPROFILE%\.lmstudio\bin\lms.exe` if it is not on PATH).
```
lms server start --port 1234
lms unload --all
lms load google/gemma-4-e4b --gpu max -c 4096 --parallel 1 -y
```

**(d) Request settings.**
- Endpoint: `POST http://127.0.0.1:1234/v1/chat/completions` (or `/api/v0/...` to get timing stats).
- Body: `{"model": "google/gemma-4-e4b", "reasoning_effort": "none", "temperature": 0.7, "max_tokens": 1024, "tools": [...]}`.
- Treat HTTP 400 `exceed_context_size_error` and `finish_reason: "length"` as explicit outcomes.

State left behind: the server is running on port 1234 with google/gemma-4-e4b loaded at ctx 4096, parallel 1. Nothing else is loaded.
