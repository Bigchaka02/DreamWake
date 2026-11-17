# 001 Baseline pilot: gemma-4-e4b never produces a valid wake on settings_gen

## Question
Can the plain agent on google/gemma-4-e4b at W=4096 complete the shared wake (s1, s2), so that `workspace` and `handoff` can be compared on the final session?

## Setup
- Model google/gemma-4-e4b (Q4_K_M) in LM Studio, ctx 4096, parallel 1, `reasoning_effort: none`, temperature 0.7. Window 4096, reserve 1024, wake_retries 3 (4 attempts per seed).
- Commit 347af6a plus the uncommitted fixes listed below (the agent.py replay fix and the designs.py handoff fix).
- `python -m dreamwake.run --task settings_gen --seeds 6 --designs workspace,handoff --window 4096 --reserve 1024 --out runs/pilot-gemma-4096`, and a 1-seed smoke run to `runs/smoke`.
- Early stop: seeds 0-2 all invalid, so the run was stopped after seed 2. **No final session ran**, so the summary table only shows `n valid 0/3` for both designs.

## Result: 0 valid wakes out of 16 attempts (pilot 12, smoke 4)
Hidden tests were scored for every attempt by replaying the logged `write_file` calls onto a fresh workspace. The last attempt per seed matches `results.jsonl`.

| where s1 ended | pilot | smoke |
|---|---|---|
| never left the trap; hard-coded a 30.0 fallback in `src/client.py`, so visible checks pass and the hidden test fails (no TIMEOUT_S) | 9 | 1 |
| never left the trap; finished with the checks failing | 0 | 1 |
| found and edited `settings.tmpl.json`, then failed the CHANGES.md audit | 3 | 2 |
| **s1 hidden pass / s2 hidden pass** | **0/12 / 0/12** | **0/4 / 0/4** |

- Every attempt wrote `src/settings.py` (16/16). Every s1 saw it undone: the next check run showed `AttributeError: ... no attribute 'TIMEOUT_S'` (16/16).
- No attempt ever read `tools/gen.py`, `tools/audit.py` or README.md (0/32 sessions).
- All 5 template edits wrote `"TIMEOUT_S": {"type": "str", "value": "30"}`.
- After an audit failure (10 sessions), the model typically re-ran the checks with no change, 2-3 times in a row. In both pilot s1 cases that reached CHANGES.md, it overwrote the whole file without reading it first, which deleted the RETRIES/BASE_URL entries.
- s2 inherits the broken s1 workspace: 9/12 pilot s2 sessions repeated the fallback pattern (`TIMEOUT_MS` in `src/settings.py`, `/1000` or fallback in client.py) and never touched the template.
- Steps used (pilot): s1 mean 15.2 of 20 (2 ran out), s2 mean 11.5 of 15 (2 ran out). Smoke: 13.2 and 14.2.

Excerpt 1, `runs/pilot-gemma-4096/seed1/wake_s1.attempt3.jsonl`. At step 10 the model says "This strongly suggests that `tools/gen.py` is running and overwriting `src/settings.py` from a template". At steps 11-12 it reads client.py and adds `except AttributeError:` returning 30.0. At step 14 it calls `finish`: "satisfying the test requirement regardless of how `tools/gen.py` populates" the settings.

Excerpt 2, `runs/pilot-gemma-4096/seed0/wake_s1.attempt3.jsonl`. At step 11 (nothing trimmed yet) it says "`tools/gen.py` might be overwriting my manual changes". At step 13 it writes the template with `"type": "str"`. Step 14 gives `audit: FAIL ... TIMEOUT_S`, and steps 15-16 re-run the checks unchanged. At step 17 it overwrites CHANGES.md with one entry, and step 18 gives `audit: FAIL ... RETRIES, BASE_URL`. It runs out at step 20.

## Judgment: model-skill limit, with two harness/task factors to weigh
- Mainly a model-skill limit. The evidence for "my edit was undone" is in plain view: the first line of every check run is `gen: regenerated src/settings.py from settings.tmpl.json`, and `list_files` shows `tools/gen.py` and `settings.tmpl.json` at step 1. The model names gen.py or the template as the likely cause in 16 of 16 s1 traces, usually with nothing trimmed (dropped=0). Its next action is still a re-read of client.py or another `settings.py` write, never a read of gen.py. When it reaches the audit, it cannot read and extend a file before overwriting it.
- Trimming is not the cause: the diagnosis moments mostly happen before any unit is dropped.
- Factor 1 (a hidden-test strictness question for the manager): hidden s1 requires `settings.TIMEOUT_S == 30` and hidden s2 requires `== 30000`, both ints, but neither prompt states a type. All 5 template edits used `str`, and `gen.py` accepts that. Fixing the audit alone would not have made s1 pass.
- Factor 2: `write_file` is whole-file overwrite only, so appending to CHANGES.md takes a read first. That is not unfair, but it made both pilot s1 audit recoveries destructive.

## Token estimate vs real
Over 320 pilot steps, real `prompt_tokens` ≈ 0.71 × the chars/3.2 estimate (0.66-0.79). The largest real prompt was 2362, so 2362 + 1024 = 3386, under 4096, with 0 overflows. Trimming therefore starts at about 2100 real tokens (73 trimmed steps, mean 2101), and about 28% of the 3072 prompt budget is unused. Handoff sleep prompts measured 0.92-0.96 × the estimate (they are prose). Same-seed runs were identical for 11 steps and then diverged: seeds do not make the server reproducible.

## What this suggests next (manager decides; nothing changed)
- Re-run the wake with the runner-up model (qwen3.8-27b iq2_xs) before changing the task. Gemma fails before any memory design matters.
- Decide whether the hidden s1/s2 checks should accept a numeric string, or the prompts should state the type. All 5 template edits used `str`.
- Calibrate the trimming estimate (for example /2.3 for JSON-encoded messages, or feedback from `usage.prompt_tokens`). The current estimate wastes about 28% of W.

## Integration fixes made
1. `dreamwake/agent.py`: a replayed tool call now always carries a JSON-object `arguments` (`json.dumps` of the recorded args, `{"raw": ...}` when unparseable). Reason: on the real server, `""` or invalid JSON in a replayed call gave HTTP 500 (which would kill the run), and a non-object silently garbled the prompt. Regression test: `test_replayed_tool_calls_always_carry_a_json_object` (red before, green after).
2. `dreamwake/designs.py` (item B): the handoff prompt is now previous note, then the session request in full, then the transcript tail that fits. Test: `test_handoff_keeps_the_session_request_in_full_between_the_note_and_the_tail`. It was verified live on smoke events: 2197 and 2777 prompt tokens.
- Checked live and fine without changes: tool_call ids, an empty assistant content, `content: None`, unicode in results, and a cut-off tool call (it returns empty content, which the agent handles as a nudge).
