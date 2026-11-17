# 1 — Project overview: goal, design, structure

This is the entry point for understanding what DreamWake is and how the
current version works. History is in `2_PROJECT_LOG.md`, the state of play and
to-do list in `3_PROJECT_STATUS_AND_TODO.md`. Evidence lives in `docs/notes/`.

## 1. Goal

Find, build and evaluate the best practical agent design achievable with a
**small, locally running language model, a limited context window, and one
desktop**. The agent must be able to put down unfinished work, lose its entire
conversation, and later resume with the right goal, constraints and next
action. The research question is whether offline "dreaming" helps it remember
what matters, learn from mistakes, and apply an earlier lesson to a new task.

The owner's motivation: their local models run with a small context window
because of hardware limits, so they trade quality for capacity. They want
better memory without giving up model size, and better answer quality
generally.

**What "optimal" means.** Successful completion of multi-session tasks under a
fixed token allowance, with the cost reported (latency, disk, offline
compute). A cheap design that resumes correctly can beat a slightly more
accurate one that needs an hour of rehearsal. A finding that a one-paragraph
handoff beats dreaming is a successful outcome.

**Milestones (from the mission).**
- *First milestone:* a fresh checkout reproduces an end-to-end comparison, and
  one success or failure across a context reset can be explained. **Reached**
  (README commands; notes 001–003).
- *First research cycle:* a recommended design for this desktop, the evidence
  for it, a condition where it breaks down, and the next experiment most likely
  to change the recommendation. **Not reached** — see file 3.

## 2. Research foundation

*Language Models Need Sleep* (arXiv 2606.03979v2). Full brief:
`docs/research/paper-brief.md`. The points that shape this project:

| Paper mechanism | Status here |
|---|---|
| Consolidate before the fast memory is overwritten | **Adapted**: sleep runs at session end on the full session log |
| New knowledge goes into a slower store; old entries pruned | **Adapted**: external rule list, merged across sessions, newer decision replaces older |
| Dreams kept only if training on them improves a held-out score (Eq. 5) | **Adapted**: a rule rewrite is kept only if a practice re-run scores better |
| Gradient-based dream selection, random-expert injection, generator training | **Not carried over**: need weight updates; effects in the paper are within noise |
| Any parametric update | **Not attempted**: installed models are GGUF, served inference-only |
| Executable scoring of practice; dreams that carry their own checks; shared-wake comparison | **Original to DreamWake** |

Every gain in the paper's Table 3 comes from weight updates, so no effect size
may be assumed for memory placed in a prompt.

## 3. Experiment design (current version)

Full protocol: `docs/design/001-experiment-protocol.md`.

**Episode = one task × one seed.**
1. **Wake (shared).** Sessions `s1` and `s2` run once, live, with no memory,
   each from an empty conversation, on one workspace. Every design gets this
   identical transcript and workspace. A wake is valid only if the hidden `s2`
   tests pass; otherwise it is redone (up to 3 retries).
2. **Sleep (per design).** After `s1` and again after `s2`, the design turns
   the recorded session into its artifact. Two sleeps are deliberate: the
   second must supersede a decision stored by the first.
3. **Reset.** Conversation discarded. Only the artifact and workspace remain.
4. **Final session (per design).** Fresh conversation, same tools and window,
   the artifact injected into the system prompt and counted against the window.
   Every design's final session runs in a directory with the same path.
5. **Evaluation.** Hidden tests run on a copy, after protected folders
   (`tests`, `tools`) are restored from the pristine snapshot and the task's
   `eval_setup` commands have run.

**Token window.** The model is loaded with context 4096, and LM Studio rejects
an over-length prompt with HTTP 400, so the window is physically enforced.
Reserve 1024 for output (`max_tokens`). Prompt budget = window − reserve. The
agent estimates tokens as chars/3.2 times a ratio calibrated from the server's
real `prompt_tokens` after each call (capped at 1.0, 10% margin). If the server
still refuses, the oldest history unit is dropped and the step retried.
History is a rolling window: system prompt and user request are pinned, the
oldest units (an assistant turn plus its tool results) go first.

**Agent scaffold** (`dreamwake/agent.py`). Five tools: `list_files`,
`read_file`, `write_file` (whole-file), `run_tests` (the task's check
command), `finish`. One step = one model call. Only the first tool call of a
reply is executed. `write_file` refuses to overwrite an existing file that was
not read in this session. A reply with no tool call gets a nudge; two in a row
end the session. Results record `ended_by` ∈ finish / max_steps /
no_tool_calls / overflow. Tool errors never expose absolute paths.

**Designs** (`dreamwake/designs.py`). All share one injected budget of 900
characters.

| Design | Sleep | Injected text |
|---|---|---|
| `workspace` | nothing | nothing |
| `handoff` | one call: previous note + the session request + the transcript tail that fits → a note of at most 120 words | the note |
| `consolidated` | reads the whole log in window-sized chunks, one extract call per chunk (rules and lessons as JSON lines), then one merge call with the current list; at most 8 items of 240 characters | `- RULE: …` / `- LESSON: …` lines |
| `replay` | consolidated, then 2 practice exercises that re-do the real request from the pre-session snapshot | same list, possibly rewritten |
| `dreaming` | consolidated, then one call invents 2 hypothetical requests from the rules, each with `must_add` / `must_not_add` strings; practised on the post-session snapshot | same list, possibly rewritten |

*Practice scoring is done by code.* An attempt = sandbox copy → agent session
with the memory under test → project check → ADDED text (lines not in the
snapshot). Score = failed check (1) + missing `must_add` + present
`must_not_add`. If any exercise scores above 0, one refine call rewrites the
rules from code-generated problem lines; both exercises are re-run; the
rewrite is accepted only if neither score got worse, the sum dropped, and no
item was lost. At most 4 practice sessions per sleep. Dream requests are never
injected; a contamination measure lists identifiers that leaked from a dream
into memory.

**Tasks** (`tasks/<id>/`). Each has the same three kinds of knowledge:

| Task | Trap (obvious fix fails) | Tool-enforced convention | User revision (silent in visible checks) |
|---|---|---|---|
| `settings_gen` (tuning) | `src/settings.py` is regenerated from `settings.tmpl.json` | every setting listed in `CHANGES.md` (audit) | durations stored in milliseconds, `_MS` suffix |
| `catalog_legacy` (held-out) | `app/messages.py` is a dead legacy module; real catalog is `locale/en.json` | catalog keys sorted (lint) | keys namespaced, UI texts under `ui.` |
| `cli_registry` (held-out) | a command exists only if listed in `cli/manifest.json` | registered functions prefixed `cmd_` (lint) | commands return `{"ok", "out"}` dicts, not strings |

Final hidden checks share five names: `test_feature_works`,
`test_honors_revision`, `test_trap_avoided`, `test_convention_followed`,
`test_no_regression`. Each task ships reference states (correct and
deliberately wrong) and validity tests proving the fixture behaves as claimed.
Held-out tasks are frozen at commit `09eb4d3`.

**Measures.** Completion (all hidden checks pass); per-check rates; repeated
the earlier mistake (wrote a trap path during the final session); steps;
`ended_by`; final tokens and seconds; sleep calls, tokens, seconds and practice
sessions; injected tokens; contamination.

## 4. Code map

| Path | Responsibility |
|---|---|
| `dreamwake/llm.py` | the only network code: `chat()` to LM Studio's OpenAI-compatible endpoint; `ContextOverflow`; token estimate |
| `dreamwake/agent.py` | tools, `run_session`, trimming, `render_events`, `render_chunks` |
| `dreamwake/designs.py` | `sleep(design, …)`, `inject(design, …)`, consolidation, practice, `contamination` |
| `dreamwake/run.py` | `run_experiment`, `summarize`, CLI (`python -m dreamwake.run`) |
| `dreamwake/tasks.py` | `load_task`, `make_workspace`, `run_check`, `evaluate`, `apply_reference` |
| `tasks/<id>/` | `task.json` (check_cmd, trap_paths, protected, eval_setup, sessions), `workspace/`, `hidden_tests/`, `reference/` |
| `tests/` | 192 tests; all model calls faked |
| `scripts/probe_models.py` | re-runnable model probe |
| `docs/` | protocol, notes, paper brief, engineering rules, run index |

`task.json` keys: `id`, `check_cmd` (list of `python …` steps), `trap_paths`,
`protected`, `eval_setup` (optional), `sessions` (`id`, `prompt`, `max_steps`
— 20 everywhere). The runner currently assumes exactly three sessions.

## 5. Environment

- Ryzen 7 9800X3D, 31 GB RAM, RTX 5080 16 GB, Windows 11, Python 3.14 venv at
  `.venv` (pytest only). Runtime code is standard library only.
- LM Studio CLI `lms` (`%USERPROFILE%\.lmstudio\bin\lms.exe`). Server at
  `http://127.0.0.1:1234` — never `localhost` (adds 2 s per request here).
- Every request sends `reasoning_effort: "none"`; nothing else disables
  reasoning on these models. Temperature 0.7. Seeds do not make the server
  reproducible.
- Models: `qwen/qwen3.5-9b` (primary; 86 tok/s), `qwen3.8-27b-gsq-rco@iq2_xs`
  (the owner's use case; 75 tok/s), `google/gemma-4-e4b` (fastest, cannot do
  the task), `qwen3.8-flash-next` (crashes on load), full `qwen/qwen3.8-27b`
  (too large for 16 GB).
- Details: `docs/notes/000-machine-inventory.md`.

## 6. Guardrails

- Held-out tasks and their outcomes never feed tuning, dream generation or
  selection. They are run once, at the end.
- Nothing is changed to make a model succeed more often without a recorded
  reason; calibration changes are listed in file 2.
- Dream content is hypothetical: only a general rule may enter memory.
- One consequential mechanism per comparison where practical; a bundled
  change is stated as bundled.
- Each note records question → result → decision changed.
- The repository is public: no personal paths or account details in tracked
  files.
- Gains are attributed to where they persist. So far: stored artifacts
  supplied as context. No parameters change.
