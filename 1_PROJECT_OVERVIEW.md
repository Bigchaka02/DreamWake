# 1 — Project overview: goal, design, structure

This is the entry point for understanding DreamWake and its current design.
History is in `2_PROJECT_LOG.md`; results and open work are in
`3_PROJECT_STATUS_AND_TODO.md`. Evidence lives in `docs/notes/`.
Updated 2026-11-18 for work through 2026-11-17, commit `f630b95`.

## 1. Goal

Find, build and evaluate the best practical agent design achievable with a
**small, locally running language model, a limited context window, and one
desktop**. The agent must put down unfinished work, lose its entire
conversation, and resume with the right goal, constraints and next action.
The research question is whether offline "dreaming" helps it remember what
matters, learn from mistakes, and apply an earlier lesson to a new task.

The owner's local models use small context windows because of hardware
limits. They want better memory without giving up model size, and better
answer quality generally.

**What "optimal" means.** Successful completion of multi-session tasks under
a fixed token allowance, with latency, disk and offline compute reported.
A cheap design that resumes correctly can beat a more expensive one.
A handoff beating dreaming is a valid result.

**Milestones.** The first milestone is reached: a fresh checkout can run an
end-to-end comparison, and a result across a reset can be explained. There
is also a live resumable demo. The first research cycle remains open:
the recommendation is provisional, the five-session comparison incomplete,
and the held-out tasks unrun (file 3).

## 2. Research foundation

*Language Models Need Sleep* (arXiv 2606.03979v2) is the starting point.
See `docs/research/paper-brief.md`. Its gains come from weight updates;
no effect size may be assumed for memory placed in a prompt.

| Paper mechanism | Status here |
|---|---|
| Consolidate before fast memory is overwritten | adapted: sleep reads the full session log |
| Slower memory; old entries pruned | adapted: external rules merged across sessions; newer decisions supersede older ones |
| Keep dreams only if they improve a held-out score | adapted: keep one rule edit only after executable practice confirms improvement |
| Gradient-based selection, expert injection, generator training | not carried over |
| Parameter updates | not attempted; installed GGUF models are served inference-only |
| Code-scored practice, grounded dream checks, shared wakes | DreamWake mechanisms |

`docs/research/literature-review.md` condenses three research reports, with
verification labels and the decisions they informed: separate content,
placement and checking; preserve the user's words; constrain and verify
exercises; edit one rule at a time; pair final repeats per seed. The model
choice is the owner's decision, not a literature finding.

## 3. Experiment design (current version)

Protocol: `docs/design/001-experiment-protocol.md`, including its amendments.

**Episode = one task × one seed.**

1. **Wake (shared).** Every task session before the last runs live, without
   memory, from an empty conversation on one workspace. All designs receive
   the same logs and snapshots. An optional `stage` chooses hidden tests;
   null is unscored. Wake validity is the last scored wake session passing,
   with up to 3 retries.
2. **Sleep (per design).** After each wake, the design builds its artifact.
   The user's requests are archived verbatim and passed to every sleep.
3. **Reset.** Conversation discarded; only workspace and artifact survive.
4. **Final repeats (per design).** `--final-repeats N` runs the last session
   N times from the same memory and workspace snapshot. Repeat seeds are
   shared across designs. Repeats do not create independent seed pairs.
   Every final uses the same workspace path.
5. **Evaluation.** Hidden tests run on a copy after protected folders
   (`tests`, `tools`) are restored and task `eval_setup` commands run.

**Token window.** LM Studio is loaded at context 4096 and rejects oversized
prompts with HTTP 400. Reserve 1024 tokens for output; prompt budget is
window minus reserve. The estimate is chars/3.2, calibrated from real
`prompt_tokens` (ratio capped at 1.0, 10% margin). On overflow the oldest
history unit is dropped and the step retried. System prompt and request are
pinned; an assistant turn and its tool results are trimmed together.

**Agent scaffold** (`dreamwake/agent.py`). Five tools: `list_files`,
`read_file`, `write_file` (whole-file), `run_tests` (task check command),
`finish`. One step is one model call. Only the first tool call in a reply
runs; the result explicitly reports ignored calls, and rows record
`ignored_calls`. Existing files must be read before overwrite. A reply
without a tool call gets a nudge; two in a row end the session. Endings are
`finish`, `max_steps`, `no_tool_calls`, `overflow`, or `interrupted`.
Ctrl+C ends a session cleanly. Tool errors omit absolute paths.
`run_session(on_event=...)` supports the demo's live display.

**Design grammar.** `store[@placement][+gate]`. Default placement is `system`.
`@request` puts memory after the request. `@tail` rebuilds a trailing user
message on every call; it is never trimmed. `+gate` answers the first
`finish` with the notes as a checklist. It is a model check, not a validator.
`workspace` accepts neither placement nor gate. The old `nearby` name was
removed; use `consolidated@request`.

**Stores** (`dreamwake/designs.py`). All injected memory shares a cap of
900 characters, including hybrid's state line.

| Store | Offline work | Injected text |
|---|---|---|
| `workspace` | none | none; project files only |
| `handoff` | one call with previous note, request and transcript tail; asks for at most 120 words | prose note |
| `consolidated` | extract rules and lessons from full log chunks; merge with current list | at most 8 items of 240 characters, within the shared cap |
| `requests` | retain requests without a model call | verbatim requests; newest prioritised within the cap |
| `oracle` | no sleep; load `tasks/<id>/oracle_memory.txt` | hand-written ideal diagnostic rules |
| `review` | consolidate; one schema call re-reads the request archive and may edit one rule | reviewed rule list |
| `hybrid` | consolidate; code adds last request and completion status | rules plus `LAST SESSION` line |
| `replay` | consolidate; practise real past request 5 times | rules, possibly edited |
| `dreaming` | consolidate; generate, verify and practise new exercises | rules, possibly edited |

Hybrid uses `- LAST SESSION (finished): <request>` with the request cut to
160 characters, or the prefix
`- LAST SESSION (NOT finished - continue this work): ` with the request cut
to 400 characters. It needs no extra model call. Rules are removed from the
end if needed to fit the combined cap.

**Dreaming, second version.**

1. One schema-enforced call sees requests verbatim. It proposes up to
   4 exercises: exact decision quote, plain-words request, `must_add` copied
   from the decision, and `must_not_add`.
2. Code requires a whitespace-normalised decision substring of an archived
   request, at least 20 characters. `must_add` must occur in the decision.
   Numeric-only strings are excluded. Requests cannot contain check strings
   or name an identifier or file.
3. A positive-control agent receives only `- USER DECISION: <quote>` after
   the request, without a gate. It must score 0. At most 3 controls run;
   stop at 2 verified dreams. No verified dream means no practice and no
   fallback to real requests.
4. First attempts use stored rules at the design's placement and gate.
   If all score 0, memory stays unchanged.
5. Otherwise one schema-enforced repair call proposes one rule edit.
   `_edit` refuses an edit that drops identifiers or does not fit. The
   failed exercise is retried twice and the other once. Both failed-exercise
   retries must score lower; the other must not score higher.
6. Protected folders are restored from the snapshot before scoring.

Dreams run on the post-session snapshot. Every dreamed request has the
suffix (including its initial space):

```text
 This is a new, separate feature: add it even if something similar already exists.
```

Replay uses the pre-session snapshot, real request and same repair machinery,
without dreams or controls.

**Practice score.** Failed project check (1), plus missing `must_add` strings,
plus present `must_not_add` strings in text added relative to the snapshot.
Lower is better. Logs preserve `items_before`, candidate edits and exercise
missing/present strings. Leading `RULE:`/`LESSON:` labels are cleaned. Sleep
seeds use `SLEEP_OFFSET=50_000`, `SLEEP_STRIDE=1_000`. Contamination lists
dream identifiers reaching memory without appearing in the real experience.
No model parameters change.

**Tasks.** Tasks can inherit checks, trap paths, protected folders, evaluator
setup, workspace, reference states, merged hidden tests and oracle memory
from a base. Child reference states and parents are looked up before the base.

| Task | Sessions and purpose |
|---|---|
| `settings_gen` | two wakes; generated-file trap, CHANGES audit, silent milliseconds policy; final adds backoff |
| `settings_resume` | inherits `settings_gen`; second wake cut off at 6 steps, unscored; final says only to continue |
| `settings_long` | inherits `settings_gen`; five wakes, then the same backoff final |
| `catalog_legacy` | held-out: dead legacy module, sorted catalog keys, `ui.` namespace policy |
| `cli_registry` | held-out: manifest registration, `cmd_` convention, dictionary return policy |

`settings_resume` validates the wake with s1 and scores the final with s2
hidden tests. `settings_long`: s1 timeout in seconds; s2 milliseconds policy;
s3 `client.is_secure()`; s4 `SERVICE_NAME` and a "doc" field policy for new
settings; s5 `retry.describe()`. Its s5 checks cumulative state. Its final
adds `test_honors_second_policy`; that check can pass by copying a neighbour
and does not measure memory.

Usual hidden checks cover feature, revision, trap, convention and regression.
The interrupted task checks feature, revision and audit. Held-out tasks
remain frozen at `09eb4d3` and have never run.

**Measures.** Completion, per-check outcomes, all repeats passing per seed,
paired seed wins/ties/losses, repeated trap writes, steps, ending, ignored
calls, final tokens/time, sleep calls/tokens/time, practice sessions, injected
tokens and contamination. Transport and evaluator faults are separate.
LM Studio is not reproducible by seed.

## 4. Code map

| Path | Responsibility |
|---|---|
| `dreamwake/llm.py` | LM Studio chat, schema response format, overflow, token estimate |
| `dreamwake/agent.py` | tools, session loop, trimming, event callback and log rendering |
| `dreamwake/designs.py` | stores, consolidation, review, practice, injection, contamination |
| `dreamwake/run.py` | design parser, shared wakes, final repeats, rows, summary, CLI |
| `dreamwake/demo.py` | interruptible work, saved state, empty-conversation continuation |
| `dreamwake/tasks.py` | inheritance, workspace, checks, evaluation, references |
| `tasks/<id>/` | metadata, optional base/stages, workspace, hidden tests, references, oracle |
| `tests/` | fake-model tests; 557 passed at `f630b95` |
| `scripts/probe_models.py` | model probe |
| `docs/` | protocol, notes, research, engineering rules, run index |

Run folders and research reports under `runs/` are local, not versioned.

## 5. Environment

- Ryzen 7 9800X3D, 31 GB RAM, RTX 5080 16 GB, Windows 11; Python 3.14
  venv at `.venv`. Runtime code is standard library only.
- Primary and only live experiment model: Qwen 3.8 27B IQ2_XS,
  `qwen3.8-27b-gsq-rco@iq2_xs`, context 4096, `--parallel 1`.
- The owner chose Qwen 3.8 only on 2026-11-16. Qwen 3.5 9B and Gemma
  results remain history; neither has a planned run.
- LM Studio CLI: `%USERPROFILE%\.lmstudio\bin\lms.exe`; endpoint
  `http://127.0.0.1:1234`. Use `127.0.0.1`, not `localhost`.
- Requests use `reasoning_effort: "none"` and temperature 0.7.
- Earlier machine and model probes: `docs/notes/000-machine-inventory.md`.

## 6. Guardrails

- Held-out outcomes never feed tuning, dream generation or selection.
  Run once at the end; do not alter an unachievable held-out task.
- Do not change tasks, checks, limits or prompts to raise success without
  recording why. Version a replacement task separately.
- Dreams are hypothetical. Only a general rule may enter memory.
- Change one consequential mechanism per comparison where practical;
  describe bundled changes as bundled.
- Notes record question, result and decision changed.
- Documentation, including experiment notes, changes only on the owner's
  explicit say-so. This update was authorised on 2026-11-18.
- The repository is public. No personal paths or account details in files.
- Gains live in stored artifacts supplied as context, plus the ignored-call
  notice in the harness. No parameters change.
