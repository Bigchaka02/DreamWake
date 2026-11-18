# 2 — Project log

Concise record of work on 2026-11-16 to 2026-11-17. Updated 2026-11-18.
Results are summarised here; the evidence is in `docs/notes/` and the run
registry in `docs/runs-index.md`.

## Commits

| Commit | What |
|---|---|
| `6c7e3db` | initial commit (README stub from GitHub) |
| `0a6d461` | experiment protocol, engineering rules, paper brief, `settings_gen` task fixture + evaluator |
| `6f6d655` | LF line endings |
| `347af6a` | machine inventory note, model probe script, agent loop, handoff baseline, experiment runner |
| `26ba3d5` | note 001 (Gemma pilot); tool-call replay fix; handoff keeps the full request |
| `d6f198f` | harness hardening after independent review + calibration from the live pilot |
| `09eb4d3` | two held-out tasks; evaluator setup step becomes per-task (`eval_setup`). **Held-out tasks frozen here** |
| `f100e14` | `consolidated` design |
| `725073c` | `dreaming` and `replay` designs, first version (LLM reviewer) |
| `47bddd3` | practice sessions get 20 steps; shorter `(replaces: …)` instruction |
| `1341a85` | note 002 |
| `0b414e5` | practice reworked: executable scoring, rule-targeted dreams, tail cleaner |
| `3096429` | README with status and commands; note 003 |
| `0c5c7d1` | practice logs retain prior items, candidate and missing/present strings |
| `664df72` | final repeats with shared seeds; temporary `nearby` design |
| `644f488` | literature review from three research reports |
| `8818724` | store/placement/gate grammar; oracle and requests diagnostics; `nearby` removed |
| `685ec11` | task-defined sessions, optional stages, inheritance; `settings_resume` |
| `3faf932` | `settings_long`; child/base reference lookup |
| `29a963b` | request archive, schema output, single-rule edits, sleep seed spacing; review store |
| `6c455a3` | ignored-tool-call notice and rows; hybrid last-request/status line |
| `82c8dda` | dreaming second version: grounded exercises, controls, confirmed single-rule repair |
| `aea50a8` | dreams ask for uncovered work; suffix requires a new separate feature |
| `f630b95` | resumable demo, event callback, clean Ctrl+C interruption |

## Phase 0 — setup and inventory

- Resolved paths: Desktop folder was empty; cloned the owner's public GitHub
  repository into it. No `gh` CLI; plain git with the stored credential.
- Identified the two method repositories: `obra/superpowers` and
  `DietrichGebert/ponytail`; distilled into `docs/engineering-rules.md`.
- Machine and model probe (note 000): three models load and call tools;
  `qwen3.8-flash-next` crashes on load; over-length prompts get HTTP 400;
  reasoning needs `reasoning_effort: "none"`; `localhost` costs 2 s per call.
- Paper brief: confirmed 46.2% vs 36.2%; dreaming's value comes from keeping
  only dreams that measurably help; all gains are parametric.

## Phase 1 — harness and first live runs

- Task fixture `settings_gen` with hidden tests, reference states and
  validity tests. Agent loop, token trimming, handoff design and runner built
  and tested against a fake model.
- **Gemma 4 E4B pilot (note 001): 0 valid wakes in 16 attempts.** It names the
  generator as the likely cause in every trace, never reads it, and usually
  hard-codes a fallback so visible checks pass. Dropped as experiment model.
- Qwen wake check (12-step sessions, pre-fix code): both Qwen models complete
  the wake; every final session hit the step cap; the 27B reads about ten
  files before writing.
- Independent code review: no critical issue; eight important ones, all
  fixed in `d6f198f`.

### Fixes and calibration made in `d6f198f` (and why)

- Evaluator restores `tests` and `tools` from the pristine snapshot (an agent
  could have rewritten the audit script).
- Tool errors and check output no longer contain absolute paths; all final
  sessions run in the same directory (the path used to name the design).
- Trap detection normalises paths and ignores failed writes.
- Rows written per completed seed; summary pairs seeds across designs;
  evaluator and transport faults recorded instead of scored as failures.
- Hidden checks compare numerically (the prompts never state a type).
- `write_file` refuses blind overwrites (the model wiped `CHANGES.md` twice).
- Token estimate calibrated from real usage (28% of the window was wasted).
- Every session gets 20 steps (12 measured reading speed, not memory).
- Earlier, in `26ba3d5`: replayed tool calls always carry JSON-object
  arguments (the server returned HTTP 500 otherwise).

## Phase 2 — baseline and consolidated memory (note 002)

Qwen 3.5 9B, window 4096, 20 steps. Completion:

| Run | workspace | handoff | consolidated |
|---|---|---|---|
| A (`d6f198f`, 8 seeds) | 0/8 | 1/8 | — |
| B (`f100e14`, 8 seeds) | 0/8 | 4/8 | 4/8 |

- The only failing check anywhere is the silent revision.
- Repeated mistake in run B: 7/8, 2/8, 0/8. Steps 17.0, 14.0, 15.5.
- All consolidated failures had the rule in memory. The problem is applying
  it, not recalling it.
- Handoff moved 1/8 → 4/8 between identical configurations: 8 seeds do not
  separate designs.

## Phase 3 — larger model and practice designs (note 003)

- **Qwen 3.8 27B IQ2_XS (run C, `47bddd3`, 6 seeds): 0/6, 5/6, 6/6.** The
  application gap is mostly a 9B weakness.
- First practice design, one-seed live smoke: replay scored everything
  perfect; dreams named the setting and invented policy reversals; the LLM
  reviewer contradicted itself; no rewrite accepted. Dreaming = consolidated
  plus cost.
- Reworked in `0b414e5`: code-scored attempts, dreams written from the rules
  with their own `must_add` / `must_not_add` strings, cross-exercise accept
  rule, `(replaces: …)` tail cleaner. **At that point: 192 tests passed;
  not yet run live.**

## Housekeeping at the first stopping point (2026-11-16)

- LM Studio model unloaded and server stopped (as found).
- Early run folders (probe, smoke, Gemma pilot, both wake checks, practice
  smoke: 641 files) zipped to `runs/archive/early_runs_2026-11-16.zip`
  (local, not versioned) after verifying the file count; the frozen git
  worktree `runs/wt-frozen` removed.
- ChatGPT command-line route verified (see file 4).
- One usage-limit interruption mid-session; three subagents resumed with no
  work lost.

## Phase 4 — research (2026-11-16)

- The owner requested extensive research on dreaming and harness design.
  Three ChatGPT 6.1 live-search jobs covered practice, rule application and
  evaluation. An Opus subagent condensed them into the literature review
  (`644f488`); eight sources' numbers were checked against page text.
- Research reports remain in `runs/research-2026-11-16_compact_protected/` (local, not
  versioned). The review records which findings were verified.
- The owner chose Qwen 3.8 27B IQ2_XS only for subsequent live experiments.
  The 9B and Gemma results remain history; no further runs are planned.

## Phase 5 — diagnostic and placement (note 004)

- Last 9B checks: reworked practice still changed no rule. The one-seed
  nearby check suggested testing placement; it was not a rate estimate.
- Added final repeats, then separated store, placement and finish gate.
  `nearby` became `consolidated@request`.
- Qwen 3.8 short diagnostic (`8818724`, 8 seeds × 3): ideal rules and raw
  requests 24/24, consolidated/request/gate 23/24 each, tail 18/24,
  handoff 15/24. Our reading: the short task is saturated except for handoff.
- Trace reading found prose rules present in failures, tail write loops,
  a rubber-stamped gate, and silently ignored extra tool calls.

## Phase 6 — longer histories and interruption (note 005)

- Generalised sessions, stages and task inheritance. Built `settings_resume`
  and `settings_long`; added request review and then hybrid's code-built
  last-request/status line. Added an explicit ignored-call notice.
- Interrupted task (`6c455a3`, 8 × 3): hybrid/handoff 24/24, requests 22/24,
  consolidated/review 15/24, workspace 0/24. Rules alone lacked the goal.
- Preliminary five-wake task (`29a963b`, 4 × 3, before the notice):
  consolidated/review 7/12, handoff 2/12, requests 1/12, workspace 0/12.
  Trace reading found a merge dropping a rule and a second-policy check
  that can pass by copying a neighbour. The task flaw remains recorded.

## Phase 7 — dreaming, second version (note 005)

- Schema-enforced exercises quote real user decisions. Code filters and
  positive controls verify dreams; one rule edit needs repeated improvement.
  Practice restores protected files. Replay uses real requests 5 times.
- One-seed live check at `82c8dda`: all four designs passed; no rule changed.
  Existing-feature dreams made controls answer "already there". `aea50a8`
  asks for uncovered work and adds a new-feature suffix to every dream.

## Phase 8 — demo and stopping point (2026-11-17; note 005)

- Built `python -m dreamwake.demo`, live event output and clean Ctrl+C.
  Live hybrid demo: stopped at 6 steps with failed checks; new process used
  stored memory, fixed template/audit work and passed in 11 steps.
- The owner asked to reach a stopping point. Work stopped cleanly at
  `f630b95` (01:52 local). The suite then passed 557 tests, about 4 minutes.
- The five-wake comparison with practice started at 01:32 local. At about
  02:07 all background jobs exited with code 4 together: session or machine
  interruption, not a code fault. `long-27b-a` has **2 of 8 seeds**, 36 rows.
  Hybrid 6/6, consolidated/review/dreaming 5/6, replay 4/6, handoff 0/6.
  These are descriptions of **2 of 8 seeds**, not rates. No dream repair ran.
- The frozen worktree was removed at the stopping point. Completing seeds
  2–7 needs a new seed-offset option or a full rerun.
- Documentation, including notes 004 and 005, was deferred until the owner's
  explicit authorisation on 2026-11-18. This update records that work.
