# Run index

`runs/` is local and not versioned. This table says what each run was, so a
number in a note can be traced to a folder and a commit. Updated 2026-11-18.
Historical experiment rows below use `settings_gen`. Current tasks are
listed explicitly in the new table. Model experiments use window 4096,
reserve 1024, temperature 0.7.

| Folder | Commit | Model | Designs | Seeds | Steps/session | Cited in |
|---|---|---|---|---|---|---|
| `probe` * | `347af6a` | four candidates | — (model probe) | — | — | note 000 |
| `smoke` * | `347af6a` + fixes later committed as `26ba3d5` | gemma-4-e4b | workspace, handoff | 1 | 20/15/12 | note 001 |
| `pilot-gemma-4096` * | same | gemma-4-e4b | workspace, handoff | 3 (stopped early) | 20/15/12 | note 001 |
| `wakecheck-qwen9b` * | `26ba3d5` | qwen3.5-9b | workspace, handoff | 3 | 20/15/12 | log, phase 1 |
| `wakecheck-qwen27b` * | `26ba3d5` | qwen3.8-27b iq2_xs | workspace, handoff | 3 | 20/15/12 | log, phase 1 |
| `base-9b-4096` | `d6f198f` | qwen3.5-9b | workspace, handoff | 8 | 20 | note 002 (run A) |
| `dev-9b-4096` | `f100e14` | qwen3.5-9b | workspace, handoff, consolidated | 8 | 20 | note 002 (run B) |
| `smoke-practice` * | `47bddd3` | qwen3.5-9b | consolidated, replay, dreaming (first version) | 1 | 20 | note 003 |
| `dev-27b-4096` | `47bddd3` | qwen3.8-27b iq2_xs | workspace, handoff, consolidated | 6 | 20 | note 003 (run C) |

\* archived in `runs/archive/early_runs_2026-11-16.zip`.

Runs before `d6f198f` used stricter hidden checks, no overwrite guard, the
uncalibrated token estimate and shorter sessions. Do not pool them with later
runs.

Each run folder holds `config.json` (arguments, model, commit, dirty flag),
`results.jsonl` (one row per seed, design and final repeat), `errors.jsonl`
if any seed faulted, and per seed: wake logs, snapshots, each design's final log
and archived workspace, and practice sandboxes for the practice designs.

Print a run's table with `python -m dreamwake.run --summarize runs/<folder>`.

## Added work, 2026-11-16 to 2026-11-17

Configuration fields below come from each experiment's `config.json`.
Q27 means `qwen3.8-27b-gsq-rco@iq2_xs` (Qwen 3.8 27B IQ2_XS);
Q9 means `qwen/qwen3.5-9b`. Current runs use Q27 only, by the owner's
2026-11-16 decision. Q9 entries are history. All folders under `runs/`
are local, not versioned; none of these names links to a published artifact.

| Folder | Commit | Model | Task | Designs | Seeds × repeats | Reports it |
|---|---|---|---|---|---|---|
| `smoke-practice2` | `0c5c7d1` | Q9 | `settings_gen` | consolidated, replay, dreaming (first rework) | 1 × 1 | [note 004](notes/004-diagnostic-placement-and-gate.md) |
| `smoke-nearby` | `664df72` | Q9 | `settings_gen` | handoff, consolidated, nearby | 1 × 2 | [note 004](notes/004-diagnostic-placement-and-gate.md) |
| `smoke-diag-27b` | `8818724` | Q27 | `settings_gen` | handoff, consolidated, consolidated@tail, consolidated@tail+gate, requests, requests@tail, oracle, oracle@request, oracle@tail | 1 × 1 | [note 004](notes/004-diagnostic-placement-and-gate.md) |
| `diag-27b-a` | `8818724` | Q27 | `settings_gen` | handoff, consolidated, consolidated@request, consolidated@tail, consolidated+gate, requests, oracle | 8 × 3 | [note 004](notes/004-diagnostic-placement-and-gate.md) |
| `smoke-resume-27b` | `29a963b` | Q27 | `settings_resume` | workspace, handoff, consolidated, requests, review | 1 × 1 | [note 005](notes/005-resumption-long-histories-and-dreaming-v2.md) |
| `smoke-long-27b` | `29a963b` | Q27 | `settings_long` | workspace, handoff, consolidated, requests, review | 1 × 1 | [note 005](notes/005-resumption-long-histories-and-dreaming-v2.md) |
| `long-27b-pre` | `29a963b` | Q27 | `settings_long` | workspace, handoff, consolidated, requests, review | 4 × 3; preliminary | [note 005](notes/005-resumption-long-histories-and-dreaming-v2.md) |
| `resume-27b-a` | `6c455a3` | Q27 | `settings_resume` | workspace, handoff, consolidated, requests, review, hybrid | 8 × 3 | [note 005](notes/005-resumption-long-histories-and-dreaming-v2.md) |
| `smoke-dream2-27b` | `82c8dda` | Q27 | `settings_long` | consolidated, hybrid, replay, dreaming | 1 × 1 | [note 005](notes/005-resumption-long-histories-and-dreaming-v2.md) |
| `long-27b-a` | `aea50a8` | Q27 | `settings_long` | handoff, consolidated, review, hybrid, replay, dreaming | planned 8 × 3; **2 of 8 seeds** completed | [note 005](notes/005-resumption-long-histories-and-dreaming-v2.md) |
| `demo-live` | `f630b95` (fact sheet; no config) | Q27 | `settings_resume` | hybrid | not applicable; two process invocations | [note 005](notes/005-resumption-long-histories-and-dreaming-v2.md) |

`smoke-practice2` predates the final-repeat flag; its single final is the
one-repeat condition. Single-seed checks and the incomplete **2 of 8 seeds**
are descriptions, not rate estimates. The incomplete run has only seeds
0 and 1, with 36 rows. It stopped with the session/machine interruption,
not a recorded code fault.

Since 2026-11-18 the six `smoke-*` folders of 2026-11-16/17 (1,556 files)
are stored in `runs/archive/smoke_runs_2026-11-17.zip`; the file count was
verified before the folders were removed. The research reports and the
documentation fact sheet are kept in folders whose names end in
`_compact_protected`, so clean-ups leave them in place.

`diag-27b-a` and `long-27b-pre` predate the ignored-tool-call notice added
in `6c455a3`. Do not pool them with later conditions without stating that.
`nearby` was removed in `8818724`; use `consolidated@request` now.

## Supporting local folders

These are not model comparison runs and have no experiment `config.json`.
Commit/model/task/design/seed settings do not apply. They are also local,
not versioned.

| Folder | Contents | Reports it |
|---|---|---|
| `research-2026-11-16_compact_protected` | three ChatGPT 6.1 live-search reports and briefs; review written by Opus at `644f488` | literature review; project log phase 4 |
| `context-2026-11-18_compact_protected` | lead's fact sheet, computed tables, three trace analyses, adversarial build brief | notes 004/005; this documentation update |
| `tmp-codex` | disposable CLI temp/test scratch | local orchestrator guide |
| `tmp-codex2` | separate disposable temp/test scratch for a concurrent builder | local orchestrator guide |
