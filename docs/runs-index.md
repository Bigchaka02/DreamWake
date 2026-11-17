# Run index

`runs/` is local and not versioned. This table says what each run was, so a
number in a note can be traced to a folder and a commit. All runs: task
`settings_gen`, window 4096, reserve 1024, temperature 0.7.

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
`results.jsonl` (one row per seed and design), `errors.jsonl` if any seed
faulted, and per seed: the wake logs, the snapshots, each design's final log
and archived workspace, and practice sandboxes for the practice designs.

Print a run's table with `python -m dreamwake.run --summarize runs/<folder>`.
