# 2 — Project log

Concise record of what was done, in order. All of it happened on 2026-11-16.
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
  rule, `(replaces: …)` tail cleaner. **Passes 192 tests; not yet run live.**

## Housekeeping at the stopping point
- LM Studio model unloaded and server stopped (as found).
- Early run folders (probe, smoke, Gemma pilot, both wake checks, practice
  smoke: 641 files) zipped to `runs/archive/early_runs_2026-11-16.zip` after
  verifying the file count; the frozen git worktree `runs/wt-frozen` removed.
- ChatGPT command-line route verified (see file 4).
- One usage-limit interruption mid-session; three subagents resumed with no
  work lost.
