# 3 — Status, analysis and to-do

Snapshot at the stopping point on 2026-11-16, commit `3096429` plus these
documents. What the project is: file 1. What was done: file 2.

## 1. Where things stand

| Area | State |
|---|---|
| Harness, evaluator, 3 tasks | working, reviewed, 192 tests pass |
| `workspace`, `handoff`, `consolidated` | built, run live on two models (tuning task only) |
| `replay`, `dreaming` (reworked) | built and unit-tested; **never run against a live model** |
| Held-out tasks | written and frozen; **never run** |
| Longer histories, interrupted sessions | not built |
| Adversarial tests (stale memory, wrong dream, changed instruction) | not built |
| 8192-token condition | not run |
| Resumable demo | not built |
| Final recommendation | not written |
| Adapter training | not attempted (needs a download the owner has not approved) |

Mission milestones: first milestone reached; first research cycle not reached.

## 2. Results so far

Task `settings_gen`, window 4096, completion after the reset:

| Model | workspace | handoff | consolidated |
|---|---|---|---|
| Qwen 3.5 9B | 0/16 | 5/16 | 4/8 |
| Qwen 3.8 27B IQ2_XS | 0/6 | 5/6 | 6/6 |
| Gemma 4 E4B | 0 valid wakes in 16 attempts | | |

Cost per episode: handoff 3.6–5.2 s of sleep and about 210–260 injected
tokens; consolidated 6.6–8.3 s and about 230–270 tokens.

## 3. What the evidence supports, and how firmly

1. **No memory, no completion** (0/22 across both models). Firm. The single
   failing check is always the revision the user stated in session 2.
2. **Tool-enforced knowledge is re-discovered without memory** when the agent
   has 20 steps; memory only saves steps and repeated mistakes there. Firm for
   this task.
3. **On the 27B a one-paragraph note is nearly as good as curated memory**
   (5/6 vs 6/6). Suggestive only: 6 seeds, one task.
4. **On the 9B the rule is in memory and often not applied.** Firm as a
   description of the traces (every failing consolidated run had the rule).
5. **Handoff vs consolidated on the 9B is not separated.** The handoff rate
   moved 1/8 → 4/8 between identical runs.
6. **The first practice design did nothing useful.** One seed; a description,
   not a rate.

## 4. Open questions, in the order they matter

1. Does the reworked dreaming close the 9B's application gap, and does replay
   with the same budget fail to? (The core dreaming claim.)
2. Does any of this hold on the two held-out tasks?
3. Does a note that is rewritten every session keep an early decision after
   several unrelated sessions? (Where consolidated should beat handoff.)
4. Can a rule list alone resume a half-finished request, or does it need a
   state note alongside? (The mission's central scenario.)
5. Do stale rules, a wrong dream or a changed instruction corrupt memory?
6. Does the ranking change at 8192 tokens?

## 5. Known weaknesses and things to improve

- **Seeds per cell are too few.** Use at least 16 for any 9B claim, or
  report paired differences with the run-to-run spread from note 002.
- **The runner assumes exactly three sessions.** Longer and interrupted
  variants need: any number of wake sessions, a per-session choice of which
  hidden stage scores it (or none), validity = all scored wake sessions pass.
- **Bundled change.** `consolidated` differs from `handoff` in three ways at
  once (whole log, rule format, merge step). If it wins, ablate.
- **Dream expectations are substring tests.** They suit naming, unit and
  format rules; they cannot express "edit file A, not file B" (the project
  check covers that only when a test exercises the new code).
- **A wrong dream can still pass the accept rule** if both exercises are
  wrong in the same direction. This is the adversarial test to run.
- **Dreams after session 1 practise the old policy**, which session 2 then
  revises. Whether a dream-strengthened stale rule survives the merge is
  untested.
- **Memory sits in the system prompt.** Placing rules next to the request is
  an untested, cheap alternative for the application gap.
- **`catalog_legacy` leaks its revision by analogy** (`app.ui.farewell()` and
  existing `ui.` keys). Expect files-only to do better there. Left as is.
- Consolidation's extract and merge calls share one seed; practice seed
  offsets assume at most 22 steps; LM Studio is not reproducible by seed.
- No design stores goal/state/next-step except `handoff`.

## 6. To-do

### For Claude, in order (commands in file 5)
1. Live one-seed smoke of the reworked `replay` and `dreaming`; read the
   practice log before anything else.
2. Dev comparison on the 9B: `handoff`, `consolidated`, `replay`, `dreaming`,
   16 seeds (two runs of 8 if time is short). At most one tuning round after.
3. Generalise the runner to N sessions; add `settings_long` (five wake
   sessions, revision in session 2) and `settings_resume` (session 2 cut off
   at 6 steps, final request "continue"); add a `hybrid` design (rule list +
   short state note within the same 900 characters).
4. Run the held-out tasks once, all designs, both models. No tuning after.
5. Adversarial tests and the 8192 condition.
6. Build `python -m dreamwake.demo`: interruptible, restarts with an empty
   conversation, shows what it remembered, finishes the work.
7. Write the recommendation note and update the README: design for this
   desktop, evidence, a condition where it breaks, the next experiment.

### For the owner (decisions; none blocks step 1)
- **Adapter training:** approve or decline an 8–16 GB download of
  unquantised weights plus a PyTorch install. Without it the "does a small
  adapter help with retrieval disabled" question stays unanswered.
- **Primary model for the final recommendation:** the 27B is the stated use
  case and already near ceiling on the short task; the 9B shows more
  separation. Current plan reports both.
- **Whether files 4 and 5 should be published.** They are kept local
  (git-ignored) because they describe the working setup on this PC.
- **`qwen3.8-flash-next` crashes on load** in the current LM Studio runtime.
  Switching runtime is a persistent setting, so it was left alone.
