# 3 — Status, analysis and to-do

Updated 2026-11-18 for work on 2026-11-16 to 2026-11-17, through `f630b95`.
The design is in file 1; history is in file 2; detailed evidence is in
notes 004 and 005.

## 1. Where things stand

| Area | State |
|---|---|
| Harness and evaluator | task-defined sessions/stages, inheritance, final repeats; 557 tests passed at `f630b95` |
| Model | Qwen 3.8 27B IQ2_XS only, context 4096, `--parallel 1`; owner's decision |
| Stores, placements, finish gate | built and checked live; overview lists the grammar |
| Interrupted work | `settings_resume` compared across 8 seeds × 3 repeats |
| Five wakes | `settings_long` preliminary comparison; practice comparison incomplete, **2 of 8 seeds** |
| Dreaming, second version | live checks and incomplete comparison; no rule changed |
| Demo | built; new process resumed unfinished work and passed project checks |
| Held-out tasks | unchanged, frozen; never run |
| Adversarial tests | not built or run |
| Hybrid on short task | not run |
| 8192-token condition | not run |
| Adapter training | not attempted |
| Recommendation | provisional; first research cycle still open |

The owner ended 9B and Gemma experiments on 2026-11-16. Their results stay
as history; neither has a planned run. This was an owner decision, not a
controlled finding about which model is best.

## 2. Results so far

All current comparisons use Qwen 3.8, window 4096, reserve 1024, temperature
0.7, normally 20 steps per session. The interrupted wake has a 6-step cap.

| Store | Short, 8 × 3 | Interrupted, 8 × 3 | Five wakes, **2 of 8 seeds**, 3 repeats |
|---|---|---|---|
| `workspace` | not in diagnostic | 0/24 | not in comparison |
| `handoff` | 15/24 | 24/24 | 0/6 |
| `consolidated` | 23/24 | 15/24 | 5/6 |
| `requests` | 24/24 | 22/24 | not in comparison |
| `oracle` | 24/24 | not run | not run |
| `review` | not run | 15/24 | 5/6 |
| `hybrid` | not run | 24/24 | 6/6 |
| `replay` | not in diagnostic | not run | 4/6 |
| `dreaming` | not in diagnostic | not run | 5/6 |

Five-session figures are a description of **2 of 8 seeds**, not a rate.
The preliminary five-session run used 4 seeds × 3 repeats: consolidated and
review 7/12 each, handoff 2/12, requests 1/12, workspace 0/12. It predates
the ignored-call notice. Do not pool it as the same condition.

Historical 9B evidence remains in notes 002–004: files-only 0/16, handoff
5/16, consolidated 4/8. Gemma had 0 valid wakes in 16 attempts (note 001).

## 3. What the evidence supports, and how firmly

1. **Short-task memory application is near ceiling on Qwen 3.8.** Ideal
   rules and raw requests completed 24/24; consolidated 23/24. Our reading:
   little room remains for wording or practice to improve this task.
   Handoff's larger gap is descriptive evidence (0/3/5 against consolidated
   per seed, sign test p = 0.06), not a general result across tasks.
2. **Placement and a checklist did not close that gap.** Request placement
   and gate each completed 23/24; tail placement 18/24. Tail traces show
   disrupted execution. The gate rubber-stamped its failing conversion.
3. **Resumption needs the unfinished goal.** Hybrid and handoff completed
   24/24; rules alone 15/24; workspace 0/24. Every failure was the unfinished
   rename. Hybrid beats consolidated 4-0 among non-tied seed pairs, without
   extra offline calls. This supports the mechanism on this task; it does
   not establish general reliability.
4. **Long histories expose both application and retention failures.** In
   the preliminary run, handoff and requests retained general policy words
   yet rarely applied them. Consolidated/review completed 7/9 with the rule
   present and 0/3 with it absent. One merge dropped the rule below the cap.
5. **Dreaming has not demonstrated a benefit.** In the live second-version
   check all designs passed. In the incomplete comparison, verified dreams
   were already passed by the rules; no repair call ran and no rule changed.
   These are small-sample descriptions, not a null for other tasks.
6. **The demo demonstrates one real continuation.** A new process received
   stored memory, corrected the template and audit, and passed checks. One
   example is not a reliability estimate.

### Recommendation as it stands — provisional

Our reading: use `hybrid` on this desktop with Qwen 3.8, keeping consolidated
rules in the system prompt plus a verbatim last-request line marked
finished/NOT finished. It completed 24/24 interrupted finals and 6/6 in the
five-session comparison (**2 of 8 seeds**), at consolidated's offline cost.
It has not run on the short task. Handoff ties it for a single interruption
but collapses over five wakes (0/6 in **2 of 8 seeds**, 2/12 preliminary).
Dreaming adds 10–15x offline cost and has changed no rule. The failure to
protect against is a merge dropping a standing decision; review does not
restore it reliably, and dreaming did not catch it in the observed sleeps.
The remaining six seeds and held-out tasks are most likely to change this
recommendation.

## 4. Open questions

- Does the five-session result survive the remaining six seeds, paired by
  seed? Does dreaming beat both consolidated and replay with sign-test
  support, or should we record a null?
- Does the recommendation transfer to the frozen held-out tasks?
- Can the leading design obey an explicit exception to a stored rule?
  Can wrong-scope or inverted dreams corrupt practice?
- How should user decisions survive a merge that drops a rule?
- Can a replacement second decision in `settings_long` measure memory
  without leaving a visible pattern to copy?
- Does hybrid also work on the short task? Does ranking change at 8192?

## 5. Known weaknesses

- `settings_long`'s "doc" field check passed 59/60 preliminary finals,
  including no-memory cases copying `SERVICE_NAME`. It does not measure
  memory. This flaw is recorded, not fixed.
- The merge can silently drop a rule. Review is unreliable protection;
  dreaming did not trigger a repair in the observed comparison.
- The gate can be rubber-stamped and costs about 1.7 steps. `@tail` disrupts
  work on this model.
- The requests store cuts old requests at 900 characters. Our reading is
  that long histories are read as finished tasks rather than standing rules.
- `diag-27b-a` and `long-27b-pre` predate the ignored-call notice. The
  effect of silently dropped calls on their outcomes is not established.
- Seeds are numbered 0..N-1, and existing `results.jsonl` is refused.
  Finishing the interrupted run needs a full rerun or a new `--first-seed`.
- First-sleep dreaming has no standing decision to practise. Agent sessions
  dominate practice cost, up to 15 per episode.
- LM Studio is not reproducible by seed. Shared repeat seeds still diverge.
- Consolidated versus handoff bundles full-log extraction, rule format and
  merging. Substring checks cannot express every kind of rule.
- Held-out `catalog_legacy` leaves an analogy for its revision in the files.
  It stays frozen; no held-out result is available.

## 6. To-do

### For the lead, in pinned order

1. Re-enter: rerun the suite (557 passed at the stopping point), start LM
   Studio, load Qwen 3.8 at 4096 / `--parallel 1`, recreate the frozen
   worktree at HEAD.
2. Build `--first-seed K`, record it in config, then run seeds 2–7 into
   `runs/long-27b-b` with the same six designs and 3 final repeats (about
   2 hours). Analyse seeds 0–7 together, paired per seed. Fixed rule:
   dreaming helps only if it beats both consolidated and replay per seed
   with sign-test support; otherwise record a null.
3. Build `settings_override` and `scripts/adversarial_dreams.py` from the
   ready adversarial brief. Then run both on Qwen 3.8.
4. Run hybrid on `settings_gen`, 8 seeds × 3 repeats.
5. Run held-out tasks once, Qwen 3.8 only: workspace, handoff, consolidated,
   hybrid, requests; dreaming only if step 2 keeps it. Use 8 × 3. First a
   one-seed wake check. If wake is unachievable, report it; do not alter the
   task. No design change after these runs.
6. Fix or replace the second decision as a new `settings_long` task version.
   Never edit the frozen held-out tasks.
7. Write the recommendation note and update README on the owner's say-so:
   design, evidence, where it breaks, next experiment, where gains live.

Commands are in file 5. Run folders and the brief under
`runs/docs-update-2026-11-18/` are local, not versioned.

### For the owner

- Adapter training is effectively moot: the reported 27B QLoRA floor is
  22 GB, beyond this 16 GB GPU. Decide only if a separate 9B training
  experiment is wanted.
- Decide whether local, git-ignored files 4 and 5 should be published.
- Decide whether the ChatGPT route's usage limit changes the delegation
  plan. Documentation changes still require explicit say-so.
