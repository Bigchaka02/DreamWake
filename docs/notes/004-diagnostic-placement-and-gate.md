# 004 Diagnostic: memory content, placement and finish gate

Written 2026-11-18. Work ran 2026-11-16 to 2026-11-17.

## Questions

1. Do ideal rules or the user's own words beat learned rules on the short
   task? Does placing rules nearer the request or checking before finish
   close the application gap?
2. What did the last 9B practice check actually change?

## Setup

- Last historical 9B checks: `smoke-practice2` at `0c5c7d1`, one seed;
  `smoke-nearby` at `664df72`, one seed × 2 repeats. The owner then chose
  Qwen 3.8 only. No 9B or Gemma run is planned.
- Main diagnostic `diag-27b-a`, commit `8818724`, Qwen 3.8 27B IQ2_XS
  (`qwen3.8-27b-gsq-rco@iq2_xs`), task `settings_gen`, 8 seeds × 3 final
  repeats. Window 4096, reserve 1024, 20 steps, temperature 0.7;
  `reasoning_effort: none`, `--parallel 1`. Runtime: 52 minutes.
- Each seed shares its wake across designs. Repeats reuse the stored memory
  and workspace, with repeat seeds shared across designs. Analyse pairs per
  seed; repeats do not add independent pairs. LM Studio is not reproducible
  by seed.
- `oracle` loads hand-written ideal rules, without sleep. `requests` keeps
  the user's words within the same 900-character injection cap, without a
  model call. These diagnose content. Placement and gate diagnose delivery.
- The one-seed `smoke-diag-27b` at `8818724` checked the new grammar before
  the comparison; its configuration is in the run index.

Commands below describe the recorded runs. Use the recorded commit and
model, replace `<abs>` with the repository directory, and choose an unused
output folder. All `runs/` paths are local, not versioned.

```powershell
python -m dreamwake.run --task settings_gen `
  --designs consolidated,replay,dreaming --seeds 1 `
  --out <abs>\runs\smoke-practice2
# At 664df72 only: nearby was later removed.
python -m dreamwake.run --task settings_gen `
  --designs handoff,consolidated,nearby --seeds 1 --final-repeats 2 `
  --out <abs>\runs\smoke-nearby
$shortDesigns = 'handoff,consolidated,consolidated@request,' +
  'consolidated@tail,consolidated+gate,requests,oracle'
python -m dreamwake.run --task settings_gen `
  --designs $shortDesigns `
  --seeds 8 --final-repeats 3 --out <abs>\runs\diag-27b-a
```

## Result 1: last 9B evidence

`smoke-practice2`: consolidated, replay and dreaming all failed the revision
and ended with identical memory. Practice changed nothing. Replay scored
all attempts perfect. After s1 the dream check strings could not match;
after s2 malformed JSON was silently dropped and fell back to real-request
replay. The stored policy was tied to `TIMEOUT`; the final wrote
`RETRY_BACKOFF_SECONDS: 2.0`. Sleep cost was 9 s / 67 s / 80 s respectively.
This is a one-seed description, not a rate.

`smoke-nearby`: handoff 1/2, consolidated 1/2, nearby 2/2 in one seed.
This suggested a diagnostic, not a rate estimate. `nearby` was removed in
`8818724`; its replacement is `consolidated@request`.

Separate lead-run prompt probes were not committed as runs. With a revised
prompt and filters the 9B produced 7 usable exercises of 16 parsed, the 27B
10 of 14. LM Studio enforced a JSON schema in 6 of 6 calls. These are small
checks, not a measured benefit from dreaming.

## Result 2: short task on Qwen 3.8

Completion means every hidden check passes. W/T/L compares each design's
completion within a seed against consolidated. All-3 counts seeds where
all three final repeats passed.

| Design | Completed | All-3 (of 8) | W/T/L | Mean steps | Cap endings |
|---|---|---|---|---|---|
| `oracle` | 24/24 | 8 | 1/7/0 | 14.3 | 0 |
| `requests` | 24/24 | 8 | 1/7/0 | 17.5 | 1 |
| `consolidated` | 23/24 | 7 | — | 16.1 | 0 |
| `consolidated@request` | 23/24 | 7 | 1/6/1 | 17.0 | 1 |
| `consolidated+gate` | 23/24 | 7 | 0/8/0 | 17.8 | 6 |
| `consolidated@tail` | 18/24 | 3 | 1/2/5 | 16.7 | 6 |
| `handoff` | 15/24 | 3 | 0/3/5 | 15.8 | 2 |

Handoff versus consolidated: sign test p = 0.06. For the gate, 8/24 reached
step 20, but 6 ended at the cap; reaching that step is not the same as
running out of steps.

## What the traces show

- **Handoff:** the policy was in all 9 failing notes, 7 as a clear rule and
  2 as finished work. Four notes failed one repeat and passed another.
  Our reading is unreliable application inside prose, not a lost rule.
- **Tail:** identical path-and-content writes occurred in 7/24 sessions,
  versus 0/24 for consolidated. Failures include disrupted work as well as
  missed rules.
- **Consolidated, request placement and gate:** each had one failure with
  the rule present. Each wrote `RETRY_BACKOFF_MS = 2.0` and divided by 1000.
  The gate fired, received "All rules satisfied", and allowed finish without
  correction. Our reading: the model rubber-stamped its own checklist.
- **Requests:** the agent touched generated `src/settings.py` in 16/24
  finals. The user's words contain no learned lesson about that trap.
- **Harness:** extra calls in a multi-call reply were silently dropped.
  `6c455a3` adds a notice and `ignored_calls` rows. This diagnostic predates
  the fix; it does not establish how many failures that defect caused.

Source records: `results.jsonl` and final traces in the named local run
folders. The read-only analyses `report_analysis_smoke.txt` and
`report_analysis_diag.txt`, and the checked `tables.txt`, are in
`runs/docs-update-2026-11-18/` (local, not versioned).

## Decisions this changed

1. Our reading: the short task is saturated on Qwen 3.8. Learned rules are
   within one session of ideal rules, leaving little for wording or practice
   to fix. The handoff gap is the only large effect here.
2. Keep store, placement and gate separate in the design grammar. Neither
   request placement nor a finish checklist improved completion here.
   Tail placement disrupted work on this model.
3. Move the next comparisons to interrupted work and longer histories.
   Keep requests and oracle as diagnostics, not evidence of learned memory.
4. Make ignored calls visible before further comparisons.

## Where the gain lives

Stored artifacts injected as context, plus the later harness notice for
ignored calls. Oracle rules are hand-written; requests are the user's own
words. No model parameters change. This run does not test hybrid on the
short task, held-out tasks, or adversarial behaviour.
