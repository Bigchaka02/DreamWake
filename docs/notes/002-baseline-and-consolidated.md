# 002 Baseline and consolidated memory on the tuning task

## Question
On `settings_gen` with Qwen 3.5 9B at a 4096-token window, how often does the
agent finish the final session after the context reset with (a) files only,
(b) a handoff note, (c) a consolidated rule list? Which hidden check fails?

## Setup
- Model `qwen/qwen3.5-9b` (Q8_0) in LM Studio, context 4096, reserve 1024,
  `reasoning_effort: none`, temperature 0.7. 20 steps per session.
- Gemma 4 E4B was dropped first: 0 valid wakes in 16 attempts (note 001).
  The IQ2_XS 27B Qwen also completes the wake and is kept for confirmation.
- Run A, commit `d6f198f`:
  `python -m dreamwake.run --task settings_gen --designs workspace,handoff --seeds 8`
- Run B, commit `f100e14`:
  `python -m dreamwake.run --task settings_gen --designs workspace,handoff,consolidated --seeds 8`
- Every wake was valid on the first attempt in both runs (16/16), and the
  agent fell into the trap during the wake every time.

## Result

| | workspace | handoff | consolidated |
|---|---|---|---|
| completion, run A | 0/8 | 1/8 | not run |
| completion, run B | 0/8 | 4/8 | 4/8 |
| honors the revision | 0/16 | 5/16 | 4/8 |
| feature works, trap avoided, convention, no regression | all pass | all pass | all pass |
| repeated the earlier mistake, run B | 7/8 | 2/8 | 0/8 |
| mean steps, run B | 17.0 | 14.0 | 15.5 |
| sleep seconds per episode, run B | 0 | 3.6 | 8.3 |
| injected tokens, run B | 0 | 214 | 268 |

Completion and "honors the revision" are the same number in every cell: the
revised requirement (durations in milliseconds, `_MS` suffix) is the only
check any design fails.

The handoff rate moved from 1/8 to 4/8 between two identical configurations.
With 8 seeds a difference of that size is noise, so handoff and consolidated
are not separated on completion by these runs.

## What the traces show
- **Files only.** The agent re-discovers the generator trap and the
  CHANGES.md audit from tool feedback, at a cost of about three steps. It
  never infers the milliseconds policy: the stale README says seconds, the
  request says "2 seconds", and it writes `RETRY_BACKOFF_SECONDS: 2.0` next
  to the existing `TIMEOUT_MS`.
- **Handoff.** Every note read states the policy, usually as "the new policy
  replaces the older store-in-seconds decision". The agent still writes a
  seconds setting about two times in three. The notes describe a finished
  task ("next step: none; task complete"), not an instruction for new work.
- **Consolidated.** All four failures had the rule in memory, for example
  "Duration settings must be stored in milliseconds with names ending in
  _MS (e.g., TIMEOUT_MS)". One failure converted the value but kept the old
  name: `BACKOFF_SECONDS: 2000`. The model has the fact and does not apply it
  to a request phrased in the old unit.
- A wart in the merge step: items grew `(replaces: <whole old rule>` tails
  until they hit the length cap. The instruction now limits that to five
  words (commit `47bddd3`).

## Decisions this changed
1. The failure to attack is **application**, not recall. That is the case
   for practice-based refinement: run the agent on a practice request with
   the memory, and rewrite a rule only when a re-run shows the rewrite
   changes what the agent does (dreaming and replay designs, commit `725073c`).
2. Completion needs more than 8 seeds per cell to separate designs. The
   five-design run uses the same seeds for all designs and reports the
   run-to-run spread seen here alongside any difference.
3. The trap and the convention are recoverable from tool feedback when the
   agent has 20 steps, so their memory benefit shows up as fewer repeated
   mistakes and fewer steps, not as completion.

## Where the gain lives
Stored artifacts (the note or rule list on disk) supplied as context at
inference. No model parameters change.
