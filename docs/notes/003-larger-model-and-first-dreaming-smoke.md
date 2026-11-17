# 003 The 27B model, and a first live look at dreaming

## Questions
1. Does the larger, heavily quantised model (Qwen 3.8 27B, IQ2_XS) show the
   same "has the rule, does not apply it" failure as Qwen 3.5 9B?
2. Does the first practice design (dreaming and replay with an LLM reviewer)
   do anything useful with a real model?

## Setup
- Run C, commit `47bddd3`, model `qwen3.8-27b-gsq-rco@iq2_xs`, window 4096,
  reserve 1024, 20 steps per session:
  `python -m dreamwake.run --task settings_gen --designs workspace,handoff,consolidated --seeds 6`
- Smoke D, same commit, model `qwen/qwen3.5-9b`, one seed:
  `python -m dreamwake.run --task settings_gen --designs consolidated,replay,dreaming --seeds 1`

## Result 1: the 27B applies stored rules

| | workspace | handoff | consolidated |
|---|---|---|---|
| completion | 0/6 | 5/6 | 6/6 |
| honors the revision | 0/6 | 5/6 | 6/6 |
| repeated the earlier mistake | 2/6 | 0/6 | 1/6 |
| sleep seconds per episode | 0 | 5.2 | 6.6 |
| injected tokens | 0 | 260 | 230 |

All six wakes were valid on the first attempt. With 12-step sessions the same
model had needed six attempts for three valid wakes (a check at commit `26ba3d5`):
it reads about ten files before writing anything.

Against the 9B (0/16, 5/16, 4/8 in note 002) the difference is in applying
the memory, not in writing it: both models' notes contain the policy.

## Result 2: the first practice design changed nothing
One seed, read from the practice log, so this is a description, not a rate.
- **Replay** scored every attempt as perfect. Re-doing a past request that
  names the setting and the unit cannot expose a rule being ignored.
- **Dreaming** invented requests in the style of the real one, including the
  setting name ("add a setting called CONNECTION_TIMEOUT_S"), so they did not
  test unprompted application either. After the second session it invented
  policy reversals ("switch all time values back to seconds"). The agent
  obeyed them, correctly, and the reviewer then counted that as a broken rule.
- **The LLM reviewer** reported problems whose own text said every step had
  passed.
- No rule rewrite was accepted. All three designs ended with the same memory
  text; dreaming and replay only added offline cost.

## Decisions this changed
1. On the 27B, the short-task comparison is close to settled: any memory
   takes completion from 0 to at least 5/6, and a note is within one seed of
   curated memory. Further work on that model should test longer histories
   and interrupted sessions, where the two designs can plausibly differ.
2. Practice is reworked so that scoring is executable (commit after
   `47bddd3`): a dream is written from the rules, phrased in everyday terms
   without naming files or conventions, and carries strings the result must
   and must not contain. A rewrite is kept only if it improves the failed
   exercise without making the other one worse. The LLM reviewer is removed.
3. A dream that invents a policy change is a hazard in its own right: a
   rewrite trained on it would teach the agent to resist a real instruction.
   The dream prompt now excludes policy changes, and this case stays on the
   list of adversarial tests.

## Not yet tested
The reworked practice design with a live model; any held-out task; whether
the note keeps an early decision across more than two sessions.
