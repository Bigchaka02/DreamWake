# 005 Resumption, long histories and dreaming, second version

Written 2026-11-18. Work ran 2026-11-16 to 2026-11-17.

## Questions

1. Can a rule list resume an unfinished request without a record of the goal?
2. Does an early decision survive five wake sessions?
3. Can grounded, verified dreaming improve rules that already work? What
   does it cost compared with consolidation and replay?

## Setup

All runs use Qwen 3.8 27B IQ2_XS (`qwen3.8-27b-gsq-rco@iq2_xs`), context
4096, reserve 1024, temperature 0.7, `reasoning_effort: none`, `--parallel 1`.
Sessions have 20 steps except the interrupted wake's 6-step cap. Wakes are
shared across designs; final repeats start from the same memory and
workspace. Repeat seeds are shared; comparison units remain seed pairs.

Two tasks inherit `settings_gen`:

- **`settings_resume`:** s1 as before; s2 asks for the milliseconds policy
  but stops after 6 steps, with `stage: null` (unscored). Wake validity is
  s1 passing. Final request: "Continue the unfinished work from your previous
  session and finish it. Make the checks pass." The final uses s2 hidden
  tests, including the unfinished rename.
- **`settings_long`:** five wakes: timeout in seconds; milliseconds policy;
  `client.is_secure()`; `SERVICE_NAME` plus the decision that each new
  template entry carries a "doc" field; `retry.describe()`. The s5 hidden
  stage checks cumulative state. The final adds backoff as on `settings_gen`
  and adds `test_honors_second_policy`.

| Run | Commit | Task | Seeds × final repeats |
|---|---|---|---|
| `smoke-resume-27b` | `29a963b` | `settings_resume` | 1 × 1 |
| `smoke-long-27b` | `29a963b` | `settings_long` | 1 × 1 |
| `long-27b-pre` | `29a963b` | `settings_long` | 4 × 3; preliminary, before ignored-call notice |
| `resume-27b-a` | `6c455a3` | `settings_resume` | 8 × 3 |
| `smoke-dream2-27b` | `82c8dda` | `settings_long` | 1 × 1 |
| `long-27b-a` | `aea50a8` | `settings_long` | planned 8 × 3; incomplete, **2 of 8 seeds** |
| `demo-live` | `f630b95` | `settings_resume` | interactive continuation; not a seed comparison |

Commands below describe those configurations. Use the listed commit and
model, replace `<abs>` with the repository directory and use an unused
output folder. All run folders are local, not versioned.

```powershell
# 29a963b: the two one-seed checks (use settings_long for the second).
python -m dreamwake.run --task settings_resume `
  --designs workspace,handoff,consolidated,requests,review `
  --seeds 1 --final-repeats 1 --out <abs>\runs\smoke-resume-27b
python -m dreamwake.run --task settings_long `
  --designs workspace,handoff,consolidated,requests,review `
  --seeds 1 --final-repeats 1 --out <abs>\runs\smoke-long-27b
# 29a963b: preliminary long-history comparison.
python -m dreamwake.run --task settings_long `
  --designs workspace,handoff,consolidated,requests,review `
  --seeds 4 --final-repeats 3 --out <abs>\runs\long-27b-pre
# 6c455a3: interrupted-session comparison.
python -m dreamwake.run --task settings_resume `
  --designs workspace,handoff,consolidated,requests,review,hybrid `
  --seeds 8 --final-repeats 3 --out <abs>\runs\resume-27b-a
# 82c8dda: dreaming second-version check.
python -m dreamwake.run --task settings_long `
  --designs consolidated,hybrid,replay,dreaming `
  --seeds 1 --final-repeats 1 --out <abs>\runs\smoke-dream2-27b
# aea50a8: configured for eight seeds; only two were completed.
python -m dreamwake.run --task settings_long `
  --designs handoff,consolidated,review,hybrid,replay,dreaming `
  --seeds 8 --final-repeats 3 --out <abs>\runs\long-27b-a
```

## Result 1: interrupted work needs the goal

In the one-seed checks, handoff and requests passed interrupted work;
workspace, consolidated and review failed. Only consolidated passed the
five-session check. These are descriptions, not rates. In the interrupted
trace the rule list, told only to continue, ran checks, saw them pass and
finished in 9 steps. It did not know work was unfinished.

`6c455a3` adds hybrid: consolidated rules plus a code-built last-request line
marked finished or NOT finished. It also reports ignored extra tool calls.
The interrupted comparison took 32 minutes.

| Design | Completed | All 3 passed (of 8) | W/T/L vs hybrid | Sleep s/episode | Injected tokens |
|---|---|---|---|---|---|
| `hybrid` | 24/24 | 8 | — | 5.8 | 300 |
| `handoff` | 24/24 | 8 | 0/8/0 | 4.9 | 263 |
| `requests` | 22/24 | 6 | 0/6/2 | 0 | 211 |
| `consolidated` | 15/24 | 4 | 0/4/4 | 5.8 | 209 |
| `review` | 15/24 | 4 | 0/4/4 | 7.1 | 225 |
| `workspace` | 0/24 | 0 | 0/0/8 | 0 | 0 |

Workspace versus hybrid has sign-test p = 0.008. Every failure is
`test_honors_revision`, the unfinished rename. Ignored calls per final
session range from about 1.1 for workspace to 2.1 for handoff.

Our reading: resuming needs a goal record. One code-built line changes the
rule list's result from 62% to 100% here, without extra offline cost.
This task-specific result does not establish general reliability.

## Result 2: five wakes, preliminary comparison

`long-27b-pre`: 4 seeds × 3 repeats, 23 minutes. This predates the
ignored-call notice and is a separate condition from the later comparison.

| Design | Completed | Honoured session-2 decision |
|---|---|---|
| `consolidated` | 7/12 | 7/12 |
| `review` | 7/12 | 8/12 |
| `handoff` | 2/12 | 2/12 |
| `requests` | 1/12 | 1/12 |
| `workspace` | 0/12 | 0/12 |

### What the traces show

- Handoff and requests state the policy generally in every memory, yet apply
  it in 2/12 and 1/12 finals. Failing traces never mention milliseconds
  before writing. Our reading is an application problem despite retention.
- Consolidated and review complete 7/9 when the rule is present, 0/3 when
  absent. In one traced seed the merge after s5 dropped the milliseconds
  rule while rewriting the list. Memory was 736 of 900 characters; the cap
  was not the cause in that trace.
- Review restored a dropped rule in one seed, but spent its edit elsewhere
  in another. It is not reliable protection for the user's decisions.
- The "doc" field check passed 59/60, including no-memory cases. The agent
  copied the field from the neighbouring `SERVICE_NAME` entry. This check
  does not measure memory. The task flaw is recorded, not fixed.

## Result 3: dreaming second version, live check

The second version uses the user's request archive as its source. One
schema call proposes up to 4 exercises with exact decision quotes. Code
rejects ungrounded quotes, numeric-only check strings, answer-bearing
requests and named identifiers/files. Positive controls receive only the
quoted decision after the request, without a gate; at most 3 controls seek
2 verified dreams. There is no fallback to real requests. Stored rules are
then tried at the design's placement/gate. A failure can trigger one rule
edit, kept only after two improved retries and a non-worsening other
exercise. Protected files are restored before scoring. Replay instead
practises the real request 5 times, without controls. Full mechanics are
in file 1.

`smoke-dream2-27b`, one seed: consolidated, hybrid, replay and dreaming all
passed. This is a description, not a rate. Dream replies quoted decisions
exactly; filters rejected one altered quote. Three of five sleeps ended
"no verified dreams": the requests asked for settings already present, and
the control agent correctly answered "already there".

| Design | Sleep s/episode |
|---|---|
| `consolidated` | 18 |
| `hybrid` | 18 |
| `replay` | 243 |
| `dreaming` | 170 |

`aea50a8` then asks the dream writer for work no archived request covered.
Every dreamed request receives the suffix:

```text
 This is a new, separate feature: add it even if something similar already exists.
```

## Result 4: five wakes with practice — incomplete, 2 of 8 seeds

`long-27b-a` started 2026-11-17 at 01:32 local. At about 02:07 all session
background jobs exited with code 4 together. The session or machine was
interrupted; this was not a code fault. Only seeds 0 and 1 have rows:
36 rows, from 2 seeds × 6 designs × 3 repeats. The configured comparison
remains **2 of 8 seeds**. These are descriptions, not rates or a final
ranking.

| Design | Completed (6) | Seed 0, seed 1 fractions | Sleep s/episode |
|---|---|---|---|
| `hybrid` | 6/6 | 1.00, 1.00 | 18.1 |
| `consolidated` | 5/6 | 1.00, 0.67 | 18.1 |
| `review` | 5/6 | 0.67, 1.00 | 22.1 |
| `dreaming` | 5/6 | 1.00, 0.67 | 282.6 |
| `replay` | 4/6 | 0.67, 0.67 | 222.5 |
| `handoff` | 0/6 | 0.00, 0.00 | 11.9 |

Dreaming's log has 10 sleeps. Session 1 always reports "no verified dreams",
expected because no standing decision exists yet. Every other sleep reports
"all exercises passed". Controls verified dreams about milliseconds and
"doc"; stored rules passed them. No repair call ran, no rule changed, and
contamination was empty.

Our reading: on this task Qwen 3.8 already applies consolidated decisions
when they are present, leaving practice nothing to repair. A merge can
still drop a standing rule, as seen in the preliminary run. Practice did
not catch that weakness in these two seeds. This is a reading of the
observed traces, not an established explanation for every failed check.

The runner cannot append later seeds: it numbers 0..N-1 and refuses an
existing `results.jsonl`. Completing seeds 2–7 requires a full rerun or a
small `--first-seed` build. No such option exists yet.

## Result 5: a live usable continuation

At `f630b95`, hybrid on `settings_resume`, state `runs/demo-live` (local,
not versioned):

```powershell
python -m dreamwake.demo --state runs/demo-live --reset --interrupt-after 6
python -m dreamwake.demo --state runs/demo-live
```

- First process read files, edited generated `src/settings.py` directly,
  ran checks and stopped at 6 steps with checks FAILED. It stored
  729 characters: a NOT-finished line and three rules, including the lesson
  to edit the template because direct generated-file edits fail.
- Second process had an empty conversation. It printed that memory and a
  continuation request, edited `settings.tmpl.json`, then updated
  `CHANGES.md` after audit failed. It finished in 11 steps; checks passed.

The demo also supports clean Ctrl+C (`ended_by="interrupted"`) and prints
live tool events through `on_event`. This live example is a description,
not a reliability estimate. It reports project checks, not a new comparison
of hidden-test completion rates.

## Cost and capacity

Full-GPU planning costs: about 6.5 minutes per seed for 7 short-task designs
with 3 repeats; about 4 minutes per seed for 6 interrupted-task designs;
about 18–20 minutes per seed for the five-session task with replay/dreaming,
each practising after five sleeps.

| Store | Offline seconds per episode |
|---|---|
| `handoff` | 5–12 |
| `consolidated`, `hybrid` | 6–18 |
| `review` | 7–22 |
| `replay` | 220–245 |
| `dreaming` | 170–285 |
| `requests`, `oracle` | 0 |

Agent sessions dominate practice cost, up to 15 per episode. First-sleep
dreaming has no decision to practise.

## Decisions this changed

1. A rule list needs a goal record for interrupted work. Hybrid adds it
   from the user's own words with no extra model call.
2. Our provisional recommendation for this desktop is hybrid, pending the
   remaining six seeds, its short-task run and held-out tasks. Handoff works
   for the interruption but loses effectiveness over longer histories.
3. Do not claim a dreaming benefit. In the observed sleeps it added cost
   and changed no rule. Keep the decision rule fixed before completing the
   comparison: dreaming helps only if it beats both consolidated and replay
   per seed with sign-test support; otherwise record a null.
4. Protecting standing decisions against merge loss remains open. Review
   is unreliable; dreaming has not demonstrated a repair for it.
5. Replace the second decision in a new task version so success needs memory.
   Keep frozen held-out tasks untouched.

## Where the gains live

Stored artifacts supplied as context, plus the ignored-call notice in the
harness. Hybrid's state is a code-built line from the user's words.
No parameters change. The demo makes those artifacts usable across process
boundaries; it does not add a new learning mechanism.

## Not yet tested

The remaining six seeds; hybrid on the short task; either frozen held-out
task; explicit rule exceptions, wrong-scope or inverted dreams; a repaired
second-decision task; the 8192-token condition; adapter training. These
remain work, not results. Documentation changes require the owner's say-so.

Evidence: `results.jsonl`, practice logs and session traces in the named run
folders. `tables.txt` and `report_analysis_long.txt` are under
`runs/context-2026-11-18_compact_protected/` (local, not versioned). The literature review
records the research that informed the second version.
