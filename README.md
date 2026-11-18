# DreamWake

Can a small, locally running language model put down unfinished work, lose
its whole conversation, and resume correctly later? DreamWake compares what
survives a reset under a fixed 4096-token window: project files, a handoff
note, rule memory, and rules refined by offline practice ("dreaming").
The starting point is *Language Models Need Sleep* (arXiv 2606.03979); see
[the paper brief](docs/research/paper-brief.md) for what carries over to an
inference-only model.

## Status (2026-11-18)

Work through 2026-11-17, commit `f630b95`. All current experiments use
Qwen 3.8 27B IQ2_XS, by the owner's decision. Completion means all hidden
checks pass. Each seed shares a wake across designs; final repeats start
from the same stored memory.

| Store | Short: `settings_gen`, 8 × 3 | Interrupted: `settings_resume`, 8 × 3 | Five wakes: `settings_long`, **2 of 8 seeds**, 3 repeats |
|---|---|---|---|
| `workspace` | not in diagnostic | 0/24 | not in this comparison |
| `handoff` | 15/24 | 24/24 | 0/6 |
| `consolidated` | 23/24 | 15/24 | 5/6 |
| `requests` | 24/24 | 22/24 | not in this comparison |
| `oracle` | 24/24 | not run | not run |
| `review` | not run | 15/24 | 5/6 |
| `hybrid` | not run | 24/24 | 6/6 |
| `replay` | not in diagnostic | not run | 4/6 |
| `dreaming` | not in diagnostic | not run | 5/6 |

The five-session run was interrupted: **2 of 8 seeds**, a description, not
a rate. A separate preliminary comparison is in
[note 005](docs/notes/005-resumption-long-histories-and-dreaming-v2.md).
Placement and gate results are in
[note 004](docs/notes/004-diagnostic-placement-and-gate.md).

History: Qwen 3.5 9B had 0/16 files-only, 5/16 handoff, 4/8 consolidated;
Gemma 4 E4B had 0 valid wakes in 16 attempts (notes 001–003).

Our reading: the short task is nearly saturated; interrupted work needs a
record of the goal. Hybrid adds the last request and completion status to
the rules without another model call. It is the provisional recommendation
for this desktop. It has not run on the short task. Dreaming's second
version has run live but changed no rule. Neither held-out task
(`catalog_legacy`, `cli_registry`) nor the adversarial tests has run.
No model parameters change.

## Reproduce

Needs Python 3.11+ and LM Studio. Runtime code is standard library only;
`pytest` is needed for the test suite. PowerShell:

```powershell
lms server start --port 1234
lms load qwen3.8-27b-gsq-rco@iq2_xs --gpu max -c 4096 --parallel 1 -y
$env:DW_MODEL = "qwen3.8-27b-gsq-rco@iq2_xs"
$shortDesigns = 'handoff,consolidated,consolidated@request,' +
  'consolidated@tail,consolidated+gate,requests,oracle'
python -m dreamwake.run --task settings_gen `
  --designs $shortDesigns `
  --seeds 8 --final-repeats 3 --out <abs>\runs\<new-short-run>
python -m dreamwake.run --task settings_resume `
  --designs workspace,handoff,consolidated,requests,review,hybrid `
  --seeds 8 --final-repeats 3 --out <abs>\runs\<new-resume-run>
python -m dreamwake.run --task settings_long `
  --designs handoff,consolidated,review,hybrid,replay,dreaming `
  --seeds 8 --final-repeats 3 --out <abs>\runs\<new-long-run>
python -m dreamwake.run --summarize runs/<run-folder>
python -m pytest -q
```

Replace `<abs>` with the absolute repository directory and choose a new
output name. These reproduce the configured comparisons on the checked-out
code; recorded commits are in [the run index](docs/runs-index.md).
The runner refuses an existing `results.jsonl`; it cannot yet resume at a
later seed. Configuration, results and logs under `runs/` are local, not
versioned.

Try the resumable demo in two separate processes:

```powershell
python -m dreamwake.demo --reset --interrupt-after 6
python -m dreamwake.demo
```

Defaults: `settings_resume`, `hybrid`, state in `runs/demo` (local, not
versioned). It prints remembered text, the request, tool events, stored
memory and project checks. Each invocation starts with an empty
conversation. Ctrl+C also ends a session cleanly. The recorded live check
stopped with failed checks, then resumed in a new process and passed
([note 005](docs/notes/005-resumption-long-histories-and-dreaming-v2.md)).

## Designs

Grammar: `store[@placement][+gate]`. Injected memory is capped at
900 characters.

| Store | Survives with the project files | Offline work |
|---|---|---|
| `workspace` | no memory text | none |
| `handoff` | prose note | one call per session |
| `consolidated` | rule list | extract from full log, merge across sessions |
| `requests` | verbatim requests, newest prioritised | no model call |
| `oracle` | hand-written ideal rules | no sleep; diagnostic |
| `review` | rule list | consolidate; one schema call may edit one rule against the request archive |
| `hybrid` | rules plus last request marked finished/NOT finished | consolidate; state line built by code |
| `replay` | possibly edited rules | repeat real request 5 times; confirm one rule repair |
| `dreaming` | possibly edited rules | verify new exercises against user's words; confirm one rule repair |

| Modifier | Meaning |
|---|---|
| `@system` (default) | memory in system prompt |
| `@request` | memory after request |
| `@tail` | trailing user message rebuilt every call, never trimmed |
| `+gate` | first finish gets notes as checklist |

`workspace` accepts no modifiers. `nearby` was removed; use
`consolidated@request`. Details: [project overview](1_PROJECT_OVERVIEW.md).

## Layout

- `1_PROJECT_OVERVIEW.md`, `2_PROJECT_LOG.md`, `3_PROJECT_STATUS_AND_TODO.md`
  — current design, history, evidence and open work.
- `dreamwake/` — loop (`agent.py`), stores (`designs.py`), runner (`run.py`),
  demo (`demo.py`), task loader/evaluator (`tasks.py`), client (`llm.py`).
- `tasks/` — `settings_gen`, `settings_resume`, `settings_long`, and frozen
  held-out tasks `catalog_legacy`, `cli_registry`.
- `docs/design/001-experiment-protocol.md` — protocol and dated amendments.
- `docs/notes/` — questions, results and decisions.
- `docs/research/literature-review.md` — research with verification labels.
- `docs/runs-index.md` — registry of local, unversioned evidence.
- `docs/engineering-rules.md` — working rules from superpowers and ponytail.
