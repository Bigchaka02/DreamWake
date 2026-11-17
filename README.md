# DreamWake

Can a small, locally running language model put down unfinished work, lose
its whole conversation, and resume correctly later? DreamWake compares what
can survive that reset under a fixed 4096-token window: nothing but the
project files, a one-paragraph handoff note, a curated rule memory, and that
memory refined by offline practice ("dreaming"). The starting point is
*Language Models Need Sleep* (arXiv 2606.03979); see
[docs/research/paper-brief.md](docs/research/paper-brief.md) for what carries
over to an inference-only model and what does not.

## Status (2026-11-16)

Tuning task `settings_gen`: two work sessions, a context reset, then a new
related change scored by hidden tests. Completion = all hidden checks pass.

| Model (LM Studio, 4096 window) | files only | handoff note | consolidated rules |
|---|---|---|---|
| Qwen 3.5 9B (Q8) | 0/16 | 5/16 | 4/8 |
| Qwen 3.8 27B (IQ2_XS) | 0/6 | 5/6 | 6/6 |
| Gemma 4 E4B | never completes the first two sessions (0 of 16 attempts) | | |

What this shows so far:

- Without memory the agent always misses a requirement the user revised in an
  earlier session. It re-discovers tool-enforced conventions on its own.
- On the 27B model a one-paragraph note is nearly as good as curated memory.
- On the 9B model the memory usually *contains* the rule and the agent still
  does not apply it. That gap is what the dreaming design targets.

Not done yet: the dreaming and replay designs have not had a full live run;
the two held-out tasks (`catalog_legacy`, `cli_registry`) are written and
frozen but not run; the resumable demo is not built. Details and evidence are
in [docs/notes](docs/notes).

## Reproduce

Needs Python 3.11+ and LM Studio with one of the models above. Runtime code
is standard library only; `pytest` is needed for the test suite.

```
lms server start --port 1234
lms load qwen/qwen3.5-9b --gpu max -c 4096 --parallel 1 -y
```

```
# PowerShell; use `export DW_MODEL=...` in a POSIX shell
$env:DW_MODEL = "qwen/qwen3.5-9b"
python -m dreamwake.run --task settings_gen --designs workspace,handoff,consolidated --seeds 8
python -m dreamwake.run --summarize runs/<run folder>
python -m pytest -q
```

Each run writes `runs/<name>/results.jsonl`, every session's step log, and a
`config.json` with the commit, model and arguments. `runs/` is not versioned.

## Designs

| Name | Survives the reset | Offline work |
|---|---|---|
| `workspace` | project files | none |
| `handoff` | files + a note of at most 900 characters | one model call per session |
| `consolidated` | files + a rule list of at most 900 characters | reads the whole session log in chunks, merges rules across sessions |
| `replay` | same rule list | re-does the real past request in a sandbox; keeps a rule rewrite only if the re-run scores better |
| `dreaming` | same rule list | the same, on invented practice requests that carry their own executable checks |

## Layout

- `1_PROJECT_OVERVIEW.md`, `2_PROJECT_LOG.md`, `3_PROJECT_STATUS_AND_TODO.md`
  — the design in full, what was done in order, and what is open.
- `dreamwake/` — agent loop (`agent.py`), designs (`designs.py`), experiment
  runner (`run.py`), task loader and evaluator (`tasks.py`), LM Studio client
  (`llm.py`).
- `tasks/<id>/` — starting workspace, session prompts, hidden tests and
  reference states for each task.
- `docs/design/001-experiment-protocol.md` — the protocol and its guardrails.
- `docs/notes/` — one note per experiment: question, result, decision.
- `docs/runs-index.md` — which local run folder backs which number.
- `docs/engineering-rules.md` — working rules distilled from the superpowers
  and ponytail repositories.
