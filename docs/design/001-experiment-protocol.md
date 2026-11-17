# 001 — Experiment protocol (pilot)

Status: draft v1, written before any agent run. Numbers marked *(calibrate)* are
fixed after the baseline pilot and frozen before held-out tasks are touched.

## Question

Under a fixed token window, which of four things that can survive a context
reset lets a small local model finish multi-session work most often, and at
what cost?

| Design | Survives the reset | Offline ("sleep") work |
|---|---|---|
| `workspace` | project files only | none |
| `handoff` | files + one short note | one LLM call per session, written from the live window |
| `consolidated` | files + curated memory store | reads the full session transcript in window-sized chunks; add / update / supersede / forget |
| `dreaming` | same store | plus practice on verified variations of experienced failures |
| `replay` (control) | same store | same offline budget as `dreaming`, spent re-reviewing real experience instead of generating new practice |

## Episode

One episode = one task × one seed.

1. **Wake (shared).** Sessions `s1` and `s2` are run ONCE, live, by the plain
   agent with no memory, each from an empty conversation. Every design
   receives this identical transcript and the identical resulting workspace.
   A wake counts as valid only if the hidden tests for `s2` pass; invalid wakes
   are re-sampled and the validity rate is reported.
2. **Sleep (per design).** After `s1` and again after `s2`, each design turns
   the recorded transcript into whatever it keeps. Sleeping twice is deliberate:
   the second sleep has to supersede a decision stored by the first.
3. **Reset.** Conversation history is discarded. Only the design's artifact
   and the workspace remain.
4. **Final session (per design).** Fresh conversation, same tools, same window.
   The design's artifact is injected and counts against the window.
5. **Evaluation.** Hidden executable tests, run on a copy after regenerating
   build outputs. Never visible to the agent or to any sleep process.

## Token window

- Window `W` is enforced physically: the model is loaded with context length
  `W`, so an over-length prompt cannot silently succeed.
- Everything counts: system prompt, tool schemas, injected memory, history,
  tool results. A fixed reserve is kept for the model's reply.
- History is a rolling window: oldest turns are dropped first. Session
  history is expected to exceed `W` (the required over-window condition).
- Pilot `W` = 4096 *(calibrate from the machine inventory)*; second condition
  at 2×W to see whether the ranking changes with budget.

## Measures

Primary: final-session completion (all hidden checks pass).
Per-check: feature works · honors the revised requirement · trap avoided ·
convention followed · no regression.
Process: did the agent repeat the earlier mistake at any point (from the trace),
steps used, whether it ran out of steps.
Cost: final-session tokens and wall time; sleep LLM calls, tokens and wall
time; artifact size on disk and in injected tokens.

## Guardrails

- Held-out task variants are written and frozen before any design is tuned;
  their hidden tests and outcomes never feed dream generation or selection.
- Dream content is tagged `hypothetical`; only a general procedure distilled
  from it may enter memory, never a project fact.
- One consequential mechanism changes per comparison where practical.
- Each note in `docs/notes/` records: question → observed result → decision
  it changed.

## Where an improvement lives

Every result names where the gain persists: model parameters (none in the
pilot — the installed models are served inference-only), stored artifacts, or
context supplied at inference.
