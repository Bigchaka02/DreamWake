# Brief: "Language Models Need Sleep" (Behrouz, Hashemi, Javanmard, Mirrokni)

Source: arXiv 2606.03979v2 (v1 2 Jun 2026, v2 10 Jul 2026; arXiv comment says a version was on OpenReview from Sep 2025). Read from the v2 HTML and the Google Research blog post, both saved in `.refs/paper/`. Section, table and equation numbers below are the paper's. Anything marked UNVERIFIED was not in either source.

## 1. What the paper claims

- **Sleep paradigm (§3.1).** The model alternates "wake" phases (taking in data, fast memory) with "sleep" phases (no external input). Sleep has two stages: consolidation (§3.2–3.3) and dreaming (§3.4). Every stage changes the weights.
- **Table 3 (knowledge incorporation, SQuAD).** The metric is mean "no-context" accuracy: questions are answered without the passage in context. n=1 is a single passage. n=200 is continued pretraining (CPT): 200 passages in one run, scored on 974 questions, with 5 dreams sampled per passage (§4.1).

| Method (Table 3) | n=1 | n=200 |
|---|---|---|
| Base model | 31.9 | 31.9 |
| Fine-tuned model with no dreaming | 33.4 | 32.0 |
| SEAL | 46.7 | 43.2 |
| Sleep (Transformer) [two-level memory, §4.1] | 48.1 | 44.3 |
| Sleep (Transformer + four-level) | 48.9 | 46.2 |
| removing gradient-based selection | 47.1 | 45.2 |
| removing random expert | 48.0 | 44.7 |
| removing Dreaming | 35.7 | 36.2 |

- **The lead's numbers are correct, with one wording fix.** 46.2% is the four-level model at n=200, and 36.2% is the "removing Dreaming" row at n=200. The paper's term is "no-context accuracy", not "passages removed". The paper never says which configuration the ablation rows start from. The four-level model is implied but not stated.
- **What the other rows show (my arithmetic, n=200):**
  - Dreaming accounts for most of the gain: removing it costs 10.0 points (46.2 to 36.2); at n=1 it costs 13.2 points.
  - Consolidation alone is worth +4.2 points over plain fine-tuning (36.2 vs 32.0).
  - Four levels beat two by +1.9 (46.2 vs 44.3).
  - Full Sleep beats SEAL by +3.0. SEAL is essentially dreaming without consolidation.
  - Gradient-based dream selection is worth 1.0 point and the random expert 1.5 points.
  - No seeds, variances or confidence intervals are reported. For a single run on 974 questions at about 45% accuracy, the binomial standard error is about 1.6 points (my calculation). So the 1.0–1.9 point differences cannot be told apart from noise. Only the roughly 10-point dreaming effect and the roughly 11–14 point gap to the non-dreaming baselines are clearly larger.
- **Other results.** Consolidation-only ablations on maths (Table 1, Qwen3-8B, AIME-24/AIME-25/HMMT-25): full Sleep 79.2/69.0/46.1; without imitation learning 76.8/67.9/45.0; without semantic reward 78.9/69.2/44.5; without expansion 78.2/67.9/44.9; OPSD 76.6/67.4/45.1. Consolidation also helps class-incremental learning (Fig. 3) and sequential translation of two unseen languages without forgetting (§4.1, "Learning a New Language In-Context"). Hope-3 "nearly recovers" its single-language performance. More consolidation levels improve long-context QA (Fig. 4).

## 2. Consolidation (§3.2–3.3): what moves, from where to where, when, with what objective

- **Structure.** The model's MLP blocks form a chain of mixture-of-experts layers (the paper's "CMS"). Each block updates at its own frequency, with chunk size C^(l). The fastest block is in-context memory.
- **Trigger.** Consolidation runs at step counts that are multiples of a block's chunk size (C^(l) × b). It always runs *before* the sender block's own weight update, so the fast block's knowledge is not lost when it is overwritten (§3.2).
- **What moves where.** Knowledge goes from the faster block MLP^(f_{l*-1}) into a newly added low-rank expert (A, B) in the next, slower block MLP^(f_{l*}) (§3.2 parameter expansion). Implementation detail: the expert's parameters are pre-allocated and masked until they are switched on.
- **Compute–consolidate–update protocol (§3.3):**
  - (a) Compute the sender's prospective base-weight update from its accumulated gradients, but do not apply it yet.
  - (b) Consolidate. The teacher is the model before the update. The student is the model with the prospective weights, the sender's experts reset, and the new expert added. Only A and B are trained; all other student parameters are frozen.
  - (c) Update: apply the base-weight update, reset the experts previously added to the sender (the paper calls this "synaptic pruning"), and switch on the new expert.
- **Objective ("Knowledge Seeding", an upward distillation from a smaller to a larger model):**
  - Training data D is sampled from the teacher. No external data is used.
  - The GKD-style on-policy distillation loss (the unnumbered equation in §3.3) mixes teacher-sampled sequences with student-sampled ones, weighted by λ. It uses a divergence between the teacher's and the student's token distributions. Gradients do not flow through the student's sampling.
  - "Learning to Imitate" (Eq. 3–4) is an RL step. The student completes a random prefix of a teacher sample. Its reward is γ·r_sem + (1−γ)·r_abs:
    - r_sem: a frozen reward model gives 1 if the meaning matches the teacher's sample, else 0.
    - r_abs: 1 minus the normalised Levenshtein distance, set to 0 above a distance threshold z0.
  - The combined objective L_KS is weighted by α.
- **What needs gradients:** the base-weight update, the distillation and the LTI policy optimisation. Every training step of consolidation needs them. Sampling from the teacher, the prefix-completion test and the similarity scoring are inference only.

## 3. Dreaming (§3.4)

- **Generation.** For a sampled task (C, τ), where C is the context and τ is a downstream performance measure, the model samples m ≥ 1 dreams (synthetic training text) with C in context. During sampling, each MoE router also routes to one random expert, which injects unrelated knowledge.
- **Filtering.** Each dream gets an importance score g = ∇θ L_SFT(dream), the gradient of the language-modelling loss. The top-k dreams by score are kept, plus b random dreams for diversity. This is gradient-based, so it needs backprop.
- **Reward: judge a dream by what the model can subsequently do.**
  - Each kept dream is used to LoRA-fine-tune an isolated copy of the model.
  - Reward (Eq. 5) is 1 if the fine-tuned copy improves on τ compared with the copy before fine-tuning, else 0.
  - The dream *generator* is then trained with ReST^EM, following SEAL: generate, keep the reward-1 samples, fine-tune on them, repeat.
- **What is kept.** Dreams that earned reward 1 become the generator's training data. The resulting weights carry the new knowledge. In the CPT setting, the 5 dreams per passage are pooled into one synthetic training set (§4.1).

## 4. Portable to an inference-only small local model?

| Mechanism (section) | Verdict | How, for DreamWake |
|---|---|---|
| Trigger at fixed schedule; consolidate *before* the fast memory is overwritten (§3.2) | (a) direct | Run consolidation at session end, before the context is wiped, while the full context still exists |
| Teacher = pre-update state, student = post-update state (§3.3 a–c) | (b) adapt | Teacher = agent with full context. Student = fresh agent given only the memory. Compare their behaviour on probe tasks |
| New low-rank expert in a slower block (§3.2) | (b) adapt | Append new entries to an external store tier. Do not overwrite older consolidated entries |
| Multi-level hierarchy, two vs four levels (§3.2, Table 3) | (b) adapt | Tiers updated at different rates: scratch, then session log, then project memory, then stable conventions. Expect a small effect (+1.9 in Table 3) |
| On-policy distillation loss on logits (§3.3) | (c) needs training | No weight updates possible. Only the *idea* survives: test the student on the student's own continuations |
| Learning to Imitate reward, semantic + Levenshtein (Eq. 3–4) | (b) adapt | As an acceptance test for a memory, not as RL: give the fresh agent a prefix of the teacher's plan or trajectory and score how its continuation matches (LLM-judge equivalence plus edit distance) |
| Synaptic pruning, reset fast experts (§3.3 c) | (a) direct | Clear scratch notes after consolidation. Delete superseded memory entries |
| Dream generation conditioned on context C (§3.4) | (a) direct | Generate candidate lessons, procedures or practice tasks from the session log |
| Random-expert injection (§3.4) | (b) adapt | Not exposed in LM Studio (UNVERIFIED for that runtime). Nearest analogue: put unrelated memory items or higher temperature into dream prompts. Low priority (−1.5, within noise) |
| Gradient-based importance score (§3.4) | (c) needs training | Needs backprop. Replace with a cheap proxy (novelty against the store, coverage of failing tests) or skip it (−1.0, within noise) |
| SFT on each dream + reward if τ improves (Eq. 5) | (b) adapt | Put the candidate lesson into a fresh agent's context instead of fine-tuning. Run executable practice tasks with and without it. Keep the lesson only if the measured pass rate improves |
| ReST^EM training of the dream generator (§3.4) | (c) needs training | Weaker analogue: reuse accepted lessons as few-shot examples in later dream prompts |

## 5. Testable hypotheses for the four designs

Designs: W = workspace-only, N = handoff note, M = consolidated external memory, D = consolidation + dreaming. Metric for every hypothesis: hidden executable test pass rate after the context wipe, at equal injected-token budget (fixed share of the ~4k window).

1. **H1 (D vs M).** D beats M only if lessons are kept by measured success on executable practice tasks (Eq. 5 analogue). An unvalidated variant of D (generated lessons, no selection) scores no better than M. Falsified if unvalidated D ≥ validated D, or validated D ≤ M.
2. **H2 (M vs N scales with horizon).** M ≈ N at 2 sessions, but M > N by a pre-set margin at 5 or more sessions, because a one-paragraph note cannot carry accumulated state. Falsified if N ≥ M at every horizon tested.
3. **H3 (consolidate before wipe, §3.2 ordering).** Memory written by the agent while it has full context beats memory rebuilt after the wipe from workspace files alone. Falsified if the rebuilt memory is equal or better.
4. **H4 (imitation agreement predicts resume success, Eq. 3–4 analogue).** Across runs, how well a fresh agent's next action, given the memory, matches the full-context agent's next action correlates positively with resume pass rate. If so, it can be a cheap gate for any note or memory. Falsified if the correlation is about zero.
5. **H5 (append + prune prevents forgetting, §3.2–3.3).** An append-only tiered store with pruning of superseded entries keeps more earlier-session regression tests passing than a store overwritten in place. Falsified if regression pass rates are equal.
6. **H6 (baseline bar).** N, M and D must each beat W by a margin chosen before the runs. Report any gain under ~2× the run-to-run standard deviation as null. Falsified for a design if it does not clear W.

## 6. Caveats: what should not be assumed to transfer

- **Every Table 3 gain comes from weight updates:** LoRA SFT on each dream, ReST^EM, the distillation into new experts. The 10.0-point dreaming effect says that *training on validated dreams* helps parametric recall. It does not say that putting dreams or lessons into a prompt helps. Treat any expected size of DreamWake's effect as unknown.
- **Task mismatch.** Table 3 measures recall of facts from SQuAD passages, with no context. DreamWake measures continuing procedural, stateful coding work with an executable check. The paper's τ is QA accuracy; ours (test pass rate) is more objective, which may make Eq. 5-style selection *easier* for us. That is unproven.
- **Model and scale.** The Table 3 backbone is not named; §4.1 says only that it follows "the experimental setup of Zweiger et al. (2025)" (SEAL). Which model that is: UNVERIFIED. Other results use Llama-3B/8B, Qwen3-1.7B/8B and Llama-3.2-1B. Dream quality from a small quantised GGUF model is untested.
- **Architecture.** "Four-level" means a four-level MoE memory chain. The two- vs four-level gap (+1.9) and the selection and random-expert ablations (−1.0, −1.5) are within about 1.6 points of single-run binomial noise. No variance is reported, and it is unclear which configuration the ablations start from. Do not build on these small effects.
- **Context budget.** The paper's consolidation targets long contexts (BABILong up to 10M tokens). We have about 4k tokens, so selecting what goes into memory matters more than storing it.
- **Closest prior work.** §2.3 cites Lin et al. (2025) "sleep-time compute", which summarises past context for the next session without training. That is closer to DreamWake than Sleep is, and the paper does not evaluate it.
- **Paper quality.** There are cross-reference slips: the text cites "Table 4.1" and "Figure 6" for results that are in Table 3/4 and Table 1. Minor, but read the numbers from the tables, not the prose.
