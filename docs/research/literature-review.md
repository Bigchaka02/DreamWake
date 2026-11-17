# Literature review: memory, practice and rule application (2026-11-16)

On 2026-11-16 the 9B model usually had the right rule in memory and still
did not apply it. Its practice exercises were often unusable, and practice
had not changed a single stored rule. Three research briefs asked what the
literature says about (1) offline dreaming, consolidation and verified
self-practice; (2) why a rule in the prompt goes unapplied, and what helps;
and (3) agent harness and evaluation design. This document condenses the
three reports, says how far each finding can be trusted, and records what we
decided. The foundation paper's gains all come from weight updates [1], so
every transfer judgement asks whether an idea survives without training.

**Evidence labels.** **M** = measured in a paper. **P** = practitioner report
or documentation. **I** = inference by the researcher who wrote the report.
"Transfer" is our judgement for a 9B–27B local model with a 4096-token
window, no weight updates and a 900-character memory. Sources are numbered
and listed at the end.

## How far to trust this

- The three reports were produced by a research assistant model with live web
  search. It was told to open each source before citing it.
- The project lead independently opened the arXiv abstract pages of five
  papers: 2609.03213 [32], 2512.14982 [28], 2508.21433 [44], 2505.16944 [24]
  and 2405.15793 [42]. All exist with the stated titles, authors and dates.
  The abstracts support the direction of each claim. The specific numbers
  quoted from them are in the full texts and were not independently checked
  at that step.
- While writing this document we fetched eight pages and checked the numbers
  against the page text:

| URL fetched | Checked | Outcome |
|---|---|---|
| https://arxiv.org/html/2506.01716v1 [10] | A6 | Confirmed |
| https://arxiv.org/html/2303.11366v4 [7] | A2, A3 | Confirmed |
| https://arxiv.org/html/2507.19457v1 [13] | A8 | Confirmed |
| https://arxiv.org/html/2510.04618v1 [5] | A4 | Confirmed; the 10.6% is an average gain over selected baselines, not over the base model, and is worded so below |
| https://arxiv.org/html/2505.16944v1 [24] | B2 | Confirmed |
| https://arxiv.org/html/2405.15793v3 [42] | C1 | Confirmed |
| https://arxiv.org/abs/2508.21433 [44] | C3 | Partly: the abstract says masking roughly halves cost against the raw agent; the report's "over 50%" was not found, so C3 says "roughly halved" |
| https://arxiv.org/html/2411.00640v1 [49] | C6 | Confirmed |

- Everything else is as reported, not independently verified. Works dated
  after mid-2026 are marked † in the source list; we cannot vouch for them
  beyond what the reports say.

## A. Dreaming, consolidation and self-generated practice

| # | Finding | Ev. | Src | Transfer |
|---|---|---|---|---|
| A1 | Many systems turn experience into stored text without training. They admit an edit by model judgement or usefulness counters, not by a behavioural test. | M; I ("ungated") | [2][3][4][5][6] | Store designs carry over; there is no tested admission gate to copy. |
| A2 | Self-reflection helped a strong model, not a small one: GPT-4 HumanEval 80 → 91%, StarChat-beta 26 → 26%. | M | [7] | Closest small-model null. Practice may do nothing on a weak model. |
| A3 | Wrong self-tests hurt: Reflexion's MBPP fell 80 → 77%, with 16.3% false-positive self-tests. Unfiltered failed experiences degrade baselines. | M | [7][8] | Model-written checks need controls. |
| A4 | Whole-memory rewrites can collapse: one context shrank from 18,282 to 122 tokens, accuracy 66.7 → 57.1. ACE's itemised deltas beat selected baselines by 10.6% on average (DeepSeek-V3.1). | M | [5] | Edit one item at a time. ACE playbooks far exceed 900 characters. |
| A5 | Keeping raw episodes unchanged and rendering them within a budget scored +11.5 points on AppWorld at 4k (commercial models). | M | [9]† | Keep the user's exact words offline. |
| A6 | Generated tasks can be policed by code: instruction, verifier, example solution, three failure cases. Only 5.2% of Llama-3.1-8B proposals survive. False positives disappear; underspecified instructions remain. | M | [10] | Filters need no stronger model. That paper's capability gains need training. |
| A7 | Test–program agreement improves selection (HumanEval 47 → 65.8%), but correlated mistakes can agree. Leakage is not fixed automatically: showing failing tests cut bug-reproduction attempts from 379/500 to 127/500. | M | [11][12] | Hide answer-bearing checks from the solver. |
| A8 | Prompt optimisation works when selection uses a separate set. GEPA: Qwen3-8B 48.85 → 61.28 aggregate, up to 16k context. TextGrad and OPRO also keep held-out data apart. | M | [13][14][15] | Treat the 900-character artifact as the candidate; keep development, selection and final tasks apart. |
| A9 | Negative: a local 27B study found GEPA MATH .828 → .904, no HotpotQA gain, and SkillOpt validation gains that reversed on held-out data. It used a stronger external reflector. | M | [16]† | Same-weak-model transfer is UNVERIFIED. |
| A10 | No matched-budget comparison of generated practice against real replay or rereading was found. Generated replay ends 2–3% below multitask training; generated negative replay often fails on harder streams. | I (absence); M | [17][18] | Generated practice is an experiment, not an established advantage. |
| A11 | Memory can be poisoned persistently. One study: accuracy .85 → .30 after poisoning 1.2% of a corpus; content screening rejected 0 of 360. | M | [19][20]† | Keep user revisions apart from inferred lessons. Protective effect of supersession is UNVERIFIED. |
| A12 | Parametric "sleep" needs training hardware: one SEAL round took six hours on two H100s. Reported minimum QLoRA VRAM: 6.5 GB for 9B, 22 GB for 27B. | M; P | [1][21][22] | Nothing here runs through an inference-only endpoint. |

## B. Having a rule in context versus applying it

The report's framing (I): the model must recognise that a rule applies,
resolve its priority, then carry it out alongside other demands. Which stage
fails on the 9B is still an inference.

| # | Finding | Ev. | Src | Transfer |
|---|---|---|---|---|
| B1 | Policies in context do not make reliable agents: τ-bench under 50% success, retail pass^8 below 25%. | M | [23] | Failures also include reasoning errors. |
| B2 | Conditional constraints are weak. Llama-3.1-8B: 53.6% per constraint, 19.9% all constraints (70B: 56.3%, 20.9%). Qwen3-32B: 41.1% conditional vs 57.5% ordinary. | M | [24] | Our silent revision is conditional. AgentIF instructions average 1,723 words. |
| B3 | More rules impair each rule: one model's line-length compliance fell from 97% alone to 2% beside five other instructions, while correctness held. | M | [25] | Score conventions separately from tests. |
| B4 | Position matters within 4k: Llama-2-7B is recency-biased; 13B base shows about a 20-point best–worst gap, about 10 after chat tuning. | M | [26] | Retrieval, not rule execution. |
| B5 | The system prompt gives priority, not obedience: the best open model scored 48% on conflicting instructions. | M | [27] | State precedence explicitly. |
| B6 | Repeating the whole input won 47 of 70 comparisons, no significant losses (API models, DeepSeek-V3). Manus recites goals at the context tail, without an effect size. | M; P | [28][29] | Test a short reminder; full duplication is too costly. |
| B7 | The chat template is part of the intervention; LM Studio tool use needs a matching template and parser. | P | [30][31] | Inspect the rendered prompt before blaming memory. |
| B8 | Explicit rules generally beat examples for 7B–32B models, Qwen and Gemma included; more examples gave no consistent gain (preprint). | M | [32]† | Keep the abstract rule. |
| B9 | Trigger wording and explicit scope exclusions are plausible remedies; no controlled evidence. | I | — | Test unseen settings plus non-duration controls. |
| B10 | Targeted checklists beat generic review: LiveBench 69.6 → 75.8 (Command-R+), 73.3 → 76.2 (GPT-4o). Plan-and-Solve: 70.4 → 76.7 on arithmetic. | M | [33][34] | Reuse stored rules as the checklist. Indirect for coding. |
| B11 | Weak self-verification barely helps: Llama-2-13B 37.2 → 37.5, Gemma-7B 36.3 → 36.7. | M | [35] | A second call by the same model is not independent. |
| B12 | Filtering context helped Llama-2-70B (62.8 → 80.3). Retrieval weakens without literal overlap, Llama-3.1-8B included. | M | [36][37] | Neither shows picking 2 of 8 rules helps. |
| B13 | LM Studio supports JSON-schema output; llama.cpp can skip unsupported schema features silently. Strict tool guarantees UNVERIFIED. Llama-3.1-8B GSM8K: 80.1% unconstrained, 82.4% constrained. | P; M | [31][38][39] | Guarantees shape, not sense. |
| B14 | A 4-bit 13B underperformed an FP16 7B on IFEval. Quantisation raised Gemma-4-31B hallucinated tool events 649 → 1,646, with no significant change for Qwen-3.5-9B. | M | [40][41]† | Neither settles 27B IQ2_XS against 9B Q8. |

## C. Agent harness and evaluation method

| # | Finding | Ev. | Src | Transfer |
|---|---|---|---|---|
| C1 | SWE-agent ablations (GPT-4 Turbo): linting 18.0 vs 15.0; 100-line viewer 18.0 vs 12.7 whole file; last five observations 18.0 vs 15.0 full history. | M | [42] | Not measured on small models. |
| C2 | Whole-file editing works for small models: Qwen2.5-Coder 7B/14B 57.9%/69.2%, Gemma2 27B Q8 36.1%, all correctly formatted. Uncontrolled runs. | P | [43] | Keep whole-file writes for short files. |
| C3 | Masking old observations matched LLM summarisation across five configurations and roughly halved cost against the raw agent. Action history is kept. | M | [44] | Trajectories much longer than our 20 steps. |
| C4 | Models degrade unevenly as input grows (18 models). Context folding cuts active context about 10×, via RL training. | M | [45][46] | Not shown at 4096 tokens; folding needs training. |
| C5 | Editing early prompt text forces reprocessing; cached and uncached runs can differ in logits. An LM Studio MLX seed bug is reported, unconfirmed. | P | [47][48]† | Keep static text first; pin backend, template, samplers. |
| C6 | Pair comparisons per item, cluster related items, resample, and do not lower temperature to cut variance. Use pair-preserving tests such as exact McNemar. | Statistical analysis; P | [49][50] | Pair per seed; keep temperature 0.7. |
| C7 | Rule of thumb: n ≈ (1.96 + 0.84)² · q / δ² independent pairs (discordance q, effect δ, 80% power, two-sided 5%). With q = 0.4–0.6: 20 points needs about 80–120 pairs, 30 points 35–55. | I | — | There is no universal seed count. |
| C8 | Report reliability (pass^k), not only the mean; correct for multiple comparisons (Holm). | M; P | [23][51][52] | Report mean success plus pass^3. |
| C9 | Evaluators need adversarial checks: fixing one benchmark cut overestimation by 33%; SWE-bench Verified removed 68.3% of samples. Log review of 21,730 rollouts found shortcuts accuracy hid. | M | [53][54][55] | Test checks against empty, hard-coded and wrong-unit submissions. |
| C10 | Memory benchmarks separate updates, conflicts and abstention; 91.3% of Mem2ActBench's 400 tool tasks were judged strongly memory-dependent. | M; design | [56][57][58] | Build executable tasks with reversals and negative cases. |

## What we measured ourselves on 2026-11-16

- The local server (LM Studio) enforced a JSON schema on chat completions in 6
  of 6 trial calls with the 9B model.
- With one revised dream-writing prompt and the same code-side filters, Qwen
  3.5 9B produced 7 usable practice exercises out of 16 parsed, and Qwen 3.8
  27B (IQ2_XS) produced 10 out of 14. These are small samples on one rule
  list, not a rate.
- In a one-seed live run on the 9B, practice changed no stored rule. After
  the second session the 9B's dream output was malformed JSON and was
  discarded; after the first, its check strings could not match. The stored
  rule in that run was tied to the one setting it was learned on, and the
  final session did not apply it to a new setting.

## What this changes in DreamWake

- **Diagnostic before more dreaming.** A hand-written ideal rule against the
  learned rule, crossed with where the memory is placed: system prompt, next
  to the request, or repeated at the end of the context on every step. Add a
  checklist the agent must pass before it may finish. Supported by B2, B4–B6,
  B8–B10 and our narrow-rule observation.
- **Zero-cost baseline that keeps the user's requests verbatim.** Nothing
  shows generated practice beating real experience (A10); keeping raw text
  helped elsewhere (A5); user revisions should stay apart from inferred
  lessons (A11).
- **Dreaming, second version.** Exercise output is schema-constrained (B13,
  our 6 of 6), with code-side filters afterwards (A6). An exercise is used
  only if an agent shown the user's original words passes its check, as a
  positive control (A6, A7, C9). Rewrites edit one rule at a time (A3, A4).
  The user's explicit decisions cannot be weakened or narrowed (A11).
- **Evaluation.** Three final sessions per seed, analysed as per-seed pairs
  (C6, C8). Each result states what effect size the sample could detect,
  using C7. Our arithmetic, inverting C7: δ ≈ 2.8·√(q/n), so with 20
  independent seed pairs and q = 0.5 only a gap of about 44 points is
  detectable. The three sessions per seed reduce noise but do not add
  independent pairs.
- **One model.** From 2026-11-16 experiments run only on Qwen 3.8 27B
  (IQ2_XS). This is the owner's decision, not a literature finding; B14 does
  not settle the 9B-versus-27B question.

## Considered, not adopted yet

- **Lint-on-write.** C1 supports it, but only on GPT-4 Turbo.
- **Masking old tool output instead of dropping steps.** C3 is promising, but
  measured on much longer trajectories.
- **A "which rules apply?" call before editing.** Only indirect support (B10)
  and one extra call per session.
- **Selective rule retrieval.** Nothing in B12 shows that picking among 8
  short rules helps, and a selector may miss the implicit trigger.
- **Parametric adapter training.** The reported QLoRA floor for a 27B (22 GB,
  A12) exceeds this 16 GB GPU, and training cannot run through the
  inference-only endpoint.

## Open questions the literature does not answer

- Are 900 characters enough for reliable rule application, and which note
  format is best at that size?
- Does practice help a frozen weak model without a stronger reflector?
- Does generated or interleaved practice beat real replay or rereading at
  equal total compute?
- Which stage fails on the 9B: abstraction, priority, unit conversion or
  editing?
- How does low-bit quantisation change rule application, and how does a 27B
  at very low bits compare with a 9B at Q8 under one budget?
- Where should memory sit during multi-turn coding with a rolling history?
- What abstraction level, scope wording and negative examples make a rule
  transfer to new settings?
- Does an applicability-planning call beat an equally costly fresh attempt?
- Can executable filtering remove incomplete instructions and answer leakage?
- Which tool set and practice curriculum suit this exact setting?

## Sources

† dated after mid-2026; as reported, not vouched for here.

1. Behrouz et al. *Language Models Need Sleep: Learning to Self-Modify and
   Consolidate Memories*. 2026. https://arxiv.org/abs/2606.03979
2. Lin et al. *Sleep-time Compute: Beyond Inference Scaling at Test-time*.
   2025. https://arxiv.org/abs/2504.13171
3. Zhao et al. *ExpeL: LLM Agents Are Experiential Learners*. 2023.
   https://arxiv.org/abs/2308.10144
4. Suzgun et al. *Dynamic Cheatsheet: Test-Time Learning with Adaptive
   Memory*. EACL 2026. https://aclanthology.org/2026.eacl-long.333.pdf
5. Zhang et al. *Agentic Context Engineering: Evolving Contexts for
   Self-Improving Language Models*. 2025. https://arxiv.org/abs/2510.04618
6. Chhikara et al. *Mem0: Building Production-Ready AI Agents with Scalable
   Long-Term Memory*. 2025. https://arxiv.org/abs/2504.19413
7. Shinn et al. *Reflexion: Language Agents with Verbal Reinforcement
   Learning*. 2023. https://arxiv.org/abs/2303.11366
8. Wei et al. *Evo-Memory: Benchmarking LLM Agent Test-time Learning with
   Self-Evolving Memory*. 2025. https://arxiv.org/abs/2511.20857
9. Xu et al. *FlexA-Mem: Budget-Adaptive Agentic Memory for LLM Agents*.
   2026†. https://arxiv.org/abs/2610.33274
10. Zhou et al. *Self-Challenging Language Model Agents*. 2025.
    https://arxiv.org/abs/2506.01716
11. Chen et al. *CodeT: Code Generation with Generated Tests*. 2022.
    https://arxiv.org/abs/2207.10397
12. Yang et al. *SWE-smith: Scaling Data for Software Engineering Agents*.
    2025. https://arxiv.org/abs/2504.21798
13. Agrawal et al. *GEPA: Reflective Prompt Evolution Can Outperform
    Reinforcement Learning*. 2025. https://arxiv.org/abs/2507.19457
14. Yuksekgonul et al. *TextGrad: Automatic "Differentiation" via Text*.
    2024. https://arxiv.org/abs/2406.07496
15. Yang et al. *Large Language Models as Optimizers*. 2023.
    https://arxiv.org/abs/2309.03409
16. Giummarra et al. *Understanding When Reflective Prompt and Skill
    Optimization Helps on Locally Served Reasoning Models: A Two-Model Study
    of GEPA and SkillOpt*. 2026†. https://arxiv.org/abs/2610.26020
17. Sun et al. *LAMOL: LAnguage MOdeling for Lifelong Language Learning*.
    2019. https://arxiv.org/abs/1909.03329
18. Graffieti et al. *Generative Negative Replay for Continual Learning*.
    2022. https://arxiv.org/abs/2204.05842
19. Srivastava and He. *MemoryGraft: Persistent Compromise of LLM Agents via
    Poisoned Experience Retrieval*. 2025. https://arxiv.org/abs/2512.16962
20. Karunanidhi. *Utility Under Attack: Agent Memory Poisoning and the Limits
    of Content Screening and Provenance Ranking*. 2026†.
    https://arxiv.org/abs/2608.21230
21. Zweiger et al. *Self-Adapting Language Models*. 2025.
    https://arxiv.org/abs/2506.10943
22. Unsloth maintainers. *Unsloth Requirements*. Documentation, accessed
    2026-11.
    https://unsloth.ai/docs/get-started/fine-tuning-for-beginners/unsloth-requirements
23. Yao et al. *τ-bench: A Benchmark for Tool-Agent-User Interaction in
    Real-World Domains*. 2024. https://arxiv.org/abs/2406.12045
24. Qi et al. *AgentIF: Benchmarking Instruction Following of Large Language
    Models in Agentic Scenarios*. 2025. https://arxiv.org/abs/2505.16944
25. Harada et al. *When Instructions Multiply: Measuring and Estimating LLM
    Capabilities of Multiple Instructions Following*. 2025.
    https://arxiv.org/abs/2509.21051
26. Liu et al. *Lost in the Middle: How Language Models Use Long Contexts*.
    2023. https://arxiv.org/abs/2307.03172
27. Zhang et al. *IHEval: Evaluating Language Models on Following the
    Instruction Hierarchy*. 2025. https://arxiv.org/abs/2502.08745
28. Leviathan et al. *Prompt Repetition Improves Non-Reasoning LLMs*. 2025.
    https://arxiv.org/abs/2512.14982
29. Ji. *Context Engineering for AI Agents: Lessons from Building Manus*.
    2025.
    https://manus.im/blog/Context-Engineering-for-AI-Agents-Lessons-from-Building-Manus
30. Qwen Team. *Qwen3 concepts and chat templates*. Documentation, n.d.
    https://github.com/QwenLM/Qwen3/blob/main/docs/source/getting_started/concepts.md
31. LM Studio. *Structured Output*; *Tool Use*. Documentation, n.d.
    https://lmstudio.ai/docs/developer/openai-compat/structured-output ;
    https://lmstudio.ai/docs/developer/openai-compat/tools
32. Fu et al. *LLMs Learn Better In-Context from Rules than from Examples*.
    2026†. https://arxiv.org/abs/2609.03213
33. Cook et al. *TICKing All the Boxes: Generated Checklists Improve LLM
    Evaluation and Generation*. 2024. https://arxiv.org/abs/2410.03608
34. Wang et al. *Plan-and-Solve Prompting: Improving Zero-Shot
    Chain-of-Thought Reasoning by Large Language Models*. ACL 2023.
    https://aclanthology.org/2023.acl-long.147/
35. Zhang et al. *Small Language Models Need Strong Verifiers to
    Self-Correct Reasoning*. 2024. https://arxiv.org/abs/2404.17140
36. Weston and Sukhbaatar. *System 2 Attention (is something you might need
    too)*. 2023. https://arxiv.org/abs/2311.11829
37. Modarressi et al. *NoLiMa: Long-Context Evaluation Beyond Literal
    Matching*. 2025. https://arxiv.org/abs/2502.05167
38. ggml-org contributors. *llama.cpp grammar guide*. Documentation, n.d.
    https://github.com/ggml-org/llama.cpp/blob/master/grammars/README.md
39. Geng et al. *Generating Structured Outputs from Language Models:
    Benchmark and Studies* (JSONSchemaBench). 2025.
    https://arxiv.org/abs/2501.10868
40. Lee et al. *Exploring the Trade-Offs: Quantization Methods, Task
    Difficulty, and Model Size in Large Language Models From Edge to Giant*.
    IJCAI 2025. https://www.ijcai.org/proceedings/2025/0902.pdf
41. Jang et al. *Flat Score, Amplified Failures: How the Error Budget Masks
    Damage in Quantized LLM Agents*. 2026†. https://arxiv.org/abs/2607.27275
42. Yang et al. *SWE-agent: Agent-Computer Interfaces Enable Automated
    Software Engineering*. 2024. https://arxiv.org/abs/2405.15793
43. Gauthier (Aider). *Code Editing Leaderboard*. n.d.
    https://aider.chat/docs/leaderboards/edit.html
44. Lindenbauer et al. *The Complexity Trap: Simple Observation Masking Is as
    Efficient as LLM Summarization for Agent Context Management*. 2025.
    https://arxiv.org/abs/2508.21433
45. Hong et al. (Chroma). *Context Rot: How Increasing Input Tokens Impacts
    LLM Performance*. 2025. https://www.trychroma.com/research/context-rot
46. Sun et al. *Scaling Long-Horizon LLM Agent via Context-Folding*. 2025.
    https://arxiv.org/abs/2510.11967
47. llama.cpp contributors. *llama-server README*. Documentation, n.d.
    https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md
48. LM Studio bug tracker. *MLX Backend Ignores the Seed Parameter, Issue
    #2399*. 2026†.
    https://github.com/lmstudio-ai/lmstudio-bug-tracker/issues/2399
49. Miller. *Adding Error Bars to Evals: A Statistical Approach to Language
    Model Evaluations*. 2024. https://arxiv.org/abs/2411.00640
50. statsmodels developers. *McNemar Test*. Documentation, n.d.
    https://www.statsmodels.org/stable/generated/statsmodels.stats.contingency_tables.mcnemar.html
51. UK AI Security Institute / Inspect contributors. *Scoring Metrics*.
    Documentation, n.d. https://inspect.aisi.org.uk/metrics.html
52. statsmodels developers. *Multiple-Testing Corrections*. Documentation,
    n.d.
    https://www.statsmodels.org/stable/generated/statsmodels.stats.multitest.multipletests.html
53. Zhu et al. *Establishing Best Practices for Building Rigorous Agentic
    Benchmarks*. 2025. https://arxiv.org/abs/2507.02825
54. OpenAI, with the SWE-bench authors. *Introducing SWE-bench Verified*.
    2024. https://openai.com/index/introducing-swe-bench-verified/
55. Kapoor et al. *Holistic Agent Leaderboard: The Missing Infrastructure for
    AI Agent Evaluation*. 2025. https://arxiv.org/abs/2510.11977
56. Wu et al. *LongMemEval*. 2024. https://arxiv.org/abs/2410.10813
57. Hu et al. *Evaluating Memory in LLM Agents via Incremental Multi-Turn
    Interactions* (MemoryAgentBench). 2025.
    https://arxiv.org/abs/2507.05257
58. Shen et al. *Mem2ActBench*. 2026. https://arxiv.org/abs/2601.19935
