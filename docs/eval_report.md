# Evaluation Report — RAGOps Copilot (W5)

> A reproducible, eval-driven assessment of the RAG pipeline over the vLLM docs.
> This report consolidates a week of work: a RAGAS baseline, a hand-built LLM-as-judge
> cross-check, and two single-variable ablation studies (chunking, then retrieval/rerank)
> that turned retrieval quality into **production decisions** — including a clean **negative
> result** (the reranker is not worth it here).
>
> Headline: eval-driven tuning lifted **context_recall** from a **0.69** answerable-only
> baseline to **0.79** at the production default (k=5), and to **0.82** at the documented
> recall-max setting (k=10) — while *shrinking* the index from 2,700 → 1,783 chunks.

---

## 1. Method

**Eval set** — `eval/eval_set.jsonl`, **36 questions** (32 answerable + 4 no-answer), validated
by `eval/validate_eval_set.py`. Each question carries:

- `gold_chunk_ids` — multiple gold chunks (a doc is split into several valid chunks; single
  `::0` matching produced false negatives in W4).
- `gold_sources` — document-level gold (stable across re-chunking; `chunk_id`s are not).
- `reference` — a one-line-to-one-paragraph reference answer (RAGAS uses it for context_recall).

Coverage: 10 doc areas (features 11, getting_started 5, serving 4, configuration 4, …); average
1.56 gold chunks / 1.09 gold sources per question. The **4 no-answer questions** (text-to-image,
TTS audio, pricing/subscription, built-in vector DB — all confirmed absent from the corpus) test
**refusal / anti-hallucination**.

> Honesty note: gold and reference text were *drafted* by an LLM from the real chunk text (not
> invented) and still warrant human review — especially `reference`, since RAGAS scores DeepSeek's
> answers against it.

**Metrics.**

| Layer | Metric | Measures | Needs |
|---|---|---|---|
| Retrieval | recall@k (doc-level) | gold doc within top-k | gold_sources |
| Retrieval | context_precision | relevance + ordering of retrieved context | question, contexts, reference |
| Retrieval | context_recall | did we retrieve the info the answer needs | contexts, reference |
| Generation | faithfulness | answer grounded in retrieved context (anti-hallucination) | response, contexts |
| Generation | answer_relevancy | answer on-topic | question, response |

Mnemonic: **precision/recall judge retrieval; faithfulness/answer_relevancy judge generation.**

**Judge & reproducibility.** RAGAS and the custom judge both use the **same** model —
DeepSeek `deepseek-v4-flash`, **temperature 0** — so a score difference reflects the
**rubric**, not the model. Embeddings are local `bge-small`. The expensive `ask()` outputs are
**frozen once** into `eval/ragas_inputs.jsonl`, so re-scoring never re-runs generation and every
ablation compares apples to apples.

---

## 2. RAGAS baseline (D3)

36-question mean, on the original 800/100 corpus:

| metric | mean |
|---|---|
| faithfulness | 0.788 |
| answer_relevancy | 0.772 |
| context_precision | 0.733 |
| **context_recall** | **0.532** |

**context_recall is the weakest link → the bottleneck is retrieval coverage.** The questions
that scored 0 across *all four* metrics (Q1 continuous batching, Q4 throughput/TTFT/p99,
Q29 logprobs) were **refused at generation time** — top-5 never retrieved a chunk that could
answer them, so every downstream metric collapsed. This is the W4 lesson restated: **recall is
the ceiling** — the LLM can only answer from what was retrieved. The 4 no-answer questions were
correctly refused (anti-hallucination works).

> The 0.532 figure is over all 36 questions *including the 4 refusals*, which drag context_recall
> down (a refusal retrieves nothing that covers its reference). The ablations below isolate the
> **32 answerable** questions, where the comparable 800/100 baseline is **0.688** — that is the
> honest starting point for the tuning story.

**How each low score is fixed** (interview shorthand): low faithfulness → tighten prompt / lower
temp / better retrieval; low context_recall → chunking / embedding / reranker / larger k (the D5–D6
target); low context_precision → rerank / smaller k / filter.

---

## 3. Judge agreement — custom LLM-as-judge vs RAGAS (D4)

`eval/llm_judge.py` is a from-scratch **reference-free faithfulness** judge with four reliability
choices baked in: **reason-before-score (CoT)**, a **discrete 1–5 rubric**, **structured JSON
output** with fault-tolerant parsing, and **temperature 0**. Refusals are handled in code (exact
match on the refusal string → score 5, no API call) — a **deliberate** point of divergence from
RAGAS.

| | custom judge | RAGAS |
|---|---|---|
| faithfulness mean | 0.917 | 0.788 |
| Spearman ρ (all 36) | 0.469 | |
| **Spearman ρ (refusals removed)** | **0.811** | |

**What the disagreement means (the real point):**

- **Every largest disagreement is a refusal** (custom = 1.0, RAGAS ≈ 0.0; 5 questions). Systematic,
  not error: I judge a "not in the docs" refusal as **trivially faithful** (no claims → no
  hallucination); RAGAS's supported-claims-over-total-claims collapses to ~0 when there are no
  extractable claims. **Neither is wrong** — they define "faithfulness of a refusal" differently.
  Remove the refusals and the two judges **strongly agree on real answers (ρ = 0.81)**.
- **PagedAttention is the one real answer both judges score low** (custom 0.25 / RAGAS 0.33) — a
  credible, genuine grounding gap (top-5 likely missed the defining chunk), i.e. real signal for the
  D5/D6 ablations, not rubric noise.
- **Judge instability, caught live:** PagedAttention scored 4 in a smoke run and 2 in the full run —
  **both at temp 0.** Concrete evidence that an LLM judge is sensitive to prompt/sampling and still
  needs **human spot-checks**.

**Why build both?** RAGAS gives fast, standardized metrics; the custom judge proves you *understand*
the scoring (can explain every point, customize dimensions) rather than treating a library as a
black box. ρ = 0.81 says the judge is trustworthy for **relative** comparison (ablations); absolute
scores and boundary cases still need human calibration.

---

## 4. Ablation 1 — chunking (D5)

Goal: raise the weak context_recall by data-driven decision. **Only chunk_size/overlap vary** —
same embedding, same k, **reranker off**, same frozen eval set, same judge — so every delta is
attributable to chunking. Main metric is **doc-level** recall (chunk_ids change every re-chunk;
gold *documents* are stable). Generation is not run (context metrics need only retrieved contexts +
reference); the 4 refusals are dropped (recall is undefined for them).

Hypothesis: low recall = answers split across chunk boundaries / chunks too fine → **bigger chunks
(C) or more overlap (D) should raise context_recall.**

**32 answerable questions, reranker off:**

| group | size/overlap | chunks | recall@1 | @3 | @5 | context_recall | context_precision | latency |
|---|---|---|---|---|---|---|---|---|
| A baseline | 800/100 | 2700 | 0.562 | 0.812 | **0.906** | 0.688 | 0.691 | 15 ms |
| B small | 400/50 | 5866 | 0.375 | 0.688 | 0.750 | 0.438 | 0.522 | 11 ms |
| **C large** | **1200/200** | **1783** | **0.594** | 0.781 | 0.844 | **0.786** | **0.758** | 11 ms |
| D more overlap | 800/200 | 2862 | 0.594 | 0.750 | 0.844 | 0.703 | 0.687 | 12 ms |

**Decision — adopt C (1200/200):**

- context_recall **0.688 → 0.786 (+14%)** *and* context_precision **0.691 → 0.758** rise
  **together** (normally recall↑ costs precision↓ — here it's a win-win), recall@1 is highest, and
  the index is **smallest** (1,783 vs 2,700 → cheaper).
- The driving variable is **chunk_size, not overlap**: overlap-only D barely moved (0.688 → 0.703);
  small-block B **collapsed on everything** (confirms over-fine chunks lose context). Hypothesis
  direction confirmed.
- **The one trade-off:** C's doc-recall@5 drops 0.906 → 0.844 (bigger chunks → fewer *distinct*
  docs in top-5). But for citation RAG, **context_recall/precision are the target-aligned signals**
  (do the retrieved contexts actually contain what the answer needs = the ceiling on faithful
  generation), and both rose. The small recall@5 regression is acceptable. → `config.py` default
  changed to 1200/200, index re-ingested. (`eval/ablation_chunking.csv`)

---

## 5. Ablation 2 — retrieval / rerank (D6, the week's deliverable)

On the D5-optimal 1200/200 corpus (**fixed**, no re-ingest), only **query-time** knobs vary, one at
a time: top-k (3/5/10), rerank (off/on), rerank top-N (20/50/100). Retrieval metrics run on all 6;
generation metrics run only on 3 representative configs (baseline / +rerank / k=10) to bound LLM cost.

Two recall lenses (don't conflate): **recall@1** = gold doc ranked first (measures *ordering* — what
rerank should move); **recall@k** = gold doc within the k chunks fed to the LLM (measures *coverage* —
known near-saturated from W4).

Hypothesis (from W4): coverage near-saturated → the reranker mainly improves ordering, not coverage;
if it doesn't even improve ordering, that reproduces "rerank isn't worth it when recall is saturated."

**Before/after table (32 answerable questions):**

| config | k | rerank | N | recall@1 | recall@k | ctx_prec | ctx_recall | faith | ans_rel | latency |
|---|---|---|---|---|---|---|---|---|---|---|
| **baseline** | 5 | off | – | 0.594 | 0.844 | **0.777** | 0.786 | 0.844 | 0.829 | 19 ms |
| k=3 | 3 | off | – | 0.594 | 0.781 | 0.763 | 0.625 | – | – | 9 ms |
| k=10 | 10 | off | – | 0.594 | **0.969** | 0.712 | **0.823** | **0.863** | **0.865** | 11 ms |
| +rerank N=50 | 5 | on | 50 | 0.438 | 0.906 | 0.700 | 0.703 | 0.786 | 0.832 | 94 ms |
| +rerank N=20 | 5 | on | 20 | 0.438 | 0.906 | 0.753 | 0.719 | – | – | 40 ms |
| +rerank N=100 | 5 | on | 100 | 0.406 | 0.875 | 0.682 | 0.672 | – | – | 146 ms |

(`recall@k` is at each config's own k, so the k=3 row is recall@3; `–` = generation metrics run only
on the 3 representative configs.) (`eval/ablation_retrieval.csv`)

**Decisions:**

- **Reranker OFF — a clean negative result.** The cross-encoder lost on *every* metric — recall@1
  0.594 → 0.438, context_precision 0.777 → 0.700, context_recall 0.786 → 0.703, faithfulness
  0.844 → 0.786 — at **5–8× the latency** (19 ms → 94/146 ms). A **deeper** candidate pool (N=100)
  was **strictly worst** and slowest: deeper pool = more noise, not more signal. A stronger repeat of
  the W4 finding — on a near-saturated doc corpus, reranking demotes the gold doc rather than
  surfacing new evidence. Removed from the production `ask()` path (was hardcoded `use_reranker=True`;
  now reads `config.use_reranker`, default **False**).
- **top_k = 5 — the balanced default.** Best context_precision (0.777) at half the token cost of
  k=10. **k=10 is documented as the recall-max option** (recall@k 0.844 → 0.969, best end-to-end
  faithfulness/answer_relevancy) for when coverage outweighs the ~2× token cost — but with n=32 the
  faith/ans_rel gains sit within noise, while the 12.5-point recall@k gap (4 more questions covered)
  is real. Safe default k=5, document k=10.
- **k=3 rejected** — context_recall 0.786 → 0.625, cuts too much evidence.

**One instructive metric clash:** rerank's doc-recall@5 *rose* (0.844 → 0.906) while context_recall
*fell* (0.786 → 0.703). doc-recall@5 is coarse ("is the gold doc in top-5", binary); context_recall
is fine ("how much of the reference content is covered"). Rerank reorders by query-chunk relevance and
can surface a *less on-point* chunk of the gold doc, or drop a useful one at the k=5 cut — the document
arrives, but content coverage worsens. Proof that **reading one number misleads**; cross-check several.

---

## 6. Recommended production config

| knob | value | why |
|---|---|---|
| chunk_size / overlap | **1200 / 200** | best context_recall + precision, smallest index (D5) |
| top_k | **5** | best precision, half the token cost of k=10; k=10 documented as recall-max |
| reranker | **off** | lost on every metric at 5–8× latency on a saturated corpus (D6 negative result) |
| judge | DeepSeek `deepseek-v4-flash`, temp 0 | reproducible scoring, frozen inputs |

**Bottom line:** single-variable, eval-driven tuning raised **context_recall** from a **0.69**
answerable-only baseline (0.53 across all 36 incl. refusals) to **0.79** at the k=5 production
default — and **0.82** at the documented k=10 recall-max — while *shrinking* the index (2,700 → 1,783
chunks) and *removing* the reranker's latency. The quality–latency–cost triangle, decided from data
rather than intuition, with the negative result reported in full.

---

## 7. Reproduce

```bash
uv run python eval/validate_eval_set.py            # eval set integrity (36 Q)
uv run python -m eval.run_ragas                    # RAGAS 4-metric baseline → ragas_baseline.csv
uv run python -m eval.llm_judge                    # custom faithfulness judge → judge_faithfulness.csv
uv run python -m eval.compare_judge_ragas          # Spearman agreement
uv run python -m eval.ablation_chunking            # chunk_size/overlap sweep → ablation_chunking.csv
uv run python -m eval.ablation_retrieval           # top_k / rerank / top_N sweep → ablation_retrieval.csv
uv run python -m eval.ablation_retrieval --no-ragas  # doc-recall + latency only (fast, no API cost)
```

All scoring reuses the frozen `eval/ragas_inputs.jsonl`, so results are reproducible without
re-running generation.
