# Agent layer (W6)

The W4–W5 system was a **fixed pipeline**: every question ran `retrieve → (rerank) →
generate`, always. W6 adds an **agent** — an LLM that *decides* whether and how to
retrieve, by calling tools. A simple lookup should still be one retrieval; a
"compare A and B" question can fan out to several. The control flow is chosen at
runtime, not hard-wired.

Both engines are reachable through one entrypoint, `src/answer.py :: answer()`,
selected by the `SERVING.use_agent` switch (see [Unified entrypoint](#unified-entrypoint--regression)).

## The ReAct graph

Built with **LangGraph** (`src/agent/graph.py`). One turn = the LLM either calls a
tool or emits a final answer; tool results are appended and fed back until it stops.

```
(START) ─▶ agent ──tool_calls?── yes ─▶ tools ─┐
             ▲                                  │
             └────────── ToolMessage(s) ────────┘
             │
             └── no tool_calls ─▶ (END)   final cited answer
```

Four LangGraph concepts carry the whole agent:

| Concept | Role here |
|---|---|
| **State** (`MessagesState`) | a `messages` list with the `add_messages` reducer — **appends** each LLM request + tool result, so the agent always sees the full trace. |
| **Node `agent`** | one LLM call with the tools bound; emits `tool_calls` or a final answer. The agent-specific system prompt (grounding + citation rules **+** the D6 tool-use policy) is prepended fresh each turn. |
| **Node `tools`** | a prebuilt `ToolNode` that runs the requested tool(s) and appends each result as a `ToolMessage`. |
| **Conditional edge** after `agent` | last message has `tool_calls` → `tools`, else → `END`. **This edge is the ReAct "keep going or stop" switch.** |

## Tools

The LLM picks a tool purely from its **name + argument schema + docstring**, so each
tool is single-purpose with a distinct docstring (`src/agent/tools.py`).

| Tool | Responsibility |
|---|---|
| `search_docs(query)` | Semantic search of the vLLM docs; returns numbered, source-tagged excerpts to answer + cite from. The agent's equivalent of the fixed pipeline's retrieval. |
| `list_sources(topic)` | Which doc **pages** cover a topic (scope / "is this even answerable?"), with a hit count per page. No excerpt text — a one-shot lookup. |
| `compare(a, b)` | Retrieve two concepts under one continuous `[n]` numbering so the LLM can synthesize a side-by-side comparison. The tool that makes the agent genuinely **multi-step**. |
| `estimate_cost` | Deferred (in the design doc, not yet wired). |

## Guardrails

An agent hands control to a free-running LLM, so it needs deterministic safety edges.
Four are implemented (config in `AgentConfig`, logic in `graph.py` / `tools.py`;
fault-injection tests in `eval/verify_guardrails.py`):

1. **Step cap → graceful fallback.** `recursion_limit=8` bounds a non-converging
   loop; `run_agent()` catches `GraphRecursionError` and returns an honest fallback
   answer (`degraded=True`) instead of a traceback. `steps` (tool-call rounds) is
   logged as a cheap health signal — a healthy lookup is 1 step.
2. **Tool retry / never-raise.** `_search_with_retry` retries only **transient**
   errors (OpenSearch down/timeout) with linear backoff; `@safe_tool` converts
   *any* escaping exception into a readable `"TOOL ERROR: …"` observation. Principle:
   a tool failure becomes something the LLM can **read and route around**, not a
   crash that kills the graph.
3. **Low-score refusal.** Pure k-NN always returns top-k, so the "miss" signal is a
   low **top-1 score**: below `min_relevance_score=0.82`, retrieval is treated as
   "not in the docs" → refuse rather than ground on noise. The 0.82 threshold is
   **eval-derived** (`eval/analyze_score_threshold.py`): on-topic top-1 min 0.834 /
   median 0.917; off-topic median 0.793 / max 0.854 — 0.82 sits just under the
   on-topic minimum (0 false refusals on the eval set) while rejecting 7/8 off-topic.
4. **Citation check (output guardrail).** `validate_citations()` reconstructs the
   numbered context the LLM was shown and splits the answer's `[n]` into **valid**
   (resolve to a shown source) vs **hallucinated** (out-of-range), and warns when a
   non-refusal answer cites nothing. It **marks**, doesn't rewrite — the caller decides.

## Agent eval (D5)

Agent quality is scored on the **process**, not just the answer — 10 tasks in four
classes (`eval/agent_eval_set.jsonl`), each with `expected_tools` + LLM-judged
`success_criteria` (`eval/eval_agent.py` → `eval/agent_eval.csv`):

| metric | value |
|---|---|
| task success rate | **50%** (5/10) |
| tool-selection accuracy | **70%** (7/10) |
| avg steps (tool rounds) | 3.4 |
| avg latency | 11.3 s |
| degraded (hit step cap) | 3 |
| citation warnings | 2 |

**Key insight:** tool selection is *not* the bottleneck (70% correct, zero
"wrong-tool" failures). The bottleneck is the **ReAct loop over-exploring** — on the
tightest step budget the agent re-searched reworded queries and hit the cap. This
directly motivated the D6 tool-use-policy fix (retrieve once, then answer).

## Trace demo — multi-step planning

`"Compare continuous batching and PagedAttention in vLLM"` — the agent plans
implicitly across **3 tool rounds**, then synthesizes one cited comparison:

```
HumanMessage : Compare continuous batching and PagedAttention in vLLM.
AIMessage    -> wants tool: compare(continuous batching, PagedAttention)   # each side retrieved once
ToolMessage  <- compare: [1] (source: …) … [6] (source: …)                 # two numbered groups
AIMessage    -> wants tool: search_docs(...) , search_docs(...)            # deepen each concept
ToolMessage  <- search_docs: [1] (source: …) …
AIMessage    -> wants tool: search_docs(overview)                          # one more for framing
ToolMessage  <- search_docs: [1] (source: …) …
AIMessage    : Continuous batching schedules … [1][3]; PagedAttention manages the KV cache … [4][6]. …
```

Contrast: a *scope* question (`"Which docs cover quantization?"`) converges in **1
round** (`list_sources`). The agent matches effort to the question — the point of
letting the LLM decide.

## Unified entrypoint + regression

`answer()` returns one shape for both engines —
`{answer, citations, tool_trace, steps, contexts, mode, degraded}` — so the
frontend / eval / monitoring never special-case which engine ran.

Before trusting the agent as the default, a **regression** ran both engines over the
36-question eval set (`eval/compare_agent_vs_rag.py`, RAGAS four metrics + refusal
accuracy):

| metric | fixed RAG | agent | Δ |
|---|---|---|---|
| faithfulness | 0.723 | 0.695 | −0.027 |
| answer_relevancy | 0.697 | 0.639 | −0.058 |
| context_precision | 0.776 | 0.525 | **−0.251** |
| context_recall | 0.708 | 0.616 | −0.093 |
| refusal_acc | 0.833 | 0.778 | −0.056 |

A tool-use-policy prompt fix cut avg tool rounds **8.0 → 2.8** and non-convergence
from 3/3 (smoke) to **8/36**. The agent reaches near-parity on faithfulness /
answer_relevancy / refusal_acc (within noise at n=36) but **still degrades** on
context_precision and leaves **22% of questions** at the step-cap fallback.

**Decision:** the runtime default stays **`use_agent=False`** (fixed RAG); the agent
is the architectural entrypoint but ships **behind the flag** until convergence
improves. *Changing the architecture needs a regression test — and if it degrades,
you don't flip the default.*
