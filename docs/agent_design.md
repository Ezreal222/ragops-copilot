# Agent Design — RAGOps Copilot (W6)

> **Status:** design (W6 D1). This is the blueprint for D2–D6, not code.
> **Goal of W6:** evolve the system from a fixed *retrieve → generate* RAG pipeline
> into an **agent** that decides which tools to call, in what order, and when to
> stop — with fallbacks and a measured success rate.

## Why an agent at all (and when *not* to)

Today's pipeline (`src/generate.ask()`) is a **single, fixed path**: embed the
question → k-NN retrieve → (optional rerank) → LLM answers with `[n]` citations.
That is the right design for a plain FAQ-style question, and we should keep using
it directly for those — it is simpler, cheaper, and lower-latency than an agent.

An **agent** earns its keep only when a task needs *multiple steps, multiple
tools, or a decision the LLM has to make at runtime* — e.g. "compare X and Y"
(two retrievals + a synthesis), or "you searched and got nothing, now what?".
The agent = the LLM **decides** the control flow (which tool, what args, when to
finish); the outer code only **executes** what it asks for and feeds results back.

So the design rule for this project: **the agent's default action is still
`search_docs`, and for a simple lookup its first tool call should behave exactly
like today's RAG.** We are adding decision-making on top, not replacing retrieval.

---

## 1. Tool inventory

Each tool is a plain Python function with a typed signature and a docstring; we
expose it to the LLM via LangGraph / LangChain `bind_tools`, which turns the
signature + docstring into the JSON schema the model sees (name / args /
description). All tools are **read-only** — there are no writes, which keeps the
guardrail surface small.

### ① `search_docs` — the agent's main weapon
Wraps the existing retrieval (and optional rerank) from `src/retrieve.search()`.

| | |
|---|---|
| **name** | `search_docs` |
| **input** | `query: str`, `k: int = 5` (chunks to return) |
| **output** | `list[{score, chunk_id, title, source, section, text}]` — the exact dicts `retrieve.search()` already returns |
| **description** | "Search the vLLM documentation for chunks relevant to a query. Returns the top-k most relevant doc excerpts with their source URLs. Use this to find grounding for any factual question about vLLM." |
| **backed by** | `src/retrieve.search()` (bi-encoder k-NN; rerank off per W5 D6 ablation) |

Note: the agent gets **chunks**, not a finished answer — it reasons over them and
decides whether to search again or answer. Final answer generation reuses the
`src/generate.py` system prompt (answer only from context, cite `[n]`, else refuse).

### ② `list_sources` — scope / grounding
Lists the distinct doc pages in the index so the agent can see what corpus exists
(useful for "which pages cover X?" and for honest "not in docs" reasoning).

| | |
|---|---|
| **name** | `list_sources` |
| **input** | *(none)*, optional `contains: str = ""` substring filter |
| **output** | `list[{source, title, n_chunks}]` |
| **description** | "List the vLLM documentation pages available in the index, with how many chunks each has. Use to see what topics the corpus covers before deciding whether a question is answerable." |
| **backed by** | an OpenSearch terms aggregation on the `source` field (`src/opensearch_client.get_client()`) |

### ③ `compare` — the multi-step driver
Compares two vLLM concepts. This is the tool that makes the agent genuinely
*plan*: it decomposes into two `search_docs` calls + a structured synthesis. It is
the showcase task for D3 (multi-step) and for the agent eval in D5.

| | |
|---|---|
| **name** | `compare` |
| **input** | `concept_a: str`, `concept_b: str` |
| **output** | `{a_chunks, b_chunks}` (retrieved context for each side); the LLM then synthesizes a cited comparison |
| **description** | "Retrieve documentation for two vLLM concepts so they can be compared side by side. Use when the user asks how two things differ or which to use." |
| **backed by** | two `search_docs` calls |

> Design choice: `compare` returns the two context sets and lets the **agent's LLM
> node** do the synthesis (staying grounded + cited), rather than baking a second
> LLM call inside the tool. Keeps all generation in one place and one prompt.

### ④ `estimate_cost` — bridge to Project 2 (vLLM serving)
A small utility that estimates token count and rough API cost for a piece of text.
Low RAG relevance, but it ties into the Project-1 → Project-2 narrative ("first the
RAG assistant, then the infra that serves/optimizes vLLM") and gives the agent a
non-retrieval tool so tool-*selection* becomes a real decision, not a formality.

| | |
|---|---|
| **name** | `estimate_cost` |
| **input** | `text: str`, `model: str = "deepseek-v4-pro"` |
| **output** | `{n_tokens, usd}` |
| **description** | "Estimate the token count and approximate USD cost of sending a piece of text to a given model. Use for questions about how expensive or long a prompt/answer is." |
| **backed by** | `tiktoken` (or the model's tokenizer) + a small price table |

---

## 2. Agent type + LangGraph graph

We build in two steps, reusing the **same** four concepts (state / node / edge /
graph) so D3 is an extension of D2, not a rewrite.

### D2 — single-step tool agent
The classic minimal ReAct loop: the LLM either calls **one** tool or answers.
For a simple lookup it should call `search_docs` once and then answer — i.e. it
reproduces today's RAG, but the *decision* to search is now the model's.

### D3 — multi-step planning + tool loop
Same graph, but the conditional edge lets the loop run **multiple** times: the LLM
can call `search_docs`, look at the result, call `compare` or `search_docs` again,
and only then answer. This is where `compare` and multi-hop questions pay off.

### The graph (identical shape for D2 and D3)

```
                 ┌──────────────────────────────────────┐
                 │                                       │
                 ▼                                       │
   (START) ──► ┌──────────┐   tool_calls?   ┌──────────┐ │  ToolMessage(s)
               │  agent   │ ───── yes ─────► │  tools   │ ┘  fed back into state
               │ (LLM node│                 │  node    │
               │  w/ tools│                 │(executes │
               │  bound)  │                 │ the call)│
               └──────────┘                 └──────────┘
                    │
                    └───────── no tool_calls ──────────► (END)  final cited answer
```

- **State** — a `messages` list (LangGraph `MessagesState`): the running
  conversation, including the LLM's tool-call requests and each tool's result
  (`ToolMessage`). Everything the agent "knows" lives here. We may add a
  `step_count` field for the max-steps guardrail.
- **Node `agent`** — one LLM call with the four tools bound. It either emits
  `tool_calls` (wants to act) or a plain final answer (done).
- **Node `tools`** — executes the requested tool(s) and appends the result(s) to
  `messages` (a `ToolNode`).
- **Conditional edge** — after `agent`: if the last message has `tool_calls` →
  go to `tools`; else → `END`. This is the ReAct "reason → act → observe → repeat"
  loop expressed as a graph.
- **Edge `tools → agent`** — always loops back so the LLM can observe the tool
  output and decide the next step.

Mapping to **ReAct**: `agent` node = *Reason*, `tools` node = *Act*, the
`ToolMessage` appended to state = *Observe*, the loop edge = *repeat until done*.

---

## 3. Guardrails / fallback plan (D4)

The whole point of "production-grade" is that the agent fails *safely*. Planned
guardrails, each mapped to a concrete failure mode:

| Failure mode | Guardrail |
|---|---|
| Infinite / runaway tool loop | **Max-steps cap** (e.g. ≤ 6 tool calls via a `step_count` in state or LangGraph `recursion_limit`); on hit → force a final answer or refuse. |
| Tool raises / times out | **Retry once**, then return a structured error `ToolMessage` so the LLM can recover or refuse — never crash the graph. |
| Retrieval returns nothing (or low scores) | **Fallback to refusal**: reuse `generate.py`'s exact "I couldn't find this in the vLLM docs." rather than letting the model invent an answer. |
| Answer not grounded / missing citations | **Output validation**: check the final answer has `[n]` markers that resolve to real retrieved chunks (reuse `_cited_sources()` logic); if not, downgrade to refusal. |
| Off-topic / abusive input | Keep the system prompt's "only answer from vLLM docs" scope; out-of-scope → refuse. |

The refusal + citation logic already exists in `src/generate.py`; D4 is mostly
*wiring the agent to honor it*, plus the loop cap.

---

## 4. Agent eval plan (D5)

We keep the W5 retrieval/RAGAS eval as-is and add an **agent-level** eval that
measures whether the agent *behaves* correctly, not just whether retrieval recalls.

- **Task set** — a small set (~8–12) of tasks that genuinely need the agent, each
  labeled with the tool(s) it *should* use:
  - simple lookup → expects `search_docs` (should behave like RAG)
  - comparison ("X vs Y") → expects `compare` / two searches
  - "what pages cover X?" → expects `list_sources`
  - "how many tokens is this prompt?" → expects `estimate_cost`
  - an out-of-docs question → expects **refusal** (fallback path)
- **Success criteria (per task):**
  1. **Tool choice** — did it call the expected tool(s)? (inspect the tool-call
     trace in state)
  2. **Final correctness** — is the final answer right + grounded + cited (or a
     correct refusal)? Judged via the W5 LLM-judge / a reference answer.
- **Headline metric:** **task success rate** = fraction of tasks passing *both*
  criteria. Report it the way W5 reports recall@k. Also log avg tool-calls/task
  (efficiency) and refusal-correctness.
- Yang writes the gold task set + expected tools (like the W5 eval set); D5
  scaffolds the harness that runs the agent and scores it.

---

## 5. D2–D6 schedule

| Day | Deliverable |
|---|---|
| **D1** (today) | Agent design doc (this file) + concepts in `notes/w6.md`. |
| **D2** | Single-step tool agent: `search_docs` + `list_sources` as tools, minimal LangGraph ReAct graph, runs end-to-end on a lookup. |
| **D3** | Multi-step planning loop: add `compare` + `estimate_cost`, let the conditional edge loop; demo a multi-hop / comparison question. |
| **D4** | Guardrails: max-steps cap, tool retry, empty-retrieval → refusal, citation/output validation. |
| **D5** | Agent eval harness: ~8–12 agent tasks + expected tools; report **task success rate** (+ avg tool-calls, refusal correctness). |
| **D6** | Wire the agent back into the end-to-end RAG entry point; final W6 write-up + metrics in `notes/w6.md`. |

---

## Code touch-points (for D2 onward)

- `src/retrieve.search()` → backs `search_docs` (already returns chunk dicts).
- `src/generate.py` (SYSTEM_PROMPT, `_cited_sources`, refusal string) → reused for
  the agent's final answer + citation validation + fallback.
- `src/opensearch_client.get_client()` → backs `list_sources` aggregation.
- `src/config.py` → add an `AgentConfig` (max_steps, tool timeout/retries) when D4
  lands, keeping the "all knobs in one config" convention.
- New file likely `src/agent.py` (graph + tools) in D2.
