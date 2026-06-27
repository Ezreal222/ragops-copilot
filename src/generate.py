"""LLM answer generation — turn retrieved chunks into a cited, grounded answer.

This is the "generation" half of RAG. Retrieval (retrieve.py) finds the most
relevant doc chunks; here we hand those chunks to an LLM as CONTEXT and ask it
to answer the question USING ONLY that context, citing each claim with [n].

Why "only from the context"? That's the core anti-hallucination lever in RAG:
  1. the system prompt forbids outside knowledge and tells the model to say
     "not found" rather than guess,
  2. requiring [n] citations makes every answer traceable back to a source
     chunk (verifiable — the thing that separates a production RAG from a
     plain chatbot),
  3. (upstream) retrieval quality has to be good enough that the right chunk is
     actually in the context — this is why we measured recall@k first.

Provider-agnostic: we talk to the LLM over the OpenAI-compatible chat API, so
DeepSeek (default), OpenAI, and similar all work by changing src/config.py's
LLMConfig only. See LLMConfig for how to switch.

`generate_answer()` does generation given chunks; `ask()` is the end-to-end
entry point that puts retrieval + rerank (retrieve.search) in front of it and
returns a structured {answer, citations, retrieved_chunks} result.
"""

from __future__ import annotations

import os
import re

from dotenv import load_dotenv
from openai import OpenAI

from src.config import LLM, RETRIEVAL, LLMConfig
from src.retrieve import search

# Load .env so the provider key is available as an env
# var. Safe to call repeatedly; does nothing if there's no .env file.
load_dotenv()

# The system prompt is the quality-critical part of D6. Every clause is a
# deliberate anti-hallucination / citation constraint, not boilerplate.
SYSTEM_PROMPT = (
    "You are RAGOps Copilot, an assistant that answers questions about vLLM "
    "strictly from the provided documentation excerpts.\n\n"
    "Rules:\n"
    "- Answer ONLY using the numbered context below. Do not use any outside "
    "knowledge, even if you think you know the answer.\n"
    '- If the context does not contain the answer, reply exactly: "I couldn\'t '
    'find this in the vLLM docs." Do not guess, infer, or fabricate.\n'
    "- Cite every claim with the bracketed number(s) of the supporting "
    "excerpt(s), e.g. [1] or [2][3].\n"
    "- Be concise and technical, and prefer the docs' own terminology."
)


def get_client(config: LLMConfig = LLM) -> OpenAI:
    """Build an OpenAI-compatible client for the configured provider.

    The same OpenAI SDK drives DeepSeek and OpenAI — only `base_url` + the key
    differ. We read the key from the env var named by `config.api_key_env` so
    no secret is ever hardcoded. An empty `base_url` falls back to the SDK's
    default (OpenAI's endpoint).
    """
    if config.provider == "anthropic":
        raise NotImplementedError(
            "anthropic provider not wired yet — use the `anthropic` SDK here, "
            "or set LLMConfig.provider to 'deepseek'/'openai'."
        )

    api_key = os.environ.get(config.api_key_env)
    if not api_key:
        raise RuntimeError(
            f"{config.api_key_env} is not set. Copy .env.example to .env and "
            f"fill in your {config.provider} key."
        )
    # base_url="" -> pass None so the SDK uses its default endpoint.
    return OpenAI(api_key=api_key, base_url=config.base_url or None)


def format_context(chunks: list[dict]) -> str:
    """Render retrieved chunks as a numbered, source-tagged context block.

    Each chunk becomes one "[n] (source: ...)\\n<text>" entry. The [n] is what
    the model cites, and we keep the same 1-based numbering the prompt asks for,
    so citation [1] maps to chunks[0].
    """
    blocks = []
    for i, c in enumerate(chunks, 1):
        blocks.append(f"[{i}] (source: {c['source']})\n{c['text']}")
    return "\n\n".join(blocks)


def build_messages(question: str, chunks: list[dict]) -> list[dict]:
    """Assemble the chat messages: system rules + context + the question."""
    context = format_context(chunks)
    user = (
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer (cite sources with [n]):"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def generate_answer(
    question: str,
    chunks: list[dict],
    *,
    client: OpenAI | None = None,
    config: LLMConfig = LLM,
) -> str:
    """Ask the LLM to answer `question` using only `chunks`; return the text.

    `client` is injectable so callers (the end-to-end ask(), eval) can build it
    once and reuse it. If no chunks are given there's nothing to ground on, so
    we short-circuit to the honest "not found" answer instead of letting the
    model invent one.
    """
    if not chunks:
        return "I couldn't find this in the vLLM docs."

    client = client or get_client(config)
    resp = client.chat.completions.create(
        model=config.model,
        messages=build_messages(question, chunks),
        max_tokens=config.max_tokens,
        temperature=config.temperature,
    )
    return resp.choices[0].message.content.strip()


# Matches citation markers like [1] or [12] in the model's answer.
_CITATION_RE = re.compile(r"\[(\d+)\]")


def _cited_sources(answer: str, chunks: list[dict]) -> list[dict]:
    """Map the [n] markers in `answer` back to the chunks they reference.

    Returns one entry per DISTINCT in-range citation, in the order it first
    appears: {n, source, title}. Out-of-range numbers (e.g. the model writes
    [9] when only 4 chunks exist) are ignored. This is what makes the answer
    traceable — every [n] resolves to a real source doc.
    """
    cited = []
    seen = set()
    for m in _CITATION_RE.findall(answer):
        i = int(m)
        if 1 <= i <= len(chunks) and i not in seen:
            seen.add(i)
            c = chunks[i - 1]
            cited.append({"n": i, "source": c["source"], "title": c["title"]})
    return cited


def ask(
    question: str,
    k: int = RETRIEVAL.top_k,
    *,
    os_client=None,
    embedder=None,
    reranker=None,
    llm_client: OpenAI | None = None,
    config: LLMConfig = LLM,
) -> dict:
    """End-to-end RAG: question -> retrieve+rerank -> grounded, cited answer.

    The full pipeline in one call:
      1. retrieve   — bi-encoder k-NN fetches top-N candidates,
      2. rerank     — cross-encoder narrows them to the top-`k`,
      3. generate   — the LLM answers using ONLY those k chunks, citing [n].

    All heavy objects (OpenSearch client, embedder, reranker, LLM client) are
    injectable so a caller answering many questions — e.g. the eval harness —
    builds them once instead of reloading per question.

    Returns a structured result for display, debugging, and later eval:
      {
        "answer":           str,                       # the cited text
        "citations":        [{n, source, title}, ...], # sources actually cited
        "retrieved_chunks": [ ...search() hit dicts ], # everything fed as context
      }
    """
    # Stages 1+2 — always rerank here: ask() is the production answer path, and
    # showed rerank improves ranking precision (better top-k for the LLM).
    chunks = search(
        question,
        k=k,
        client=os_client,
        embedder=embedder,
        use_reranker=True,
        reranker=reranker,
    )

    # Stage 3 — grounded generation over exactly those k chunks.
    llm_client = llm_client or get_client(config)
    answer = generate_answer(question, chunks, client=llm_client, config=config)

    return {
        "answer": answer,
        "citations": _cited_sources(answer, chunks),
        "retrieved_chunks": chunks,
    }


if __name__ == "__main__":
    #   uv run python -m src.generate
    # End-to-end smoke test of ask(): real retrieval+rerank over the index,
    # then grounded generation. Checks (a) a real question gets a cited answer
    # and (b) an out-of-docs question is honestly refused, not fabricated.
    print(f"provider={LLM.provider} model={LLM.model}\n")

    for q in [
        "What is PagedAttention and what problem does it solve?",
        "Can vLLM make coffee?",  # not in the docs -> should refuse
    ]:
        print(f"=== {q}")
        result = ask(q)
        print(result["answer"])
        if result["citations"]:
            print("  cited:")
            for c in result["citations"]:
                print(f"    [{c['n']}] {c['title']}  ({c['source']})")
        print()
