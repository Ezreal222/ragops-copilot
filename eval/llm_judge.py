"""Custom LLM-as-judge for FAITHFULNESS — the D4 "open the RAGAS black box" deliverable.

RAGAS gives us four numbers (run_ragas.py) but hides HOW it turns an answer into a
score. Here we hand-build the same idea for one dimension — faithfulness

What "faithfulness" means here (reference-FREE, same as RAGAS's faithfulness):
    Is every factual claim in the ANSWER supported by the retrieved CONTEXT?
    It measures HALLUCINATION, not helpfulness — a wrong-but-grounded answer can
    still be faithful, and a correct answer that adds facts NOT in the context is
    NOT faithful. We deliberately do NOT show the judge the gold `reference`, so it
    can only reward what the retrieved context actually supports.

The four techniques that make an LLM judge more reliable (each is a design choice
below, not decoration):
    1. Chain-of-thought — the judge REASONS first, then scores. Judging before
       thinking is where LLM judges are least accurate.
    2. Discrete rubric — a 1–5 scale with anchored levels is far more stable
       across runs than "give a float 0..1".
    3. Structured output — force JSON so we can parse claims + score mechanically.
    4. temperature=0 — reproducible scoring (same design choice RAGAS makes).

Refusals ("I couldn't find this in the vLLM docs.") are handled in CODE, not by the
LLM: a refusal asserts nothing, so nothing can be UNsupported → it is trivially
faithful (score 5). We flag these (`is_refusal`) and skip the API call. This is
also a deliberate divergence point vs RAGAS, which tends to return NaN for a
no-claim answer — a good thing to discuss in the D4 write-up.

Judge model: DeepSeek `deepseek-v4-flash` — the SAME cheap/fast model RAGAS uses,
so the comparison isolates "our prompt vs RAGAS's prompt", not "flash vs pro".

Inputs come from the frozen `eval/ragas_inputs.jsonl` (built once in D3), so our
judge sees the EXACT question/answer/contexts RAGAS scored — an apples-to-apples set.

Run:
    uv run python -m eval.llm_judge --smoke   # 3 samples, to confirm wiring
    uv run python -m eval.llm_judge           # full set -> eval/judge_faithfulness.csv
"""

from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

from dotenv import load_dotenv

# Reuse the exact provider client the rest of the project uses (DeepSeek,
# OpenAI-compatible). We only swap in the cheaper judge model id below.
from src.generate import get_client

REPO_ROOT = Path(__file__).resolve().parent.parent
INPUTS_PATH = REPO_ROOT / "eval" / "ragas_inputs.jsonl"
OUT_CSV = REPO_ROOT / "eval" / "judge_faithfulness.csv"

# Cheap/fast judge — same model RAGAS's run_ragas.py uses, so the only variable
# between the two faithfulness scores is the PROMPT, not the model.
JUDGE_MODEL = "deepseek-v4-flash"

# The exact string generate.py emits when it has nothing to ground on. Detected
# in code so we never pay an API call to "judge" an answer that makes no claims.
REFUSAL_MARKER = "I couldn't find this in the vLLM docs."

# The judge prompt IS the deliverable — every clause encodes one reliability
# technique (CoT, discrete rubric, JSON-only). Kept verbose on purpose: a vague
# rubric is the #1 cause of a noisy LLM judge.
JUDGE_PROMPT = """You are a strict, impartial evaluator of an AI assistant's answer.

You are given a QUESTION, a numbered CONTEXT (documentation excerpts the assistant
was allowed to use), and the assistant's ANSWER.

Your ONLY job is to judge FAITHFULNESS: is every factual claim in the ANSWER
supported by the CONTEXT? Judge grounding, NOT correctness or helpfulness:
- A claim counts as UNSUPPORTED if it is not stated in or directly entailed by the
  CONTEXT — even if you happen to know it is true from outside knowledge.
- General connective phrasing, restating the question, and citation markers like
  [1] are NOT claims; ignore them.
- If the ANSWER declines to answer / says the information is not in the docs and
  makes no factual claims, it is trivially faithful (nothing to be unsupported).

Reason FIRST, then score (do not score before reasoning). Use this 1-5 rubric:
  5 = every factual claim is fully supported by the context
  4 = supported overall; at most a minor detail is unsupported
  3 = roughly half of the claims are supported
  2 = mostly unsupported; only a small part is grounded
  1 = no claim is supported, or the answer contradicts the context

Return ONLY a single JSON object, no prose outside it, in this exact shape:
{{"reason": "<one or two sentences of reasoning>", "unsupported_claims": ["<claim>", ...], "score": <int 1-5>}}

QUESTION:
{q}

CONTEXT:
{ctx}

ANSWER:
{ans}
"""


def format_contexts(contexts: list[str]) -> str:
    """Render the retrieved contexts as a numbered block, matching how the
    generator saw them ([1], [2], ...) so the judge reads the same layout."""
    return "\n\n".join(f"[{i}] {c}" for i, c in enumerate(contexts, 1))


# The judge may wrap its JSON in ```json fences or add stray text; grab the first
# {...} block and parse that. Robust parsing keeps one malformed reply from
# killing the whole run.
_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_judgement(raw: str) -> dict:
    """Extract {reason, unsupported_claims, score} from the judge's raw reply.

    Returns score=None on a parse failure so the caller can flag the row instead
    of crashing — a judge that occasionally returns junk is expected at temp 0.
    """
    m = _JSON_RE.search(raw)
    if not m:
        return {"reason": f"PARSE_FAIL: {raw[:200]}", "unsupported_claims": [], "score": None}
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return {"reason": f"PARSE_FAIL: {raw[:200]}", "unsupported_claims": [], "score": None}

    score = obj.get("score")
    if isinstance(score, (int, float)):
        score = int(round(score))
        score = max(1, min(5, score))  # clamp into the rubric range
    else:
        score = None
    return {
        "reason": str(obj.get("reason", "")),
        "unsupported_claims": obj.get("unsupported_claims", []) or [],
        "score": score,
    }


def judge_faithfulness(question: str, contexts: list[str], answer: str, *, client) -> dict:
    """Score one (question, contexts, answer) triple for faithfulness.

    Returns {score 1-5|None, unsupported_claims, reason, is_refusal}. Refusals are
    resolved in code (trivially faithful, no API call); everything else goes to
    the LLM judge at temperature=0.
    """
    if answer.strip() == REFUSAL_MARKER:
        return {
            "score": 5,
            "unsupported_claims": [],
            "reason": "Refusal: answer makes no factual claims, so nothing is unsupported.",
            "is_refusal": True,
        }

    prompt = JUDGE_PROMPT.format(q=question, ctx=format_contexts(contexts), ans=answer)
    resp = client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,  # reproducible scoring
    )
    result = parse_judgement(resp.choices[0].message.content or "")
    result["is_refusal"] = False
    return result


def load_samples(limit: int | None = None) -> list[dict]:
    if not INPUTS_PATH.exists():
        raise FileNotFoundError(
            f"{INPUTS_PATH} not found — run `uv run python -m eval.collect_ragas_inputs` first."
        )
    rows = [json.loads(line) for line in INPUTS_PATH.open(encoding="utf-8") if line.strip()]
    return rows[:limit] if limit else rows


def main() -> None:
    load_dotenv()
    smoke = "--smoke" in sys.argv
    samples = load_samples(limit=3 if smoke else None)
    mode = "SMOKE (3 samples)" if smoke else f"FULL ({len(samples)} samples)"
    print(f"Custom faithfulness judge — {mode} | judge={JUDGE_MODEL}\n")

    client = get_client()  # DeepSeek, OpenAI-compatible (from src.generate)
    out_rows = []
    for i, s in enumerate(samples, 1):
        q = s["user_input"]
        j = judge_faithfulness(q, s["retrieved_contexts"], s["response"], client=client)
        # Normalize 1-5 -> 0..1 so it lines up with RAGAS's 0..1 faithfulness.
        norm = (j["score"] - 1) / 4 if j["score"] is not None else None
        out_rows.append(
            {
                "user_input": q,
                "judge_score": j["score"],
                "judge_score_norm": norm,
                "is_refusal": j["is_refusal"],
                "n_unsupported": len(j["unsupported_claims"]),
                "unsupported_claims": json.dumps(j["unsupported_claims"], ensure_ascii=False),
                "reason": j["reason"],
            }
        )
        tag = "refusal" if j["is_refusal"] else f"score={j['score']}"
        print(f"  [{i:>2}/{len(samples)}] {tag:>10}  {q[:64]}")

    if not smoke:
        with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
            writer.writeheader()
            writer.writerows(out_rows)
        print(f"\nper-question judgements -> {OUT_CSV}")

    scored = [r["judge_score"] for r in out_rows if r["judge_score"] is not None]
    if scored:
        print(f"\nmean judge score: {sum(scored) / len(scored):.2f}/5  "
              f"({sum(scored) / len(scored) / 5 * 100:.0f}% grounded)  "
              f"| refusals: {sum(r['is_refusal'] for r in out_rows)}  "
              f"| parse-fails: {sum(r['judge_score'] is None for r in out_rows)}")


if __name__ == "__main__":
    main()
