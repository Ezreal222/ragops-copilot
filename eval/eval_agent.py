"""Agent eval — task success rate, tool-selection accuracy, failure analysis (W6 D5).

RAG eval (eval/eval_retrieval.py, run_ragas.py, llm_judge.py) measures a FIXED
pipeline: given a question, did retrieval find the right chunks and did the answer
stay grounded? An agent is different — it *decides* what to do (which tool, how
many steps), so the same "final answer only" lens misses half the story. Here we
evaluate the PROCESS as well as the RESULT:

  - tool-selection accuracy : did the agent call the tool(s) the task needs?
    (compare tool for a comparison, list_sources for a scope question, search_docs
    for a lookup). A wrong tool = a planning/description problem, invisible if you
    only read the final text.
  - task success rate        : does the final answer satisfy the task's
    pre-defined `success_criteria`? Judged by an LLM (reusing the D4 judge
    recipe: reason-first, JSON, temperature 0) so it's mechanical and repeatable.
  - agent-health signals     : steps (tool-call rounds), latency, whether a
    guardrail fired (degraded run / citation warnings). Cheap indicators that the
    agent is thrashing or degrading even when the answer happens to pass.

Why judge success against `success_criteria` and NOT exact tool path: an agent
task often has several correct routes (compare vs two search_docs), so success is
"did it achieve the result", while tool accuracy is a SEPARATE, softer diagnostic.

Inputs: eval/agent_eval_set.jsonl (the 10 hand-labeled tasks). The agent is the
guardrailed src.agent.graph.run_agent — the exact entry point a caller/API uses,
so we measure the real system, not a stripped-down copy.

Run:
    uv run python -m eval.eval_agent --smoke   # first 3 tasks, confirm wiring
    uv run python -m eval.eval_agent           # full set -> eval/agent_eval.csv
"""

from __future__ import annotations

import csv
import json
import re
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

from src.agent.graph import build_graph, run_agent
# Same OpenAI-compatible client + cheap judge model the D4 faithfulness judge uses,
# so the judge here is consistent with the rest of the eval suite.
from src.generate import get_client

REPO_ROOT = Path(__file__).resolve().parent.parent
TASKS_PATH = REPO_ROOT / "eval" / "agent_eval_set.jsonl"
OUT_CSV = REPO_ROOT / "eval" / "agent_eval.csv"

JUDGE_MODEL = "deepseek-v4-flash"

# The success judge: unlike the faithfulness judge (grounding vs context), this one
# grades the answer against the task's own `success_criteria` — a binary pass/fail.
# Same reliability recipe: reason first, discrete verdict, JSON-only, temperature 0.
JUDGE_PROMPT = """You are a strict, impartial grader of an AI agent's answer to a task.

You are given a TASK (what the user asked), the SUCCESS CRITERIA (the specific,
pre-defined conditions the answer must meet to count as a success), and the
agent's ANSWER.

Decide ONLY whether the ANSWER satisfies the SUCCESS CRITERIA as written — not
your own idea of a good answer.
- If the criteria require covering specific points, ALL required points must be
  present to pass.
- If the criteria require a refusal / "not in the docs", then a correct refusal
  is a PASS and a fabricated answer is a FAIL.
- If the criteria require citations (markers like [1]), their absence is a FAIL.
- Ignore writing style; judge substance against the criteria.

Reason FIRST, then give the verdict (do not decide before reasoning).

Return ONLY a single JSON object, no prose outside it, in this exact shape:
{{"reason": "<one or two sentences>", "pass": <true or false>}}

TASK:
{task}

SUCCESS CRITERIA:
{criteria}

ANSWER:
{answer}
"""

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def load_tasks(limit: int | None = None) -> list[dict]:
    if not TASKS_PATH.exists():
        raise FileNotFoundError(f"{TASKS_PATH} not found — the agent eval set is missing.")
    rows = [json.loads(line) for line in TASKS_PATH.open(encoding="utf-8") if line.strip()]
    return rows[:limit] if limit else rows


def tool_trace(messages: list) -> list[str]:
    """The ordered list of tool NAMES the agent actually called this run.

    Each AIMessage that requested action carries `.tool_calls` (dicts with a
    'name'); we flatten them in order. This IS the "process" we score against
    expected_tools. Empty for a degraded run (we bailed before producing a trace).
    """
    names: list[str] = []
    for m in messages:
        for tc in getattr(m, "tool_calls", None) or []:
            names.append(tc["name"])
    return names


def tools_ok(task: dict, trace: list[str]) -> bool:
    """Did the agent pick the right tool(s) for this task type?

    Base rule (matches the D5 spec skeleton): the actual tools are a SUPERSET of
    expected_tools — the agent called at least what the task needs (extra
    exploratory calls don't fail it). Special case: a multi_step comparison has
    TWO valid routes, so we accept either the `compare` tool OR two+ `search_docs`
    calls (retrieve-each-then-synthesize). We score the result-achieving path, not
    one blessed sequence.
    """
    actual = set(trace)
    if task["type"] == "multi_step":
        return "compare" in actual or trace.count("search_docs") >= 2
    return set(task["expected_tools"]) <= actual


def judge_success(task: str, criteria: str, answer: str, *, client) -> dict:
    """LLM-grade one answer against its success_criteria -> {pass: bool|None, reason}.

    pass=None signals a judge parse failure (rare at temp 0) so the caller can flag
    the row instead of silently counting it as a fail.
    """
    prompt = JUDGE_PROMPT.format(task=task, criteria=criteria, answer=answer)
    resp = client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,  # reproducible grading
    )
    raw = resp.choices[0].message.content or ""
    m = _JSON_RE.search(raw)
    if not m:
        return {"pass": None, "reason": f"PARSE_FAIL: {raw[:160]}"}
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return {"pass": None, "reason": f"PARSE_FAIL: {raw[:160]}"}
    verdict = obj.get("pass")
    return {
        "pass": bool(verdict) if isinstance(verdict, bool) else None,
        "reason": str(obj.get("reason", "")),
    }


def classify_failure(tools_correct: bool, degraded: bool, passed) -> str:
    """Coarse failure-type hint for a task that did NOT pass (guides Step-3 fixes).

    Maps to the D5 failure taxonomy so aggregate counts point at WHAT to fix:
      - non_convergence : hit the step cap (degraded) — the answer never landed.
      - tool_selection  : wrong/insufficient tools — a description/planning issue
                          (fix the tool docstrings or the prompt).
      - generation      : right tools, but the answer misses the criteria — a
                          synthesis/generation issue (fix the system prompt / model).
      - judge_parse_fail: the judge returned junk; not a real agent failure.
    A hint, not a verdict — the human confirms the root cause in the write-up.
    """
    if passed is None:
        return "judge_parse_fail"
    if degraded:
        return "non_convergence"
    if not tools_correct:
        return "tool_selection"
    return "generation"


def evaluate(tasks: list[dict], *, write_csv: bool) -> list[dict]:
    load_dotenv()
    client = get_client()          # judge client (DeepSeek, OpenAI-compatible)
    app = build_graph()            # compile the agent graph ONCE, reuse per task

    rows = []
    for i, t in enumerate(tasks, 1):
        # Wall-clock latency. NOTE: the first task also pays lazy model/index load
        # (embedder + OpenSearch client build on the first search_docs), so treat
        # task 1's latency as a warm-up outlier when reading the numbers.
        t0 = time.perf_counter()
        out = run_agent(t["task"], app=app)
        latency = time.perf_counter() - t0

        trace = tool_trace(out["messages"])
        t_ok = tools_ok(t, trace)
        j = judge_success(t["task"], t["success_criteria"], out["answer"], client=client)
        passed = j["pass"]
        failure_type = "" if passed else classify_failure(t_ok, out["degraded"], passed)

        rows.append({
            "id": t["id"],
            "type": t["type"],
            "task": t["task"],
            "expected_tools": " ".join(t["expected_tools"]),
            "actual_tools": " ".join(trace) or "(none)",
            "tools_ok": t_ok,
            "success": bool(passed),               # None (parse fail) counts as not-success
            "steps": out["steps"],
            "degraded": out["degraded"],
            "citation_ok": out["citation_report"]["ok"],
            "latency_s": round(latency, 2),
            "failure_type": failure_type,
            "judge_reason": j["reason"],
        })

        flag = "PASS" if passed else ("PARSE?" if passed is None else "FAIL")
        tick = "ok " if t_ok else "BAD"
        print(f"  [{i:>2}/{len(tasks)}] {flag:<6} tools:{tick} "
              f"steps={out['steps']} {latency:5.1f}s  {t['id']}")
        if not passed:
            print(f"          -> {failure_type}: {j['reason'][:110]}")

    if write_csv:
        with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nper-task results -> {OUT_CSV}")

    return rows


def summarize(rows: list[dict]) -> None:
    n = len(rows)
    success_rate = sum(r["success"] for r in rows) / n
    tool_acc = sum(r["tools_ok"] for r in rows) / n
    avg_steps = sum(r["steps"] for r in rows) / n
    avg_latency = sum(r["latency_s"] for r in rows) / n

    print(f"\n=== agent eval summary ({n} tasks) ===")
    print(f"  task success rate       : {success_rate:.0%}  ({sum(r['success'] for r in rows)}/{n})")
    print(f"  tool-selection accuracy : {tool_acc:.0%}  ({sum(r['tools_ok'] for r in rows)}/{n})")
    print(f"  avg steps (tool rounds) : {avg_steps:.1f}")
    print(f"  avg latency             : {avg_latency:.1f}s")
    print(f"  degraded (hit step cap) : {sum(r['degraded'] for r in rows)}")
    print(f"  citation warnings       : {sum(not r['citation_ok'] for r in rows)}")

    failures = [r for r in rows if not r["success"]]
    if failures:
        print(f"\n  failing tasks ({len(failures)}):")
        for r in failures:
            print(f"    - {r['id']:<24} [{r['failure_type']}] "
                  f"tools={r['actual_tools']}  ({r['judge_reason'][:70]})")
    else:
        print("\n  no failing tasks.")

    # Failure-type breakdown -> where to spend fixing effort (Step 3 attribution).
    if failures:
        from collections import Counter
        by_type = Counter(r["failure_type"] for r in failures)
        print("\n  failure types:", dict(by_type))


def main() -> None:
    smoke = "--smoke" in sys.argv
    tasks = load_tasks(limit=3 if smoke else None)
    mode = "SMOKE (3 tasks)" if smoke else f"FULL ({len(tasks)} tasks)"
    print(f"Agent eval — {mode} | judge={JUDGE_MODEL}\n")
    rows = evaluate(tasks, write_csv=not smoke)
    summarize(rows)


if __name__ == "__main__":
    main()
