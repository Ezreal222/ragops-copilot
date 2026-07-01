"""Compare our custom faithfulness judge against RAGAS's faithfulness — D4 Step 2.

Two independent estimates of the SAME thing (is the answer grounded in the
retrieved context?), scored over the SAME 36 frozen samples with the SAME judge
model (deepseek-v4-flash). The only difference is the PROMPT/rubric. So this
answers: does our hand-written judge track RAGAS, and where do they disagree?

Inputs (both keyed on `user_input`):
    eval/judge_faithfulness.csv  — our judge  (judge_score_norm, 0..1)
    eval/ragas_baseline.csv      — RAGAS      (faithfulness, 0..1)

Agreement metric: Spearman rank correlation (do the two rank questions the same
way?). We use RANK correlation, not exact-value agreement, because the two rubrics
live on different scales — what matters is "both say question A is worse than B".
Implemented by hand (average ranks for ties → Pearson on the ranks) so we don't
pull in scipy just for one number.

Run:
    uv run python -m eval.compare_judge_ragas
"""

from __future__ import annotations

import csv
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
JUDGE_CSV = REPO_ROOT / "eval" / "judge_faithfulness.csv"
RAGAS_CSV = REPO_ROOT / "eval" / "ragas_baseline.csv"


def _to_float(s: str) -> float | None:
    s = (s or "").strip()
    if s == "" or s.lower() == "nan":
        return None
    try:
        return float(s)
    except ValueError:
        return None


def load_joined() -> list[dict]:
    """Join judge + RAGAS rows on the question text. Returns one dict per matched
    question with both faithfulness estimates on the 0..1 scale."""
    judge = {r["user_input"]: r for r in csv.DictReader(JUDGE_CSV.open(encoding="utf-8"))}
    ragas = {r["user_input"]: r for r in csv.DictReader(RAGAS_CSV.open(encoding="utf-8"))}

    joined = []
    for q, jr in judge.items():
        rr = ragas.get(q)
        if rr is None:
            continue  # question not in the RAGAS run (shouldn't happen — same set)
        joined.append(
            {
                "q": q,
                "judge": _to_float(jr["judge_score_norm"]),
                "judge_raw": jr["judge_score"],
                "is_refusal": jr["is_refusal"] == "True",
                "ragas": _to_float(rr["faithfulness"]),
            }
        )
    return joined


def _avg_ranks(values: list[float]) -> list[float]:
    """Rank values ascending, assigning the average rank to ties (standard for
    Spearman)."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1  # 1-based average rank over the tie group
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def _pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs) ** 0.5
    vy = sum((y - my) ** 2 for y in ys) ** 0.5
    if vx == 0 or vy == 0:
        return float("nan")
    return cov / (vx * vy)


def spearman(xs: list[float], ys: list[float]) -> float:
    """Spearman rho = Pearson correlation of the average ranks."""
    return _pearson(_avg_ranks(xs), _avg_ranks(ys))


def main() -> None:
    rows = load_joined()
    both = [r for r in rows if r["judge"] is not None and r["ragas"] is not None]

    # Spearman over ALL matched rows, and again EXCLUDING refusals — refusals are
    # the known systematic disagreement (we score them 1.0, RAGAS ~0.0), so the
    # "excl. refusals" number shows how well the judges agree on real answers.
    non_refusal = [r for r in both if not r["is_refusal"]]
    rho_all = spearman([r["judge"] for r in both], [r["ragas"] for r in both])
    rho_nr = spearman([r["judge"] for r in non_refusal], [r["ragas"] for r in non_refusal])

    j_mean = sum(r["judge"] for r in both) / len(both)
    r_mean = sum(r["ragas"] for r in both) / len(both)

    print(f"matched questions: {len(both)}  ({sum(r['is_refusal'] for r in both)} refusals)\n")
    print(f"mean faithfulness   custom judge: {j_mean:.3f}   RAGAS: {r_mean:.3f}")
    print(f"Spearman rho        all rows:     {rho_all:.3f}   excl. refusals: {rho_nr:.3f}\n")

    # Biggest disagreements (|judge - ragas| on the 0..1 scale). This is the
    # interesting part: where does a hand-written rubric part ways with RAGAS?
    ranked = sorted(both, key=lambda r: abs(r["judge"] - r["ragas"]), reverse=True)
    print("=== largest disagreements (|judge - RAGAS|) ===")
    print(f"{'delta':>5}  {'judge':>5} {'ragas':>5}  {'refusal':>7}  question")
    for r in ranked[:10]:
        d = abs(r["judge"] - r["ragas"])
        flag = "yes" if r["is_refusal"] else ""
        print(f"{d:>5.2f}  {r['judge']:>5.2f} {r['ragas']:>5.2f}  {flag:>7}  {r['q'][:58]}")

    # Both-agree-low: questions BOTH judges score poorly are the trustworthy
    # problem cases (real grounding issues, not rubric noise).
    print("\n=== both score low (trustworthy problem questions) ===")
    low = sorted(
        [r for r in non_refusal if r["judge"] <= 0.5 and r["ragas"] <= 0.5],
        key=lambda r: r["judge"] + r["ragas"],
    )
    if not low:
        print("  (none — no real answer scored low by both)")
    for r in low:
        print(f"  judge={r['judge']:.2f} ragas={r['ragas']:.2f}  {r['q'][:60]}")


if __name__ == "__main__":
    main()
