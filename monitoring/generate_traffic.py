"""Traffic generator — give the dashboard something real to show (W7 D4 step 3).

    uv run python -m monitoring.generate_traffic              # both engines
    uv run python -m monitoring.generate_traffic --mode rag   # one engine
    uv run python -m monitoring.generate_traffic --repeat 3   # more samples

Why a Python script and not `ab` / `hey`: those measure a URL. We need to measure
a *RAG system*, where the question mix matters as much as the request rate. A load
generator firing one identical question would warm every cache and report a
beautiful, meaningless p95. So this sends a deliberate MIX:

  - answerable technical lookups  -> the normal case (expect: answered, cited),
  - a comparison question         -> the multi-step case (the agent's reason to
                                     exist; on the fixed path it's one retrieval),
  - an out-of-corpus question     -> the refusal case (expect: an honest "not in
                                     the docs", NOT a fabricated answer).

and runs the whole mix through BOTH engines, so "agent vs rag" is a controlled
comparison on identical questions rather than two different workloads.

The summary is measured two ways on purpose:
  - client-side : wall-clock latency per request, as a user experiences it
    (includes network + JSON, so it reads slightly above the server's own number),
  - server-side : the /metrics counters, read BEFORE and AFTER, differenced. That
    delta is where token/cost-per-query comes from — the API is the only thing
    that sees the provider's usage numbers.

Reading the counters at both ends and subtracting is what makes this safe to run
against a service that already has history: we report what THIS run cost, not the
process's lifetime totals.
"""

from __future__ import annotations

import argparse
import statistics
import time
from urllib.request import urlopen

import requests
from prometheus_client.parser import text_string_to_metric_families

# The question mix. Hand-picked rather than generated: each line exercises a
# different behaviour we claim the system has. The D3 fix already established that
# the corpus genuinely cannot answer some plausible-sounding questions, so the
# refusal below is a real refusal rather than a retrieval bug.
QUESTIONS = [
    ("technical", "What is PagedAttention?"),
    ("technical", "How does vLLM do continuous batching?"),
    ("technical", "How does vLLM handle quantization?"),
    ("compare", "Compare continuous batching and PagedAttention."),
    ("refusal", "Can vLLM make coffee?"),  # not in the docs -> expect a refusal
]


def _scrape(base_url: str) -> dict[str, float]:
    """Read /metrics and flatten it to {metric{labels}: value}.

    Uses prometheus_client's own parser rather than a regex, because the text
    exposition format has real rules (HELP/TYPE lines, escaping, histogram
    suffixes) and hand-parsing it is how you get subtly wrong numbers.
    """
    text = urlopen(f"{base_url}/metrics", timeout=10).read().decode()
    out: dict[str, float] = {}
    for family in text_string_to_metric_families(text):
        for sample in family.samples:
            labels = ",".join(f"{k}={v}" for k, v in sorted(sample.labels.items()))
            out[f"{sample.name}{{{labels}}}"] = sample.value
    return out


def _delta(before: dict[str, float], after: dict[str, float], prefix: str) -> float:
    """Sum the increase of every series whose key starts with `prefix`.

    Counters are cumulative, so the only honest way to ask "what did this run
    cost" is after-minus-before. Series absent from `before` default to 0 — a
    label combination appearing for the first time during the run (say, the first
    refusal) has genuinely gone from nothing to something.
    """
    total = 0.0
    for key, value in after.items():
        if key.startswith(prefix):
            total += value - before.get(key, 0.0)
    return total


def run_mix(base_url: str, mode: str, repeat: int) -> list[dict]:
    """Send the question mix through one engine `repeat` times; return per-request rows."""
    use_agent = mode == "agent"
    rows: list[dict] = []
    for _ in range(repeat):
        for kind, question in QUESTIONS:
            t0 = time.perf_counter()
            resp = requests.post(
                f"{base_url}/ask",
                json={"question": question, "use_agent": use_agent},
                timeout=180,
            )
            latency = time.perf_counter() - t0

            if resp.status_code != 200:
                # A failed request is DATA, not a reason to crash the run: it is
                # exactly what ask_exceptions_total counts, and the run should
                # still report everything else.
                print(f"  [{mode:5}] {kind:9} HTTP {resp.status_code} in {latency:5.1f}s")
                rows.append({"kind": kind, "latency": latency, "ok": False})
                continue

            body = resp.json()
            refused = "couldn't find this in the vLLM docs" in body["answer"]
            rows.append(
                {
                    "kind": kind,
                    "latency": latency,
                    "ok": True,
                    "refused": refused,
                    "degraded": body["degraded"],
                    "steps": body["steps"],
                    "citations": len(body["citations"]),
                }
            )
            status = "REFUSED" if refused else f"cited {len(body['citations'])}"
            print(
                f"  [{mode:5}] {kind:9} {latency:5.1f}s  steps={body['steps']}  {status}"
            )
    return rows


def summarize(
    mode: str, rows: list[dict], cost: float, prompt_tok: float, completion_tok: float
) -> None:
    """Print the numbers D4 asks us to record for this engine."""
    ok = [r for r in rows if r["ok"]]
    if not ok:
        print(f"\n  {mode}: no successful requests")
        return

    lat = sorted(r["latency"] for r in ok)
    # p95 by nearest-rank. Honest naming matters: with ~10 requests this is
    # really "the slowest one", not a statistically meaningful p95 — the
    # DASHBOARD's histogram_quantile is the real thing. This is a sanity check.
    p95 = lat[min(len(lat) - 1, int(round(0.95 * len(lat))) - 1)]
    refused = sum(1 for r in ok if r["refused"])

    # The raw refusal rate — what the dashboard's panel shows — is NOT a quality
    # score, and reporting it alone would be misleading in both directions. Some
    # refusals are the system working (the corpus really can't answer "can vLLM
    # make coffee?"); a FALSE refusal, where we turn down a question the docs do
    # cover, is a straight bug. The dashboard can't tell them apart because it
    # doesn't know the ground truth. Here we do — QUESTIONS tags each question —
    # so we split them, and the false-refusal line is the one that matters.
    answerable = [r for r in ok if r["kind"] != "refusal"]
    false_refusals = sum(1 for r in answerable if r["refused"])
    expected = [r for r in ok if r["kind"] == "refusal"]
    correct_refusals = sum(1 for r in expected if r["refused"])

    print(f"\n  {mode}")
    print(f"    requests        : {len(ok)}")
    print(
        f"    latency p50/p95 : {statistics.median(lat):.2f}s / {p95:.2f}s"
        f"   (min {lat[0]:.2f}s, max {lat[-1]:.2f}s)"
    )
    print(f"    refusal rate    : {refused}/{len(ok)} = {refused / len(ok):.0%}"
          f"   (what the dashboard panel shows)")
    if answerable:
        print(
            f"    FALSE refusals  : {false_refusals}/{len(answerable)} = "
            f"{false_refusals / len(answerable):.0%} of answerable questions  <- bug rate"
        )
    if expected:
        print(
            f"    correct refusals: {correct_refusals}/{len(expected)} of "
            f"out-of-corpus questions  <- working as intended"
        )
    # Degraded = the agent burned its step cap and returned the fallback. Reported
    # separately from refusals because it is NEITHER: the answer text isn't the
    # "not in the docs" sentence, so a naive refusal check misses it entirely, and
    # it's the agent path's own failure mode (the W6 D6 non-convergence finding).
    # Leaving it out of this summary would flatter the agent.
    degraded = sum(1 for r in ok if r["degraded"])
    print(f"    degraded        : {degraded}/{len(ok)} hit the step cap")
    print(f"    avg steps       : {statistics.mean(r['steps'] for r in ok):.2f}")
    print(
        f"    tokens/query    : {prompt_tok / len(ok):,.0f} prompt + "
        f"{completion_tok / len(ok):,.0f} completion"
    )
    print(f"    cost/query      : ${cost / len(ok):.5f}   (run total ${cost:.4f})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default="http://localhost:8000", help="API base URL")
    ap.add_argument("--mode", choices=["rag", "agent", "both"], default="both")
    ap.add_argument(
        "--repeat",
        type=int,
        default=2,
        help="How many times to send the whole question mix per engine.",
    )
    args = ap.parse_args()

    modes = ["rag", "agent"] if args.mode == "both" else [args.mode]
    results = {}

    for mode in modes:
        print(f"\n=== {mode} ===")
        # Scrape around each engine separately, so the cost delta is attributable
        # to THIS engine even though both increment the same counter family.
        before = _scrape(args.url)
        rows = run_mix(args.url, mode, args.repeat)
        after = _scrape(args.url)
        results[mode] = (
            rows,
            _delta(before, after, f"llm_cost_usd_total{{mode={mode}}}"),
            _delta(before, after, f"llm_tokens_total{{mode={mode},type=prompt}}"),
            _delta(before, after, f"llm_tokens_total{{mode={mode},type=completion}}"),
        )

    print("\n" + "=" * 62)
    print("SUMMARY (client-side latency; token/cost deltas from /metrics)")
    print("=" * 62)
    for mode, (rows, cost, ptok, ctok) in results.items():
        summarize(mode, rows, cost, ptok, ctok)

    # The headline D4 comparison, printed only when we actually ran both — the
    # quality/latency/COST trade-off of the agent, in two numbers.
    if len(results) == 2:
        rag_rows, rag_cost, *_ = results["rag"]
        agent_rows, agent_cost, *_ = results["agent"]
        rag_ok = [r["latency"] for r in rag_rows if r["ok"]]
        agent_ok = [r["latency"] for r in agent_rows if r["ok"]]
        if rag_ok and agent_ok and rag_cost:
            print("\n  agent vs rag")
            print(
                f"    latency : {statistics.median(agent_ok) / statistics.median(rag_ok):.1f}x"
                f" slower (median)"
            )
            print(f"    cost    : {agent_cost / rag_cost:.1f}x more expensive (run total)")


if __name__ == "__main__":
    main()
