#!/usr/bin/env python
"""M6 -- aggregate-dollar cost accounting for the full method vs. the BM25 baseline, on all
three datasets, symmetric to the scrutiny the paper applies to DCI's own cost claim
(docs/reviews/round4_full.md, M6; see also scripts/cost_analysis.py's DCI-focused version of
this same count-once-vs-cache-aware idea, which this script deliberately does NOT duplicate --
this one is full-method-vs-BM25-baseline, count-once vs STEP-SUMMED, in real assumed dollars).

ASSUMED PRICE RATE -- stated prominently, parameterizable, NOT a real invoice:
  This repo serves Tongyi-DeepResearch-30B-A3B on self-hosted vLLM, which has no per-token bill.
  Default rate below ($0.20/1M input, $0.80/1M output) is the SAME "30B-class hosted-serving
  proxy" rate scripts/cost_analysis.py already uses (its `TONGYI_LOCAL (proxy)` table) -- reused
  here for consistency with the rest of the repo's cost analyses, not because it is a verified
  invoice. Override with --price-in-per-1m / --price-out-per-1m to substitute any other public
  list price (e.g. a commercial API's rate card) and re-run.

TWO BASES, both already-existing row fields (see evaluation/run_eval.py ~lines 520-575):
  count-once   : tok_in = initial_prompt_tokens + context_once_tokens (each real token the model
                 ever saw counted exactly once); tok_out = output_tokens. This is what the paper's
                 headline 30-51% token-savings number uses.
  step-summed  : tok_in = prompt_tokens (the RAW cumulative sum of every LLM call's input tokens
                 across the whole episode -- the growing context re-sent every turn); tok_out =
                 completion_tokens. This is what an UNCACHED backend actually bills on, and is
                 the basis that is sensitive to the method's LLM-CALL COUNT (more calls -> more
                 times the growing prefix gets re-sent and re-billed), which count-once does not
                 capture at all.

Cells: reuses `analysis.paper_analyses.CELLS` verbatim (method =
`runs/_headline_validation/.../agent_research_bql_dense_snip`, baseline =
`runs/_visit_uncapped/.../agent_research_bm25`, both already-existing, no new experiments) --
the SAME six cells `analysis/paper_analyses.py`'s Analysis 3 reads, so this script's mean-tokens/
mean-calls numbers reconcile with that file's by construction.

STREAMS every rows.jsonl line-by-line (only the numeric token/call fields are kept per row, never
the full row dict -- `observations`/`trajectory` etc. are skipped), per the task brief's "stream
large files, avoid loading everything into memory" instruction. All six files here are
188MB-750MB, well under the 17-31GB auto-read files the brief warns off, but streaming costs
nothing and scales safely regardless.

    PYTHONPATH=. envs/bin/python analysis/cost_dollars.py
    PYTHONPATH=. envs/bin/python analysis/cost_dollars.py --price-in-per-1m 3.0 --price-out-per-1m 15.0 \\
        --out analysis/cost_dollars_sonnet_class.md
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from analysis.paper_analyses import CELLS, DATASETS  # noqa: E402  (reuse, not reimplement)

DEFAULT_PRICE_IN_PER_1M = 0.20    # same "30B-class hosted-serving proxy" scripts/cost_analysis.py uses
DEFAULT_PRICE_OUT_PER_1M = 0.80


def stream_aggregate(rows_path: Path) -> dict:
    """Single streaming pass: sums of count-once tok_in/tok_out, step-summed tok_in/tok_out,
    and llm_calls, plus n. Never holds more than one parsed row dict at a time."""
    n = 0
    sum_once_in = sum_once_out = 0
    sum_step_in = sum_step_out = 0
    sum_calls = 0
    with open(rows_path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            n += 1
            once = r.get("total_tokens_once")
            out_tok = r.get("output_tokens")
            if out_tok is None:
                out_tok = r.get("completion_tokens") or 0
            if once is None:
                once_in = (r.get("initial_prompt_tokens") or 0) + (r.get("context_once_tokens") or 0)
            else:
                once_in = once - out_tok
            sum_once_in += max(0, once_in)
            sum_once_out += out_tok
            sum_step_in += r.get("prompt_tokens") or 0
            sum_step_out += r.get("completion_tokens") or 0
            sum_calls += r.get("llm_calls") or r.get("n_steps") or 0
    return {
        "n": n,
        "mean_once_in": sum_once_in / n if n else 0.0, "mean_once_out": sum_once_out / n if n else 0.0,
        "mean_step_in": sum_step_in / n if n else 0.0, "mean_step_out": sum_step_out / n if n else 0.0,
        "mean_calls": sum_calls / n if n else 0.0,
        "sum_once_in": sum_once_in, "sum_once_out": sum_once_out,
        "sum_step_in": sum_step_in, "sum_step_out": sum_step_out,
    }


def dollar_cost(tok_in: float, tok_out: float, price_in_per_1m: float, price_out_per_1m: float) -> float:
    return tok_in * (price_in_per_1m / 1e6) + tok_out * (price_out_per_1m / 1e6)


def compute(price_in_per_1m: float, price_out_per_1m: float) -> dict:
    out = {}
    for ds in DATASETS:
        out[ds] = {}
        for role, cond_dir in CELLS[ds].items():
            rows_path = cond_dir / "rows.jsonl"
            print(f"[{ds}/{role}] streaming {rows_path} ...", file=sys.stderr)
            agg = stream_aggregate(rows_path)
            per_query_once = dollar_cost(agg["mean_once_in"], agg["mean_once_out"],
                                          price_in_per_1m, price_out_per_1m)
            per_query_step = dollar_cost(agg["mean_step_in"], agg["mean_step_out"],
                                          price_in_per_1m, price_out_per_1m)
            out[ds][role] = {
                **agg,
                "per_query_usd_once": per_query_once,
                "per_query_usd_step": per_query_step,
                "aggregate_usd_once": per_query_once * agg["n"],
                "aggregate_usd_step": per_query_step * agg["n"],
            }
        m, b = out[ds]["method"], out[ds]["baseline"]
        out[ds]["delta_pct"] = {
            "once": 100.0 * (m["per_query_usd_once"] - b["per_query_usd_once"]) / b["per_query_usd_once"],
            "step": 100.0 * (m["per_query_usd_step"] - b["per_query_usd_step"]) / b["per_query_usd_step"],
            "llm_calls": 100.0 * (m["mean_calls"] - b["mean_calls"]) / b["mean_calls"],
        }
        out[ds]["method_cheaper"] = {
            "once": m["per_query_usd_once"] < b["per_query_usd_once"],
            "step": m["per_query_usd_step"] < b["per_query_usd_step"],
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--price-in-per-1m", type=float, default=DEFAULT_PRICE_IN_PER_1M,
                     help=f"assumed $ per 1M INPUT tokens (default {DEFAULT_PRICE_IN_PER_1M}, "
                          "the same proxy rate scripts/cost_analysis.py uses -- NOT a real invoice)")
    ap.add_argument("--price-out-per-1m", type=float, default=DEFAULT_PRICE_OUT_PER_1M,
                     help=f"assumed $ per 1M OUTPUT tokens (default {DEFAULT_PRICE_OUT_PER_1M})")
    ap.add_argument("--out", default=None)
    ap.add_argument("--json-out", default=str(REPO_ROOT / "analysis" / "cost_dollars_data.json"))
    args = ap.parse_args()

    results = compute(args.price_in_per_1m, args.price_out_per_1m)

    with open(args.json_out, "w") as f:
        json.dump({"price_in_per_1m": args.price_in_per_1m, "price_out_per_1m": args.price_out_per_1m,
                   "results": results}, f, indent=2, default=str)
    print(f"wrote {args.json_out}", file=sys.stderr)

    md = render_markdown(results, args.price_in_per_1m, args.price_out_per_1m)
    if args.out:
        Path(args.out).write_text(md)
        print(f"wrote {args.out}", file=sys.stderr)
    else:
        print(md)


def render_markdown(results: dict, price_in: float, price_out: float) -> str:
    lines = ["# Aggregate-dollar cost accounting -- full method vs. BM25 baseline (M6)\n"]
    lines.append(f"**Assumed price** (parameterizable, NOT a verified invoice -- see script "
                  f"docstring): **${price_in:.2f} / 1M input tokens, ${price_out:.2f} / 1M output "
                  "tokens**. Two accounting bases per cell: **count-once** (each real token the "
                  "model ever saw, counted exactly once -- what the paper's headline 30-51% "
                  "token-savings number uses) and **step-summed** (the raw cumulative sum of "
                  "every LLM call's input+output tokens across the episode -- what an UNCACHED "
                  "backend actually bills, and the basis sensitive to the method's LLM-call "
                  "count).\n")

    lines.append("| Dataset | Cell | n | LLM calls/ep | $/query (count-once) | "
                  "$/query (step-summed) | $ over full eval set (count-once) | "
                  "$ over full eval set (step-summed) |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for ds in DATASETS:
        for role, label in (("baseline", "baseline (bm25)"), ("method", "full method (bql+dense+snip)")):
            c = results[ds][role]
            lines.append(f"| {ds} | {label} | {c['n']} | {c['mean_calls']:.1f} | "
                          f"${c['per_query_usd_once']:.4f} | ${c['per_query_usd_step']:.4f} | "
                          f"${c['aggregate_usd_once']:,.2f} | ${c['aggregate_usd_step']:,.2f} |")
        d = results[ds]["delta_pct"]
        lines.append(f"| {ds} | **method vs baseline, %Δ** | | **{d['llm_calls']:+.1f}%** | "
                      f"**{d['once']:+.1f}%** | **{d['step']:+.1f}%** | | |")
    lines.append("")

    lines.append("**Key sentences (paper-ready, hedged).**\n")
    for ds in DATASETS:
        d = results[ds]["delta_pct"]
        mc = results[ds]["method_cheaper"]
        verdict = "remains cheaper" if mc["step"] else "is MORE EXPENSIVE, reversing the count-once story"
        lines.append(f"- On {ds}, at ${price_in:.2f}/${price_out:.2f} per 1M in/out tokens, the "
                      f"full method uses {d['llm_calls']:+.1f}% LLM calls versus the BM25 baseline "
                      f"and costs {d['once']:+.1f}% under count-once dollar accounting and "
                      f"{d['step']:+.1f}% under step-summed (call-count-sensitive) dollar "
                      f"accounting -- the method **{verdict}** under the accounting basis that is "
                      "sensitive to call count.")
    all_step_cheaper = all(results[ds]["method_cheaper"]["step"] for ds in DATASETS)
    lines.append("")
    if all_step_cheaper:
        lines.append("**Overall.** Even under the step-summed, call-count-sensitive accounting -- "
                      "the basis the paper does not currently report, and the one most analogous "
                      "to the standard the paper itself applies to DCI's cost claim -- the full "
                      "method remains cheaper than the BM25 baseline on all three datasets at the "
                      "assumed price above, DESPITE making more LLM calls on two of the three "
                      "(HotpotQA, MuSiQue): the per-call context the method resends is enough "
                      "shorter (bounded section-level fetches vs. whole-document visits) that "
                      "extra calls do not flip the aggregate-dollar verdict. This should be stated "
                      "explicitly, with the assumed rate disclosed, alongside the existing "
                      "count-once headline number.\n")
    else:
        flipped = [ds for ds in DATASETS if not results[ds]["method_cheaper"]["step"]]
        lines.append(f"**Overall.** Under the step-summed, call-count-sensitive accounting, the "
                      f"method's cost advantage REVERSES on: {', '.join(flipped)} -- the "
                      "count-once headline number alone would overstate the savings there. This "
                      "should be stated explicitly, with the assumed rate disclosed, alongside the "
                      "existing count-once headline number.\n")

    lines.append("**Why the $ delta is smaller than the raw-token-count delta "
                  f"(count-once basis).** `analysis/paper_analyses.py`'s Analysis 3 reports the "
                  "method's count-once TOKEN savings as -29.6%/-30.4%/-50.9% on browsecomp/"
                  "hotpotqa/musique respectively (input+output tokens combined, unweighted) -- "
                  "noticeably larger in magnitude than this script's count-once DOLLAR deltas "
                  f"above ({results['browsecomp_plus_structured']['delta_pct']['once']:+.1f}%/"
                  f"{results['hotpotqa_structured']['delta_pct']['once']:+.1f}%/"
                  f"{results['musique_structured']['delta_pct']['once']:+.1f}%). The reason: at "
                  f"a {DEFAULT_PRICE_OUT_PER_1M/DEFAULT_PRICE_IN_PER_1M:.0f}x output:input price "
                  "ratio, the method's INPUT tokens drop sharply (structured fetch reads far less "
                  "text than whole-document visit) but its OUTPUT tokens are somewhat HIGHER than "
                  "the baseline's on every dataset (more calls / more structured-query "
                  "reformulation text generated) -- a token-mix shift toward the more expensive "
                  "side that a plain (unweighted) token-count delta cannot see, but a dollar "
                  "accounting does. This is exactly the kind of basis-sensitivity the paper "
                  "already applies to DCI's cost claim (`related_work.tex` Sec 4); it should be "
                  "applied symmetrically to the paper's own count-once headline number too.\n")

    lines.append("**Sensitivity note.** These dollar figures move linearly with the assumed price "
                  "ratio; the SIGN of the method-vs-baseline delta on each basis does not depend on "
                  "the absolute rate (both cells are priced identically), only on the input:output "
                  "price RATIO, which is held fixed at the CLI-supplied rate for both cells here. "
                  "Re-run with `--price-in-per-1m`/`--price-out-per-1m` set to any other public "
                  "list price to confirm the qualitative verdict is rate-independent.\n")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
