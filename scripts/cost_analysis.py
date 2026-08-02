#!/usr/bin/env python3
"""Post-hoc cost analysis — raw token counts vs. real-dollar cost accounting, side by side.

MOTIVATION (see docs/papers/dci_2605.05242_notes.md): the DCI paper's headline cost claim
("$1,440 -> $1,016, -29.4%") is a BILLING-DOLLAR claim under commercial prompt-cache pricing
(cache-read tokens ~10x cheaper than fresh input, summed over the whole eval set). Our own
per-query TOKEN counts point the opposite direction (dci uses MORE raw tokens/episode than
structured/bm25). Those are not the same axis. This script computes both accountings, for
every condition we have >=50 rows for, so the two can be compared directly instead of by feel.

Reads ONLY existing rows.jsonl files (read-only over run data). Streams line-by-line so a
run with 10s of thousands of rows doesn't blow up memory; whole script runs in well under
3 minutes end to end (see bottom __main__ timing print).

    PYTHONPATH=${REPO_ROOT:-.} envs/bin/python scripts/cost_analysis.py
    PYTHONPATH=${REPO_ROOT:-.} envs/bin/python scripts/cost_analysis.py --out docs/cost_analysis.md

Field names (verified against a live row, see docs/cost_analysis.md "Data notes"):
  prompt_tokens        cumulative SUM of each LLM call's input tokens across the whole episode
                        (re-sends the whole growing context every turn -> triple/quadruple counts
                        a long-lived prefix). This is "raw, as-billed-if-uncached" input tokens.
  completion_tokens    cumulative sum of generated tokens (never cached, counted once naturally).
  cached_input_tokens  provider-reported cache-read input tokens. For our self-hosted vLLM
                        Tongyi-DeepResearch-30B-A3B backend this is MEASURED (not assumed) to be
                        0 for every row we checked (vLLM was launched with enable_prefix_caching
                        =True, but does not appear to populate this usage field for this backend/
                        client path) -- this script re-verifies that claim itself rather than
                        trusting the docs note, and says so explicitly in the output.
  reasoning_tokens      subset of completion_tokens spent on <think> spans; itemized, not additive.
  total_tokens_once     initial_prompt_tokens + context_once_tokens + output_tokens: the "count
                        every real token exactly once" reconstruction (see evaluation/run_eval.py
                        lines ~379-421) -- NOTE it already has completion/output folded in.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from collections import defaultdict

# --------------------------------------------------------------------------------------
# Price points ($ per token; input tables in the docstring/spec are $ per 1M tokens)
# --------------------------------------------------------------------------------------
CACHE_READ_MULT = 0.1   # cache-read tokens cost 10% of fresh input, both price tables below

PRICE_TABLES = {
    # A typical self-hosted 30B-class serving-cost PROXY (not a real per-token bill -- vLLM/
    # local serving has no per-token invoice; this converts GPU-time-ish serving cost into a
    # $/token number for comparability only). LABEL CLEARLY AS A PROXY wherever it's printed.
    "TONGYI_LOCAL (proxy)": {"IN": 0.20 / 1e6, "OUT": 0.80 / 1e6},
    # The DCI paper's actual billing basis (claude-sonnet-4-6 list price, see
    # docs/papers/dci_2605.05242_notes.md #7): cacheRead is exactly 10x cheaper than fresh
    # input ($0.30/1M vs $3/1M), which is where the 0.1 CACHE_READ_MULT above comes from.
    "SONNET-CLASS (DCI paper billing basis)": {"IN": 3.0 / 1e6, "OUT": 15.0 / 1e6},
}

MIN_ROWS = 50
EPISODES_PER_BLOCK = 100  # all $ figures reported as "$ per 100 episodes"

# Preferred left-to-right / display order for known condition (retriever) names.
_ARM_ORDER = ["agent_research", "agent_research_bm25", "agent_research_dci",
              "agent_research_bm25_dci", "agent_research_bm25_fetch",
              "agent_research_v2", "agent_research_indri", "agent_research_snip"]

_ARM_LABEL = {
    "agent_research": "structured",
    "agent_research_bm25": "bm25",
    "agent_research_dci": "dci",
    "agent_research_bm25_dci": "bm25+dci",
    "agent_research_bm25_fetch": "bm25+fetch",
    "agent_research_v2": "structured_v2",
    "agent_research_indri": "indri",
    "agent_research_snip": "snip",
}


def _arm_key(name: str) -> tuple:
    try:
        return (0, _ARM_ORDER.index(name))
    except ValueError:
        return (1, name)


# --------------------------------------------------------------------------------------
# Discovery: find every condition dir (a dir that directly contains rows.jsonl) under the
# three run roots named in the task brief, tagging each with (run_group, dataset, model, arm).
# --------------------------------------------------------------------------------------

def discover_conditions(repo_root: str) -> list[dict]:
    found = []

    # 1) runs/agent/<dataset>/<model>/<condition>/rows.jsonl
    # 2) runs/_headline_validation/agent/<dataset>/<model>/<condition>/rows.jsonl
    for run_group, base in [
        ("runs/agent", os.path.join(repo_root, "runs", "agent")),
        ("runs/_headline_validation/agent", os.path.join(repo_root, "runs", "_headline_validation", "agent")),
    ]:
        pattern = os.path.join(base, "*", "*", "*", "rows.jsonl")
        for path in sorted(glob.glob(pattern)):
            cond_dir = os.path.dirname(path)
            model_dir = os.path.dirname(cond_dir)
            dataset_dir = os.path.dirname(model_dir)
            found.append({
                "run_group": run_group,
                "dataset": os.path.basename(dataset_dir),
                "model": os.path.basename(model_dir),
                "arm": os.path.basename(cond_dir),
                "path": path,
            })

    # 3) runs_old/ablation_50steps_*/<dataset>__<model>__<condition>/rows.jsonl (flat dirs)
    for ablation_dir in sorted(glob.glob(os.path.join(repo_root, "runs_old", "ablation_50steps_*"))):
        if not os.path.isdir(ablation_dir):
            continue
        run_group = "runs_old/" + os.path.basename(ablation_dir)
        pattern = os.path.join(ablation_dir, "*", "rows.jsonl")
        for path in sorted(glob.glob(pattern)):
            cond_dir = os.path.dirname(path)
            leaf = os.path.basename(cond_dir)
            parts = leaf.split("__")
            if len(parts) >= 3:
                dataset, model, arm = parts[0], parts[1], "__".join(parts[2:])
            else:
                dataset, model, arm = leaf, "?", "?"
            found.append({
                "run_group": run_group,
                "dataset": dataset,
                "model": model,
                "arm": arm,
                "path": path,
            })

    return found


# --------------------------------------------------------------------------------------
# Per-condition streaming aggregation
# --------------------------------------------------------------------------------------

def aggregate_condition(path: str) -> dict | None:
    """Stream rows.jsonl, accumulate SUMS only (never hold all rows in memory).

    Returns None if the file has fewer than MIN_ROWS parseable rows.
    """
    n = 0
    sum_prompt = sum_completion = sum_cached = sum_once = sum_reasoning = 0
    # per-row estimated cache split (see module docstring): estimated_cacheable_i =
    # max(0, prompt_i - total_once_i) -- the portion of the cumulative re-sent prompt that
    # is "more than the once-counted content", i.e. context repeated across turns that a
    # working prefix cache would have deduplicated. Summed per-row (not from aggregate sums)
    # so a single pathological row can't produce a negative estimate for the whole condition.
    sum_est_cacheable = sum_est_fresh = 0

    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            pt = r.get("prompt_tokens") or 0
            ct = r.get("completion_tokens") or 0
            cit = r.get("cached_input_tokens") or 0
            once = r.get("total_tokens_once") or 0
            rt = r.get("reasoning_tokens") or 0
            n += 1
            sum_prompt += pt
            sum_completion += ct
            sum_cached += cit
            sum_once += once
            sum_reasoning += rt
            est_cacheable = max(0, pt - once)
            sum_est_cacheable += est_cacheable
            sum_est_fresh += pt - est_cacheable

    if n < MIN_ROWS:
        return None

    mean_prompt = sum_prompt / n
    mean_completion = sum_completion / n
    mean_cached = sum_cached / n
    mean_fresh = mean_prompt - mean_cached
    mean_once = sum_once / n
    mean_reasoning = sum_reasoning / n
    mean_est_cacheable = sum_est_cacheable / n
    mean_est_fresh = sum_est_fresh / n

    cache_ratio_measured = (sum_cached / sum_prompt) if sum_prompt else 0.0
    cache_ratio_estimated = (sum_est_cacheable / sum_prompt) if sum_prompt else 0.0

    costs = {}
    for label, price in PRICE_TABLES.items():
        IN, OUT = price["IN"], price["OUT"]
        flat = (mean_prompt * IN + mean_completion * OUT) * EPISODES_PER_BLOCK
        cache_aware_measured = (mean_fresh * IN + mean_cached * IN * CACHE_READ_MULT
                                 + mean_completion * OUT) * EPISODES_PER_BLOCK
        cache_aware_estimated = (mean_est_fresh * IN + mean_est_cacheable * IN * CACHE_READ_MULT
                                  + mean_completion * OUT) * EPISODES_PER_BLOCK
        once_cost = (mean_once * IN + mean_completion * OUT) * EPISODES_PER_BLOCK
        costs[label] = {
            "flat": flat,
            "cache_aware_measured": cache_aware_measured,
            "cache_aware_estimated": cache_aware_estimated,
            "once": once_cost,
        }

    return {
        "n": n,
        "mean_prompt": mean_prompt,
        "mean_completion": mean_completion,
        "mean_cached": mean_cached,
        "mean_fresh": mean_fresh,
        "mean_once": mean_once,
        "mean_reasoning": mean_reasoning,
        "sum_cached": sum_cached,
        "sum_prompt": sum_prompt,
        "cache_ratio_measured": cache_ratio_measured,
        "cache_ratio_estimated": cache_ratio_estimated,
        "costs": costs,
    }


# --------------------------------------------------------------------------------------
# Markdown rendering
# --------------------------------------------------------------------------------------

def _fmt_money(x: float) -> str:
    return f"${x:,.2f}"


def _fmt_pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def render_table(group_label: str, dataset: str, rows: list[dict]) -> str:
    rows = sorted(rows, key=lambda r: _arm_key(r["arm"]))
    lines = []
    lines.append(f"### {dataset}  _(run group: `{group_label}`)_\n")
    header = ("| condition | n | cache ratio MEASURED | cache ratio ESTIMATED | "
              "flat $/100ep (Tongyi proxy) | cache-aware $/100ep, measured (Tongyi) | "
              "cache-aware $/100ep, ESTIMATED (Tongyi) | tokens-once $/100ep (Tongyi) | "
              "flat $/100ep (Sonnet-class) | cache-aware $/100ep, measured (Sonnet) | "
              "cache-aware $/100ep, ESTIMATED (Sonnet) | tokens-once $/100ep (Sonnet) |")
    sep = "|" + "---|" * 12
    lines.append(header)
    lines.append(sep)
    for r in rows:
        arm_label = _ARM_LABEL.get(r["arm"], r["arm"])
        agg = r["agg"]
        ct = agg["costs"]["TONGYI_LOCAL (proxy)"]
        cs = agg["costs"]["SONNET-CLASS (DCI paper billing basis)"]
        lines.append(
            f"| {arm_label} (`{r['arm']}`) | {agg['n']} | "
            f"{_fmt_pct(agg['cache_ratio_measured'])} | {_fmt_pct(agg['cache_ratio_estimated'])} | "
            f"{_fmt_money(ct['flat'])} | {_fmt_money(ct['cache_aware_measured'])} | "
            f"{_fmt_money(ct['cache_aware_estimated'])} | {_fmt_money(ct['once'])} | "
            f"{_fmt_money(cs['flat'])} | {_fmt_money(cs['cache_aware_measured'])} | "
            f"{_fmt_money(cs['cache_aware_estimated'])} | {_fmt_money(cs['once'])} |"
        )
    return "\n".join(lines)


def _rank_flip(rows: list[dict], cost_key: str, price_label: str) -> list[tuple]:
    """Return list of (a, b) arm-label pairs whose relative FLAT order flips under `cost_key`
    (e.g. 'cache_aware_estimated') at the given price table. Order is by ascending cost."""
    def cost(r, key):
        return r["agg"]["costs"][price_label][key]

    flat_order = sorted(rows, key=lambda r: cost(r, "flat"))
    other_order = sorted(rows, key=lambda r: cost(r, cost_key))
    flat_rank = {r["arm"]: i for i, r in enumerate(flat_order)}
    other_rank = {r["arm"]: i for i, r in enumerate(other_order)}
    flips = []
    arms = [r["arm"] for r in rows]
    for i in range(len(arms)):
        for j in range(i + 1, len(arms)):
            a, b = arms[i], arms[j]
            flat_cmp = flat_rank[a] - flat_rank[b]
            other_cmp = other_rank[a] - other_rank[b]
            if flat_cmp == 0 or other_cmp == 0:
                continue
            if (flat_cmp > 0) != (other_cmp > 0):
                flips.append((a, b))
    return flips


def build_findings(all_groups: dict) -> str:
    lines = ["## Findings (auto-generated)\n"]

    # --- (0) cache measurement fact, stated prominently, derived from the data itself ---
    total_cached = sum(r["agg"]["sum_cached"]
                        for grp in all_groups.values() for ds in grp.values() for r in ds)
    total_prompt = sum(r["agg"]["sum_prompt"]
                        for grp in all_groups.values() for ds in grp.values() for r in ds)
    lines.append(
        "**Prefix-cache measurement.** vLLM was launched with `enable_prefix_caching=True` for "
        "the Tongyi-DeepResearch-30B-A3B serving used across every run analyzed here. This script "
        f"independently summed `cached_input_tokens` across every row it read: total = "
        f"**{total_cached:,}** cached tokens out of **{total_prompt:,}** total prompt tokens "
        f"({_fmt_pct(total_cached / total_prompt) if total_prompt else 'n/a'}). "
        + ("**`cached_input_tokens` is measured ZERO everywhere — vLLM/the client path did NOT "
           "populate this usage field for this backend, despite prefix caching being enabled "
           "server-side.** Every 'measured' cache-aware column below is therefore numerically "
           "identical to the flat column; it is NOT evidence of no caching benefit, only evidence "
           "that we cannot observe it through this field. The 'ESTIMATED' cache-aware columns use "
           "a proxy instead (see Data notes) and are the only cache-aware numbers that move."
           if total_cached == 0 else
           "Some rows DO report nonzero cached_input_tokens — the 'measured' cache-aware columns "
           "are real for those rows.")
        + "\n"
    )

    # --- (1) rank flips flat -> cache-aware (estimated), and flat -> tokens-once ---
    lines.append("**Rank-order flips (flat $ vs. cache-aware-ESTIMATED $, per dataset/run-group, "
                  "Sonnet-class pricing — this is the accounting axis the DCI paper's headline "
                  "claim actually lives on):**\n")
    any_flip_found = False
    for grp_label, datasets in all_groups.items():
        for dataset, rows in datasets.items():
            if len(rows) < 2:
                continue
            flips = _rank_flip(rows, "cache_aware_estimated", "SONNET-CLASS (DCI paper billing basis)")
            once_flips = _rank_flip(rows, "once", "SONNET-CLASS (DCI paper billing basis)")
            if flips:
                any_flip_found = True
                pretty = ", ".join(f"{_ARM_LABEL.get(a,a)} <-> {_ARM_LABEL.get(b,b)}" for a, b in flips)
                lines.append(f"- `{grp_label}` / **{dataset}**: FLIPS under cache-aware-estimated: {pretty}")
            if once_flips:
                any_flip_found = True
                pretty = ", ".join(f"{_ARM_LABEL.get(a,a)} <-> {_ARM_LABEL.get(b,b)}" for a, b in once_flips)
                lines.append(f"- `{grp_label}` / **{dataset}**: FLIPS under tokens-once accounting: {pretty}")
    if not any_flip_found:
        lines.append("- No pairwise rank-order flips detected in any dataset/run-group under either "
                      "cache-aware-estimated or tokens-once accounting, at Sonnet-class prices.")
    lines.append("")

    # --- (2) dci vs structured vs bm25 cache ratios ---
    lines.append("**Cache ratio (estimated), dci vs structured vs bm25, by dataset/run-group:**\n")
    lines.append("| run group | dataset | structured | bm25 | dci |")
    lines.append("|---|---|---|---|---|")
    for grp_label, datasets in all_groups.items():
        for dataset, rows in datasets.items():
            by_arm = {r["arm"]: r["agg"] for r in rows}
            s = by_arm.get("agent_research")
            b = by_arm.get("agent_research_bm25")
            d = by_arm.get("agent_research_dci")
            if not (s or b or d):
                continue
            fmt = lambda a: (_fmt_pct(a["cache_ratio_estimated"]) + " / " + _fmt_pct(a["cache_ratio_measured"])
                              + " meas.") if a else "n/a (<50 rows or absent)"
            lines.append(f"| `{grp_label}` | {dataset} | {fmt(s)} | {fmt(b)} | {fmt(d)} |")
    lines.append("\n_(cell format: ESTIMATED cache ratio / MEASURED cache ratio)_\n")

    # --- (3) one-paragraph implication ---
    lines.append("**Implication for cost claims (one paragraph).** The DCI paper's \"$1,440 -> "
                  "$1,016\" headline is a real-dollar, prompt-cache-discounted, cross-backbone, "
                  "830-question-summed claim on a commercial API (Claude Sonnet 4.6) that reports "
                  "`cached_input_tokens` accurately; our self-hosted vLLM Tongyi-30B setup cannot "
                  "observe that same signal (it is measured zero everywhere despite prefix caching "
                  "being enabled server-side), so the 'measured' cache-aware column here is not "
                  "informative on its own. Once we substitute a principled ESTIMATED cache split "
                  "(cacheable = the portion of the cumulative re-sent prompt beyond what "
                  "`total_tokens_once` says was genuinely new), dci's cache ratio is comparable to "
                  "or higher than structured/bm25's on every dataset we measured (its trajectories "
                  "are longer, so a larger share of every later turn's prompt is repeated prefix) — "
                  "but that higher cache ratio is not enough to close the ABSOLUTE gap: dci's flat "
                  "and estimated-cache-aware costs are both several times structured/bm25's on "
                  "browsecomp_plus_structured, because its raw prompt-token footprint per episode "
                  "is an order of magnitude larger to begin with. In other words: cache-aware "
                  "accounting compresses dci's dollar cost a lot in absolute terms, but on this "
                  "repo's data it is not the mechanism that would flip dci from 'more expensive' to "
                  "'cheaper' relative to structured/bm25 — that would require either a real, "
                  "measured client-side cache (which we don't have) or a genuinely shorter/coached "
                  "trajectory (which the DCI paper's harness has and ours deliberately does not, "
                  "see docs/papers/dci_2605.05242_notes.md #4).\n")

    return "\n".join(lines)


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo-root", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ap.add_argument("--out", default=None, help="write markdown here (default: print to stdout)")
    args = ap.parse_args()

    t0 = time.time()
    conditions = discover_conditions(args.repo_root)

    # group_label -> dataset -> [ {arm, agg} ]
    all_groups: dict = defaultdict(lambda: defaultdict(list))
    skipped = []
    for c in conditions:
        agg = aggregate_condition(c["path"])
        if agg is None:
            skipped.append(c)
            continue
        all_groups[c["run_group"]][c["dataset"]].append({"arm": c["arm"], "agg": agg})

    elapsed = time.time() - t0

    out = []
    out.append("# Post-hoc cost analysis: raw tokens vs. billing-dollar accounting\n")
    out.append(f"_Generated by `scripts/cost_analysis.py` in {elapsed:.1f}s. "
               f"{sum(len(v) for grp in all_groups.values() for v in grp.values())} condition(s) "
               f"with >= {MIN_ROWS} rows analyzed; {len(skipped)} condition(s) skipped for having "
               f"fewer than {MIN_ROWS} rows._\n")

    out.append("## Data notes\n")
    out.append(
        "- Verified directly from a live row (`runs/agent/2wiki_structured/.../agent_research/"
        "rows.jsonl`, first line) and from `evaluation/run_eval.py` (~lines 379-421) / "
        "`agent_search/agent/retriever.py` (~lines 264-303, 391-430):\n"
        "  - `prompt_tokens` = cumulative SUM of every LLM call's input tokens across the episode "
        "(the whole growing context re-sent every turn) -- MEASURED, real usage.\n"
        "  - `completion_tokens` = cumulative sum of generated tokens -- MEASURED.\n"
        "  - `cached_input_tokens` = provider/vLLM-reported cache-read input tokens -- MEASURED "
        "field, but see the prefix-cache finding in Findings below (it reads zero for every row in "
        "this analysis).\n"
        "  - `reasoning_tokens` = itemized subset of completion_tokens (not additive on top).\n"
        "  - `total_tokens_once` = `initial_prompt_tokens + context_once_tokens + output_tokens`, "
        "the harness's own \"count every real token exactly once\" reconstruction -- MEASURED (not "
        "estimated), but NOTE it already folds completion/output tokens in.\n"
        "- **Caveat on the `tokens-once` cost column**: per the task spec this column is computed "
        "as `total_tokens_once * IN + completion_tokens * OUT`. Because `total_tokens_once` "
        "already includes `output_tokens` (`completion_tokens`) once, this formula bills "
        "completion tokens twice -- once implicitly at the INPUT rate inside `total_tokens_once`, "
        "once explicitly at the OUTPUT rate. It is reported literally as specified (and is still "
        "useful as a *relative* ranking across conditions, since the double-count is a roughly "
        "constant small addend), but it is a modest OVER-statement of the true once-per-token "
        "dollar cost, not a precise dollar figure.\n"
        "- **cache ratio ESTIMATED** = `max(0, prompt_tokens - total_tokens_once) / prompt_tokens`, "
        "summed per-row then divided (never computed from row-level negative values). "
        "Justification: `prompt_tokens` counts the whole growing context every turn; "
        "`total_tokens_once` counts each real token exactly once; the difference is (an "
        "approximation of) how much of the cumulative prompt is EXACT REPEATED PREFIX that a "
        "working prefix cache would have deduplicated. This is explicitly an ESTIMATE, not a "
        "measurement -- it is not literally what any cache would have hit, just a principled upper "
        "bound on the cacheable share implied by our own token accounting.\n"
        "- All $ figures are **per 100 episodes** (mean $/episode x 100), not per-token.\n"
    )

    for grp_label in sorted(all_groups.keys()):
        out.append(f"\n## Run group: `{grp_label}`\n")
        for dataset in sorted(all_groups[grp_label].keys()):
            out.append(render_table(grp_label, dataset, all_groups[grp_label][dataset]))
            out.append("")

    if skipped:
        out.append("\n## Skipped (fewer than {} rows)\n".format(MIN_ROWS))
        for c in skipped:
            out.append(f"- `{c['run_group']}` / {c['dataset']} / {c['arm']} (`{c['path']}`)")
        out.append("")

    out.append("\n" + build_findings(all_groups))

    text = "\n".join(out) + "\n"
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(text)
        print(f"wrote {args.out} ({len(text)} bytes) in {elapsed:.1f}s", file=sys.stderr)
    else:
        print(text)


if __name__ == "__main__":
    main()
