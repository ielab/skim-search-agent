#!/usr/bin/env python
"""Fetch-error audit: quantify the damage from the stale `bql_browsecomp.md` manual.

BACKGROUND (verified separately, not re-derived here): `agent_search/prompts/skills/
bql_browsecomp.md` tells the agent repeatedly that BrowseComp-Plus documents have NO named
sections and no infobox ("there are no named sections", "This corpus has NO `section` or
`infobox` field", "there is nothing else to fetch ... don't hunt for a 'section' or
'infobox' name"). That manual was written against the PRE-correction corpus build. The
corpus actually in `data/browsecomp_plus_structured/corpus.jsonl` has real, varied section
names on the large majority of documents (94.2% measured over 2000 docs), and the `search`
tool's listing prints those section names, and `fetch` REQUIRES one of them (or errors).
So an agent following the manual's advice literally (fetch "body" — the manual's own
suggested part name, since it insists there is only one flat slice) will 0-hit/error against
a real, multi-section document.

THIS SCRIPT quantifies:
  Part 1 -- fetch/read tool-call error rates, per cell (5 cells: 3 on
            browsecomp_plus_structured, 1 each on hotpotqa_structured/musique_structured, all
            model dir Tongyi-DeepResearch-30B-A3B, tier `_headline_validation`).
  Part 2 -- which per-dataset MANUAL each cell's system prompt actually renders (via
            `agent_search.prompts.loader.load_condition`, the same composer the eval harness
            uses -- not a re-implementation), and whether browsecomp's error rate is
            anomalously high relative to the two Wikipedia-backed datasets.
  Part 3 -- LLM-call and (where honestly attributable) token cost of the errored fetch calls,
            within the browsecomp Sieve cell (`agent_research_bql_dense_snip`).
  Part 4 -- a DESCRIPTIVE (explicitly non-causal) split of exact-match accuracy by whether an
            episode had >=1 fetch error, within the browsecomp Sieve cell only.

SCHEMA, established by direct inspection (`rows.jsonl`, one row printed with `json.loads`
before writing any parsing logic -- see the conversation this script was written from):
  row['trajectory'] : list[dict], one entry per LLM-driven agent step, IN ORDER. Each entry
    has (among other keys) 'action' (str, the tool name the model's call was DISPATCHED as
    -- may be a canonical tool name like 'fetch_bqlds', a cross-condition alias like plain
    'fetch', or a hallucinated/typo'd name like 'fetch_bqhs' or 'functions.fetch_bqlds'),
    'observation' (str, the literal tool-result text appended to the transcript), and the
    PER-STEP (step-summed, i.e. NOT deduplicated) 'prompt_tokens'/'completion_tokens' for
    the single LLM call that produced this step. Confirmed: sum(step['prompt_tokens'] for
    step in row['trajectory']) == row['prompt_tokens'] exactly (and same for
    completion_tokens) on every row sampled -- i.e. this is the STEP-SUMMED accounting basis
    `analysis/cost_dollars.py` documents, not count-once.
  row['llm_calls'] == row['n_steps'] == len(row['trajectory']) on every row sampled (the
    'answer' action is itself the final step/call).
  row['answer_em'] : float, 0.0 or 1.0 (binary exact match) -- confirmed by inspection.
  row['total_tokens_once'] / row['context_once_tokens'] / row['initial_prompt_tokens'] :
    EPISODE-level count-once fields (see analysis/cost_dollars.py's documented definition:
    tok_in_once = initial_prompt_tokens + context_once_tokens, each real token the model
    ever saw counted exactly once across the WHOLE episode). There is no per-call count-once
    field anywhere in the schema -- count-once dedup is a whole-episode computation (it
    dedupes repeated document text reappearing in the growing context across calls), so it
    is NOT decomposable per call. Part 3 states this plainly and does not fabricate a
    per-call count-once split.

HOW A "FETCH CALL" IS IDENTIFIED: a trajectory step whose `action` contains the substring
"fetch" (case-insensitive). Verified against `agent_search/agent/tools/doc_research.py`'s
`DocSearchFetch.run()`: the tool names 'fetch', 'fetch_v2', 'fetch_s', 'fetch_bqlds',
'fetch_bqldf' all dispatch to the SAME `fetch()` method, whose response always starts with
the literal prefix "fetch:\\n" -- confirmed by grep (doc_research.py:593) and by sampling
observations for each of these action strings. Off-list variants that still contain "fetch"
(typos/hallucinations such as 'fetch_bqhs', or a badly-formed 'functions.fetch_bqlds') fall
through `run()`'s dispatch to "ERROR: unknown tool ...— these are counted as fetch calls
(the model was clearly attempting one) and as errors, broken out under the 'unknown_tool'
kind so they are not silently merged with genuine section-lookup errors.

HOW AN "ERROR" IS IDENTIFIED: the literal substring "ERROR:" (colon included) anywhere in
the step's observation. Verified against doc_research.py: every error path in `fetch()`/
`run()` emits exactly this prefix ("ERROR: no section ...", "ERROR: ... is ambiguous:",
"ERROR: no such doc ...", "ERROR: ambiguous doc ...", "ERROR: rank ... out of range",
"ERROR: fetch needs at least one ...", "ERROR: bad spec ...", "ERROR: unknown tool ...",
"ERROR: <ExceptionType>: ..."), and manual sampling of ~15 matched observations found no
false positive (no fetched document body was observed to contain the literal string
"ERROR:"). A CALL is scored as erroring if "ERROR:" appears ANYWHERE in its observation --
i.e. at call granularity, not per-spec: multi-spec fetch calls (>1 [rank,section] pair in
one call) are rare (6/11856 = 0.05% on the browsecomp Sieve cell) and a call with a mix of
succeeding and erroring specs (1/11856 calls) is counted as an erroring call either way.

Run:
    PYTHONPATH=. envs/bin/python analysis/fetch_error_audit.py
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

RUNS = REPO_ROOT / "runs" / "_headline_validation" / "agent"
MODEL_DIR = "Tongyi-DeepResearch-30B-A3B"

# (label, dataset, condition, field_profile) -- field_profile is what the eval harness passes
# as `profile=` to `agent_search.prompts.loader.load_condition` for THIS dataset (see
# evaluation/datasets.py's `register_dataset(..., field_profile=...)` calls); None means "no
# override, falls back to the domain" (== "general" for every dataset here).
CELLS = [
    ("browsecomp_plus_structured / agent_research_bql_dense_snip (Sieve)",
     "browsecomp_plus_structured", "agent_research_bql_dense_snip", "browsecomp"),
    ("browsecomp_plus_structured / agent_research_bm25_fetch_snip (sparse-only control)",
     "browsecomp_plus_structured", "agent_research_bm25_fetch_snip", "browsecomp"),
    ("browsecomp_plus_structured / agent_research_snip (no-dense rung)",
     "browsecomp_plus_structured", "agent_research_snip", "browsecomp"),
    ("hotpotqa_structured / agent_research_bql_dense_snip",
     "hotpotqa_structured", "agent_research_bql_dense_snip", "wiki"),
    ("musique_structured / agent_research_bql_dense_snip",
     "musique_structured", "agent_research_bql_dense_snip", "wiki"),
]

SIEVE_BROWSECOMP = CELLS[0]

ERROR_KIND_PATTERNS = [
    ("no_section", "no section "),
    ("ambiguous_section", "is ambiguous:"),
    ("ambiguous_doc", "ambiguous doc "),
    ("no_such_doc", "no such doc "),
    ("no_prior_search", "no prior search"),
    ("rank_out_of_range", "out of range"),
    ("empty_specs", "needs at least one"),
    ("bad_spec", "bad spec "),
    ("unknown_tool", "unknown tool "),
]


def classify_error(obs: str) -> tuple[str, str]:
    """Return (kind, the ERROR:...  snippet used to classify it) for the FIRST ERROR: in obs."""
    idx = obs.find("ERROR:")
    if idx == -1:
        return "", ""
    snippet = obs[idx:idx + 160]
    for kind, needle in ERROR_KIND_PATTERNS:
        if needle in snippet:
            return kind, snippet
    return "other", snippet


def rows_path(dataset: str, condition: str) -> Path:
    return RUNS / dataset / MODEL_DIR / condition / "rows.jsonl"


def audit_cell(dataset: str, condition: str) -> dict:
    """Single streaming pass over rows.jsonl. Returns fetch-call / error / episode stats,
    error-kind breakdown, and (for Part 4) per-episode (has_fetch_error, answer_em)."""
    path = rows_path(dataset, condition)
    n_episodes = 0
    fetch_calls_total = 0
    fetch_calls_error = 0
    episodes_with_error = 0
    kind_counts: Counter = Counter()
    exact_string_counts: Counter = Counter()
    llm_calls_total = 0
    error_step_llm_calls = 0          # = error_step count, 1 LLM call per step
    error_step_tokens = 0             # step-summed prompt+completion tokens on erroring steps
    total_step_tokens = 0             # step-summed prompt+completion tokens, whole cell
    episode_records = []              # (has_fetch_error: bool, answer_em: float, n_fetch_calls: int)

    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            n_episodes += 1
            llm_calls_total += row.get("llm_calls") or len(row.get("trajectory") or [])
            has_error_this_ep = False
            n_fetch_this_ep = 0
            for step in row.get("trajectory") or []:
                action = step.get("action") or ""
                p_tok = step.get("prompt_tokens") or 0
                c_tok = step.get("completion_tokens") or 0
                total_step_tokens += p_tok + c_tok
                if "fetch" not in action.lower():
                    continue
                fetch_calls_total += 1
                n_fetch_this_ep += 1
                obs = step.get("observation") or ""
                if "ERROR:" in obs:
                    fetch_calls_error += 1
                    has_error_this_ep = True
                    kind, snippet = classify_error(obs)
                    kind_counts[kind] += 1
                    # normalize the exact-string tally to the first line only (the doc id
                    # inside varies per call; the literal template + argument is still kept,
                    # exact repeats DO occur e.g. same bad section retried).
                    first_line = obs.splitlines()[-1] if obs.splitlines() else obs
                    exact_string_counts[first_line.strip()] += 1
                    error_step_llm_calls += 1
                    error_step_tokens += p_tok + c_tok
            if has_error_this_ep:
                episodes_with_error += 1
            episode_records.append((has_error_this_ep, row.get("answer_em"), n_fetch_this_ep))

    return {
        "dataset": dataset,
        "condition": condition,
        "path": str(path),
        "n_episodes": n_episodes,
        "fetch_calls_total": fetch_calls_total,
        "fetch_calls_error": fetch_calls_error,
        "fetch_error_pct": 100.0 * fetch_calls_error / fetch_calls_total if fetch_calls_total else 0.0,
        "episodes_with_error": episodes_with_error,
        "episodes_with_error_pct": 100.0 * episodes_with_error / n_episodes if n_episodes else 0.0,
        "error_kind_counts": dict(kind_counts.most_common()),
        "most_common_error_string": (exact_string_counts.most_common(1)[0] if exact_string_counts else None),
        "llm_calls_total": llm_calls_total,
        "error_step_llm_calls": error_step_llm_calls,
        "error_step_tokens_step_summed": error_step_tokens,
        "total_step_tokens_step_summed": total_step_tokens,
        "episode_records": episode_records,
    }


def manual_report() -> dict:
    """What manual each cell's condition x field_profile actually renders, via the SAME
    composer (`agent_search.prompts.loader.load_condition`) the eval harness uses -- not a
    re-implementation. Flags whether the composed system prompt contains the stale
    "no named sections" BrowseComp-only claim."""
    from agent_search.prompts.loader import load_condition

    out = {}
    for label, dataset, condition, profile in CELLS:
        cond_name = condition.replace("agent_", "", 1) if condition.startswith("agent_") else condition
        # conditions.yaml condition names drop the "agent_" run-dir prefix; verify below.
        prof = load_condition(cond_name, domain="general", profile=profile)
        sys_text = prof.system
        stale_claim = ("no named sections" in sys_text.lower()
                        or "there is nothing else to fetch" in sys_text.lower())
        # which manual file(s) actually got concatenated in -- infer from toolset's tool list
        out[label] = {
            "condition_name_resolved": cond_name,
            "toolset": prof.toolset,
            "tool_names": list(prof.tool_names),
            "field_profile_passed": profile,
            "system_chars": len(sys_text),
            "contains_stale_no_sections_claim": stale_claim,
        }
    return out


def chi_square_or_fisher(a_err_correct, a_err_wrong, b_noerr_correct, b_noerr_wrong):
    from scipy.stats import chi2_contingency, fisher_exact
    table = [[a_err_correct, a_err_wrong], [b_noerr_correct, b_noerr_wrong]]
    chi2, chi_p, dof, expected = chi2_contingency(table, correction=True)
    _, fisher_p = fisher_exact(table)
    min_expected = expected.min()
    return {
        "table": table,
        "chi2_stat": chi2,
        "chi2_p": chi_p,
        "fisher_p": fisher_p,
        "min_expected_cell": float(min_expected),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-json", default=str(REPO_ROOT / "analysis" / "fetch_error_audit_data.json"))
    ap.add_argument("--out-md", default=str(REPO_ROOT / "analysis" / "fetch_error_audit.md"))
    args = ap.parse_args()

    print("Auditing 5 cells (this streams rows.jsonl files up to ~1GB; may take a minute)...",
          file=sys.stderr)
    cell_results = {}
    for label, dataset, condition, profile in CELLS:
        print(f"  {label} ...", file=sys.stderr)
        r = audit_cell(dataset, condition)
        cell_results[label] = r

    print("Composing system prompts for the manual report...", file=sys.stderr)
    manuals = manual_report()

    # ---- Part 3: wasted-effort estimate, browsecomp Sieve cell -------------------------------
    sieve_label = SIEVE_BROWSECOMP[0]
    sv = cell_results[sieve_label]
    part3 = {
        "fetch_calls_total": sv["fetch_calls_total"],
        "fetch_calls_error": sv["fetch_calls_error"],
        "fetch_call_error_fraction": sv["fetch_calls_error"] / sv["fetch_calls_total"] if sv["fetch_calls_total"] else 0.0,
        "llm_calls_total_cell": sv["llm_calls_total"],
        "llm_calls_that_produced_an_errored_fetch": sv["error_step_llm_calls"],
        "llm_call_fraction_on_errored_fetch": sv["error_step_llm_calls"] / sv["llm_calls_total"] if sv["llm_calls_total"] else 0.0,
        "count_once_tokens_attributable_per_call": False,
        "count_once_note": (
            "row['total_tokens_once']/'context_once_tokens' are EPISODE-level dedup "
            "quantities (each real token the model ever saw, counted once across the WHOLE "
            "episode) -- there is no per-call count-once field, and count-once dedup cannot "
            "be decomposed to a single call's share without re-simulating the dedup logic "
            "against a specific call's marginal contribution, which this script does not "
            "attempt. NOT reported. The step-summed figures below are reported instead, "
            "explicitly labeled as step-summed (NOT count-once, per analysis/cost_dollars.py's "
            "existing two-basis distinction)."
        ),
        "step_summed_tokens_on_errored_fetch_steps": sv["error_step_tokens_step_summed"],
        "step_summed_tokens_total_cell": sv["total_step_tokens_step_summed"],
        "step_summed_token_fraction_on_errored_fetch": (
            sv["error_step_tokens_step_summed"] / sv["total_step_tokens_step_summed"]
            if sv["total_step_tokens_step_summed"] else 0.0
        ),
    }

    # ---- Part 4: descriptive accuracy split, browsecomp Sieve cell only ----------------------
    recs = sv["episode_records"]
    err_em = [em for has_err, em, _ in recs if has_err and em is not None]
    noerr_em = [em for has_err, em, _ in recs if not has_err and em is not None]
    n_err, n_noerr = len(err_em), len(noerr_em)
    err_correct = sum(1 for v in err_em if v == 1.0)
    noerr_correct = sum(1 for v in noerr_em if v == 1.0)
    err_acc = err_correct / n_err if n_err else 0.0
    noerr_acc = noerr_correct / n_noerr if n_noerr else 0.0
    stats = chi_square_or_fisher(err_correct, n_err - err_correct, noerr_correct, n_noerr - noerr_correct)
    # A confound check: does the "no fetch error" group just mean "barely fetched at all"?
    noerr_nfetch = [nf for has_err, em, nf in recs if not has_err]
    noerr_zero_fetch = sum(1 for nf in noerr_nfetch if nf == 0)
    err_nfetch = [nf for has_err, em, nf in recs if has_err]
    part4 = {
        "n_episodes_with_fetch_error": n_err,
        "n_episodes_without_fetch_error": n_noerr,
        "accuracy_em_with_fetch_error": err_acc,
        "accuracy_em_without_fetch_error": noerr_acc,
        "accuracy_delta_pp": 100.0 * (noerr_acc - err_acc),
        **stats,
        "confound_check_no_error_group_zero_fetch_calls": noerr_zero_fetch,
        "confound_check_no_error_group_zero_fetch_pct": (
            100.0 * noerr_zero_fetch / n_noerr if n_noerr else 0.0),
        "confound_check_mean_fetch_calls_no_error_group": (
            sum(noerr_nfetch) / len(noerr_nfetch) if noerr_nfetch else 0.0),
        "confound_check_mean_fetch_calls_error_group": (
            sum(err_nfetch) / len(err_nfetch) if err_nfetch else 0.0),
        "CAVEAT": (
            "OBSERVATIONAL split on a POST-TREATMENT variable, not a randomized or matched "
            "comparison. Whether an episode ever hits a fetch error is itself driven by "
            "question difficulty and how many fetch attempts the episode needed -- harder "
            "questions plausibly cause BOTH more fetch errors (more exploratory fetching) "
            "AND more wrong answers, independent of any causal harm from the errors "
            "themselves. This comparison is DESCRIPTIVE ONLY and CANNOT establish that fetch "
            "errors caused the accuracy difference. Do not read it as an effect estimate. "
            "CONCRETELY here: the 'no fetch error' group is almost entirely episodes that made "
            "(near-)ZERO fetch calls in the first place, not episodes that fetched cleanly -- "
            "so its low accuracy reflects giving up / never reading a document, not the ABSENCE "
            "of fetch errors helping. The direction of the raw delta (no-error group scoring "
            "LOWER) is a Simpson's-paradox-style artifact of this confound, not evidence that "
            "fetch errors improve accuracy."
        ),
    }

    # ---- Part 2: is browsecomp anomalous vs the two Wikipedia-backed datasets? ----------------
    bc_sieve = cell_results[CELLS[0][0]]
    hp = cell_results[CELLS[3][0]]
    mu = cell_results[CELLS[4][0]]
    part2 = {
        "browsecomp_sieve_fetch_error_pct": bc_sieve["fetch_error_pct"],
        "hotpotqa_sieve_fetch_error_pct": hp["fetch_error_pct"],
        "musique_sieve_fetch_error_pct": mu["fetch_error_pct"],
        "browsecomp_vs_hotpotqa_ratio": (bc_sieve["fetch_error_pct"] / hp["fetch_error_pct"]
                                          if hp["fetch_error_pct"] else None),
        "browsecomp_vs_musique_ratio": (bc_sieve["fetch_error_pct"] / mu["fetch_error_pct"]
                                         if mu["fetch_error_pct"] else None),
        "manuals": manuals,
    }

    data = {
        "cells": {k: {kk: vv for kk, vv in v.items() if kk != "episode_records"}
                  for k, v in cell_results.items()},
        "part2_browsecomp_vs_wikipedia": part2,
        "part3_wasted_effort_sieve_browsecomp": part3,
        "part4_descriptive_accuracy_split_sieve_browsecomp": part4,
    }

    Path(args.out_json).write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"Wrote {args.out_json}", file=sys.stderr)

    write_markdown(args.out_md, cell_results, manuals, part2, part3, part4)
    print(f"Wrote {args.out_md}", file=sys.stderr)


def write_markdown(path, cell_results, manuals, part2, part3, part4):
    lines = []
    lines.append("# Fetch-error audit: cost of the stale BrowseComp-Plus manual\n")
    lines.append(
        "Quantifies fetch/read tool-call error rates across 5 cells (tier "
        "`_headline_validation`, model dir `Tongyi-DeepResearch-30B-A3B`), whether "
        "BrowseComp-Plus is anomalous relative to the two Wikipedia-backed datasets, the "
        "wasted-effort cost in the browsecomp Sieve cell, and a strictly descriptive "
        "accuracy split. See `analysis/fetch_error_audit.py`'s module docstring for the exact "
        "schema inspection and fetch-call / error identification rules used below.\n"
    )

    lines.append("## Part 1 -- per-cell fetch-call error rates\n")
    lines.append("| Cell | episodes | fetch calls | errors | error % | episodes w/ >=1 error | episode error % |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for label, r in cell_results.items():
        lines.append(f"| {label} | {r['n_episodes']} | {r['fetch_calls_total']} | "
                      f"{r['fetch_calls_error']} | {r['fetch_error_pct']:.1f}% | "
                      f"{r['episodes_with_error']} | {r['episodes_with_error_pct']:.1f}% |")
    lines.append("")

    for label, r in cell_results.items():
        lines.append(f"### Error kinds -- {label}\n")
        lines.append("| kind | count |")
        lines.append("|---|---:|")
        for kind, cnt in r["error_kind_counts"].items():
            lines.append(f"| {kind} | {cnt} |")
        mc = r["most_common_error_string"]
        if mc:
            lines.append(f"\nMost common exact error string ({mc[1]}x): `{mc[0]}`\n")
        else:
            lines.append("\n(no errors)\n")

    lines.append("## Part 2 -- is BrowseComp anomalous?\n")
    lines.append(f"- BrowseComp Sieve fetch error rate: **{part2['browsecomp_sieve_fetch_error_pct']:.1f}%**")
    lines.append(f"- HotpotQA Sieve fetch error rate: **{part2['hotpotqa_sieve_fetch_error_pct']:.1f}%**")
    lines.append(f"- MuSiQue Sieve fetch error rate: **{part2['musique_sieve_fetch_error_pct']:.1f}%**")
    if part2["browsecomp_vs_hotpotqa_ratio"] is not None:
        lines.append(f"- BrowseComp is {part2['browsecomp_vs_hotpotqa_ratio']:.1f}x HotpotQA's rate, "
                      f"{part2['browsecomp_vs_musique_ratio']:.1f}x MuSiQue's rate.\n")
    lines.append("### Manual each cell actually renders (via `load_condition`, same composer the harness uses)\n")
    lines.append("| Cell | toolset | field_profile passed | contains stale \"no named sections\" claim |")
    lines.append("|---|---|---|---|")
    for label, m in manuals.items():
        lines.append(f"| {label} | {m['toolset']} | {m['field_profile_passed']} | "
                      f"{'YES' if m['contains_stale_no_sections_claim'] else 'no'} |")
    lines.append("")

    lines.append("## Part 3 -- wasted effort, browsecomp Sieve cell\n")
    p3 = part3
    lines.append(f"- Fetch calls: {p3['fetch_calls_total']}, of which {p3['fetch_calls_error']} errored "
                 f"({100*p3['fetch_call_error_fraction']:.1f}%).")
    lines.append(f"- LLM calls in the cell: {p3['llm_calls_total_cell']}; calls that produced an errored "
                 f"fetch: {p3['llm_calls_that_produced_an_errored_fetch']} "
                 f"({100*p3['llm_call_fraction_on_errored_fetch']:.1f}% of all LLM calls in the cell).")
    lines.append(f"- **Count-once tokens are NOT attributable per call** -- {p3['count_once_note']}")
    lines.append(f"- Secondary, STEP-SUMMED (not count-once) approximation: tokens on the specific LLM-call "
                 f"steps that produced an errored fetch = {p3['step_summed_tokens_on_errored_fetch_steps']:,} "
                 f"of {p3['step_summed_tokens_total_cell']:,} step-summed tokens in the cell "
                 f"({100*p3['step_summed_token_fraction_on_errored_fetch']:.1f}%).\n")

    lines.append("## Part 4 -- descriptive accuracy split, browsecomp Sieve cell only\n")
    p4 = part4
    lines.append(f"- Episodes WITH >=1 fetch error: n={p4['n_episodes_with_fetch_error']}, "
                 f"EM accuracy = {100*p4['accuracy_em_with_fetch_error']:.1f}%")
    lines.append(f"- Episodes WITHOUT any fetch error: n={p4['n_episodes_without_fetch_error']}, "
                 f"EM accuracy = {100*p4['accuracy_em_without_fetch_error']:.1f}%")
    lines.append(f"- Delta (no-error minus error): {p4['accuracy_delta_pp']:+.1f} pp")
    lines.append(f"- 2x2 table {p4['table']}: chi2={p4['chi2_stat']:.3f}, chi2 p={p4['chi2_p']:.4f}, "
                 f"Fisher exact p={p4['fisher_p']:.4f} (min expected cell = {p4['min_expected_cell']:.1f})")
    lines.append(f"- Confound check: {p4['confound_check_no_error_group_zero_fetch_calls']}/"
                 f"{p4['n_episodes_without_fetch_error']} "
                 f"({p4['confound_check_no_error_group_zero_fetch_pct']:.1f}%) of the 'no fetch error' "
                 f"group made ZERO fetch calls at all (mean fetch calls in that group = "
                 f"{p4['confound_check_mean_fetch_calls_no_error_group']:.2f}, vs "
                 f"{p4['confound_check_mean_fetch_calls_error_group']:.1f} in the 'has error' group).")
    lines.append(f"\n**CAVEAT: {p4['CAVEAT']}**\n")

    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
