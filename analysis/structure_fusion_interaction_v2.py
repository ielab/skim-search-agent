#!/usr/bin/env python
"""Structure x dense-FUSION interaction, v2 — CORRECTED cell identity, plus an IDENTIFICATION audit.

WHY v2 EXISTS. `analysis/structure_fusion_interaction.py` (v1, left untouched for the record)
instantiated the (structure=0, fusion=1) corner with `agent_research_dense_fetch` — "dense only,
same interface". That is dense REPLACING BM25, not dense ADDED TO BM25 by RRF. Because the
(structure=1, fusion=1) corner (`agent_research_bql_dense_snip`) adds dense to an existing
lexical ranking by RRF, factor B in v1 was "add dense" at one level of A and "delete BM25" at the
other. The resulting interaction is uninterpretable. This file fixes the cell:

    (0,1) fusion alone := agent_research_hybrid_fetch_snip   (BM25 (+) dense by RRF, k=60)

and additionally AUDITS whether the resulting 2x2 is IDENTIFIED at all, i.e. whether "structure"
is confounded with (a) the BQL skill manual's ANSWER-FORMAT COACHING and (b) the retrieval POOL
SIZE the listing shows.

METHOD PARITY (nothing reimplemented):
  - evaluation.metrics.answer_em                          (via scripts.compare_cells.metrics)
  - scripts.force_answer_backfill.load_rows_with_recovery (via scripts.compare_cells.cell_rows)
  - scripts.compare_cells.{cell_dir, cell_rows, metrics, mcnemar_p, load_qrels, pct}
  - statsmodels (logit + GEE)

Run: PYTHONPATH=. envs/bin/python analysis/structure_fusion_interaction_v2.py
Writes analysis/structure_fusion_interaction_v2_data.json (the .md report is written by hand
from that JSON).
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402
import statsmodels.formula.api as smf  # noqa: E402
from statsmodels.genmod.cov_struct import Exchangeable  # noqa: E402
from statsmodels.genmod.families import Binomial  # noqa: E402
from statsmodels.genmod.generalized_estimating_equations import GEE  # noqa: E402

from agent_search.prompts.loader import load_condition  # noqa: E402
from scripts.compare_cells import (  # noqa: E402
    cell_dir, cell_rows, load_qrels, mcnemar_p, metrics, pct,
)

DATASET = "browsecomp_plus_structured"
SUBDIR = "_headline_validation"

# label -> (condition, structure, fusion, REGISTRY label, prompt-condition name)
CELLS = {
    "neither":   ("agent_research_bm25_fetch_snip",   0, 0, "bm25+snip fetch",
                  "research_bm25_fetch_snip"),
    "structure": ("agent_research_snip",              1, 0, "bql+snip fetch",
                  "research_snip"),
    "fusion":    ("agent_research_hybrid_fetch_snip", 0, 1, "hybrid+snip fetch",
                  "research_hybrid_fetch_snip"),
    "both":      ("agent_research_bql_dense_snip",    1, 1, "bql+dense+snip fetch",
                  "research_bql_dense_snip"),
}
# the v1 (wrong) fusion-alone cell, kept so the report can quantify what the correction changes
V1_FUSION_CELL = ("agent_research_dense_fetch", "dense+snip fetch", "research_dense_fetch")

# --- mandatory sanity gates -------------------------------------------------------------------
GATE_1 = dict(
    name="Sieve vs 'No dense evidence', browsecomp_plus_structured (both cap 100)",
    subdir_a=SUBDIR, cond_a="agent_research_bql_dense_snip", ds="browsecomp_plus_structured",
    subdir_b=SUBDIR, cond_b="agent_research_snip",
    exp_n=830, exp_delta=4.7, exp_p=0.0104, tol_delta=0.05, tol_p=0.001,
)
# Wikipedia cap-100 quantity: Sieve on hotpotqa lives in the _budget100 tier at cap 100 (the
# _headline_validation hotpotqa Sieve cell is the published cap-50 run); the hybrid control is
# _headline_validation at cap 100. See analysis/matched_cap100_results_data.json.
GATE_2 = dict(
    name="Sieve@cap100 vs hybrid control, hotpotqa_structured (both cap 100)",
    subdir_a="_budget100", cond_a="agent_research_bql_dense_snip", ds="hotpotqa_structured",
    subdir_b=SUBDIR, cond_b="agent_research_hybrid_fetch_snip",
    exp_n=7343, exp_delta=5.84, exp_p=2.8387770386540747e-29, tol_delta=0.02, tol_p=1e-30,
)


def paired(m_a: dict, m_b: dict) -> dict:
    """EM% of A and B on their mutual instance ids + exact McNemar. b = B-right/A-wrong,
    c = A-right/B-wrong (same orientation as scripts/compare_cells.py's own table)."""
    mut = sorted(set(m_a) & set(m_b))
    b = sum(1 for i in mut if m_b[i]["em"] and not m_a[i]["em"])
    c = sum(1 for i in mut if m_a[i]["em"] and not m_b[i]["em"])
    em_a = pct([m_a[i]["em"] for i in mut])
    em_b = pct([m_b[i]["em"] for i in mut])
    return dict(n=len(mut), em_a=em_a, em_b=em_b, delta=em_a - em_b,
                b=int(b), c=int(c), p=float(mcnemar_p(b, c)))


def run_gate(g: dict, qrels_cache: dict) -> dict:
    qrels = qrels_cache.setdefault(g["ds"], load_qrels(g["ds"]))
    m_a = metrics(cell_rows(g["subdir_a"], g["ds"], g["cond_a"]), qrels, g["ds"])
    m_b = metrics(cell_rows(g["subdir_b"], g["ds"], g["cond_b"]), qrels, g["ds"])
    res = paired(m_a, m_b)
    res["name"] = g["name"]
    res["expected"] = dict(n=g["exp_n"], delta=g["exp_delta"], p=g["exp_p"])
    res["pass"] = bool(res["n"] == g["exp_n"]
                       and abs(res["delta"] - g["exp_delta"]) < g["tol_delta"]
                       and abs(res["p"] - g["exp_p"]) < max(g["tol_p"], g["exp_p"] * 0.02))
    return res


# --- identification audit ---------------------------------------------------------------------
# Both listing renderers stamp the depth in the observation HEADER, which is never truncated:
#   BQL cells (DocSearchFetch):        "search: <q>  ->  <expr>   (1568 matches, top 5):"
#       -> "1568 matches" is the FILTER's total hit count; "top 5" is the LISTING DEPTH.
#   bm25/hybrid/dense fetch-snip cells: "search: <q>   (10 matches):"
#       -> here `len(ids)` IS the returned list (see Bm25FetchSnipWorkspace.search /
#          HybridFetchSnipWorkspace.search: `lines = [f"... ({len(ids)} matches):"]`), so this
#          number IS the listing depth.
# (`trajectory[*].observation` is stored truncated, so counting rendered rank lines would
# undercount — the header is the only reliable source.)
# Every listing header shape emitted by agent_search/agent/tools/doc_research.py:
#   "(<n_hits> matches, top <K>):"                                 -> depth K   (BQL exact)
#   "(0 exact matches — showing top <K> CLOSEST docs by term ...)" -> depth K   (BQL soft fallback)
#   "(0 exact matches — closest by CONSTRAINT COVERAGE; ...)"      -> depth NOT stamped
#   "(<len(ids)> matches):"                                        -> depth len(ids) (bm25/hybrid)
#   "(0 matches)"                                                  -> depth 0
#   anything else (ERROR:, "BQL parse error")                      -> no listing at all
_RE_TOPK = re.compile(r"\(\d+ matches, top (\d+)\)")
_RE_SOFT = re.compile(r"\(0 exact matches — showing top (\d+) CLOSEST")
_RE_COV = re.compile(r"\(0 exact matches — closest by CONSTRAINT COVERAGE")
_RE_PLAIN = re.compile(r"\((\d+) matches\)")


def listing_depth_stats(rows, search_actions: set, header_is_depth: bool) -> dict:
    """Empirical listing depth the agent actually SAW per search call, from the observation
    header, plus the distribution of explicit `k` arguments the agent passed."""
    depths, k_args = Counter(), Counter()
    n_searches = 0
    for r in rows:
        for step in r.get("trajectory") or []:
            if step.get("action") not in search_actions:
                continue
            n_searches += 1
            k_args[str((step.get("args") or {}).get("k"))] += 1
            obs = step.get("observation") or ""
            m = _RE_TOPK.search(obs) or _RE_SOFT.search(obs)
            if m:
                depths[int(m.group(1))] += 1
            elif _RE_COV.search(obs):
                depths["coverage-fallback (depth not stamped)"] += 1
            elif (mp := _RE_PLAIN.search(obs)):
                n = int(mp.group(1))
                depths[n if (header_is_depth or n == 0) else "no-depth"] += 1
            else:
                depths["no listing (error / parse failure)"] += 1
    return dict(n_search_calls=n_searches,
                listing_depth_hist={str(k): v for k, v in
                                    sorted(depths.items(), key=lambda kv: str(kv[0]))},
                mean_listing_depth=(
                    sum(int(k) * v for k, v in depths.items() if isinstance(k, int))
                    / max(1, sum(v for k, v in depths.items() if isinstance(k, int)))),
                explicit_k_arg_hist=dict(k_args))


def audit_cell(label: str, cond: str, prompt_cond: str, rows) -> dict:
    prof = load_condition(prompt_cond, domain="general", profile="browsecomp")
    manual_names = []
    from agent_search.prompts.loader import _registry  # noqa: PLC0415
    tools = (_registry().get("tools") or {})
    for t in prof.tool_names:
        man = (tools.get(t) or {}).get("manual")
        if man:
            manual_names.append(f"{t}:{man.get('browsecomp') if isinstance(man, dict) else man}")
    search_actions = {t for t in prof.tool_names if "search" in t}
    # BQL cells stamp ", top K"; the bm25/hybrid/dense fetch-snip cells' "(N matches)" IS the
    # returned list length (len(ids)), so the first header number is the depth for those.
    header_is_depth = not bool(manual_names)
    ipt = [r.get("initial_prompt_tokens") for r in rows
           if r.get("initial_prompt_tokens") is not None]
    cfg = json.loads((cell_dir(SUBDIR, DATASET, cond) / "config.json").read_text())
    knobs = cfg.get("env_knobs") or {}
    # answer-format coaching: does the rendered manual carry EM-relevant answer instructions?
    coaching_lines = [ln.strip() for ln in prof.system.splitlines()
                      if re.search(r"shortest span|COPIED VERBATIM|asked-for (?:unit|TYPE)"
                                   r"|not a compound|from memory", ln)]
    return dict(
        cond=cond, prompt_condition=prompt_cond, toolset=prof.toolset,
        tool_names=list(prof.tool_names),
        renders_manual=bool(manual_names), manuals=manual_names,
        answer_format_coaching_lines=coaching_lines,
        n_answer_format_coaching_lines=len(coaching_lines),
        composed_system_chars_TODAY=len(prof.system),
        composed_system_sha_TODAY=prof.system_sha256,
        config_prompt_sha_AT_RUN_TIME=cfg.get("prompt_sha256"),
        prompt_sha_matches_today=(cfg.get("prompt_sha256") == prof.system_sha256),
        initial_prompt_tokens_mean=(sum(ipt) / len(ipt)) if ipt else None,
        initial_prompt_tokens_min=min(ipt) if ipt else None,
        initial_prompt_tokens_max=max(ipt) if ipt else None,
        tool_schema_default_k=None,  # filled by caller from tools.yaml text
        env_knobs_topk={k: v for k, v in knobs.items() if k.endswith("_TOPK")},
        **listing_depth_stats(rows, search_actions, header_is_depth),
    )


def main() -> dict:
    qrels_cache: dict = {}

    print("== MANDATORY SANITY GATES ==")
    gates = [run_gate(GATE_1, qrels_cache), run_gate(GATE_2, qrels_cache)]
    for g in gates:
        print(f"  {'PASS' if g['pass'] else 'FAIL'}  {g['name']}\n"
              f"        n={g['n']} EM_a={g['em_a']:.2f} EM_b={g['em_b']:.2f} "
              f"delta={g['delta']:+.2f} b={g['b']} c={g['c']} p={g['p']:.4g} "
              f"(expected delta={g['expected']['delta']}, p={g['expected']['p']:.4g})")
    if not all(g["pass"] for g in gates):
        print("SANITY GATE FAILED — STOPPING.", file=sys.stderr)
        sys.exit(1)
    print("Both gates PASS.\n")

    qrels = qrels_cache[DATASET]

    # --- load rows + budget parity -----------------------------------------------------------
    rows_by_label, cell_metrics, budget = {}, {}, {}
    for label, (cond, s, f, reg, pc) in CELLS.items():
        rows = cell_rows(SUBDIR, DATASET, cond)
        if rows is None:
            print(f"MISSING rows for {label} ({cond}) — STOPPING.", file=sys.stderr)
            sys.exit(1)
        rows_by_label[label] = rows
        cell_metrics[label] = metrics(rows, qrels, DATASET)
        cfg = json.loads((cell_dir(SUBDIR, DATASET, cond) / "config.json").read_text())
        caps = Counter(r.get("max_steps") for r in rows)
        budget[label] = dict(cond=cond, config_max_steps=cfg.get("max_steps"),
                             row_max_steps_hist={str(k): v for k, v in caps.items()},
                             n_rows=len(rows))
        print(f"  {label:10s} {cond:34s} n={len(rows):5d} "
              f"config.max_steps={cfg.get('max_steps')} row_caps={dict(caps)}")
    cap_set = {b["config_max_steps"] for b in budget.values()}
    row_cap_set = {tuple(sorted(b["row_max_steps_hist"])) for b in budget.values()}
    budget_ok = len(cap_set) == 1 and len(row_cap_set) == 1 and cap_set == {100}
    print(f"Budget parity (cap 100 everywhere): {'PASS' if budget_ok else 'FAIL'}\n")
    if not budget_ok:
        print("BUDGET PARITY FAILED — STOPPING.", file=sys.stderr)
        sys.exit(1)

    shared = sorted(set.intersection(*(set(m) for m in cell_metrics.values())))
    n_shared = len(shared)
    print(f"n_shared (inner join over all four cells) = {n_shared}\n")

    # --- 2x2 table ----------------------------------------------------------------------------
    table = {}
    for label, (cond, s, f, reg, pc) in CELLS.items():
        ems = [cell_metrics[label][i]["em"] for i in shared]
        table[label] = dict(structure=s, fusion=f, cond=cond, reg_label=reg,
                            n=len(ems), n_correct=int(sum(ems)), em_pct=pct(ems))
        print(f"  {label:10s} structure={s} fusion={f}  EM={pct(ems):6.2f}%  "
              f"({int(sum(ems))}/{len(ems)})")

    # what v1's wrong cell would have given, for the report
    v1_rows = cell_rows(SUBDIR, DATASET, V1_FUSION_CELL[0])
    v1_m = metrics(v1_rows, qrels, DATASET)
    v1_shared = sorted(set(v1_m) & set(shared))
    v1_cell = dict(cond=V1_FUSION_CELL[0], reg_label=V1_FUSION_CELL[1],
                   n=len(v1_shared), em_pct=pct([v1_m[i]["em"] for i in v1_shared]))

    # --- long frame + fits ---------------------------------------------------------------------
    records = [dict(instance_id=i, cell=label, structure=CELLS[label][1],
                    fusion=CELLS[label][2], em=int(bool(cell_metrics[label][i]["em"])))
               for label in CELLS for i in shared]
    df = pd.DataFrame.from_records(records)
    assert len(df) == 4 * n_shared

    print("\n== Logistic regression (independence assumed) ==")
    logit = smf.logit("em ~ structure * fusion", data=df).fit(disp=0)
    print(logit.summary())
    naive = {nm: dict(coef=float(logit.params[nm]), se=float(logit.bse[nm]),
                      z=float(logit.tvalues[nm]), p=float(logit.pvalues[nm]))
             for nm in logit.params.index}

    print("\n== GEE (binomial logit, exchangeable, cluster=instance_id) ==")
    gee = GEE.from_formula("em ~ structure * fusion", groups="instance_id", data=df,
                           family=Binomial(), cov_struct=Exchangeable()).fit()
    print(gee.summary())
    gee_out = {nm: dict(coef=float(gee.params[nm]), se=float(gee.bse[nm]),
                        z=float(gee.tvalues[nm]), p=float(gee.pvalues[nm]))
               for nm in gee.params.index}
    try:
        within_corr = float(gee.cov_struct.dep_params)
    except Exception:
        within_corr = None
    print(f"estimated within-instance (exchangeable) correlation = {within_corr}")

    # --- paired contrasts ----------------------------------------------------------------------
    def contrast(a: str, b: str) -> dict:
        r = paired(cell_metrics[a], cell_metrics[b])
        r.update(cell_a=a, cell_b=b, cond_a=CELLS[a][0], cond_b=CELLS[b][0])
        return r

    contrasts = {
        "neither_to_structure": contrast("structure", "neither"),
        "neither_to_fusion":    contrast("fusion", "neither"),
        "structure_to_both":    contrast("both", "structure"),
        "fusion_to_both":       contrast("both", "fusion"),
        "neither_to_both":      contrast("both", "neither"),
    }
    print("\n== Paired contrasts (exact McNemar) ==")
    for k, v in contrasts.items():
        print(f"  {k:22s} {v['em_a']:6.2f} vs {v['em_b']:6.2f}  delta={v['delta']:+5.2f}  "
              f"b={v['b']:4d} c={v['c']:4d}  p={v['p']:.4g}")

    raw_inter = ((table["both"]["em_pct"] - table["structure"]["em_pct"])
                 - (table["fusion"]["em_pct"] - table["neither"]["em_pct"]))
    print(f"\nRaw-scale interaction contrast (both-structure)-(fusion-neither) = {raw_inter:+.2f} EM pts")

    # --- identification audit -------------------------------------------------------------------
    print("\n== IDENTIFICATION AUDIT (coaching + pool size) ==")
    tools_yaml = (Path("agent_search/prompts/tools.yaml")).read_text()
    audit = {}
    for label, (cond, s, f, reg, pc) in CELLS.items():
        a = audit_cell(label, cond, pc, rows_by_label[label])
        # tool-schema advertised default k, straight out of tools.yaml
        st = [t for t in a["tool_names"] if "search" in t]
        ks = {}
        for t in st:
            m = re.search(rf"^  {re.escape(t)}:\n(?:.*\n)*?.*?default (\d+)\)", tools_yaml, re.M)
            ks[t] = int(m.group(1)) if m else None
        a["tool_schema_default_k"] = ks
        # how often the agent WIDENED its own pool past the schema default
        dflt = next((v for v in ks.values() if v), None)
        widened = sum(v for kk, v in a["explicit_k_arg_hist"].items()
                      if kk != "None" and dflt and int(kk) > dflt)
        a["schema_default_k_value"] = dflt
        a["n_calls_widened_past_default"] = widened
        a["pct_calls_widened_past_default"] = 100.0 * widened / max(1, a["n_search_calls"])
        a["structure"], a["fusion"] = s, f
        audit[label] = a
        top = sorted(a["listing_depth_hist"].items(), key=lambda kv: -kv[1])[:4]
        print(f"  {label:10s} manual={a['renders_manual']!s:5s} "
              f"coach_lines={a['n_answer_format_coaching_lines']} "
              f"init_prompt_tok~{a['initial_prompt_tokens_mean']:.0f} "
              f"schema_default_k={ks} mean_depth={a['mean_listing_depth']:.2f} "
              f"depth_top={top}")

    coaching_collinear = ({l for l in CELLS if audit[l]["renders_manual"]}
                          == {l for l in CELLS if CELLS[l][1] == 1})
    k_by_struct = {}
    for label in CELLS:
        ks = [v for v in audit[label]["tool_schema_default_k"].values() if v]
        k_by_struct[label] = ks[0] if ks else None
    pool_collinear = ({k_by_struct[l] for l in CELLS if CELLS[l][1] == 1}
                      != {k_by_struct[l] for l in CELLS if CELLS[l][1] == 0})

    out = dict(
        generated_by="analysis/structure_fusion_interaction_v2.py",
        dataset=DATASET, subdir=SUBDIR, model="Tongyi-DeepResearch-30B-A3B",
        sanity_gates=gates,
        budget_parity=dict(cells=budget, ok=budget_ok),
        cells={l: dict(cond=CELLS[l][0], structure=CELLS[l][1], fusion=CELLS[l][2],
                       reg_label=CELLS[l][3],
                       run_dir=str(cell_dir(SUBDIR, DATASET, CELLS[l][0])))
               for l in CELLS},
        v1_wrong_fusion_cell=v1_cell,
        n_shared=n_shared,
        two_by_two=table,
        logit=naive,
        gee=dict(params=gee_out, within_instance_corr=within_corr),
        contrasts=contrasts,
        raw_scale_interaction_contrast_em_pts=raw_inter,
        identification=dict(
            per_cell=audit,
            coaching_perfectly_collinear_with_structure=coaching_collinear,
            pool_size_differs_by_structure=pool_collinear,
            fusion_mechanism_differs_by_structure=dict(
                at_structure0=("global RRF (k=60) over TWO independent top-100 pools "
                               "(pyserini/Lucene BM25 and FAISS/bge-base dense) — dense CAN "
                               "introduce documents BM25 never ranked "
                               "[doc_research.HybridFetchSnipWorkspace.search + rrf_fuse]"),
                at_structure1=("RRF (k=60) of BM25 order against dense order RESTRICTED to the "
                               "BQL filter-passing candidate ids — dense can NEVER introduce a "
                               "document, only reorder (and only within a coverage tier on the "
                               "0-exact-hit fallback) "
                               "[retrievers/structural/bql/dense_fuse.py]"),
                same_manipulation=False,
            ),
            identified=not (coaching_collinear or pool_collinear),
        ),
    )
    p = Path(__file__).resolve().parent / "structure_fusion_interaction_v2_data.json"
    p.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {p}")
    return out


if __name__ == "__main__":
    main()
