#!/usr/bin/env python
"""Structure x dense-fusion interaction test, at a MATCHED read interface, on
`browsecomp_plus_structured` (n=830).

WHY: latex/sections/appendix.tex's Table~\\ref{tab:factorial-grid} (App. G, "Full SEARCH x READ
Factorial Grid") states the paper "ran no test of an interaction term" and only describes,
descriptively, "what an interaction ... would look like." A reviewer pointed out the 830 paired
instances needed to actually run that test are sitting on disk. This script runs it.

**A NOTE ON THE PROMPT'S OWN NUMBERS.** The task that produced this script was briefed with
approximate figures ("neither ~29.6, structure-alone ~+1.9, fusion-alone ~+1.9, both ~38.9,
+9.3, p=2.8e-6"). Those four numbers are REAL numbers in App. G's table (Table
tab:factorial-grid) -- but they are NOT a matched-read-interface structure x fusion 2x2. They are
the "Query surf. + dense (RRF)" ROW of that table read at its two different READ levels: 29.6 is
that row's own VISIT cell, 38.9 is that SAME row's FETCH+snippets cell. The "+1.9" figures belong
to two OTHER rows' own visit->fetch deltas ("Query surface (no dense)": 32.3->34.2; "Hybrid
sparse-dense RRF": 33.7->35.7) -- not a "fusion-alone" cell of any 2x2 sharing a baseline with the
other three numbers. In other words, the prompt's own numbers are confounded with the READ axis
(visit vs. fetch+snippets), which is exactly the trap this script's task spec warns about ("If you
cannot identify a clean 2x2 ... STOP").

Because of that, this script does NOT reuse those four numbers. Instead it builds the ACTUAL
clean, matched-read-interface 2x2 that the run directories on disk support: all four cells run
under `_headline_validation` (Sieve's own snippet-listing + section-fetch READ interface) on
`browsecomp_plus_structured`, differing ONLY in whether the QUERY SURFACE is the structured `bql`
condition (structure=1) or plain `bm25` (structure=0), and whether a dense retrieval channel is
fused in (fusion=1) or not (fusion=0):

    structure=0, fusion=0 (neither): _headline_validation/agent_research_bm25_fetch_snip   (REGISTRY "bm25+snip fetch")
    structure=1, fusion=0 (structure alone): _headline_validation/agent_research_snip        (REGISTRY "bql+snip fetch", paper's "No dense evidence")
    structure=0, fusion=1 (fusion alone): _headline_validation/agent_research_dense_fetch     (REGISTRY "dense+snip fetch")
    structure=1, fusion=1 (both = Sieve): _headline_validation/agent_research_bql_dense_snip  (REGISTRY "bql+dense+snip fetch", = Sieve)

All four: same dataset, same `_headline_validation` run subdir (same snippet-listing + section-fetch
READ interface), same n=830, same configured max_steps=100 (verified from config.json AND the
per-row stamped `max_steps` field -- see `_check_budget_parity` below). This is the grid this
script actually tests.

METHOD PARITY (mandatory, no metric reimplemented):
  - evaluation.metrics.answer_em            (via scripts.compare_cells.metrics())
  - scripts.force_answer_backfill.load_rows_with_recovery (via scripts.compare_cells.cell_rows())
  - scripts.compare_cells.{cell_dir, cell_rows, metrics, mcnemar_p, load_qrels}
  - statsmodels for both regressions (naive logit + clustered GEE)

Run: PYTHONPATH=. envs/bin/python analysis/structure_fusion_interaction.py
Writes:
  - analysis/structure_fusion_interaction_data.json
  - analysis/structure_fusion_interaction.md
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402
import statsmodels.api as sm  # noqa: E402
import statsmodels.formula.api as smf  # noqa: E402
from statsmodels.genmod.cov_struct import Exchangeable  # noqa: E402
from statsmodels.genmod.families import Binomial  # noqa: E402
from statsmodels.genmod.generalized_estimating_equations import GEE  # noqa: E402

from scripts.compare_cells import (  # noqa: E402
    cell_dir, cell_rows, load_qrels, metrics, mcnemar_p, pct,
)

DATASET = "browsecomp_plus_structured"
SUBDIR = "_headline_validation"

# label -> (condition, structure flag, fusion flag, REGISTRY label, why-this-cell)
CELLS = {
    "neither": (
        "agent_research_bm25_fetch_snip", 0, 0, "bm25+snip fetch",
        "plain BM25 query surface (no structured bql query), no dense channel -- the (0,0) corner",
    ),
    "structure": (
        "agent_research_snip", 1, 0, "bql+snip fetch",
        "structured bql query surface, no dense channel -- structure ON, fusion OFF "
        "(this is the paper's own 'No dense evidence' ablation cell)",
    ),
    "fusion": (
        "agent_research_dense_fetch", 0, 1, "dense+snip fetch",
        "plain dense-embedding query surface (no bql), no structured query -- structure OFF, "
        "fusion ON (this is the paper's own 'Dense only, same interface' ablation cell)",
    ),
    "both": (
        "agent_research_bql_dense_snip", 1, 1, "bql+dense+snip fetch",
        "structured bql query surface WITH a fused dense channel -- structure ON, fusion ON; "
        "this is Sieve itself",
    ),
}

# Published sanity-gate number (analysis/ablation_deltas.md / analysis/multiplicity_census.md /
# analysis/step_budget_audit.md all reproduce this): Sieve vs "No dense evidence" on
# browsecomp_plus_structured.
GATE_A = ("agent_research_bql_dense_snip", "Sieve")
GATE_B = ("agent_research_snip", "No dense evidence")
GATE_EXPECT = dict(n=830, em_a=38.9, em_b=34.2, delta=4.7, p=0.0104)


def _paired(m_a: dict, m_b: dict):
    mut = sorted(set(m_a) & set(m_b))
    b = sum(1 for i in mut if m_b[i]["em"] and not m_a[i]["em"])
    c = sum(1 for i in mut if m_a[i]["em"] and not m_b[i]["em"])
    em_a = pct([m_a[i]["em"] for i in mut])
    em_b = pct([m_b[i]["em"] for i in mut])
    p = float(mcnemar_p(b, c))
    return dict(n=len(mut), em_a=em_a, em_b=em_b, delta=em_a - em_b, b=int(b), c=int(c), p=p)


def sanity_gate(qrels: dict) -> dict:
    rows_a = cell_rows(SUBDIR, DATASET, GATE_A[0])
    rows_b = cell_rows(SUBDIR, DATASET, GATE_B[0])
    m_a = metrics(rows_a, qrels, DATASET)
    m_b = metrics(rows_b, qrels, DATASET)
    res = _paired(m_a, m_b)
    ok = (
        res["n"] == GATE_EXPECT["n"]
        and abs(res["em_a"] - GATE_EXPECT["em_a"]) < 0.05
        and abs(res["em_b"] - GATE_EXPECT["em_b"]) < 0.05
        and abs(res["delta"] - GATE_EXPECT["delta"]) < 0.05
        and abs(res["p"] - GATE_EXPECT["p"]) < 0.001
    )
    res["pass"] = bool(ok)
    return res


def _config_max_steps(cond: str) -> dict:
    d = cell_dir(SUBDIR, DATASET, cond)
    cfg = json.loads((d / "config.json").read_text())
    row_caps = set()
    for r in cell_rows(SUBDIR, DATASET, cond) or []:
        v = r.get("max_steps")
        if v is not None:
            row_caps.add(v)
    return dict(config_max_steps=cfg.get("max_steps"), config_limit=cfg.get("limit"),
                row_max_steps_values=sorted(row_caps))


def check_budget_parity() -> dict:
    out = {}
    for label, (cond, *_r) in CELLS.items():
        out[label] = dict(cond=cond, **_config_max_steps(cond))
    caps = {v["config_max_steps"] for v in out.values()}
    row_caps = {tuple(v["row_max_steps_values"]) for v in out.values()}
    out["_parity_ok"] = (len(caps) == 1) and (len(row_caps) == 1)
    return out


def main():
    print(f"== Sanity gate: {GATE_A[1]} vs {GATE_B[1]} on {DATASET} ==")
    qrels = load_qrels(DATASET)
    gate = sanity_gate(qrels)
    print(json.dumps(gate, indent=2))
    if not gate["pass"]:
        print("SANITY GATE FAILED -- stopping before computing anything new.", file=sys.stderr)
        sys.exit(1)
    print("Sanity gate: PASS\n")

    print("== Budget parity check (config.json max_steps + per-row stamped max_steps) ==")
    parity = check_budget_parity()
    for label, v in parity.items():
        if label == "_parity_ok":
            continue
        print(f"  {label:10s} {v['cond']:32s} config.max_steps={v['config_max_steps']} "
              f"config.limit={v['config_limit']} row_max_steps={v['row_max_steps_values']}")
    if not parity["_parity_ok"]:
        print("BUDGET PARITY FAILED -- cells ran under different step caps. STOPPING.",
              file=sys.stderr)
        sys.exit(1)
    print("Budget parity: PASS (all four cells: max_steps=100, matched)\n")

    print("== Loading rows + metrics for the four cells ==")
    cell_metrics = {}
    for label, (cond, structure, fusion, reg_label, why) in CELLS.items():
        rows = cell_rows(SUBDIR, DATASET, cond)
        if rows is None:
            print(f"MISSING rows.jsonl for {label} ({cond}) -- STOPPING.", file=sys.stderr)
            sys.exit(1)
        m = metrics(rows, qrels, DATASET)
        cell_metrics[label] = m
        print(f"  {label:10s} n={len(m)} EM%={pct([v['em'] for v in m.values()]):.1f}  "
              f"dir={cell_dir(SUBDIR, DATASET, cond)}")

    shared = sorted(set(cell_metrics["neither"]) & set(cell_metrics["structure"])
                     & set(cell_metrics["fusion"]) & set(cell_metrics["both"]))
    n_shared = len(shared)
    print(f"\nn_shared (inner join, all four cells) = {n_shared}\n")

    # --- 2x2 table of means -----------------------------------------------------------------
    table = {}
    for label, (cond, structure, fusion, reg_label, why) in CELLS.items():
        m = cell_metrics[label]
        ems = [m[i]["em"] for i in shared]
        table[label] = dict(
            structure=structure, fusion=fusion, cond=cond, reg_label=reg_label,
            n=len(ems), em_pct=pct(ems), em_n_correct=sum(ems),
        )

    print("== 2x2 table of means (shared instance set, n={}) ==".format(n_shared))
    for label in ["neither", "structure", "fusion", "both"]:
        t = table[label]
        print(f"  {label:10s} structure={t['structure']} fusion={t['fusion']}  "
              f"n={t['n']}  EM%={t['em_pct']:.1f}  ({t['em_n_correct']}/{t['n']})")

    # --- long-format dataframe for the regressions --------------------------------------------
    records = []
    for label, (cond, structure, fusion, reg_label, why) in CELLS.items():
        m = cell_metrics[label]
        for iid in shared:
            records.append(dict(
                instance_id=iid, cell=label, structure=structure, fusion=fusion,
                em=int(bool(m[iid]["em"])),
            ))
    df = pd.DataFrame.from_records(records)
    assert len(df) == 4 * n_shared

    # --- (1) naive logistic regression (each of the 4*n_shared rows treated independently) -----
    print("\n== Naive logistic regression: em ~ structure * fusion (independence assumed) ==")
    logit_res = smf.logit("em ~ structure * fusion", data=df).fit(disp=0)
    print(logit_res.summary())

    def _extract(res, names):
        return {
            nm: dict(coef=float(res.params[nm]), se=float(res.bse[nm]),
                     z=float(res.tvalues[nm]), p=float(res.pvalues[nm]))
            for nm in names
        }

    naive_names = list(logit_res.params.index)
    naive_out = _extract(logit_res, naive_names)

    # --- (2) clustered: GEE, exchangeable correlation, cluster = instance_id ------------------
    print("\n== Clustered logistic regression: GEE, exchangeable corr., cluster=instance_id ==")
    gee_model = GEE.from_formula("em ~ structure * fusion", groups="instance_id", data=df,
                                  family=Binomial(), cov_struct=Exchangeable())
    gee_res = gee_model.fit()
    print(gee_res.summary())
    gee_names = list(gee_res.params.index)
    gee_out = {
        nm: dict(coef=float(gee_res.params[nm]), se=float(gee_res.bse[nm]),
                 z=float(gee_res.tvalues[nm]), p=float(gee_res.pvalues[nm]))
        for nm in gee_names
    }
    est_corr = None
    try:
        est_corr = float(gee_res.cov_struct.dep_params)
    except Exception:
        pass

    interaction_name = "structure:fusion"
    naive_p = naive_out[interaction_name]["p"]
    gee_p = gee_out[interaction_name]["p"]
    agree = (naive_p < 0.05) == (gee_p < 0.05)

    # --- McNemar: neither vs both, exact reviewer-highlighted contrast ------------------------
    print("\n== McNemar: neither vs both (reviewer-highlighted contrast) ==")
    mcnemar_res = _paired(cell_metrics["both"], cell_metrics["neither"])
    print(json.dumps(mcnemar_res, indent=2))
    # Does the neither-vs-both contrast WITHIN THIS matched 2x2 reproduce the brief's approximate
    # "+9.3 EM, p=2.8e-6" figure? (It should NOT -- that figure belongs to a different, READ-axis
    # contrast; see the module docstring and the .md report's section 0/6.)
    reproduces_appendix = (
        abs(mcnemar_res["delta"] - 9.3) < 1.0 and abs(mcnemar_res["p"] - 2.8e-6) < 1e-4
    )

    # --- assemble output ------------------------------------------------------------------------
    out = dict(
        dataset=DATASET, subdir=SUBDIR,
        sanity_gate=gate,
        budget_parity=parity,
        cells={label: dict(cond=CELLS[label][0], structure=CELLS[label][1],
                            fusion=CELLS[label][2], reg_label=CELLS[label][3],
                            why=CELLS[label][4],
                            run_dir=str(cell_dir(SUBDIR, DATASET, CELLS[label][0])))
               for label in CELLS},
        n_shared=n_shared,
        two_by_two=table,
        naive_logit=naive_out,
        gee_clustered=dict(params=gee_out, estimated_within_instance_corr=est_corr),
        interaction_agreement=dict(naive_p=naive_p, gee_p=gee_p, both_sig_or_both_ns=agree),
        mcnemar_neither_vs_both=mcnemar_res,
        mcnemar_reproduces_appendix_9_3=reproduces_appendix,
        note_on_prompt_numbers=(
            "The task brief's approximate figures (neither~29.6, structure-alone~+1.9, "
            "fusion-alone~+1.9, both~38.9, +9.3, p=2.8e-6) are real numbers from Appendix G's "
            "SEARCH x READ grid (Table tab:factorial-grid) but are NOT a matched-read-interface "
            "2x2: 29.6 and 38.9 are the SAME search-engine row ('Query surf. + dense (RRF)') at "
            "two DIFFERENT read levels (visit vs fetch+snippets), and the two '+1.9' figures "
            "belong to two OTHER rows' own visit->fetch deltas. That set of four numbers is "
            "confounded with the READ axis and was NOT reused here. The 2x2 actually tested in "
            "this file holds READ interface fixed at fetch+snippets (_headline_validation) for "
            "all four cells and varies only structure (bql vs bm25 query surface) and fusion "
            "(dense channel present or not)."
        ),
    )
    out_json = Path(__file__).resolve().parent / "structure_fusion_interaction_data.json"
    out_json.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {out_json}")
    return out


if __name__ == "__main__":
    main()
