# Recall p-value discrepancy: reconciliation

Two artifacts reported different exact-McNemar p-values for the same nominal comparison (Sieve vs the hybrid sparse-dense RRF + snippets + section-fetch control, gold-document recall, per-instance binary indicator) on the two Wikipedia datasets:

| source | HotpotQA p | MuSiQue p |
|---|---|---|
| `analysis/recall_inversion_test.md` | 3.52e-08 | 0.0115 |
| `latex/sections/results.tex` S6.1 | 2.5e-08 | 0.0149 |

## Cause

**Confirmed: the step-budget-mismatch hypothesis.** There are two Sieve arms on hotpotqa_structured / musique_structured, run at different configured step caps (verified below from the per-row `max_steps` CONFIGURATION stamp, not observed step counts):

- `runs/_headline_validation/.../agent_research_bql_dense_snip` -- **cap 50** (the originally-published headline cell)
- `runs/_budget100/.../agent_research_bql_dense_snip` -- **cap 100** (the later budget-repair rerun)

The hybrid control (`agent_research_hybrid_fetch_snip`, tier `_headline_validation`) is uniformly **cap 100** on both Wikipedia datasets.

`analysis/recall_inversion_test.py` imports `TIER = "_headline_validation"` from `analysis/structured_surface_control.py` and applies that single tier to ALL THREE datasets -- so on HotpotQA/MuSiQue it paired the **cap-50** Sieve arm against the **cap-100** hybrid control: a budget-MISMATCHED comparison. (On BrowseComp-Plus this is harmless -- there is no `_budget100` rerun there and both cells are cap 100, which is exactly why the sanity gate below reproduces cleanly even though the script has this defect.)

`analysis/matched_cap100_results.py` -- the script that actually produced the numbers `latex/sections/results.tex` S6.1 prints (the text there says "at the matched cap-100 arm") -- instead paired the **cap-100** `_budget100` Sieve rerun against the same cap-100 hybrid control: the budget-MATCHED comparison.

Both scripts reuse the identical `scripts.compare_cells.gold_doc_recall` binary indicator, `mcnemar_p` exact McNemar, the same recovery overlay (`force_answer_backfill.load_rows_with_recovery` via `cell_rows`), and the same inner-join shared-instance-set logic (`analysis.recall_inversion_test.paired_recall`, which `matched_cap100_results.py` imports and calls unmodified). The gap is NOT a different recall definition, join logic, McNemar variant, or qrels/dataset variant -- it is that the two artifacts drew Sieve's rows from two different underlying runs of the agent (different step budgets, hence different trajectories, hence a handful of different per-instance discordant pairs), while the recall PERCENTAGES happen to come out nearly identical at both caps (94.8/94.8 on HotpotQA, 95.4/95.4 on MuSiQue -- confirmed by `analysis/matched_cap100_results.md`'s own Sieve@100-vs-Sieve@50 budget-sensitivity row, recall delta 0.0 and n.s. both datasets), which is exactly why the point estimates in the two artifacts agree while only the p-values diverge.

## Candidate explanations ruled out

- **Different recall definition/indicator**: no -- both call `scripts.compare_cells.gold_doc_recall` via the same `cell_metrics` wrapper; both confirmed binary at runtime (`_is_binary_indicator`).
- **Different shared-instance set / recovery overlay**: no -- both use `cell_rows`/`cell_dir` (recovery overlay applied identically in both) and the same inner-join (`paired_recall`); n_shared matches per dataset per arm (see table below).
- **Different McNemar variant**: no -- both call `scripts.compare_cells.mcnemar_p` (exact, two-sided) unmodified.
- **Different qrels / dataset variant (structured vs flat)**: no -- both use `hotpotqa_structured` / `musique_structured` and `load_qrels` unmodified.

## Mandatory sanity gate

BrowseComp reproduced: n_shared=830, recall 61.2 (Sieve) vs 71.1 (hybrid), b=178, c=96, p=8.28e-07 (expected p=8.28e-07) -- **PASS**.

## Recomputed: all four combinations (2 datasets x 2 Sieve arms), vs the same hybrid control

| dataset | Sieve arm | Sieve cap | control cap | budget matched | n_shared | recall Sieve | recall control | delta | b | c | McNemar p | sig @ 0.05 |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| hotpotqa_structured | Sieve@cap50 (_headline_validation) | 50 | 100 | **NO** | 7343 | 94.8 | 96.4 | -1.6 | 278 | 162 | 3.52e-08 | yes |
| hotpotqa_structured | Sieve@cap100 (_budget100) | 100 | 100 | yes | 7343 | 94.8 | 96.4 | -1.6 | 286 | 167 | 2.47e-08 | yes |
| musique_structured | Sieve@cap50 (_headline_validation) | 50 | 100 | **NO** | 2409 | 95.4 | 96.4 | -1.0 | 58 | 33 | 0.0115 | yes |
| musique_structured | Sieve@cap100 (_budget100) | 100 | 100 | yes | 2409 | 95.4 | 96.4 | -1.0 | 57 | 33 | 0.0149 | yes |

## Verdict

The paper (`latex/sections/results.tex` S6.1) should print the **budget-matched, cap-100 vs cap-100** row for HotpotQA and MuSiQue -- i.e. HotpotQA p=2.5e-08 (recomputed here as shown above), MuSiQue p=0.0149 -- because it is the only apples-to-apples comparison of the two Wikipedia datasets against a control that is itself cap 100; the cap-50-vs-cap-100 numbers in `analysis/recall_inversion_test.md` compare Sieve run under half the control's step budget and must not be quoted for these two datasets. Both conclusions happen to remain 'significant' in this instance (unlike a scenario where the mismatch could have flipped significance), so the paper's qualitative claim was never wrong -- but the specific p-values it must cite are the matched ones, and `analysis/recall_inversion_test.md` should be corrected or superseded (e.g. by pointing its HotpotQA/MuSiQue rows at the `_budget100` Sieve cell) so it stops disagreeing with the paper on a comparison that is not actually a like-for-like rerun of the same test. BrowseComp-Plus is unaffected either way (no `_budget100` rerun exists there; both artifacts already agree at p=8.28e-07).

