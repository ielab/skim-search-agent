# Paired bootstrap 95% CIs: Sieve vs hybrid sparse-dense RRF + snip + fetch, all three datasets

Reviewer-requested power check on the browsecomp_plus_structured null (EM delta +3.3, McNemar p=0.102). Contrast: **Sieve** (`agent_research_bql_dense_snip`) vs **hybrid sparse-dense RRF + snip + fetch** (`agent_research_hybrid_fetch_snip`), tier `_headline_validation`, model dir `Tongyi-DeepResearch-30B-A3B`. Paired bootstrap over instances (resample the shared instance set with replacement, recompute the paired EM/judge delta each draw), 10000 resamples, seed=20260723 (reused as-is from `analysis/equivalence_and_latency.py`'s `BOOT_SEED`), percentile 2.5/97.5 CI. Bootstrap machinery reused verbatim from `analysis.equivalence_and_latency.{bootstrap_diff_pp, bootstrap_ci}`; EM/judge/recovery machinery reused verbatim from `analysis.structured_surface_control.{cell_metrics, paired_em, paired_judge}` (transitively `scripts.compare_cells` and `evaluation.metrics.answer_em`).

## Mandatory sanity gate (run before trusting any CI)

Reproduce the three published EM point estimates from a fresh shared-set computation (tolerance ±0.15 pp):

| dataset | n_shared | reproduced delta | published delta | McNemar p (reproduced) | published p | gate |
|---|---:|---:|---:|---:|---:|---|
| browsecomp_plus_structured | 830 | +3.25 | +3.3 | 0.102 | 0.102 | PASS |
| hotpotqa_structured | 7343 | +5.75 | +5.7 | 1.62e-27 | 1.62e-27 | PASS |
| musique_structured | 2409 | +5.52 | +5.5 | 1.58e-11 | 1.58e-11 | PASS |

**Sanity gate overall: PASS**

## Paired bootstrap 95% CIs on the EM delta (Sieve − control), all three datasets

| dataset | n_shared | point estimate (pp) | bootstrap mean (pp) | 95% CI (pp) | McNemar p |
|---|---:|---:|---:|---|---:|
| browsecomp_plus_structured | 830 | +3.25 | +3.25 | [-0.48, +6.99] | 0.102 |
| hotpotqa_structured | 7343 | +5.75 | +5.75 | [+4.70, +6.75] | 1.62e-27 |
| musique_structured | 2409 | +5.52 | +5.52 | [+3.94, +7.14] | 1.58e-11 |

## BrowseComp-Plus judge-delta paired bootstrap 95% CI (both cells fully judged)

n_shared=830 (judge coverage full=100.0%, control=100.0%), point estimate=+2.41 pp, bootstrap mean=+2.42 pp, **95% CI [-1.45, +6.27] pp**, McNemar p=0.248 (published: +2.4, p=0.248).

## Does the BrowseComp-Plus CI overlap the HotpotQA / MuSiQue point estimates?

BrowseComp-Plus EM-delta 95% CI: **[-0.48, +6.99] pp**. HotpotQA point estimate: +5.75 pp -- INSIDE the BrowseComp-Plus CI. MuSiQue point estimate: +5.52 pp -- INSIDE the BrowseComp-Plus CI.

**Both the HotpotQA and MuSiQue point estimates fall inside the BrowseComp-Plus 95% CI.** This is consistent with BrowseComp-Plus being underpowered to detect an effect of the same size the method shows on the other two datasets (n=830 vs n=7343/2409), rather than evidence that the true effect on BrowseComp-Plus is absent or smaller. This is NOT proof the underlying effect is equal on BrowseComp-Plus -- overlap of a wide CI with another dataset's point estimate is consistent with, not evidence for, an equal true effect; the interval is also consistent with a genuinely smaller (or zero) BrowseComp-Plus effect. The honest statement is: the BrowseComp-Plus data cannot statistically distinguish 'the effect is the same size here but n is too small to detect it' from 'the effect really is smaller/absent on this harder benchmark'.

