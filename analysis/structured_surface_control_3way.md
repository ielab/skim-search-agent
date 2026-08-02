# Structured query surface control, three datasets: Sieve vs hybrid sparse-dense RRF (same read interface)

Isolates the structured query surface: **Sieve** (`agent_research_bql_dense_snip`) vs the strongest non-structured control at the *same* read interface -- plain BM25+dense RRF with snippets and section fetch (`agent_research_hybrid_fetch_snip`). Tier `_headline_validation`, model dir `Tongyi-DeepResearch-30B-A3B`, on all three datasets. Previously this contrast existed only on browsecomp_plus_structured (n=830), reported as a null; hotpotqa_structured (n=7343) and musique_structured (n=2409) cells have now been merged, completing it on all three.

## Mandatory sanity gate (run before trusting anything new)

**(a)** Sieve vs "No dense evidence" (`agent_research_snip`) on browsecomp_plus_structured reproduces the published +4.7 EM / p=0.0104: n=830, EM 38.9 vs 34.2, delta=+4.7, p=0.0104 -- **PASS**.

**(b)** The existing published browsecomp hybrid-control result reproduces at +3.3 EM (p=0.102) / +2.4 judge (p=0.248): EM delta=+3.3 (p=0.102), judge delta=+2.4 (p=0.248) -- **PASS**.

**Sanity gate overall: PASS**

## Results (all three datasets)

| dataset | n_shared | EM Sieve | EM control | EM delta | b | c | McNemar p | sig@.05 | judge Sieve | judge control | judge delta | judge p | judge sig@.05 | judge status | tok Sieve | tok control | recall Sieve % | recall control % |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---|---|---:|---:|---:|---:|
| browsecomp_plus_structured | 830 | 38.9 | 35.7 | +3.3 | 113 | 140 | 0.102 | no | 41.3 | 38.9 | +2.4 | 0.248 | no | computed | 44,910 | 44,448 | 61.2 | 71.1 |
| hotpotqa_structured | 7343 | 45.3 | 39.5 | +5.7 | 549 | 971 | 1.62e-27 | yes | -- | -- | -- | -- | -- | not judged by design (EM/F1 benchmark, not the LLM-judge benchmark) | 13,873 | 18,152 | 94.8 | 96.4 |
| musique_structured | 2409 | 29.2 | 23.7 | +5.5 | 129 | 262 | 1.58e-11 | yes | -- | -- | -- | -- | -- | not judged by design (EM/F1 benchmark, not the LLM-judge benchmark) | 21,268 | 31,209 | 95.4 | 96.4 |

## Significance statements (per dataset)

- **browsecomp_plus_structured**: EM delta +3.3 points (Sieve 38.9% vs control 35.7%, McNemar p=0.102, n=830) is NOT statistically significant at alpha=0.05. Judge: computed.
- **hotpotqa_structured**: EM delta +5.7 points (Sieve 45.3% vs control 39.5%, McNemar p=1.62e-27, n=7343) is STATISTICALLY SIGNIFICANT at alpha=0.05. Judge: not judged by design (EM/F1 benchmark, not the LLM-judge benchmark).
- **musique_structured**: EM delta +5.5 points (Sieve 29.2% vs control 23.7%, McNemar p=1.58e-11, n=2409) is STATISTICALLY SIGNIFICANT at alpha=0.05. Judge: not judged by design (EM/F1 benchmark, not the LLM-judge benchmark).

Convention (matches `structured_surface_control.py` / `ablation_deltas.py`): `b` = instances the control cell got right and Sieve got wrong; `c` = instances Sieve got right and the control got wrong. No multiple-comparison correction is applied here -- raw per-comparison exact McNemar p-values only.

