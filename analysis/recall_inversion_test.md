# Recall/accuracy inversion: paired significance test (Sieve vs hybrid sparse-dense RRF)

The paper currently states, DESCRIPTIVELY and with no test, that the hybrid control (`agent_research_hybrid_fetch_snip`) "retrieves more of the right documents" than Sieve (`agent_research_bql_dense_snip`) while scoring lower accuracy -- the recall/accuracy inversion. Reviewer m2 (carried several rounds) asked for a paired test of that recall difference. This script computes it on the shared instance set (inner join on instance_id) for all three datasets, tier `_headline_validation`, model `Tongyi-DeepResearch-30B-A3B`. Zero new compute -- all data already exists; this only reuses `scripts.compare_cells`'s recall machinery.

## Mandatory sanity gate (run first)

**(a)** Sieve vs "No dense evidence" (`agent_research_snip`) EM contrast on browsecomp_plus_structured reproduces the published +4.7 EM / p=0.0104: n=830, EM 38.9 vs 34.2, delta=+4.7, p=0.0104 -- **PASS**.

**(b)** Published recall pairs reproduced (own shared-set computation, tol=0.1):

| dataset | n_shared | recall Sieve (computed / published) | recall control (computed / published) | gate |
|---|---:|---:|---:|---|
| browsecomp_plus_structured | 830 | 61.2 / 61.2 | 71.1 / 71.1 | **PASS** |
| hotpotqa_structured | 7343 | 94.8 / 94.8 | 96.4 / 96.4 | **PASS** |
| musique_structured | 2409 | 95.4 / 95.4 | 96.4 / 96.4 | **PASS** |

**Sanity gate overall: PASS**

## Results: paired recall test, all three datasets

| dataset | n_shared | recall Sieve | recall control | difference | test used | statistic | p | significant @ 0.05 |
|---|---:|---:|---:|---:|---|---|---:|---|
| browsecomp_plus_structured | 830 | 61.2 | 71.1 | -9.9 | exact McNemar | b=178, c=96 | 8.28e-07 | yes |
| hotpotqa_structured | 7343 | 94.8 | 96.4 | -1.6 | exact McNemar | b=278, c=162 | 3.52e-08 | yes |
| musique_structured | 2409 | 95.4 | 96.4 | -1.0 | exact McNemar | b=58, c=33 | 0.0115 | yes |

## Test choice, per dataset

- **browsecomp_plus_structured**: scripts.compare_cells.gold_doc_recall() returns a per-instance bool ("any gold doc surfaced"), confirmed empirically over this dataset's shared instance set -- the paired outcome is binary, so exact McNemar on the discordant pair counts (b, c) is the correct paired test (same test already used for paired EM/judge comparisons elsewhere in this repo).
- **hotpotqa_structured**: scripts.compare_cells.gold_doc_recall() returns a per-instance bool ("any gold doc surfaced"), confirmed empirically over this dataset's shared instance set -- the paired outcome is binary, so exact McNemar on the discordant pair counts (b, c) is the correct paired test (same test already used for paired EM/judge comparisons elsewhere in this repo).
- **musique_structured**: scripts.compare_cells.gold_doc_recall() returns a per-instance bool ("any gold doc surfaced"), confirmed empirically over this dataset's shared instance set -- the paired outcome is binary, so exact McNemar on the discordant pair counts (b, c) is the correct paired test (same test already used for paired EM/judge comparisons elsewhere in this repo).

## Significance statements (per dataset, plain language)

- **browsecomp_plus_structured** (n_shared=830): Sieve recall 61.2% vs control recall 71.1% -- the control's recall advantage (-9.9 points, exact McNemar p=8.28e-07) IS statistically significant at alpha=0.05 -- the sentence may say "significantly more" for this dataset.
- **hotpotqa_structured** (n_shared=7343): Sieve recall 94.8% vs control recall 96.4% -- the control's recall advantage (-1.6 points, exact McNemar p=3.52e-08) IS statistically significant at alpha=0.05 -- the sentence may say "significantly more" for this dataset.
- **musique_structured** (n_shared=2409): Sieve recall 95.4% vs control recall 96.4% -- the control's recall advantage (-1.0 points, exact McNemar p=0.0115) IS statistically significant at alpha=0.05 -- the sentence may say "significantly more" for this dataset.

Convention: `b` = instances the control surfaced a gold doc and Sieve did not; `c` = instances Sieve surfaced a gold doc and the control did not (matches the b/c convention used throughout this repo's paired EM/judge comparisons). No multiple-comparison correction is applied -- raw per-dataset exact McNemar p-values only.

