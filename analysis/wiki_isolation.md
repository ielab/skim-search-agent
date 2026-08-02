# Wiki component isolation (round8 Job 2)

Same family of DIRECT paired, same-interface isolation contrasts `analysis/round5_analyses.py` computes on browsecomp_plus_structured (the dataset where the method's headline effect is weakest), now computed on hotpotqa_structured and musique_structured wherever the required cells have crossed a 90% completion gate (checked live against runs/ at script run time, not assumed). Neither side of any contrast below is the dataset's SERP-bm25 baseline or the full method compared to that baseline -- those pairings are `comparison_result.md`'s and `analysis/round5_analyses.py`'s M3, respectively.


## Cost of removing dense fusion: full method vs No dense evidence

**Isolates:** What removing dense (embedding) fusion costs the full method, holding the structured (BQL) query surface AND the snip+fetch read interface fixed on both sides. Both cells share: BQL query surface, snippet display, section-fetch reading; they differ only in whether dense retrieval is fused in.

**Does NOT isolate:** Whether the structured query surface itself helps (neither removed here -- both sides have BQL); that isolation is contrast 2 below. Does not isolate dense retrieval's value WITHOUT the structured surface either -- that is contrast 3.

A = Full method (bql+dense+snip fetch); B = No dense evidence (bql+snip fetch).

| dataset | n_shared | EM_A% | EM_B% | ΔEM (A-B) | McNemar p_em | judge_A% | judge_B% | Δjudge | McNemar p_judge |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| hotpotqa_structured | 7343 | 45.3 | 44.5 | +0.8 | 0.103 | pending (cov A=100%, B=0%) | pending | -- | -- |
| musique_structured | 2409 | 29.2 | 27.9 | +1.3 | 0.0949 | pending (cov A=100%, B=0%) | pending | -- | -- |
| browsecomp_plus_structured | 830 | 38.9 | 34.2 | +4.7 | 0.0104 | 41.3 | 36.7 | +4.6 | 0.0145 |


## Query surface alone: No dense evidence vs Sparse-only, same interface

**Isolates:** Whether the field-tagged Boolean query SURFACE itself (vs plain keyword BM25) helps, holding the read interface (snippet-bearing listing + named-section fetch) and the absence of dense fusion fixed on both sides. Identical pairing to round5_analyses.py's M5 query_surface isolation, now computed on this dataset.

**Does NOT isolate:** Anything about dense fusion (neither side has it) or the 'visit whole document' reading style (neither side uses it). Does not separate BQL's Boolean filter from BQL's field-tag vocabulary -- BQL bundles both.

A = No dense evidence (bql+snip fetch); B = Sparse only, same interface (bm25+snip fetch).

| dataset | n_shared | EM_A% | EM_B% | ΔEM (A-B) | McNemar p_em | judge_A% | judge_B% | Δjudge | McNemar p_judge |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| hotpotqa_structured | 7343 | 44.5 | 41.2 | +3.3 | 3.6e-10 | pending (cov A=0%, B=0%) | pending | -- | -- |
| musique_structured | 2409 | 27.9 | 23.5 | +4.4 | 3.02e-08 | pending (cov A=0%, B=0%) | pending | -- | -- |
| browsecomp_plus_structured | 830 | 34.2 | 32.9 | +1.3 | 0.531 | 36.7 | 35.3 | +1.4 | 0.506 |


## Dense fusion alone: Dense-only same interface vs Sparse-only same interface

**Isolates:** Whether swapping the query engine from plain BM25 to pure dense retrieval helps, holding the read interface (snippet listing + section-fetch) fixed and the BQL structured query surface absent from both sides. Identical pairing to round5_analyses.py's M5 dense_fusion isolation, now computed on this dataset.

**Does NOT isolate:** Anything about the BQL structured query surface (neither side has it) or about dense fusion INSIDE a structured query (that is contrast 1's 'full' cell vs the 'no_dense' cell, i.e. contrast 1 itself, not this pair).

A = Dense only, same interface (dense+snip fetch); B = Sparse only, same interface (bm25+snip fetch).

| dataset | n_shared | EM_A% | EM_B% | ΔEM (A-B) | McNemar p_em | judge_A% | judge_B% | Δjudge | McNemar p_judge |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| hotpotqa_structured | 7343 | 39.4 | 41.2 | -1.8 | 0.00112 | pending (cov A=0%, B=0%) | pending | -- | -- |
| musique_structured | 2409 | 22.4 | 23.5 | -1.0 | 0.236 | pending (cov A=0%, B=0%) | pending | -- | -- |
| browsecomp_plus_structured | 830 | 27.7 | 32.9 | -5.2 | 0.00893 | 31.4 | 35.3 | -3.9 | 0.0628 |


## Live cell-completion status (all datasets, all cells this script uses)

| dataset | cell | condition | status | n | total_n | % complete | canonical merged | shard dirs found |
|---|---|---|---|---:|---:|---:|---|---:|
| hotpotqa_structured | Full method (bql+dense+snip fetch) | `agent_research_bql_dense_snip` | ok | 7343 | 7343 | 100.0% | True | 0 |
| hotpotqa_structured | No dense evidence (bql+snip fetch) | `agent_research_snip` | ok | 7343 | 7343 | 100.0% | True | 20 |
| hotpotqa_structured | Sparse only, same interface (bm25+snip fetch) | `agent_research_bm25_fetch_snip` | ok | 7343 | 7343 | 100.0% | True | 20 |
| hotpotqa_structured | Dense only, same interface (dense+snip fetch) | `agent_research_dense_fetch` | ok | 7343 | 7343 | 100.0% | True | 20 |
| musique_structured | Full method (bql+dense+snip fetch) | `agent_research_bql_dense_snip` | ok | 2409 | 2409 | 100.0% | True | 0 |
| musique_structured | No dense evidence (bql+snip fetch) | `agent_research_snip` | ok | 2409 | 2409 | 100.0% | True | 15 |
| musique_structured | Sparse only, same interface (bm25+snip fetch) | `agent_research_bm25_fetch_snip` | ok | 2409 | 2409 | 100.0% | True | 12 |
| musique_structured | Dense only, same interface (dense+snip fetch) | `agent_research_dense_fetch` | ok | 2409 | 2409 | 100.0% | True | 15 |
| browsecomp_plus_structured | Full method (bql+dense+snip fetch) | `agent_research_bql_dense_snip` | ok | 830 | 830 | 100.0% | True | 0 |
| browsecomp_plus_structured | No dense evidence (bql+snip fetch) | `agent_research_snip` | ok | 830 | 830 | 100.0% | True | 0 |
| browsecomp_plus_structured | Sparse only, same interface (bm25+snip fetch) | `agent_research_bm25_fetch_snip` | ok | 830 | 830 | 100.0% | True | 0 |
| browsecomp_plus_structured | Dense only, same interface (dense+snip fetch) | `agent_research_dense_fetch` | ok | 830 | 830 | 100.0% | True | 0 |

## Lift-able sentences

- Cost of removing dense fusion on hotpotqa_structured: Full method (bql+dense+snip fetch) scores 45.3% EM vs No dense evidence (bql+snip fetch)'s 44.5% EM on the 7343 shared instances (+0.8 pts; exact McNemar p=0.103, not statistically significant).

- Cost of removing dense fusion on musique_structured: Full method (bql+dense+snip fetch) scores 29.2% EM vs No dense evidence (bql+snip fetch)'s 27.9% EM on the 2409 shared instances (+1.3 pts; exact McNemar p=0.0949, not statistically significant).

- Cost of removing dense fusion on browsecomp_plus_structured: Full method (bql+dense+snip fetch) scores 38.9% EM vs No dense evidence (bql+snip fetch)'s 34.2% EM on the 830 shared instances (+4.7 pts; exact McNemar p=0.0104, statistically significant).

- Query surface alone on hotpotqa_structured: No dense evidence (bql+snip fetch) scores 44.5% EM vs Sparse only, same interface (bm25+snip fetch)'s 41.2% EM on the 7343 shared instances (+3.3 pts; exact McNemar p=3.6e-10, statistically significant).

- Query surface alone on musique_structured: No dense evidence (bql+snip fetch) scores 27.9% EM vs Sparse only, same interface (bm25+snip fetch)'s 23.5% EM on the 2409 shared instances (+4.4 pts; exact McNemar p=3.02e-08, statistically significant).

- Query surface alone on browsecomp_plus_structured: No dense evidence (bql+snip fetch) scores 34.2% EM vs Sparse only, same interface (bm25+snip fetch)'s 32.9% EM on the 830 shared instances (+1.3 pts; exact McNemar p=0.531, not statistically significant).

- Dense fusion alone on hotpotqa_structured: Dense only, same interface (dense+snip fetch) scores 39.4% EM vs Sparse only, same interface (bm25+snip fetch)'s 41.2% EM on the 7343 shared instances (-1.8 pts; exact McNemar p=0.00112, statistically significant).

- Dense fusion alone on musique_structured: Dense only, same interface (dense+snip fetch) scores 22.4% EM vs Sparse only, same interface (bm25+snip fetch)'s 23.5% EM on the 2409 shared instances (-1.0 pts; exact McNemar p=0.236, not statistically significant).

- Dense fusion alone on browsecomp_plus_structured: Dense only, same interface (dense+snip fetch) scores 27.7% EM vs Sparse only, same interface (bm25+snip fetch)'s 32.9% EM on the 830 shared instances (-5.2 pts; exact McNemar p=0.00893, statistically significant).
