# Structured query surface control: Sieve vs hybrid sparse-dense RRF (same read interface)

Isolates the structured query surface: **Sieve** (`agent_research_bql_dense_snip`) vs the strongest non-structured control at the *same* read interface -- plain BM25+dense RRF with snippets and section fetch (`agent_research_hybrid_fetch_snip`). Both on `browsecomp_plus_structured`, tier `_headline_validation`, n=830 / n=830 (n_shared=830 after inner join on instance_id).

**Sanity check** (must reproduce a published number before this comparison is trusted): Sieve vs "No dense evidence" (`agent_research_snip`) reproduced at n=830, EM 38.9 vs 34.2, delta=+4.7, p=0.0104 (published: +4.7 EM, p=0.0104) -- **PASS**.

## Results

| metric | Sieve (full) | hybrid+snip+fetch (control) | delta (full-control) | b | c | McNemar p | sig @ 0.05 |
|---|---:|---:|---:|---:|---:|---:|---|
| EM % | 38.9 | 35.7 | +3.3 | 113 | 140 | 0.102 | no |
| Judge accuracy % | 41.3 | 38.9 | +2.4 | 125 | 145 | 0.248 | no |
| Mean count-once tokens/episode | 44,910 | 44,448 | +462 | | | | |
| Mean gold-doc recall % | 61.2 | 71.1 | -9.9 | | | | |

Judge coverage: full=100.0%, control=100.0% (gate: >= 90%).

Convention (matches `ablation_deltas.py`): `b` = instances the control cell got right and Sieve got wrong; `c` = instances Sieve got right and the control got wrong. McNemar's test is symmetric in {b, c} so `p` is unaffected by labeling, but note the reviewer's ad hoc b/c labels below are swapped relative to this convention (140/113 vs this script's 113/140 for EM) -- the underlying discordant-pair counts {113, 140} match exactly; only which one is called b vs c differs.

## Comparison to the reviewer's ad hoc numbers

| metric | ad hoc (reviewer) | this script |
|---|---:|---:|
| EM full | 38.9 | 38.9 |
| EM control | 35.7 | 35.7 |
| EM delta | +3.3 | +3.3 |
| EM b/c | 140/113 | 113/140 |
| EM McNemar p | 0.102 | 0.102 |
| Judge full | 41.3 | 41.3 |
| Judge control | 38.9 | 38.9 |
| Judge delta | +2.4 | +2.4 |
| Judge b/c | 145/125 | 125/145 |
| Judge McNemar p | 0.248 | 0.248 |
| Mean tokens full | ~45k | 44,910 |
| Mean tokens control | ~44k | 44,448 |
| Mean recall full | 61.2% | 61.2% |
| Mean recall control | 71.1% | 71.1% |

## Significance statements

- EM delta (+3.3 points, Sieve 38.9% vs control 35.7%, McNemar p=0.102, n=830) is NOT statistically significant at alpha=0.05.
- Judge-accuracy delta (+2.4 points, Sieve 41.3% vs control 38.9%, McNemar p=0.248, n=830) is NOT statistically significant at alpha=0.05.
