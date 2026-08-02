# Flat-vs-structured comparison (C1)

| flat dataset | structured dataset | condition | run group | n_shared | flat EM% | struct EM% | ΔEM (pp) | McNemar p (EM) | flat judge% | struct judge% | Δjudge (pp) | McNemar p (judge) | n judged | flat recall% | struct recall% | flat tok/ep | struct tok/ep | flat calls/ep | struct calls/ep |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| browsecomp_plus_flat | browsecomp_plus_structured | `agent_research_bm25` | `runs/_visit_uncapped` | 830 | 34.8 | 36.4 | +1.6 | 0.372 | 37.3 | 39.3 | +1.9 | 0.263 | 830 | 58.7 | 58.9 | 64,586 | 63,805 | 63.2 | 62.5 |
| hotpotqa_flat | hotpotqa_structured | `agent_research_bm25` | `runs/_visit_uncapped` | 7343 | 42.4 | 43.7 | +1.3 | 0.00978 | pending (no judge_cache on one/both sides) | pending | -- | -- | 0 | 95.3 | 95.2 | 21,817 | 19,939 | 23.9 | 15.9 |
| musique_flat | musique_structured | `agent_research_bm25` | `runs/_visit_uncapped` | 2409 | 25.3 | 26.4 | +1.2 | 0.132 | pending (no judge_cache on one/both sides) | pending | -- | -- | 0 | 94.3 | 95.2 | 46,594 | 43,329 | 45.3 | 29.6 |

**Interpretation.** Every condition in this table is the SAME agent run over structured vs. flat (byte-identical) text; none of these conditions query the structured surface (fielded/Boolean search, section-level fetch) -- they are plain BM25 keyword search with whole-document visit on both sides. Any delta here is therefore attributable to the corpus/ranking alone (e.g. BM25 term statistics shifting slightly because the structured corpus's indexed text differs in incidental ways -- section markers, metadata fields folded into the index), NOT to the method's interface, which is not in use in any of these cells.

- **browsecomp_plus_structured vs browsecomp_plus_flat**, `agent_research_bm25` (n_shared=830): structure alone helps this structure-blind agent by +1.6 pp EM (36.4% vs 34.8%, McNemar b=84/c=97, p=0.372, not significant at α=0.05) and +1.9 pp judge accuracy (39.3% vs 37.3%, n_judged=830, McNemar p=0.263), while gold-doc recall is 58.9% (structured) vs 58.7% (flat) -- nearly flat, so the EM/judge gap is not obviously a retrieval-recall effect.
- **hotpotqa_structured vs hotpotqa_flat**, `agent_research_bm25` (n_shared=7343): structure alone helps this structure-blind agent by +1.3 pp EM (43.7% vs 42.4%, McNemar b=588/c=681, p=0.00978, significant at α=0.05) and judge accuracy is not comparable (no judge_cache on the flat side yet), while gold-doc recall is 95.2% (structured) vs 95.3% (flat) -- nearly flat, so the EM/judge gap is not obviously a retrieval-recall effect.
- **musique_structured vs musique_flat**, `agent_research_bm25` (n_shared=2409): structure alone helps this structure-blind agent by +1.2 pp EM (26.4% vs 25.3%, McNemar b=147/c=175, p=0.132, not significant at α=0.05) and judge accuracy is not comparable (no judge_cache on the flat side yet), while gold-doc recall is 95.2% (structured) vs 94.3% (flat) -- nearly flat, so the EM/judge gap is not obviously a retrieval-recall effect.

---

# Corpus-vs-interface decomposition (extension)

For each dataset with a flat/structured control pair (above): flat-corpus baseline EM (structure-blind BM25 agent on the flat corpus) -> structured-corpus baseline EM (SAME agent, structured corpus -- **the corpus step**, identical to the `agent_research_bm25` row above) -> structured-corpus full-method EM (bql+dense+snip fetch -- **the interface step**). Both steps' paired significance (exact McNemar) is computed on that step's own shared-instance set (the two steps use slightly different instance sets in general -- corpus step is flat∩structured, interface step is structured-baseline∩structured-full-method -- both are the FULL dataset for datasets with complete runs, so this is a minor caveat in practice here, not a real discrepancy; noted per-row below).

**Assumption flagged, not tested:** the total is computed as corpus-step delta + interface-step delta (additive). The full method has never been run on the flat corpus, so there is no cell that directly measures whether the corpus effect and the interface effect interact -- this decomposition assumes they do not. Treat 'total' as an estimate under that assumption, not a directly observed quantity.

| dataset | n (corpus step) | flat-corpus baseline EM% | structured-corpus baseline EM% | Δ corpus (pp) | p (corpus) | n (interface step) | structured-corpus full-method EM% | Δ interface (pp) | p (interface) | Δ total (pp) | corpus share % | interface share % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| browsecomp_plus_structured | 830 | 34.8 | 36.4 | +1.6 | 0.372 | 830 | 38.9 | +2.5 | 0.217 | +4.1 | 38% | 62% |
| hotpotqa_structured | 7343 | 42.4 | 43.7 | +1.3 | 0.00978 | 7343 | 45.3 | +1.6 | 0.00186 | +2.8 | 45% | 55% |
| musique_structured | 2409 | 25.3 | 26.4 | +1.2 | 0.132 | 2409 | 29.2 | +2.7 | 0.00073 | +3.9 | 30% | 70% |

## Per-dataset reading

- **browsecomp_plus_structured**: flat-corpus baseline 34.8% -> structured-corpus baseline 36.4% (+1.6 pp, McNemar p=0.372, not significant, n=830) -> structured-corpus full method 38.9% (+2.5 pp, McNemar p=0.217, not significant, n=830). Of the +4.1 pp total (additive estimate), the corpus step accounts for 38% and the interface step for 62%.
- **hotpotqa_structured**: flat-corpus baseline 42.4% -> structured-corpus baseline 43.7% (+1.3 pp, McNemar p=0.00978, significant, n=7343) -> structured-corpus full method 45.3% (+1.6 pp, McNemar p=0.00186, significant, n=7343). Of the +2.8 pp total (additive estimate), the corpus step accounts for 45% and the interface step for 55%.
- **musique_structured**: flat-corpus baseline 25.3% -> structured-corpus baseline 26.4% (+1.2 pp, McNemar p=0.132, not significant, n=2409) -> structured-corpus full method 29.2% (+2.7 pp, McNemar p=0.00073, significant, n=2409). Of the +3.9 pp total (additive estimate), the corpus step accounts for 30% and the interface step for 70%.

**Cross-dataset pattern:** corpus share of the total ranges 30%-45% across 3 datasets (browsecomp_plus_structured=38%, hotpotqa_structured=45%, musique_structured=30%) -- roughly consistent across datasets (corpus share spread <= 20 points).
