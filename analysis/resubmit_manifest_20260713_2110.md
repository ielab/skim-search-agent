# Resubmit manifest — captured 2026-07-13 21:10 before workers-config switch
Per-instance resume keyed on run-dir path: resubmitting the SAME command continues from current count.
Switch to the winning WORKERS (and AGENT_SEARCH_FLAT_FAISS=1 for retrieval) is result-preserving.

| cell | RUNS_DIR | dataset | condition | n_now | target | key env |
|---|---|---|---|--:|--:|---|
| agent_research_bm25 | _visit_uncapped | browsecomp_plus_structured | agent_research_bm25 | 691 | 830 | STRUCTURED_BACKEND=python |
| agent_research_bm25 | _visit_uncapped_k10 | browsecomp_plus_structured | agent_research_bm25 | 295 | 830 | STRUCTURED_BACKEND=lucene |
| agent_research_bm25 | _visit_uncapped | browsecomp_plus_flat | agent_research_bm25 | 601 | 830 | STRUCTURED_BACKEND=python |
| agent_research_bm25_autoread | _visit_uncapped | browsecomp_plus_structured | agent_research_bm25_autoread | 70 | 100 | STRUCTURED_BACKEND=python |
| agent_research_dense | _fullvisit | browsecomp_plus_structured | agent_research_dense | 780 | 830 | STRUCTURED_BACKEND=lucene |
| agent_research_indri_visit | _fullvisit | browsecomp_plus_structured | agent_research_indri_visit | 264 | 830 | STRUCTURED_BACKEND=lucene |
| agent_research_bql_visit | _fullvisit | browsecomp_plus_structured | agent_research_bql_visit | 530 | 830 | STRUCTURED_BACKEND=lucene |
| agent_research_hybrid | _fullvisit | browsecomp_plus_structured | agent_research_hybrid | 336 | 830 | STRUCTURED_BACKEND=lucene |
| agent_research_bql_dense_visit | _fullvisit | browsecomp_plus_structured | agent_research_bql_dense_visit | 266 | 830 | STRUCTURED_BACKEND=lucene |
| agent_research_indri_visit | _fullvisit_dense | browsecomp_plus_structured | agent_research_indri_visit | 646 | 830 | INDRI_DENSE=True STRUCTURED_BACKEND=lucene |
| agent_research_indri_visit | _qwen_fullvisit_dense | browsecomp_plus_structured | agent_research_indri_visit | 252 | 830 | INDRI_DENSE=True DENSE_MODEL=Qwen/Qwen3-Embedding-0.6B STRUCTURED_BACKEND=lucene |
| agent_research | _headline_validation | browsecomp_plus_structured | agent_research | 48 | 830 | STRUCTURED_BACKEND=lucene |
| agent_research_snip | _headline_validation | browsecomp_plus_structured | agent_research_snip | 300 | 830 | STRUCTURED_BACKEND=lucene |
| agent_research_indri | _headline_validation | browsecomp_plus_structured | agent_research_indri | 324 | 830 | STRUCTURED_BACKEND=lucene |
| agent_research_indri_snip | _headline_validation | browsecomp_plus_structured | agent_research_indri_snip | 369 | 830 | STRUCTURED_BACKEND=lucene |
| agent_research_dense_fetch | _headline_validation | browsecomp_plus_structured | agent_research_dense_fetch | 328 | 830 | STRUCTURED_BACKEND=lucene |
| agent_research_bm25_fetch_snip | _headline_validation | browsecomp_plus_structured | agent_research_bm25_fetch_snip | 317 | 830 | STRUCTURED_BACKEND=lucene |
| agent_research_hybrid_fetch_snip | _headline_validation | browsecomp_plus_structured | agent_research_hybrid_fetch_snip | 296 | 830 | STRUCTURED_BACKEND=lucene |
| agent_research_bql_dense_snip | _headline_validation | browsecomp_plus_structured | agent_research_bql_dense_snip | 830 | 830 | STRUCTURED_BACKEND=lucene |
| agent_research_indri_snip | _dense_validation | browsecomp_plus_structured | agent_research_indri_snip | 382 | 830 | INDRI_DENSE=True STRUCTURED_BACKEND=lucene |
| agent_research_dci | agent | browsecomp_plus_structured | agent_research_dci | 209 | 830 | STRUCTURED_BACKEND=lucene |
| agent_research_bm25_dci | agent | browsecomp_plus_structured | agent_research_bm25_dci | 604 | 830 | STRUCTURED_BACKEND=lucene |
| agent_research_bm25 | _visit_uncapped | hotpotqa_structured | agent_research_bm25 | 7343 | full | STRUCTURED_BACKEND=lucene |
| agent_research_bm25 | _visit_uncapped_k10 | hotpotqa_structured | agent_research_bm25 | 1204 | full | STRUCTURED_BACKEND=lucene |
| agent_research_indri_snip | _dense_validation | hotpotqa_structured | agent_research_indri_snip | 7343 | full | INDRI_DENSE=True STRUCTURED_BACKEND=lucene |
| agent_research_indri_visit | _fullvisit_dense | hotpotqa_structured | agent_research_indri_visit | 6501 | full | INDRI_DENSE=True STRUCTURED_BACKEND=lucene |
| agent_research_bql_dense_snip | _headline_validation | hotpotqa_structured | agent_research_bql_dense_snip | 2154 | full | STRUCTURED_BACKEND=lucene |
| agent_research_bm25 | _visit_uncapped | musique_structured | agent_research_bm25 | 2342 | full | STRUCTURED_BACKEND=lucene |
| agent_research_bm25 | _visit_uncapped_k10 | musique_structured | agent_research_bm25 | 535 | full | STRUCTURED_BACKEND=lucene |
| agent_research_indri_snip | _dense_validation | musique_structured | agent_research_indri_snip | 2409 | full | INDRI_DENSE=True STRUCTURED_BACKEND=lucene |
| agent_research_indri_visit | _fullvisit_dense | musique_structured | agent_research_indri_visit | 2339 | full | INDRI_DENSE=True STRUCTURED_BACKEND=lucene |
| agent_research_bql_dense_snip | _headline_validation | musique_structured | agent_research_bql_dense_snip | 1411 | full | STRUCTURED_BACKEND=lucene |
