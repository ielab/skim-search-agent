# Fetch-error audit: cost of the stale BrowseComp-Plus manual

Quantifies fetch/read tool-call error rates across 5 cells (tier `_headline_validation`, model dir `Tongyi-DeepResearch-30B-A3B`), whether BrowseComp-Plus is anomalous relative to the two Wikipedia-backed datasets, the wasted-effort cost in the browsecomp Sieve cell, and a strictly descriptive accuracy split. See `analysis/fetch_error_audit.py`'s module docstring for the exact schema inspection and fetch-call / error identification rules used below.

## Part 1 -- per-cell fetch-call error rates

| Cell | episodes | fetch calls | errors | error % | episodes w/ >=1 error | episode error % |
|---|---:|---:|---:|---:|---:|---:|
| browsecomp_plus_structured / agent_research_bql_dense_snip (Sieve) | 830 | 11856 | 3978 | 33.6% | 699 | 84.2% |
| browsecomp_plus_structured / agent_research_bm25_fetch_snip (sparse-only control) | 830 | 9148 | 2640 | 28.9% | 538 | 64.8% |
| browsecomp_plus_structured / agent_research_snip (no-dense rung) | 830 | 10518 | 3449 | 32.8% | 647 | 78.0% |
| hotpotqa_structured / agent_research_bql_dense_snip | 7343 | 57141 | 9638 | 16.9% | 3420 | 46.6% |
| musique_structured / agent_research_bql_dense_snip | 2409 | 26566 | 5264 | 19.8% | 1538 | 63.8% |

### Error kinds -- browsecomp_plus_structured / agent_research_bql_dense_snip (Sieve)

| kind | count |
|---|---:|
| no_section | 3847 |
| rank_out_of_range | 66 |
| ambiguous_section | 51 |
| no_such_doc | 5 |
| other | 4 |
| empty_specs | 3 |
| ambiguous_doc | 1 |
| bad_spec | 1 |

Most common exact error string (103x): `[5 §body]  ERROR: no section 'body' on 5. Available: (intro)·News·Fantasy·Competitions·Community·How to Watch·Governance·About`

### Error kinds -- browsecomp_plus_structured / agent_research_bm25_fetch_snip (sparse-only control)

| kind | count |
|---|---:|
| no_section | 2077 |
| empty_specs | 397 |
| bad_spec | 50 |
| ambiguous_doc | 44 |
| ambiguous_section | 31 |
| other | 19 |
| no_such_doc | 16 |
| rank_out_of_range | 6 |

Most common exact error string (397x): `ERROR: fetch needs at least one (doc, section) pair.`

### Error kinds -- browsecomp_plus_structured / agent_research_snip (no-dense rung)

| kind | count |
|---|---:|
| no_section | 3345 |
| rank_out_of_range | 46 |
| ambiguous_section | 37 |
| no_such_doc | 8 |
| empty_specs | 6 |
| other | 5 |
| bad_spec | 2 |

Most common exact error string (93x): `[922 §Overview and background]  ERROR: no section 'Overview and background' on 922. Available: (intro)·Early and personal life·Domestic career·Indian Premier League·Middlesex·International career·Early one-day seasons·First World Cup success·Test debut·2001 Ashes·2003 World Cup·Decline and revival`

### Error kinds -- hotpotqa_structured / agent_research_bql_dense_snip

| kind | count |
|---|---:|
| no_section | 8681 |
| rank_out_of_range | 821 |
| ambiguous_section | 59 |
| no_such_doc | 39 |
| bad_spec | 25 |
| empty_specs | 6 |
| ambiguous_doc | 5 |
| unknown_tool | 2 |

Most common exact error string (386x): `[2]  ERROR: rank 2 out of range (last search had 1 results).`

### Error kinds -- musique_structured / agent_research_bql_dense_snip

| kind | count |
|---|---:|
| no_section | 4777 |
| rank_out_of_range | 378 |
| ambiguous_section | 52 |
| no_such_doc | 44 |
| bad_spec | 10 |
| unknown_tool | 2 |
| ambiguous_doc | 1 |

Most common exact error string (168x): `[2]  ERROR: rank 2 out of range (last search had 1 results).`

## Part 2 -- is BrowseComp anomalous?

- BrowseComp Sieve fetch error rate: **33.6%**
- HotpotQA Sieve fetch error rate: **16.9%**
- MuSiQue Sieve fetch error rate: **19.8%**
- BrowseComp is 2.0x HotpotQA's rate, 1.7x MuSiQue's rate.

### Manual each cell actually renders (via `load_condition`, same composer the harness uses)

| Cell | toolset | field_profile passed | contains stale "no named sections" claim |
|---|---|---|---|
| browsecomp_plus_structured / agent_research_bql_dense_snip (Sieve) | bql_dense_snip | browsecomp | YES |
| browsecomp_plus_structured / agent_research_bm25_fetch_snip (sparse-only control) | bm25_fetch_snip | browsecomp | no |
| browsecomp_plus_structured / agent_research_snip (no-dense rung) | search_fetch_s | browsecomp | YES |
| hotpotqa_structured / agent_research_bql_dense_snip | bql_dense_snip | wiki | no |
| musique_structured / agent_research_bql_dense_snip | bql_dense_snip | wiki | no |

## Part 3 -- wasted effort, browsecomp Sieve cell

- Fetch calls: 11856, of which 3978 errored (33.6%).
- LLM calls in the cell: 50370; calls that produced an errored fetch: 3978 (7.9% of all LLM calls in the cell).
- **Count-once tokens are NOT attributable per call** -- row['total_tokens_once']/'context_once_tokens' are EPISODE-level dedup quantities (each real token the model ever saw, counted once across the WHOLE episode) -- there is no per-call count-once field, and count-once dedup cannot be decomposed to a single call's share without re-simulating the dedup logic against a specific call's marginal contribution, which this script does not attempt. NOT reported. The step-summed figures below are reported instead, explicitly labeled as step-summed (NOT count-once, per analysis/cost_dollars.py's existing two-basis distinction).
- Secondary, STEP-SUMMED (not count-once) approximation: tokens on the specific LLM-call steps that produced an errored fetch = 79,271,972 of 1,267,745,222 step-summed tokens in the cell (6.3%).

## Part 4 -- descriptive accuracy split, browsecomp Sieve cell only

- Episodes WITH >=1 fetch error: n=699, EM accuracy = 44.1%
- Episodes WITHOUT any fetch error: n=131, EM accuracy = 9.2%
- Delta (no-error minus error): -34.9 pp
- 2x2 table [[308, 391], [12, 119]]: chi2=55.268, chi2 p=0.0000, Fisher exact p=0.0000 (min expected cell = 50.5)
- Confound check: 127/131 (96.9%) of the 'no fetch error' group made ZERO fetch calls at all (mean fetch calls in that group = 0.04, vs 17.0 in the 'has error' group).

**CAVEAT: OBSERVATIONAL split on a POST-TREATMENT variable, not a randomized or matched comparison. Whether an episode ever hits a fetch error is itself driven by question difficulty and how many fetch attempts the episode needed -- harder questions plausibly cause BOTH more fetch errors (more exploratory fetching) AND more wrong answers, independent of any causal harm from the errors themselves. This comparison is DESCRIPTIVE ONLY and CANNOT establish that fetch errors caused the accuracy difference. Do not read it as an effect estimate. CONCRETELY here: the 'no fetch error' group is almost entirely episodes that made (near-)ZERO fetch calls in the first place, not episodes that fetched cleanly -- so its low accuracy reflects giving up / never reading a document, not the ABSENCE of fetch errors helping. The direction of the raw delta (no-error group scoring LOWER) is a Simpson's-paradox-style artifact of this confound, not evidence that fetch errors improve accuracy.**

