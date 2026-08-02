# Cell comparison (auto-generated)

## browsecomp_plus_flat

| cell | n | judge% | Δjudge | p_judge | EM% | ΔEM | p_em | recall% | surfaced% | avg_tok/inst | avg_llm_calls | empty% | recov | n_mut |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| flat-twin bm25 | 519 | 36.5 | — | — | 32.8 | — | — | 56.5 | 46.6 | 2,361k | 64.4 | 6.0 | 77 | — |

## browsecomp_plus_structured

| cell | n | judge% | Δjudge | p_judge | EM% | ΔEM | p_em | recall% | surfaced% | avg_tok/inst | avg_llm_calls | empty% | recov | n_mut |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SERP bm25 [BASELINE] | 621 | 38.8 | — | — | 36.7 | — | — | 59.1 | 50.4 | 2,324k | 61.9 | 7.4 | 72 | — |
| indri+snip fetch | 200 | 30.0 | -9.0 | 0.0153 | 27.0 | -9.0 | 0.0133 | 47.0 | 38.5 | 1,627k | 70.5 | 3.0 | 62 | 200 |
| one-shot dense | 830 | 8.1 | -30.6 | 7.69e-32 | 6.7 | -30.0 | 1.28e-41 | 0.0 | nan | 21k | 1.0 | 2.7 | 0 | 621 |
| one-shot bm25 | 830 | 2.9 | -36.5 | 6.45e-49 | 1.8 | -35.6 | 9.43e-64 | 0.0 | nan | 117k | 1.0 | 3.7 | 0 | 621 |
| SERP bm25 k=10 | — | | | | | | | | | | | | | (no rows) |
| bm25 auto-read | — | | | | | | | | | | | | | (no rows) |
| dense visit | — | | | | | | | | | | | | | (no rows) |
| indri visit | — | | | | | | | | | | | | | (no rows) |
| bql visit | — | | | | | | | | | | | | | (no rows) |
| indri+dense visit | — | | | | | | | | | | | | | (no rows) |
| hybrid rrf visit | — | | | | | | | | | | | | | (no rows) |
| bql+dense visit | — | | | | | | | | | | | | | (no rows) |
| hybrid+snip fetch | — | | | | | | | | | | | | | (no rows) |
| bql+dense+snip fetch | — | | | | | | | | | | | | | (no rows) |
| qwen dense visit | — | | | | | | | | | | | | | (no rows) |
| qwen hybrid visit | — | | | | | | | | | | | | | (no rows) |
| qwen bql+dense visit | — | | | | | | | | | | | | | (no rows) |
| qwen indri+dense visit | — | | | | | | | | | | | | | (no rows) |
| qwen indri+dense+snip fetch | — | | | | | | | | | | | | | (no rows) |
| bql fetch | — | | | | | | | | | | | | | (no rows) |
| bql+snip fetch | — | | | | | | | | | | | | | (no rows) |
| indri fetch | — | | | | | | | | | | | | | (no rows) |
| dense fetch | — | | | | | | | | | | | | | (no rows) |
| bm25+snip fetch | — | | | | | | | | | | | | | (no rows) |
| indri+dense+snip fetch | — | | | | | | | | | | | | | (no rows) |
| dci | — | | | | | | | | | | | | | (no rows) |
| bm25->dci | — | | | | | | | | | | | | | (no rows) |

## hotpotqa_structured

| cell | n | judge% | Δjudge | p_judge | EM% | ΔEM | p_em | recall% | surfaced% | avg_tok/inst | avg_llm_calls | empty% | recov | n_mut |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SERP bm25 k=10 | 11 | — | — | — | 36.4 | — | — | 54.5 | 63.6 | 39k | 5.3 | 27.3 | 0 | — |
| SERP bm25 [BASELINE] | — | | | | | | | | | | | | | (no rows) |
| indri+dense+snip fetch | — | | | | | | | | | | | | | (no rows) |
| indri+dense visit | — | | | | | | | | | | | | | (no rows) |
| bql+dense+snip fetch | — | | | | | | | | | | | | | (no rows) |

## musique_structured

| cell | n | judge% | Δjudge | p_judge | EM% | ΔEM | p_em | recall% | surfaced% | avg_tok/inst | avg_llm_calls | empty% | recov | n_mut |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bql+dense+snip fetch | 71 | — | — | — | 42.3 | — | — | 91.5 | 77.5 | 224k | 22.7 | 11.3 | 0 | — |
| SERP bm25 [BASELINE] | — | | | | | | | | | | | | | (no rows) |
| SERP bm25 k=10 | — | | | | | | | | | | | | | (no rows) |
| indri+dense+snip fetch | — | | | | | | | | | | | | | (no rows) |
| indri+dense visit | — | | | | | | | | | | | | | (no rows) |

## Legend

### Cells

- **SERP bm25 [BASELINE]**: The reference condition every other cell is compared against. The agent searches with plain keyword search (BM25, the classic ranked-keyword algorithm) over the document collection, gets back a listing of the top 5 results (title + a short preview snippet), then chooses results to 'visit' — open and read the entire document. No coaching/prompt hints beyond the tool descriptions themselves.
- **flat-twin bm25**: The exact same plain-keyword-search-then-visit-whole-document agent as the baseline above, but run against a different copy of the corpus where every document's internal structure (section headings, infobox fields, etc.) has been stripped out, leaving flat unstructured text. Comparing this cell to the baseline shows what having document structure is worth by itself, before any structure-aware search method is even applied.
- **SERP bm25 k=10**: Identical to the baseline, except the search listing shows 10 results per query instead of the baseline's 5. Tests whether simply showing the agent more candidates (with no change to search or reading method) changes accuracy.
- **bm25 auto-read**: Plain keyword search, but there is no separate 'open and read' step at all: each search call immediately dumps the full text of the top 5 results into the response. This removes the agent's choice of which result to open, isolating how much value that preview-then-pick step adds over just reading everything found.
- **dense visit**: Same shape as the baseline (search a ranked listing, then visit whole documents) but the search itself is swapped from keyword matching to dense (semantic/meaning-based) search: documents are ranked by how close their meaning is to the query, via an embedding model (BAAI/bge-base-en-v1.5), rather than by shared words. Useful when the right document uses different wording than the query.
- **indri visit**: Search uses a graded query language: instead of a document either matching or not matching (like plain keyword search), each document gets a score for how well it satisfies the query's operators (required/weighted/proximity terms, field and date filters), so a query never comes back with zero results — it always returns the closest matches. The listing shows a query-relevant snippet per result, and reading is whole-document 'visit' as in the baseline. Isolates the effect of the graded search language alone, holding the reading method (whole-document) fixed for a fair comparison to the baseline.
- **bql visit**: Search uses our Boolean field-tagged query language: the agent can require or exclude specific terms and filter by fields such as title, date, or section, similar to an advanced library search box. If a strict AND query would return zero hits, it automatically falls back to ranking documents by how many of the requirements each one still satisfies, rather than giving up. Reading is whole-document 'visit', matching the baseline's reading method so only the search language differs.
- **indri+dense visit**: Same graded query-language search as 'indri visit', but each candidate's score is re-scored by blending it with a meaning-based (dense/embedding) similarity score, and a few extra meaning-similar documents that the keyword-style query missed are pulled into the candidate pool too. This is a fusion INSIDE one already-retrieved candidate list, not a merge of two separately-ranked lists (contrast with 'hybrid rrf visit' below). Reading is whole-document visit, as in 'indri visit'.
- **hybrid rrf visit**: Runs plain keyword search and dense (meaning-based) search as two completely SEPARATE ranked lists, then blends them by combining each document's rank position in each list (reciprocal rank fusion) rather than by directly mixing their scores. Because it blends by rank rather than score, it works best when the two rankers tend to agree; when they disagree, the blend can be noisy. This is the control for isolating plain keyword+meaning fusion from any benefit of a genuinely structured query language. Reading is whole-document visit.
- **bql+dense visit**: Uses the SAME Boolean field-tagged filter as 'bql visit' to decide which documents are eligible at all — a document that fails the filter can never appear here no matter how similar its meaning is. Among the documents that pass the filter, only their ORDER changes: it is re-ranked by blending plain-keyword rank with meaning-based (dense) rank, the same rank-blending as 'hybrid rrf visit' but restricted to already-filtered candidates. Reading is whole-document visit.
- **hybrid+snip fetch**: Same keyword+meaning rank-fusion search as 'hybrid rrf visit', but each listing result now also shows a one-line excerpt — the single sentence from that document that best matches the query — and instead of opening the whole document, the agent pulls specific named sections of it ('fetch'), which is cheaper to read but requires picking the right section name.
- **bql+dense+snip fetch**: Same Boolean-field-tagged-filter-then-rank-blend search as 'bql+dense visit' (filter first, then re-rank the survivors by keyword+meaning), but each listing result shows a one-line best-matching excerpt, and reading pulls specific named sections instead of the whole document.
- **qwen dense visit**: Identical to 'dense visit', except the meaning-based search uses a stronger embedding model (Qwen3-Embedding-0.6B) instead of the default one (bge-base) — tests whether a better embedding model changes the result.
- **qwen hybrid visit**: Identical to 'hybrid rrf visit', except the meaning-based half of the fusion uses the stronger Qwen3-Embedding-0.6B embedding model instead of the default bge-base.
- **qwen bql+dense visit**: Identical to 'bql+dense visit', except the meaning-based re-ranking uses the stronger Qwen3-Embedding-0.6B embedding model instead of the default bge-base.
- **qwen indri+dense visit**: Identical to 'indri+dense visit', except the meaning-based score blended into the graded query-language search uses the stronger Qwen3-Embedding-0.6B embedding model instead of the default bge-base.
- **qwen indri+dense+snip fetch**: Identical to 'indri+dense+snip fetch' (below), except the meaning-based score blended in uses the stronger Qwen3-Embedding-0.6B embedding model instead of the default bge-base.
- **bql fetch**: Search uses our Boolean field-tagged query language (same as 'bql visit'): the agent can require or exclude specific terms and filter by fields such as title, date, or section, and if a strict query would return zero hits it automatically falls back to ranking documents by how many of the requirements each one still satisfies. The listing is plain (title + section outline, no per-result excerpt), and reading pulls specific named sections of a document ('fetch') instead of opening the whole thing. This is the direct Boolean-query analog of 'indri fetch' below, and the no-snippet / no-dense sibling of 'bql+snip fetch' below and 'bql+dense+snip fetch' above — included so the BQL read-interface family (visit / plain-fetch / snippet-fetch) is complete.
- **bql+snip fetch**: Search uses our Boolean field-tagged query language (same as 'bql visit'), and each listing result also shows a one-line best-matching excerpt of its actual content — added because without it, a correct document whose match is only in its title or section names (not visible content) could get skipped by the agent. Reading pulls specific named sections instead of the whole document.
- **indri fetch**: Graded query-language search (same scoring style as 'indri visit': every candidate scored for how well it satisfies the query, never zero results), with a plain listing (no per-result excerpt). Reading pulls specific named sections of a document instead of opening the whole thing.
- **indri+snip fetch**: Same graded query-language search as 'indri fetch', plus a one-line best-matching excerpt shown per listing result. Reading pulls specific named sections instead of the whole document.
- **dense fetch**: Meaning-based (dense/embedding) search, same as 'dense visit', but each listing result shows a one-line best-matching excerpt and reading pulls specific named sections instead of opening the whole document.
- **bm25+snip fetch**: Plain keyword search (BM25), but each listing result now shows a one-line excerpt chosen for relevance to the query (instead of just the document's fixed opening lines) — a fairness fix so this baseline's listing is as informative as the other excerpt-showing cells. Reading pulls specific named sections instead of opening the whole document.
- **indri+dense+snip fetch**: Same graded-query-language-plus-meaning-based-blend search as 'indri+dense visit' (candidate scores blended with dense similarity, extra meaning-similar candidates pulled in), with a one-line best-matching excerpt shown per listing result. Reading pulls specific named sections instead of opening the whole document.
- **dci**: No search tool at all. The agent gets a plain command line (bash) and a file-reader over the raw document collection's files, and has to grep/list/read its way to the answer directly — the brute-force, closed-book baseline with no ranking or retrieval help whatsoever.
- **bm25->dci**: Plain keyword search first narrows the collection down to its top results, and only THEN does the agent get a command line and file-reader restricted to just those narrowed-down files — a middle ground between the pure 'dci' brute-force baseline (no narrowing at all) and the visit/fetch cells (a guided listing, not raw file access).
- **one-shot bm25**: No agent loop and no tool calls at all: the top results from a single plain keyword search are pasted directly into one prompt, and the model produces one answer in a single call — the simplest possible retrieval-augmented setup, used to show what an agent loop adds over one-shot retrieval.
- **one-shot dense**: Same single-prompt, single-call setup as 'one-shot bm25', but the documents pasted into the prompt come from a meaning-based (dense/embedding, bge-base) search instead of plain keyword search.

### Columns

- **n**: Number of questions scored in this cell.
- **judge%**: Percent of answers a GPT-4o-mini judge model (following the BrowseComp-Plus grading protocol) rated as correct. An answer that is an exact string match to the gold answer is auto-accepted without spending a judge call.
- **Δjudge / p_judge**: The judge-accuracy gap versus this dataset's baseline cell, computed only on the questions both cells answered, plus how statistically significant that gap is (exact McNemar test — a small p-value means the gap is unlikely to be chance; a p-value near 1 means no real difference).
- **EM%**: Percent of answers that exactly match the gold answer string (strict exact-match scoring, no judge model involved).
- **ΔEM / p_em**: Same as Δjudge / p_judge, but using strict exact-match instead of the judge model's rating.
- **recall%**: Percent of questions where a document actually known to contain the answer (from the gold reference list) shows up somewhere the agent retrieved — in a search result listing, a section it fetched, or a file it opened — not just mentioned in passing.
- **surfaced%**: Percent of questions where the gold answer's exact text appeared anywhere in what the agent saw during the episode, regardless of whether it came from a known-correct source document. A weaker, looser signal than recall%.
- **avg_tok/inst**: Average total tokens (prompt + completion) processed per question, summed across every step of the episode — re-reading the same context in a later step counts again each time. This is the actual compute/billing cost, so it is normal for it to be much larger than one model context window; reported this way to match how prior agent-cost papers measure it.
- **avg_llm_calls**: Average number of separate model calls made per question.
- **empty%**: Percent of final answers that came back blank, whitespace-only, or a placeholder like "..." — measured AFTER an offline recovery pass has already tried to refill blanks by re-asking the model for a forced answer.
- **recov**: Number of rows in this cell whose blank/placeholder answer was successfully refilled by that offline recovery pass.
- **n_mut**: Number of questions this cell has in common with the dataset's baseline cell — the shared set the Δ / p-value columns are computed over.
