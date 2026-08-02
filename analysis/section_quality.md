# Section-quality audit -- BrowseComp-Plus-Structured (M2)

Random sample: n=200 documents, seed=42, reservoir-sampled in one streaming pass over `${REPO_ROOT:-.}/data/browsecomp_plus_structured/corpus.jsonl` (67,707 docs total).

| signal | value |
|---|---|
| sections/doc: mean / median / p10-p90 / min-max | 16.2 / 9.0 / 3.0-28.1 / 1-398 |
| docs with <=1 section (not genuinely multi-section) | 2.0% |
| section length, chars: mean / median / p10-p90 | 1893 / 760 / 127-3527 |
| section length, tokens (o200k_base): mean / median / p10-p90 | 429 / 163 / 26-726 |
| heading-body lexical consistency (non-generic heading term found in body) | 87.1% (of 2990 testable non-intro section headings; 45 more had only generic/stoplisted heading words and are untestable) |
| near-empty sections (<15 stripped chars) | 1.9% of all sections |
| docs with a duplicate heading (same heading text twice+ in one doc) | 3.5% of docs |
| boundary sanity: clean / large-gap / overlap / not-found | 100.0% / 0.0% / 0.0% / 0.0% |

**Method notes.** Lexical consistency: a non-`(intro)` section "passes" if any casefolded heading token (len>=3, excluding a small generic-word stoplist -- see `_GENERIC_HEADING_WORDS` in the script) appears as a whole word in that section's own body text; headings whose only content words are stoplisted are reported separately as untestable, not counted as failures. Boundary sanity: each section's `text` is located as a substring of the doc's own `text` field, walking forward from the previous match end -- `clean` = all sections found, in order, with no overlap and no gap over 200 chars (small gaps are EXPECTED: the doc `text` interleaves a `## heading` markup line the section's own `text` field omits); `large_gap` = found in order, no overlap, but at least one inter-section gap exceeds 200 chars; `overlap` = a section's text starts before the previous section's text ended; `not_found` = a section's `text` does not occur verbatim in the doc's `text` field at all past the current position.

## Hand-inspected examples

**Good:** `_id=33075` "World Heritage Site" -- 53 sections, boundary=clean, lexical 52/52, dup_heading_pairs=0, near_empty=0
    - **(intro)**: "Timeline  This timeline was developed collaboratively by our community who assigned each WHS to a historical/geological "period".  All sites are shown in chrono..."
    - **Triassic entries**: "| Dolomites: Das Gebirge besteht zu grossen Teilen aus Sedimentgestein, das deutlich typische Schichtungen aufweist. Dazwischen findet man auch Lagen aus verste..."
    - **Jurassic entries**: "Jurassic | |  |---|---|  | Dorset and East Devon Coast: The property's geology displays approximately 185 million years of the Earth's history, including a numb..."
    - **Cretaceous entries**: "Cretaceous | |  |---|---|  | Dinosaur Provincial Park: The Dinosaur Park Formation, which contains most of the fossils from articulated skeletons, was primarily..."
    - ... (49 more sections)

**Good:** `_id=34504` "Macrophages in immunoregulation and therapeutics" -- 51 sections, boundary=clean, lexical 50/50, dup_heading_pairs=0, near_empty=0
    - **(intro)**: "Abstract  Macrophages exist in various tissues, several body cavities, and around mucosal surfaces and are a vital part of the innate immune system for host def..."
    - **Macrophage activation overview**: "Activation and polarization of macrophages  An essential function of macrophages is to sanitize the cellular fragments produced by tissue remodeling and apoptos..."
    - **Macrophage polarization**: "Macrophage polarization  The macrophages' destiny relies on various environmental conditions that fuel polarization to any of the classically triggered pro-infl..."
    - **M1 macrophage stimulation**: "Stimulation of M1  Microbial products or pro-inflammatory cytokines induce M1 polarized macrophage (M1). The critical Th1 cells-derived inflammatory mediator th..."
    - ... (47 more sections)

**Worst-case:** `_id=77391` "Class of 1994" -- 144 sections, boundary=clean, lexical 118/143, dup_heading_pairs=86, near_empty=0
    - heading **"Back issue archive"** repeats 83 times in this one document with DIFFERENT bodies each time (a heading naming a recurring newsletter/list FORMAT rather than distinguishing individual items):
        - "From the November/December 2017 Issue"
        - "From the September/October 2017 Issue"
        - "From the July/August 2017 Issue"
        - ... (80 more sections share this exact heading)

**Key sentence (paper-ready, hedged).** Beyond the paper's non-empty/multi-section counts, an automatic audit of 200 sampled documents finds the LLM-inserted sections are largely well-formed structurally (100% partition the document text cleanly with no overlap or large gap) and 87% of testable section headings share at least one content word with their own body text (a necessary, not sufficient, condition for topical coherence) -- but this remains a shallow, automatic proxy for section quality, not a substitute for human topical-coherence judgment, and 1.9% of sections are near-empty and 3.5% of documents contain a duplicated heading string, both of which point to a non-trivial tail of degenerate LLM-inferred structure the paper's own counts do not surface.
