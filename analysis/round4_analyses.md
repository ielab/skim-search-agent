# Round-4 review analyses: C1, M2, M6

Three analyses requested against `docs/reviews/round4_full.md` (C1, M2, M6). All computed from
existing rows / existing corpus files — no new experiments, no GPU. Each section below is also
generated standalone by its own script (`analysis/flat_vs_structured.py`,
`analysis/section_quality.py`, `analysis/cost_dollars.py`); this file is the combined,
paper-ready version. Re-run commands are given per section so any of the three can be refreshed
independently as new runs land.

Generated 2026-07-22.

---

## 1. The flat-vs-structured comparison (C1)

**Reviewer's finding.** The paper builds the structured/flat twin specifically to isolate
structure as a causal axis ("byte-identical... attributable to the interface, not to a change in
the underlying data," `benchmark.tex` §3.1) and then never reports the one number that instrument
exists to produce. The row already existed in the project's comparison registry
(`flat-twin bm25`) but was never surfaced or significance-tested.

**Script.** `analysis/flat_vs_structured.py` — reuses `evaluation.metrics.answer_em`,
`scripts.force_answer_backfill.load_rows_with_recovery`, and
`scripts.compare_cells.{gold_doc_recall, mcnemar_p, load_qrels, load_judge_cache}` throughout.
It does not hardcode a dataset list: it walks `runs/` for every `*_flat` dataset dir, requires a
same-run-group/same-model/same-condition `*_structured` sibling with ≥100 rows on both sides, and
compares every such pair it finds — so re-running it later, once the in-flight `hotpotqa_flat`/
`musique_flat` runs have filled in, picks them up with no code change.

```
PYTHONPATH=. envs/bin/python analysis/flat_vs_structured.py --out analysis/flat_vs_structured.md
```

**Result (as of 2026-07-22 — the only pair with usable data today):**

| flat dataset | structured dataset | condition | run group | n_shared | flat EM% | struct EM% | ΔEM (pp) | McNemar p (EM) | flat judge% | struct judge% | Δjudge (pp) | McNemar p (judge) | n judged | flat recall% | struct recall% | flat tok/ep | struct tok/ep | flat calls/ep | struct calls/ep |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| browsecomp_plus_flat | browsecomp_plus_structured | `agent_research_bm25` | `runs/_visit_uncapped` | 830 | 34.8 | 36.4 | +1.6 | 0.372 | 37.3 | 39.3 | +1.9 | 0.263 | 830 | 58.7 | 58.9 | 64,586 | 63,805 | 63.2 | 62.5 |

**Critical interpretive point.** This condition (`agent_research_bm25`, BM25 keyword search +
whole-document visit) does **not** use the structured query surface — no fielded/Boolean search,
no section-level fetch — on either side. It is the same agent over byte-identical underlying
text, one side stripped of the `sections` field. Any delta here therefore isolates what the
**corpus/ranking alone** contributes, cleanly separated from what the method's **interface**
contributes (which is measured elsewhere in the paper, e.g. the ablation ladder).

**Paper-ready sentences (hedged):**

> On BrowseComp-Plus, for a BM25-search-and-visit agent that never queries the structured surface,
> adding structure to the corpus alone (structured vs. flat, byte-identical text otherwise) moves
> judge accuracy from 37.3% to 39.3% (+1.9 pp) and EM from 34.8% to 36.4% (+1.6 pp, n=830 paired).
> Neither delta reaches significance (exact McNemar p=0.263 judge, p=0.372 EM), and gold-document
> recall is essentially unchanged (58.7% flat vs. 58.9% structured), so this is best read as a
> small, non-significant, possibly-noise-level nudge from the corpus alone — **not** evidence that
> structure "helps because the corpus got richer," and not evidence it "helps because the
> interface exploits it" either, since the interface is unused here. Whatever the paper's real
> structured-vs-flat advantage turns out to be once the interface is used, it is not explained by
> this baseline-corpus effect, which is small and statistically indistinguishable from zero.

**Caveat.** n=1 condition-pair today; the wiki-twin extension this script is designed to pick up
automatically (`hotpotqa_flat`, `musique_flat`) had not produced ≥100-row cells as of this
writing. Re-run the script once those fill in; no code change needed.

---

## 2. Section-quality audit of the LLM-inserted structure (M2)

**Reviewer's finding.** The paper defends the BrowseComp-Plus-Structured sectioning only with
"100% non-empty, 94.9% multi-section, mean 14.7 sections/doc" (`benchmark.tex` §3.1) — necessary
but not sufficient conditions for the sections to be the semantically load-bearing units the
method's fetch mechanism assumes. No quality/coherence check is reported anywhere.

**Corpus format.** `data/browsecomp_plus_structured/corpus.jsonl` (67,707 docs, 3.7GB): each doc
carries `text` (the full document, with inline `## heading` markup) and `sections` (a parsed list
of `{heading, text}`, LLM-inferred). `sections.jsonl` in the same dir is a redundant `_id`-keyed
copy of the same field — not separately used.

**Script.** `analysis/section_quality.py` — reservoir-samples `n` docs (default 200, seed=42) in
one streaming pass (never loads the 67,707-doc / 3.7GB file whole), then computes purely automatic
signals: section-count and length (chars + o200k_base tokens, the same tokenizer ruler
`evaluation/run_eval.py`'s `_obs_token_count` uses elsewhere in this repo) distributions,
heading-body lexical consistency, degenerate-section rate, and boundary-partition sanity.

```
PYTHONPATH=. envs/bin/python analysis/section_quality.py --out analysis/section_quality.md
PYTHONPATH=. envs/bin/python analysis/section_quality.py --n 500 --seed 7   # different sample, sanity check
```

**Result (n=200, seed=42):**

| signal | value |
|---|---|
| sections/doc: mean / median / p10–p90 / min–max | 16.2 / 9.0 / 3.0–28.1 / 1–398 |
| docs with ≤1 section (not genuinely multi-section) | 2.0% |
| section length, chars: mean / median / p10–p90 | 1893 / 760 / 127–3527 |
| section length, tokens (o200k_base): mean / median / p10–p90 | 429 / 163 / 26–726 |
| heading-body lexical consistency (non-generic heading term found in body) | 87.1% (of 2990 testable headings; 45 more untestable — generic-only heading vocabulary) |
| near-empty sections (<15 stripped chars) | 1.9% of all sections |
| docs with a duplicate heading (same heading text 2+ times in one doc) | 3.5% of docs |
| boundary sanity: clean / large-gap / overlap / not-found | 100.0% / 0.0% / 0.0% / 0.0% |

**Method notes.** *Lexical consistency*: a non-`(intro)` section passes if any casefolded,
non-generic heading token (len≥3, stoplist excludes ~25 structural/connective words) appears as a
whole word in that section's own body; headings whose only content words are stoplisted are
reported as untestable, not as failures. *Boundary sanity*: each section's `text` is located as a
substring of the doc's own `text`, walking forward from the previous match end — `clean` means all
sections are found, in order, non-overlapping, with every inter-section gap ≤200 chars (small gaps
are *expected*: `text` interleaves a `## heading` markup line the section's own `text` omits).

**Hand-inspected examples** (from the same n=200 sample):

- **Good** — `_id=33075`, "World Heritage Site" (53 sections, boundary=clean, lexical 52/52): a
  cleanly chronological breakdown (`Triassic entries`, `Jurassic entries`, `Cretaceous entries`,
  …), each section's body topically matching its heading.
- **Good** — `_id=34504`, "Macrophages in immunoregulation and therapeutics" (51 sections,
  boundary=clean, lexical 50/50): a coherent biomedical-review structure (`Macrophage activation
  overview` → `Macrophage polarization` → `M1 macrophage stimulation` → …), each heading a genuine
  topical label for its body.
- **Worst-case** — `_id=77391`, "Class of 1994" (144 sections, boundary=clean, lexical 118/143,
  **86 duplicate-heading pairs**): the heading **"Back issue archive"** is reused verbatim 83
  times in one document, each instance with different body content ("From the November/December
  2017 Issue", "From the September/October 2017 Issue", …) — a heading naming a recurring
  newsletter *format*, not distinguishing the individual sections it labels. This is a genuine,
  concrete example of the degeneracy mode the paper's non-empty/multi-section counts cannot catch:
  every one of these 144 sections is non-empty and the doc is "multi-section" by the paper's own
  test, yet 83 of its headings carry no content-discriminating information at all.

**Paper-ready sentence (hedged):**

> Beyond the non-empty/multi-section counts already reported, an automatic audit of 200 randomly
> sampled BrowseComp-Plus-Structured documents finds the LLM-inserted sections are largely
> well-formed structurally — every sampled document's sections partition its own text cleanly,
> with no overlap or large gap — and 87% of testable section headings share at least one content
> word with their own section body (a necessary, not sufficient, proxy for topical coherence).
> This remains a shallow automatic signal, not a substitute for human coherence judgment, and the
> audit also surfaces a non-trivial degenerate tail the paper's existing counts do not see: 1.9%
> of sections are near-empty and 3.5% of documents contain a heading repeated across multiple,
> content-distinct sections (worst observed case: one heading reused 83 times in a single
> document, each instance non-empty and topically distinct from the others, i.e. a heading that
> names a recurring *format* rather than distinguishing its own section).

---

## 3. Aggregate-dollar cost accounting, full method vs. BM25 baseline (M6)

**Reviewer's finding.** The paper's own headline token-savings claim (30–51% fewer tokens) is
reported on a count-once basis, with step-summed as a secondary check (§7.5, Table 5) — but the
paper never converts either into a real-dollar accounting the way it holds DCI's cost claim to a
"basis matters" standard (`related_work.tex` §4), despite the method itself making more LLM calls
on two of three datasets (+31.6% HotpotQA, +9.9% MuSiQue).

**Assumed price** (parameterizable, **not** a verified invoice): **$0.20 / 1M input tokens,
$0.80 / 1M output tokens** — the same "30B-class hosted-serving proxy" rate `scripts/
cost_analysis.py` already uses elsewhere in this repo, reused here for consistency, not because
it is a real bill (this repo self-hosts Tongyi-DeepResearch-30B-A3B on vLLM, which has no
per-token invoice). Override with `--price-in-per-1m`/`--price-out-per-1m` for any other public
rate card.

**Bases**: *count-once* = `initial_prompt_tokens + context_once_tokens` (input) +
`output_tokens` — what the paper's headline number uses. *Step-summed* = `prompt_tokens`
(raw cumulative sum of every LLM call's input across the episode) + `completion_tokens` — what an
**uncached** backend actually bills, and the basis sensitive to the method's call count.

**Script.** `analysis/cost_dollars.py` — reuses `analysis.paper_analyses.CELLS` (method =
`runs/_headline_validation/.../agent_research_bql_dense_snip`, baseline =
`runs/_visit_uncapped/.../agent_research_bm25`, the same six cells `analysis/paper_analyses.py`'s
own Analysis 3 already reads) and streams every `rows.jsonl` line-by-line, keeping only the
numeric token/call fields per row (never the full row dict).

```
PYTHONPATH=. envs/bin/python analysis/cost_dollars.py --out analysis/cost_dollars.md
PYTHONPATH=. envs/bin/python analysis/cost_dollars.py --price-in-per-1m 3.0 --price-out-per-1m 15.0 \
    --out analysis/cost_dollars_sonnet_class.md   # any other public rate card
```

**Result ($0.20/$0.80 per 1M in/out):**

| Dataset | Cell | n | LLM calls/ep | $/query (count-once) | $/query (step-summed) | $ over full eval set (count-once) | $ over full eval set (step-summed) |
|---|---|---:|---:|---:|---:|---:|---:|
| browsecomp_plus_structured | baseline (bm25) | 830 | 62.5 | $0.0209 | $0.4772 | $17.34 | $396.05 |
| browsecomp_plus_structured | full method (bql+dense+snip) | 830 | 60.7 | $0.0186 | $0.3151 | $15.41 | $261.50 |
| browsecomp_plus_structured | **method vs baseline, %Δ** | | **-2.8%** | **-11.2%** | **-34.0%** | | |
| hotpotqa_structured | baseline (bm25) | 7343 | 15.9 | $0.0052 | $0.0602 | $38.51 | $442.33 |
| hotpotqa_structured | full method (bql+dense+snip) | 7343 | 20.9 | $0.0043 | $0.0463 | $31.70 | $339.99 |
| hotpotqa_structured | **method vs baseline, %Δ** | | **+31.6%** | **-17.7%** | **-23.1%** | | |
| musique_structured | baseline (bm25) | 2409 | 29.6 | $0.0111 | $0.1759 | $26.79 | $423.71 |
| musique_structured | full method (bql+dense+snip) | 2409 | 32.6 | $0.0066 | $0.0879 | $15.97 | $211.78 |
| musique_structured | **method vs baseline, %Δ** | | **+9.9%** | **-40.4%** | **-50.0%** | | |

**Does the method remain cheaper under a call-count-sensitive accounting? Yes, on all three
datasets, at the assumed rate.** Even step-summed — the basis that directly penalizes extra LLM
calls, since every call resends the whole growing context — the method is cheaper than the BM25
baseline by 23–50%, despite making 31.6% more calls on HotpotQA and 9.9% more on MuSiQue: the
per-call context it resends is short enough (bounded section-level fetches vs. whole-document
visits) that the extra calls do not flip the sign of the dollar delta.

**A subtlety worth stating alongside this.** The **count-once dollar** deltas above (-11.2%/
-17.7%/-40.4%) are noticeably *smaller in magnitude* than the paper's own count-once **raw token**
deltas (-29.6%/-30.4%/-50.9%, from `analysis/paper_analyses.py`'s Analysis 3, input+output tokens
combined unweighted). The reason: at a 4× output:input price ratio, the method's input tokens drop
sharply (structured fetch reads far less text than whole-document visit) but its **output** tokens
are somewhat *higher* than the baseline's on every dataset — a token-mix shift toward the more
expensive side that an unweighted token-count delta cannot see but a dollar accounting does. This
is exactly the basis-sensitivity the paper already applies to DCI's own cost claim; the same
standard should be applied to the paper's own count-once headline number.

**Paper-ready sentence (hedged):**

> Under an assumed $0.20/$0.80 per-1M-token rate (stated explicitly as an assumption, not a
> verified invoice) and a step-summed accounting that is sensitive to the method's LLM-call count
> — the standard this paper already applies to DCI's own cost claim — the full method remains
> 23–50% cheaper per query than the BM25 baseline on all three evaluation datasets, despite issuing
> 10–32% more LLM calls on two of them; the shorter per-call context the method resends (bounded
> section-level fetches vs. whole-document visits) more than offsets the extra calls. The
> count-once dollar savings are real but smaller than the paper's raw-token-count savings suggest
> (11–40% vs. 30–51%), because the method's output-token count is slightly higher than the
> baseline's on every dataset and output tokens are priced above input in every plausible rate
> card — a basis-sensitivity that should be disclosed with the same rigor this paper already
> applies to DCI's cost claim.

---

## Reproducing all three

```bash
PYTHONPATH=. envs/bin/python analysis/flat_vs_structured.py --out analysis/flat_vs_structured.md
PYTHONPATH=. envs/bin/python analysis/section_quality.py --out analysis/section_quality.md
PYTHONPATH=. envs/bin/python analysis/cost_dollars.py --out analysis/cost_dollars.md
```

Each script also writes a JSON sidecar (`analysis/flat_vs_structured_data.json`,
`analysis/section_quality_data.json`, `analysis/cost_dollars_data.json`) with the full numbers,
for the writer or any figure code later.
