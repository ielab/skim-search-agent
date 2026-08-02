# Pre-gate sanity profile — 2026-07-12

Scope: every cell in `scripts/compare_cells.py` `REGISTRY` (29 entries) + `ONESHOT` (2 entries) = 31
registered cells, plus 3 `agent_research_dci` cells found on disk for `2wiki_structured`,
`hotpotqa_structured`, `musique_structured` that are **not** in `REGISTRY` (`runs/agent/<dataset>/…`)
but exist as real `rows.jsonl` — included here since the task asked for "all datasets, including
wiki." Total: **34 cells**, all with `rows.jsonl` present (0 registry entries missing on disk).

Method notes:
- **rows.jsonl parsing**: split on `\n`, parse each non-blank line; a parse failure on the final
  line is counted as a "torn last line" (tolerated/skipped, not a failure); a parse failure on any
  other line is a real integrity failure. Snapshot taken 2026-07-12; per the task brief, a few
  `rows.jsonl` may still be growing (confirmed live: `dci [browsecomp_plus_structured]` at 158/200
  rows, several `_headline_validation`/`_fullvisit`/`_visit_uncapped` cells at 189–199/200).
- **Expected id universe**: for each dataset, read `data/<dataset>/queries.jsonl` in file order and
  take the first 200 `_id` values as the validation-tier universe (derivation verified directly:
  `runs/_visit_uncapped/.../browsecomp_plus_structured/.../agent_research_bm25/rows.jsonl`'s 198
  instance ids are a strict subset of the first 200 `queries.jsonl` ids with 0 ids outside). `n_steps`
  is used for `llm_calls` where `llm_calls` is absent; `stopped` field carries the
  `answer` vs `max_steps` termination reason directly (no fallback needed — present on all
  non-oneshot rows).
- **Degenerate-answer check**: share of rows whose *non-empty* `final_answer` string exactly matches
  the most common non-empty answer in the cell.
- Full per-cell computed metrics are in the sibling `analysis/gate_sanity_20260712.json` (not
  hand-verified line-by-line beyond the spot checks described below; treat as the audit trail).

---

## 1. Integrity — result: clean

Across all 34 cells, **zero** JSON parse failures (torn or otherwise), **zero** duplicate
`instance_id`s within any cell, and **zero** ids outside the expected first-200 universe for any of
the 31 *registered* cells (verified against each cell's own dataset's `queries.jsonl`). No row is
missing `final_answer`, `observations` (except the one-shot cells, which never carry an
`observations` field by design — they use `retrieved_ids` instead — this is a schema difference, not
data loss), or `gold_answer`.

The 3 unregistered `agent_research_dci` cells (2wiki/hotpotqa/musique) show large "outside universe"
counts (7581/2508/585 ids) — this is expected, not corruption: these are **full-collection** runs
(`limit: null` in config, 785–7781 rows, started 2026-07-10) sitting in the same registry-style path
convention as a would-be validation-tier cell, not n=200 subsets. See BLOCKERS.

## 2. Distribution profile

Full mean/median/p95 token and llm_calls numbers, n, `max_steps`%, pre-overlay empty%, recovered
count, and judge coverage per cell are in `analysis/gate_sanity_20260712.json`. Headline patterns:

- Token totals (`prompt_tokens+completion_tokens`, cumulative across the trajectory) range from
  ~130k (one-shot dense on browsecomp) to multi-million cumulative sums on 100-step agent runs —
  spot-checked (`musique_structured` SERP-bm25 baseline, highest-token row
  `musique_structured__2hop__488811_30078`, 10.2M tokens, `n_steps=101`, `stopped=max_steps`,
  empty final_answer) confirms the mega-token rows are legitimate cumulative-context runs that
  exhausted the step budget, not a units bug (lowest-token rows in the same cell resolve in 3 steps
  at ~4.2k tokens).
- `stopped=="max_steps"` rates range ~6%–52% across cells; `dci [browsecomp_plus_structured]` is the
  outlier at 52% (see Anomalies).
- Pre-overlay empty-`final_answer`% ranges 1.5%–53.2%; several `_headline_validation` "fetch" style
  cells and the `dci` cell run 40–53% empty, well above the sibling "visit" cells' 1.5–15% typical
  range for the same runs_subdir (`bql visit`, `bql+dense visit`).
- `recovered_answers.jsonl` present for 19/34 cells sampled here (mostly `_visit_uncapped`,
  `_headline_validation`, and the `dci`/`dci-unregistered` cells); recovered counts run 22–98% of
  scale reasonable for a backfill sidecar (13–26% of rows recovered where present, 98/785=12.5% max).
- `judge_cache.jsonl` present and 94–100%-covering for all 23 `browsecomp_plus_structured` +
  `browsecomp_plus_flat` cells (registry + oneshot). **`musique_structured` and `hotpotqa_structured`
  registered cells (6 cells total: baseline + 2 conditions × 2 datasets) have no `judge_cache.jsonl`
  at all** — not yet judged. See BLOCKERS.

## 3. Anomaly flags

**`dci [browsecomp_plus_structured]`** (`runs/agent/browsecomp_plus_structured/.../agent_research_dci`,
live, 158/200 rows) is the most anomalous cell in the registry:
- 53.2% empty final_answer vs its direct sibling `bm25->dci` (same runs_subdir, same dataset) at 9.6%.
- 51.9% `stopped=="max_steps"` vs `bm25->dci`'s 37.4%.
- Mean cumulative tokens 5.61M vs `bm25->dci`'s 2.88M — roughly double, and per-call token draw
  (mean tok / mean llm_calls) ~71k/call vs `bm25->dci`'s ~42k/call.
- This cell is incomplete/live (guardrail note: jobs still appending) — the 158-row snapshot may not
  represent the finished distribution, but the empty-rate / max-steps-rate gap from its sibling is
  already large enough at n=158 to be worth a hand-read before gating, not just a "wait for it to
  finish" deferral.

**BQL-family conditions** (`bql visit`, `bql+dense visit`, `qwen bql+dense visit` — 3 of the 3 BQL
visit cells that exist) show a reproducible bimodal token/step pattern distinct from their
`_fullvisit`/`_qwen_fullvisit` siblings: median cumulative tokens ~410k–490k vs mean ~2.1M–2.3M (mean
4–5× median), and `llm_calls` mean ~48–49 vs ~59–73 for `dense`/`indri`/`hybrid` siblings in the same
runs_subdir. Because the pattern reproduces identically across all 3 independent BQL cells (not a
one-off), this reads as a structural property of BQL's early-exit/prefilter behavior (resolves most
queries fast, long tail of hard ones) rather than corruption — flagged as a NOTE for a hand-read spot
check, not a blocker.

**`dense fetch [browsecomp_plus_structured]`** (`_headline_validation`) has the highest empty-rate
(47.7%) of the 7 `_headline_validation` siblings (range 7.0%–47.7%); worth a spot check alongside
`dci`'s 53.2% and `bm25+snip fetch`'s 42.5% before treating "fetch"-style empty rates as expected
baseline noise vs a real regression.

**No cell** exceeds the 20% degenerate-identical-answer threshold — highest observed is 12.5%
(`indri+dense+snip fetch [musique_structured]`), well within normal range; no evidence of
degenerate/templated answering anywhere.

**Config/env_knobs consistency** — checked every `config.json`'s `env_knobs` against the naming-convention rules:
- All `_qwen_*` dirs (`_qwen_fullvisit`, `_qwen_fullvisit_dense`, `_qwen_dense_validation`) correctly
  carry `DENSE_MODEL=Qwen/Qwen3-Embedding-0.6B`. No violations.
- All `_dense_validation` / `_fullvisit_dense` dirs correctly carry `INDRI_DENSE=true`. No violations.
- `BM25_BACKEND=pyserini` on every cell that has `env_knobs` at all (32/34 — see below). No
  non-pyserini backend anywhere.
- `STRUCTURED_BACKEND=lucene` on every structured-engine cell outside `_visit_uncapped`. The 4
  `_visit_uncapped` cells split: `browsecomp_plus_structured` and `browsecomp_plus_flat` baselines
  record `STRUCTURED_BACKEND=python` (explicitly exempted/legitimate per spec), while the
  `musique_structured` and `hotpotqa_structured` `_visit_uncapped` baselines record
  `STRUCTURED_BACKEND=lucene` instead. Both values are permitted for `_visit_uncapped` per the given
  rule (only non-`_visit_uncapped` non-lucene is a violation), so **not flagged as a violation**, but
  noting the within-runs_subdir inconsistency across datasets in case it wasn't intentional.
- **3 cells have no `env_knobs` block in `config.json` at all**: the unregistered
  `agent_research_dci` cells for `2wiki_structured`, `hotpotqa_structured`, `musique_structured`
  (all `started_at: 2026-07-10T18:0x`). Every other config on disk with `started_at` on/after
  2026-07-11 has a populated `env_knobs` block, including the sibling
  `browsecomp_plus_structured` `agent_research_dci`/`agent_research_bm25_dci` cells started
  2026-07-12 — so this reads as "these 3 runs predate env_knobs logging being added to the harness,"
  not a per-run data problem. But it means **backend (pyserini/lucene vs. anything else) cannot be
  verified from config for these 3 runs** — see BLOCKERS.

---

## Cell table

| cell | n | integrity | anomalies |
|---|---:|---|---|
| SERP bm25 [BASELINE] [browsecomp_plus_structured] | 198 | OK | incomplete 198/200 (live); empty 40% |
| flat-twin bm25 [browsecomp_plus_flat] | 189 | OK | incomplete 189/200 (live); empty 43% |
| dense visit [browsecomp_plus_structured] | 198 | OK | incomplete 198/200 (live); empty 42% |
| indri visit [browsecomp_plus_structured] | 200 | OK | none |
| bql visit [browsecomp_plus_structured] | 200 | OK | bimodal token dist. (see NOTES) |
| indri+dense visit [browsecomp_plus_structured] | 200 | OK | none |
| hybrid rrf visit [browsecomp_plus_structured] | 198 | OK | incomplete 198/200 (live) |
| bql+dense visit [browsecomp_plus_structured] | 200 | OK | bimodal token dist. (see NOTES) |
| hybrid+snip fetch [browsecomp_plus_structured] | 200 | OK | none |
| bql+dense+snip fetch [browsecomp_plus_structured] | 199 | OK | incomplete 199/200 (live) |
| qwen dense visit [browsecomp_plus_structured] | 199 | OK | incomplete 199/200 (live) |
| qwen hybrid visit [browsecomp_plus_structured] | 200 | OK | none |
| qwen bql+dense visit [browsecomp_plus_structured] | 200 | OK | bimodal token dist. (see NOTES) |
| qwen indri+dense visit [browsecomp_plus_structured] | 200 | OK | none |
| qwen indri+dense+snip fetch [browsecomp_plus_structured] | 200 | OK | none |
| bql+snip fetch [browsecomp_plus_structured] | 199 | OK | incomplete 199/200 (live) |
| indri fetch [browsecomp_plus_structured] | 200 | OK | none |
| indri+snip fetch [browsecomp_plus_structured] | 200 | OK | none |
| dense fetch [browsecomp_plus_structured] | 199 | OK | incomplete 199/200 (live); empty 48% (highest of _headline_validation siblings) |
| bm25+snip fetch [browsecomp_plus_structured] | 200 | OK | empty 42% |
| indri+dense+snip fetch [browsecomp_plus_structured] | 200 | OK | none |
| dci [browsecomp_plus_structured] | 158 | OK | incomplete 158/200 (live); empty 53%, max_steps 52% — both ~2× sibling bm25->dci |
| bm25->dci [browsecomp_plus_structured] | 198 | OK | incomplete 198/200 (live) |
| SERP bm25 [BASELINE] [musique_structured] | 200 | OK | no judge_cache.jsonl yet |
| indri+dense+snip fetch [musique_structured] | 200 | OK | no judge_cache.jsonl yet |
| indri+dense visit [musique_structured] | 200 | OK | no judge_cache.jsonl yet |
| SERP bm25 [BASELINE] [hotpotqa_structured] | 200 | OK | no judge_cache.jsonl yet |
| indri+dense+snip fetch [hotpotqa_structured] | 200 | OK | no judge_cache.jsonl yet |
| indri+dense visit [hotpotqa_structured] | 200 | OK | no judge_cache.jsonl yet |
| one-shot bm25 [browsecomp_plus_structured/oneshot] | 200 | OK | none (no `observations` field by design) |
| one-shot dense [browsecomp_plus_structured/oneshot] | 200 | OK | none (no `observations` field by design) |
| dci (unregistered) [2wiki_structured] | 7781 | OK | UNREGISTERED full-collection run, not n=200 tier; no env_knobs (pre-dates capture) |
| dci (unregistered) [hotpotqa_structured] | 2708 | OK | UNREGISTERED full-collection run, not n=200 tier; no env_knobs (pre-dates capture) |
| dci (unregistered) [musique_structured] | 785 | OK | UNREGISTERED full-collection run, not n=200 tier; no env_knobs (pre-dates capture) |

---

## BLOCKERS

1. **`musique_structured` and `hotpotqa_structured` registered cells are unjudged.** All 6 cells
   (`SERP bm25 [BASELINE]`, `indri+dense+snip fetch`, `indri+dense visit` × 2 datasets) have no
   `judge_cache.jsonl`. Per practices (`02-experiment-integrity.md`: "Gate it (judge + significance
   + sanity review)"), these cannot be gated on judge-based metrics until judging is run — EM-only
   comparison is available but incomplete for a gate decision.
2. **Orphaned full-collection `agent_research_dci` runs for `2wiki_structured`,
   `hotpotqa_structured`, `musique_structured`** sit at `runs/agent/<dataset>/Tongyi-DeepResearch-30B-A3B/agent_research_dci/`
   — the exact path convention a validation-tier registry cell for these datasets would use. They
   are full-collection runs (7781/2708/785 rows, `limit: null`), started 2026-07-10, **predate
   env_knobs logging** so their BM25/structured backend cannot be verified from config (unlike every
   other cell on disk, including the browsecomp `dci`/`bm25_dci` siblings run today which do confirm
   pyserini+lucene). Per the "Suite submission logic" and "Additive backends" practices (resume is
   keyed on run-dir path only), if `dci` for these 3 datasets is ever added to `REGISTRY` pointing at
   this same path, resume logic will collide with this pre-existing, unverified, non-n=200 data.
   **Action before gating/registering wiki or musique/hotpotqa dci: `mv` these 3 dirs aside (or
   re-verify+relabel) before any new dci validation-tier run starts writing there.**
3. **`dci [browsecomp_plus_structured]` looks degenerate relative to its sibling and is still
   live** (53% empty answers, 52% max_steps exhaustion, both ~2× `bm25->dci` at n=158/200). Recommend
   holding this cell out of any gate table until (a) the run completes and (b) a hand-read of a few
   empty-answer episodes confirms whether this is a real quality problem with the `dci` condition or
   an artifact of the run still being in progress.

## NOTES

- **BQL-family bimodal token distribution** (`bql visit`, `bql+dense visit`, `qwen bql+dense visit`):
  median cumulative tokens 4–5× lower than mean, `llm_calls` mean ~48–49 vs ~59–73 for
  dense/indri/hybrid siblings in the same runs_subdir. Reproduces identically across all 3
  independent BQL cells — most likely a real property of BQL's early-exit behavior, not corruption.
  Worth one hand-read episode to confirm before citing BQL step-efficiency as a finding.
- **`dense fetch [browsecomp_plus_structured]`** has the highest empty-rate (47.7%) of the 7
  `_headline_validation` siblings (range 7.0%–47.7%). Not flagged as broken, but the widest outlier
  in that family — spot-check recommended.
- **9 cells are still short of n=200** (188–199 rows: baseline/flat-twin/dense-visit/hybrid-visit/
  several `_headline_validation` fetch cells and both `dci` cells) — consistent with the stated
  "jobs still appending" caveat; re-run this profile once they finish before finalizing gate numbers.
- **`_visit_uncapped` `STRUCTURED_BACKEND` diverges by dataset**: `python` for
  `browsecomp_plus_structured`/`browsecomp_plus_flat`, `lucene` for `musique_structured`/
  `hotpotqa_structured`. Both are permitted values per the exemption rule for `_visit_uncapped`, but
  flagging the inconsistency in case it wasn't an intentional per-dataset design choice.
- **Extreme cumulative-token cells are legitimate, not a units bug.** Spot-checked the highest-token
  row in `musique_structured` SERP-bm25 baseline (10.24M cumulative tokens, `n_steps=101`,
  `stopped=max_steps`, empty final_answer) against the lowest-token row in the same cell (4.2k
  tokens, `n_steps=3`, answered) — confirms cumulative-context accounting scales as expected with
  step count, no encoding/units anomaly.

---

Output files: `analysis/gate_sanity_20260712.md` (this report), `analysis/gate_sanity_20260712.json`
(full per-cell computed metrics — the audit trail backing every number above).
