# Budget-mismatch blast radius: the hardcoded `TIER = "_headline_validation"` constant

**Date:** 2026-07-26. Read-only over `runs/`, `latex/`, `latex_acl8/`. No existing artifact was modified.

## 0. Mandatory sanity gate — PASS

Before any number below was trusted, the published, unaffected contrast was reproduced from the
repo's own primitives (`analysis.structured_surface_control.run_sanity_check`, i.e.
`scripts.compare_cells.metrics`/`mcnemar_p`/`cell_rows` → `evaluation.metrics.answer_em` →
`scripts.force_answer_backfill.load_rows_with_recovery`):

| check | expected | reproduced | result |
|---|---|---|---|
| Sieve vs "No dense evidence" (`agent_research_snip`), `browsecomp_plus_structured` | n=830, EM 38.9 vs 34.2, Δ=+4.7, p=0.0104 | n=830, EM 38.916 vs 34.217, Δ=+4.699, b=91, c=130, p=0.010421 | **PASS** |

Everything below is computed with the identical stack. EM, McNemar and the recovery overlay were
reused, never reimplemented.

## 1. The bug, and the exact rule that decides who it bites

`analysis/structured_surface_control.py:47` defines `TIER = "_headline_validation"`. Several scripts
import or copy it and apply it to all three datasets. On `browsecomp_plus_structured` this is
harmless forever. On the two Wikipedia datasets a budget-repair rerun created a **second Sieve arm**,
so the tier name no longer uniquely identifies "the Sieve cell":

| dataset | cell | configured cap (per-row `max_steps` stamp) | n |
|---|---|---:|---:|
| hotpotqa_structured | `_headline_validation/…/agent_research_bql_dense_snip` | **50** | 7343 |
| hotpotqa_structured | `_budget100/…/agent_research_bql_dense_snip` | **100** | 7343 |
| musique_structured | `_headline_validation/…/agent_research_bql_dense_snip` | **50** | 2409 |
| musique_structured | `_budget100/…/agent_research_bql_dense_snip` | **100** | 2409 |
| browsecomp_plus_structured | `_headline_validation/…/agent_research_bql_dense_snip` | 100 | 830 |

Cap audit re-verified independently for this report from the CONFIGURATION stamp (`max_steps` on
each row), never from observed step counts. Every cell is internally uniform (single-valued cap
distribution); no mixed cells among these:

| dataset | `hybrid_fetch_snip` | `snip` | `bm25_fetch_snip` | `dense_fetch` |
|---|---:|---:|---:|---:|
| browsecomp_plus_structured | 100 | 100 | 100 | 100 |
| hotpotqa_structured | 100 | 100 | 100 | 100 |
| musique_structured | 100 | 100 | 100 | 100 |

**The rule.** A comparison is mismatched **iff** it pairs the `_headline_validation` Sieve arm
against one of the **four Wikipedia controls that ran at cap 100** — `hybrid_fetch_snip`, `snip`
(no dense evidence), `bm25_fetch_snip` (sparse only), `dense_fetch` (dense only) — on
`hotpotqa_structured` or `musique_structured`. Pairing the cap-50 Sieve arm against a **cap-50**
comparator (the BM25-search-and-visit baseline on HotpotQA, `bql_dense_fetch` "No snippets",
`dense_fetch_plain` "Dense only, no snippets") is legitimately matched and is *not* affected by this
bug. BrowseComp-Plus is never affected.

## 2. Per-script table: which reported numbers are mismatched

### 2a. The four scripts named in the brief

| script | reported quantity | datasets affected | status |
|---|---|---|---|
| `analysis/structured_surface_control.py` | EM Δ +3.3 / p=0.102; judge Δ +2.4 / p=0.248; tokens 44,910 vs 44,448; recall 61.2 vs 71.1 | browsecomp only (script is single-dataset) | **MATCHED** — never at risk |
| `analysis/structured_surface_control_3way.py` | EM Δ **+5.7 / p=1.62e-27** (HotpotQA), **+5.5 / p=1.58e-11** (MuSiQue); b/c 549/971 and 129/262; EM Sieve **29.2** (MuSiQue); tokens **13,873** / **21,268** vs 18,152 / 31,209 | hotpotqa, musique | **MISMATCHED** |
| ″ | same quantities, browsecomp row (+3.3 / 0.102 / judge +2.4 / 0.248 / 61.2 vs 71.1) | browsecomp | **MATCHED** |
| ″ | recall 94.8 vs 96.4, 95.4 vs 96.4 (percentages only, no test) | hotpotqa, musique | **MISMATCHED source, IDENTICAL value** — Sieve recall is 94.8 / 95.4 at *both* caps (`matched_cap100_results.md` §5: Δrecall 0.0, n.s. on both) |
| `analysis/hybrid_control_ci.py` | 95% CI **[+4.70, +6.75]** (HotpotQA), **[+3.94, +7.14]** (MuSiQue); point estimates +5.75 / +5.52; its own sanity gate targets (+5.7 / +5.5) | hotpotqa, musique | **MISMATCHED** |
| ″ | 95% CI [−0.48, +6.99] and judge CI [−1.45, +6.27], browsecomp | browsecomp | **MATCHED** |
| ″ | overlap verdict ("both Wikipedia point estimates fall inside the BrowseComp CI") | derived | **MISMATCHED INPUT, CONCLUSION UNCHANGED** — verified below at matched values |
| `analysis/recall_inversion_test.py` | recall McNemar **p=3.52e-08** (HotpotQA), **p=0.0115** (MuSiQue); b/c 278/162 and 58/33 | hotpotqa, musique | **MISMATCHED** (already diagnosed) |
| ″ | recall p=8.28e-07, b/c 178/96, browsecomp | browsecomp | **MATCHED** |
| `analysis/recall_pvalue_reconciliation.py` | the four-way table (2 datasets × 2 Sieve arms) | both | **NOT A DEFECT** — it computes both arms deliberately and labels the cap-50 rows "budget matched: **NO**"; it is the diagnosis, not a victim |

### 2b. Additional affected scripts found during the sweep (not in the brief)

Same defect, same rule. None of these feed a mismatched number into `latex/` (see §3), but the
artifacts on disk are stale:

| script | affected contrasts | datasets |
|---|---|---|
| `analysis/ablation_deltas.py` | `FULL = ("_headline_validation", …)` vs **"Sparse same interface" (`bm25_fetch_snip`, cap 100)** — 1 row per Wikipedia dataset | hotpotqa, musique |
| ″ | its other Wikipedia rows — "No snippets" (`bql_dense_fetch`) and "Dense no snippets" (`dense_fetch_plain`) — are cap **50** on both wiki sets, so those rows are **MATCHED** | — |
| `analysis/wiki_isolation.py` | `SUBDIR = "_headline_validation"` with `full` vs `no_dense` / `sparse_same_iface` / `dense_same_iface`, all cap 100 → **6 mismatched contrasts** | hotpotqa, musique |
| `analysis/no_dense_evidence_3way.py` | `SUBDIR = "_headline_validation"`, Sieve vs `agent_research_snip` (cap 100) → **2 mismatched contrasts** | hotpotqa, musique |
| `analysis/multiplicity_census.py` | registry hardcodes `bql_dense_snip → _headline_validation`; any wiki row it pairs against `snip` / `bm25_snip` / `hybrid_snip` / `dense_snip` inherits the mismatch | hotpotqa, musique |

**Confirmed NOT affected** (checked, not assumed): `structure_fusion_interaction.py` and
`read_axis_grid.py` (both `DATASET = "browsecomp_plus_structured"`, single-dataset);
`matched_cap100_results.py` (correctly uses `SIEVE100 = ("_budget100", …)`);
`step_budget_audit.py` and `matched_budget_reanalysis.py` (deliberately read the cap-50 arm as the
"published/confounded basis" and say so). `paired_ttests.py`, `token_efficiency.py`,
`cost_dollars.py`, `equivalence_and_latency.py`, `method_vs_dci.py`, `paper_analyses.py` pair Sieve
against the **BM25-search-and-visit baseline**, which is cap 50 on HotpotQA (matched) and *mixed* on
MuSiQue — a separate, already-disclosed caveat, not this bug.

## 3. THE CRITICAL QUESTION: is any mismatched number still in the paper?

### 3a. `latex/` (the 32-page version) — **CLEAN. Verified, not assumed.**

Every affected quantity was traced to its `latex/` occurrence and checked against
`analysis/matched_cap100_results.md`. The writer round folded in the matched recomputation
**completely**:

| quantity | mismatched artifact value | `latex/` prints | file:line | verdict |
|---|---|---|---|---|
| Hybrid contrast, HotpotQA | +5.70, p=1.62e-27 | **+5.84, p=2.8×10⁻²⁹** | `sections/results.tex:405` | matched |
| Hybrid contrast, MuSiQue | +5.50, p=1.58e-11, Sieve EM 29.2 | **+4.57, p=1.4×10⁻⁸, Sieve EM 28.2** | `sections/results.tex:405–406` | matched |
| Hybrid contrast, abstract/intro | 5.7 and 5.5 | **5.8 and 4.6, p=2.8×10⁻²⁹ / 1.4×10⁻⁸** | `sections/introduction.tex:90` | matched |
| Sparse-only contrast, wiki | +4.00 / 1.19e-14, +5.70 / 1.37e-12 | **4.14 / 7.3×10⁻¹⁶, 4.77 / 2.7×10⁻⁹** | `sections/results.tex:407–408`; `sections/introduction.tex:84` (4.1, 4.8); `tables/ablation_family.tex` (7.29e-16, 2.70e-9) | matched |
| Dense-only contrast, wiki | +5.82 / 1.73e-28, +6.77 / 1.33e-16 | **5.91 / 4.5×10⁻²⁹, 5.81 / 1.1×10⁻¹²** | `sections/results.tex:409` | matched |
| No-dense rung, wiki | 0.103, 0.0949 | **0.0641, 0.694** | `tables/ablation_family.tex` | matched |
| Token reduction vs hybrid, wiki | +23.6% / +31.9%, 13.9k / 21.3k | **18.2% / 23.6%, 14.8k vs 18.2k, 23.8k vs 31.2k, p=2.6×10⁻⁷⁶ / 5.9×10⁻⁸¹** | `sections/results.tex:428–431`; `sections/conclusion.tex:75` ("18–24%") | matched |
| Token reduction vs sparse/dense, wiki | +22.4/31.4%, +25.3/30.5% | **17.0%/23.1% and 20.0%/22.1%** | `sections/results.tex:431` | matched |
| **Recall inversion p, HotpotQA** | **3.52e-08** | **−1.62, p=2.5×10⁻⁸** | `sections/results.tex:449` | matched |
| **Recall inversion p, MuSiQue** | **0.0115** | **−1.00, p=0.0149** | `sections/results.tex:450` | matched |
| Recall, sparse-only / dense-only / no-dense, wiki | — | 0.0017, 0.020, 0.121 & 0.644 | `sections/results.tex:451–454` | matched (`matched_cap100_results.md` §4) |
| Bootstrap CI, browsecomp | [−0.48, +6.99] | **[−0.5, +7.0] pp, 10,000 resamples** | `sections/results.tex:422` | unaffected (cap 100 both sides) |
| Bootstrap CIs, HotpotQA / MuSiQue | [+4.70,+6.75], [+3.94,+7.14] | **not quoted anywhere** | — | not exposed |
| Judge CI, browsecomp | [−1.45, +6.27] | **not quoted anywhere** | — | not exposed |

Three further points that make the `latex/` result stronger than "no mismatched digits survived":

1. **`latex/sections/results.tex:403–404` states the rule explicitly** — *"every contrast here read
   against the cap-100 Sieve arm so that both sides run at the controls' own step budget of 100"* —
   and `:447–448` repeats it for recall (*"at the matched cap-100 arm"*).
2. **`latex/sections/results.tex:202–211`** is a dedicated paragraph, *"Two step budgets for
   \textsc{Sieve}, and which contrast reads which"*, that discloses both arms, reports they are
   statistically indistinguishable (+0.10, p=0.861; −0.95, p=0.219), and states the resolution rule:
   tables and baseline comparisons from the cap-50 arm, the four cap-100 Wikipedia controls from the
   cap-100 arm. `latex/sections/appendix.tex:29–86` gives the per-cell cap table
   (`tab:step-caps`) with a `†` footnote on the Sieve row.
3. **`latex/tables/ablation_family.tex`'s caption** names the four affected rows and says they are
   computed against the cap-100 arm.

The one *derived* claim worth checking rather than assuming is
`latex/sections/results.tex:422–424`: the browsecomp CI *"contains both Wikipedia point estimates"*.
That sentence came from `hybrid_control_ci.py`'s overlap check, which ran on the **mismatched**
points. Recomputed here at the matched points the paper actually quotes: **+5.84 ∈ [−0.48, +6.99] →
true; +4.57 ∈ [−0.48, +6.99] → true.** The claim holds. (It is also now *more* comfortably true on
MuSiQue, the matched point being smaller.)

The `latex/tables/wiki_results.tex` **Sieve** rows (HotpotQA 45.3 / 94.8 / 14k / 20.9; MuSiQue 29.2 /
95.4 / 21k / 32.6) are the **cap-50 arm** — deliberately, per the rule at `results.tex:207–209`, and
flagged with `†` in `tab:step-caps`. Their ΔEM column is against the cap-50 (HotpotQA) / mixed
(MuSiQue) baseline, which the appendix discloses at `sections/appendix.tex:41–49`. This is disclosed
tabulation, not a product of the four at-risk scripts, and not an exposure.

### 3b. `latex_acl8/` (the 8-page ACL body) — **NOT CLEAN. This is the live exposure.**

The matched-cap repair landed in `latex/` only. `latex_acl8/` was last touched **2026-07-25 09:41–09:57**;
`latex/` was last touched **2026-07-26 01:36–01:45**. `latex_acl8/` is one writer round behind and
still prints the budget-mismatched artifact values, with **no step-cap disclosure anywhere** (no
`tab:step-caps`, no "two step budgets" paragraph, no "matched cap-100 arm" language — grep-verified).

| # | file:line | prints (MISMATCHED) | should print (MATCHED) |
|---|---|---|---|
| 1 | `latex_acl8/sections/results.tex:96` | HotpotQA `$45.3\%$ … $39.5\%$ ($+5.7$, $p{=}1.6\times10^{-27}$)` | `$+5.84$, $p{=}2.8\times10^{-29}$` |
| 2 | `latex_acl8/sections/results.tex:96–97` | MuSiQue `$29.2\%$ … $23.7\%$ ($+5.5$, $p{=}1.6\times10^{-11}$)` | `$28.2\%$ … $23.7\%$ ($+4.57$, $p{=}1.4\times10^{-8}$)` |
| 3 | `latex_acl8/sections/results.tex:97` | `both on $24$--$32\%$ fewer tokens` | `both on $18$--$24\%$ fewer tokens` |
| 4 | `latex_acl8/sections/results.tex:102–103` | `gains $6.0$/$4.0$/$5.7$ EM … ($p{=}0.0025$; $1.2\times10^{-14}$; $1.4\times10^{-12}$)` | `gains $6.0$/$4.1$/$4.8$ EM … ($p{=}0.0025$; $7.3\times10^{-16}$; $2.7\times10^{-9}$)` |
| 5 | `latex_acl8/main.tex:78–79` (abstract/intro) | `the structured query surface gains $5.7$ and $5.5$ exact-match points` | `gains $5.8$ and $4.6$ exact-match points` |
| 6 | `latex_acl8/main.tex:80` | `against plain BM25 at that same interface it gains $6.0$/$4.0$/$5.7$` | `gains $6.0$/$4.1$/$4.8$` |
| 7 | `latex_acl8/tables/ablation_family.tex` — HotpotQA "Sparse only, same interface" | `$1.19\times10^{-14}$` | `$7.29\times10^{-16}$` |
| 8 | `latex_acl8/tables/ablation_family.tex` — MuSiQue "Sparse only, same interface" | `$1.37\times10^{-12}$` | `$2.70\times10^{-9}$` |
| 9 | `latex_acl8/tables/ablation_family.tex` — HotpotQA "No dense evidence" | `$0.103$` | `$0.0641$` |
| 10 | `latex_acl8/tables/ablation_family.tex` — MuSiQue "No dense evidence" | `$0.0949$` | `$0.694$` |

Items 9–10 are the ones that carry a **substantive** risk, not just a digit change: the MuSiQue
no-dense rung moves 0.0949 → **0.694**, i.e. from "nearly significant, could be argued as
suggestive" to "flatly null". `latex/`'s prose already reflects this (the dense-fusion rung is
scoped to BrowseComp-Plus alone); `latex_acl8/`'s caption still reads *"neither is significant"*,
which stays literally true, so no `latex_acl8` **verdict** flips — every mismatched number there is
a magnitude/precision error, not a sign or significance error. `latex_acl8` also does **not** quote
the Wikipedia recall-inversion p-values at all, so the already-diagnosed 3.52e-08 / 0.0115 pair is
absent from both trees.

`latex_acl8/sections/results.tex:100` prints the browsecomp CI `[-0.5,+7.0]` — unaffected (cap 100
both sides).

## 4. Recomputed values (step 3)

Every mismatched paper-facing quantity already had a matched counterpart in
`analysis/matched_cap100_results.md`, so nothing in §3b required new computation. The one class of
quantity with **no** matched counterpart anywhere was `hybrid_control_ci.py`'s Wikipedia bootstrap
CIs. Those are not in either LaTeX tree, but they are recomputed here so the gap is closed and the
overlap claim is verified rather than assumed.

Method: identical machinery to `hybrid_control_ci.py` — `analysis.structured_surface_control.
{cell_metrics, paired_em}` and `analysis.equivalence_and_latency.{bootstrap_diff_pp, bootstrap_ci}`,
`n_boot=10000`, `seed=BOOT_SEED=20260723`, percentile 2.5/97.5 — with the **only** change being that
the Sieve arm is drawn from `_budget100` on the two Wikipedia datasets.

| dataset | Sieve arm | n_shared | Δ EM (pp) | McNemar p | 95% CI (pp) |
|---|---|---:|---:|---:|---|
| browsecomp_plus_structured | `_headline_validation` (cap 100) | 830 | +3.25 | 0.102 | [−0.48, +6.99] |
| hotpotqa_structured | **`_budget100` (cap 100)** | 7343 | **+5.84** | 2.84e-29 | **[+4.81, +6.86]** |
| musique_structured | **`_budget100` (cap 100)** | 2409 | **+4.57** | 1.38e-08 | **[+2.99, +6.14]** |

Superseding `analysis/hybrid_control_ci.md`'s mismatched rows: HotpotQA [+4.70, +6.75] → **[+4.81,
+6.86]**; MuSiQue [+3.94, +7.14] → **[+2.99, +6.14]**. The browsecomp row is byte-identical, which
is itself a check that only the Wikipedia Sieve arm changed.

**Overlap claim, re-verified at matched values:** BrowseComp-Plus CI [−0.48, +6.99] contains the
matched HotpotQA point (+5.84 → inside) and the matched MuSiQue point (+4.57 → inside). The
statement at `latex/sections/results.tex:422–424` and `latex_acl8/sections/results.tex:99–100` is
**correct as written**.

## 5. Recommended permanent fix (specified, deliberately NOT implemented)

A background experiment is writing to `runs/`; nothing here was changed. Apply later.

**Root cause.** A tier *name* was used as a proxy for a cell *identity*. The name was unique when
written; a later sibling tier silently made it ambiguous, and nothing in the code could notice —
`cell_dir()` happily returned the first path that existed.

**Fix: resolve the Sieve arm per-dataset by REQUIRED CAP, and fail loudly on ambiguity.** Replace
the module-level `TIER` constant in `analysis/structured_surface_control.py` with a resolver, and
have every consumer call it instead of importing `TIER`.

```python
SIEVE_TIER_CANDIDATES = ("_headline_validation", "_budget100")

def configured_cap(subdir, dataset, cond):
    """The single configured max_steps for a cell, from the per-row CONFIGURATION stamp.
    Returns None if the cell is absent; raises if the cell is internally mixed."""
    rows = cell_rows(subdir, dataset, cond)
    if rows is None:
        return None
    caps = {r.get("max_steps") for r in rows}
    if len(caps) != 1:
        raise ValueError(f"{subdir}/{dataset}/{cond} is cap-MIXED: {sorted(caps)}; "
                         f"it cannot be used as one side of a matched comparison")
    return caps.pop()

def resolve_sieve_arm(dataset, required_cap, cond=FULL_COND):
    """Return the (tier, cond) Sieve cell whose CONFIGURED cap == required_cap.
    Fails loudly if zero or more than one candidate matches."""
    hits = [(t, configured_cap(t, dataset, cond)) for t in SIEVE_TIER_CANDIDATES]
    hits = [(t, c) for t, c in hits if c == required_cap]
    if len(hits) == 0:
        raise LookupError(
            f"no Sieve arm at cap {required_cap} for {dataset}: candidates = "
            + repr({t: configured_cap(t, dataset, cond) for t in SIEVE_TIER_CANDIDATES}))
    if len(hits) > 1:
        raise LookupError(
            f"AMBIGUOUS: {len(hits)} Sieve arms at cap {required_cap} for {dataset}: {hits}")
    return hits[0][0], cond
```

Call-site shape — the required cap is **derived from the control**, never hardcoded, so the pairing
is matched by construction:

```python
ctrl_cap  = configured_cap(CONTROL_TIER, dataset, CONTROL_COND)
sieve_tier, sieve_cond = resolve_sieve_arm(dataset, required_cap=ctrl_cap)
full_m, ... = cell_metrics(dataset, sieve_tier, sieve_cond, qrels)
ctrl_m, ... = cell_metrics(dataset, CONTROL_TIER, CONTROL_COND, qrels)
```

Four properties this buys, each addressing a way the present code failed silently:

1. **No tier name in any comparison script.** `TIER` stops being importable as a Sieve locator;
   `hybrid_control_ci.py`, `structured_surface_control_3way.py`, `recall_inversion_test.py`,
   `ablation_deltas.py`, `wiki_isolation.py`, `no_dense_evidence_3way.py` and
   `multiplicity_census.py` all change from `import TIER` to `resolve_sieve_arm(ds, ctrl_cap)`.
2. **A third Sieve tier appearing later is a crash, not a wrong number.** Add the new tier to
   `SIEVE_TIER_CANDIDATES` and any dataset with two arms at the same cap raises `AMBIGUOUS`.
   The failure mode inverts from silent-and-wrong to loud-and-obvious.
3. **Cap comes from CONFIGURATION only.** `configured_cap` reads the `max_steps` stamp, never
   observed step counts (a post-treatment outcome — see `analysis/step_budget_audit.py`).
4. **Mixed cells cannot enter a matched comparison at all.** `configured_cap` raises on a
   multi-valued cap set, so the four known mixed cells (HotpotQA `DCI`, the three MuSiQue cells)
   can never silently become one side of a "matched" pairing.

Two cheap companions worth adding at the same time:

- **Assert-and-record in every comparison script.** Have each one assert
  `configured_cap(sieve) == configured_cap(control)` before computing, and write both caps into its
  `*_data.json` and a "configured cap" column in its `.md`. Then any future artifact is
  self-describing about its budget and a reader can see a mismatch without re-deriving it.
- **Keep the existing sanity-gate pattern, but gate on a Wikipedia number too.** Every affected
  script's gate reproduces a *BrowseComp* number, which is exactly the dataset the bug cannot touch
  — which is why all of them passed their own gates while producing mismatched Wikipedia output.
  Adding one cap-100 Wikipedia target (e.g. HotpotQA hybrid Δ=+5.84, p=2.84e-29) makes the gate
  sensitive to the failure it is supposed to catch.

## 6. Bottom line

- Sanity gate: **PASS** (+4.70 EM, p=0.010421 vs published +4.7 / 0.0104).
- `latex/` (32-page): **CLEAN** — every affected quantity traced and verified to be the matched
  cap-100 recomputation, with the budget situation explicitly disclosed in prose, in the ablation
  table caption, and in an appendix cap table. The one derived claim (CI overlap) re-verified true
  at matched values.
- `latex_acl8/` (8-page ACL body): **10 mismatched numbers still present** (§3b), plus no step-cap
  disclosure at all. No verdict flips, but the MuSiQue hybrid margin is overstated (+5.5 vs +4.57),
  the MuSiQue sparse-only margin is overstated (+5.7 vs +4.77), the token claim is overstated
  (24–32% vs 18–24%), and the MuSiQue no-dense p is 0.0949 where it should be 0.694.
- Blast radius beyond the four named scripts: `ablation_deltas.py` (2 rows), `wiki_isolation.py`
  (6 contrasts), `no_dense_evidence_3way.py` (2 contrasts), `multiplicity_census.py` (registry).
  Their on-disk `.md`/`.json` artifacts are stale, but none feeds a mismatched number into `latex/`.
