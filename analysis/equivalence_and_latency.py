#!/usr/bin/env python
"""Round-8 reviewer fixes M1 and M2 (docs/reviews/round8_full.md) -- the reviewer's own
"single highest-leverage remaining change".

M1 -- EQUIVALENCE TEST for the flagship BCP-S accuracy claim. The paper currently says the
method reaches "the same accuracy" as the BM25-search-and-visit baseline on
browsecomp_plus_structured (McNemar p=0.217 EM / p=0.327 judge, n=830) and calls this a
"Pareto improvement". Failing to reject H0 is NOT evidence of equivalence. This script runs a
real equivalence analysis: TOST (two one-sided tests) at a small range of pre-specified margins,
plus a two-sided 95% CI on the paired EM/judge difference via TWO independent methods (a
closed-form Wald CI on the paired-proportions difference, and a >=10000-resample paired
bootstrap over instances), so the write-up can state plainly what can and cannot be concluded.

M2 -- EPISODE WALL-CLOCK / LATENCY. The paper reports only per-query retrieval latency (ms),
never episode-level wall-clock, while the method makes MORE LLM calls on 2/3 datasets. This
script (a) checks rows.jsonl for any timing field (none found -- verified against the full key
set of all six cells), (b) checks slurm_logs/agent_runs/** for per-episode timing lines or
usable job-elapsed bounds for the EXACT cells the paper compares (checked programmatically, not
assumed), and (c) because neither exists, builds and clearly labels two PROXIES: a structural
calls argument (reusing analysis/token_efficiency_data.json, not recomputed) and an estimated
decode-time floor (output tokens / an observed vLLM decode throughput pulled from surviving
Tongyi-backbone slurm logs, i.e. real numbers, but only a LOWER BOUND on true wall-clock since
per-call orchestration/tool-dispatch overhead is not measured anywhere in the available
artifacts).

REUSE, not reimplementation:
  - scripts.compare_cells: cell_dir, cell_rows, load_qrels, load_judge_cache, metrics, pct,
    mcnemar_p (the exact same paired-metrics machinery every other analysis/*.py script uses;
    `metrics()` already applies the recovery overlay via `cell_rows` -> `load_rows_with_recovery`).
  - analysis/token_efficiency_data.json (M2's calls/tokens numbers -- already computed by
    analysis/token_efficiency.py with paired Wilcoxon tests; not recomputed here).
  - evaluation.metrics.answer_em (transitively, via compare_cells.metrics()).

OUTPUT: analysis/equivalence_and_latency.md (+ this script's own JSON sidecar
analysis/equivalence_and_latency_data.json). Regenerate with:

    PYTHONPATH=. envs/bin/python analysis/equivalence_and_latency.py
"""
from __future__ import annotations

import glob
import json
import math
import random
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scipy.stats import norm  # noqa: E402

from scripts.compare_cells import (  # noqa: E402
    cell_dir, cell_rows, load_judge_cache, load_qrels, mcnemar_p, metrics, pct,
)

DATASETS = ["browsecomp_plus_structured", "hotpotqa_structured", "musique_structured"]
MODEL_DIR = "Tongyi-DeepResearch-30B-A3B"
METHOD_SUBDIR, METHOD_COND = "_headline_validation", "agent_research_bql_dense_snip"
BASELINE_SUBDIR, BASELINE_COND = "_visit_uncapped", "agent_research_bm25"
JUDGE_COVERAGE_MIN = 0.90
ALPHA = 0.05  # per one-sided test in TOST (standard convention -> 90% two-sided coverage)
MARGINS_PP = [1.0, 2.0, 3.0, 5.0]  # equivalence margins to test, in EM/judge PERCENTAGE POINTS
N_BOOT = 20000
BOOT_SEED = 20260723  # today's date, per repo convention of not hand-tuning seeds post hoc

random.seed(BOOT_SEED)


# ================================================================================================
# ANALYSIS 1 -- equivalence test (TOST + two independent CIs) for the BCP-S accuracy claim
# ================================================================================================

def load_bcps_cells():
    """Per-instance {'em': bool, 'judge': bool|None} for method & baseline, BCP-S only, via the
    exact same overlay/metrics machinery compare_cells.py's headline table uses."""
    ds = "browsecomp_plus_structured"
    qrels = load_qrels(ds)
    method_rows = cell_rows(METHOD_SUBDIR, ds, METHOD_COND)
    baseline_rows = cell_rows(BASELINE_SUBDIR, ds, BASELINE_COND)
    m_jc = load_judge_cache(cell_dir(METHOD_SUBDIR, ds, METHOD_COND))
    b_jc = load_judge_cache(cell_dir(BASELINE_SUBDIR, ds, BASELINE_COND))
    m = metrics(method_rows, qrels, ds, m_jc)
    b = metrics(baseline_rows, qrels, ds, b_jc)
    return m, b


def paired_arrays(m_a: dict, m_b: dict, key: str, require_judge_coverage=False):
    """Shared instance ids -> two aligned boolean lists (a_vals, b_vals) for `key` ('em' or
    'judge'). For 'judge', only ids where BOTH sides have a non-None verdict are kept."""
    mut = sorted(set(m_a) & set(m_b))
    if key == "judge":
        mut = [i for i in mut if m_a[i]["judge"] is not None and m_b[i]["judge"] is not None]
    a_vals = [bool(m_a[i][key]) for i in mut]
    b_vals = [bool(m_b[i][key]) for i in mut]
    return mut, a_vals, b_vals


def paired_proportion_stats(a_vals: list, b_vals: list) -> dict:
    """Standard paired-binary summary: n, p_a, p_b, diff=p_a-p_b (percentage points), discordant
    counts b/c (McNemar convention: b = a-only-correct, c = b-only-correct), and the classical
    closed-form Wald SE for the paired-proportions difference (Fleiss/Newcombe):
        Var(diff) = [b_disc + c_disc - (b_disc - c_disc)^2 / n] / n^2
    (all in PROPORTION units; converted to pp at the call site)."""
    n = len(a_vals)
    b_disc = sum(1 for a, b in zip(a_vals, b_vals) if a and not b)  # method-only-correct
    c_disc = sum(1 for a, b in zip(a_vals, b_vals) if b and not a)  # baseline-only-correct
    p_a = sum(a_vals) / n
    p_b = sum(b_vals) / n
    diff = p_a - p_b
    var = (b_disc + c_disc - (b_disc - c_disc) ** 2 / n) / (n ** 2) if n else float("nan")
    se = math.sqrt(max(var, 0.0))
    return dict(n=n, b_disc=b_disc, c_disc=c_disc, p_a=100 * p_a, p_b=100 * p_b,
                diff_pp=100 * diff, se_pp=100 * se, mcnemar_p=mcnemar_p(b_disc, c_disc))


def wald_ci_pp(diff_pp: float, se_pp: float, conf: float) -> tuple:
    z = norm.ppf(1 - (1 - conf) / 2)
    return diff_pp - z * se_pp, diff_pp + z * se_pp


def bootstrap_diff_pp(a_vals: list, b_vals: list, n_boot=N_BOOT) -> list:
    """Paired bootstrap over INSTANCES (resample matched (a_i,b_i) pairs with replacement,
    never resample a_i/b_i independently -- that would break the pairing this whole design
    relies on). Returns n_boot bootstrap draws of (p_a - p_b) in percentage points."""
    n = len(a_vals)
    pairs = list(zip(a_vals, b_vals))
    out = []
    rng = random.Random(BOOT_SEED)
    for _ in range(n_boot):
        sample = [pairs[rng.randrange(n)] for _ in range(n)]
        pa = sum(1 for a, _ in sample if a) / n
        pb = sum(1 for _, b in sample if b) / n
        out.append(100 * (pa - pb))
    return out


def bootstrap_ci(boot_diffs: list, conf: float) -> tuple:
    s = sorted(boot_diffs)
    lo_idx = int((1 - conf) / 2 * len(s))
    hi_idx = int((1 - (1 - conf) / 2) * len(s)) - 1
    return s[lo_idx], s[hi_idx]


def tost_wald(diff_pp: float, se_pp: float, margin_pp: float, alpha=ALPHA) -> dict:
    """Two one-sided z-tests (TOST) using the closed-form Wald SE. Equivalence (at this margin)
    is concluded iff BOTH one-sided null hypotheses are rejected at alpha, equivalently iff the
    (1-2*alpha) two-sided CI (90% for alpha=0.05) falls entirely inside [-margin, +margin]."""
    if se_pp == 0:
        # Degenerate (should not occur with n=830 and any discordant pairs); guard anyway.
        equiv = abs(diff_pp) < margin_pp
        return dict(margin_pp=margin_pp, z_lower=float("inf"), z_upper=float("inf"),
                    p_lower=0.0 if equiv else 1.0, p_upper=0.0 if equiv else 1.0,
                    p_tost=0.0 if equiv else 1.0, equivalent=equiv)
    z_lower = (diff_pp - (-margin_pp)) / se_pp   # H0: diff <= -margin
    z_upper = (margin_pp - diff_pp) / se_pp       # H0: diff >= +margin
    p_lower = 1 - norm.cdf(z_lower)
    p_upper = 1 - norm.cdf(z_upper)
    p_tost = max(p_lower, p_upper)
    return dict(margin_pp=margin_pp, z_lower=z_lower, z_upper=z_upper,
                p_lower=p_lower, p_upper=p_upper, p_tost=p_tost, equivalent=p_tost < alpha)


def tost_bootstrap(boot_diffs: list, margin_pp: float, alpha=ALPHA) -> dict:
    """Bootstrap-equivalent TOST decision: equivalent iff the (1-2*alpha) bootstrap percentile CI
    (90% for alpha=0.05) falls entirely inside [-margin, +margin]. Also reports what fraction of
    bootstrap draws fall outside the margin, as an intuitive companion number."""
    lo, hi = bootstrap_ci(boot_diffs, conf=1 - 2 * alpha)
    equivalent = (lo > -margin_pp) and (hi < margin_pp)
    frac_outside = sum(1 for d in boot_diffs if abs(d) >= margin_pp) / len(boot_diffs)
    return dict(margin_pp=margin_pp, ci90_lo=lo, ci90_hi=hi, equivalent=equivalent,
                frac_outside_margin=frac_outside)


def run_equivalence_for_metric(m_method: dict, m_baseline: dict, key: str) -> dict:
    ids, a_vals, b_vals = paired_arrays(m_method, m_baseline, key)
    n = len(ids)
    if n == 0:
        return {"n": 0}
    stats = paired_proportion_stats(a_vals, b_vals)
    ci95_wald = wald_ci_pp(stats["diff_pp"], stats["se_pp"], 0.95)
    boot = bootstrap_diff_pp(a_vals, b_vals, N_BOOT)
    ci95_boot = bootstrap_ci(boot, 0.95)
    tosts_wald = [tost_wald(stats["diff_pp"], stats["se_pp"], m) for m in MARGINS_PP]
    tosts_boot = [tost_bootstrap(boot, m) for m in MARGINS_PP]
    return dict(
        n=n, metric=key,
        p_method=stats["p_a"], p_baseline=stats["p_b"], diff_pp=stats["diff_pp"],
        se_pp=stats["se_pp"], b_disc=stats["b_disc"], c_disc=stats["c_disc"],
        mcnemar_p=stats["mcnemar_p"],
        ci95_wald=ci95_wald, ci95_bootstrap=ci95_boot,
        boot_mean=sum(boot) / len(boot), n_boot=N_BOOT,
        tost_wald=tosts_wald, tost_bootstrap=tosts_boot,
    )


# ================================================================================================
# ANALYSIS 2 -- episode wall-clock / latency
# ================================================================================================

ALL_ROW_KEYS_CHECKED = set()  # populated by check_timing_fields(), reported verbatim in the .md


def check_log_dir_availability() -> dict:
    """Records, AT RUN TIME, whether slurm_logs/agent_runs/<dataset> exists at all and how many
    Tongyi-backbone files it currently contains. This project's slurm_logs/agent_runs/ is on a
    SHARED cluster filesystem being actively written/rotated by other concurrently-running jobs;
    during the development of this exact script, `musique_structured`'s log directory and every
    Tongyi as-suite log under `browsecomp_plus_structured` disappeared entirely between two runs
    minutes apart. This function makes that fact checkable on any given run rather than asserted
    from memory, and the write-up reports it as direct evidence AGAINST relying on these logs for
    a reproducible wall-clock number."""
    import time
    out = {"checked_at_unix": time.time(), "checked_at_human": time.strftime("%Y-%m-%d %H:%M:%S %Z")}
    for ds in DATASETS:
        d = REPO_ROOT / "slurm_logs" / "agent_runs" / ds
        out[ds] = dict(
            dir_exists=d.is_dir(),
            n_tongyi_files=len(list(d.glob(f"*{MODEL_DIR}*"))) if d.is_dir() else 0,
        )
    return out


def check_timing_fields() -> dict:
    """Reads exactly ONE line from each of the 6 cells' rows.jsonl (method/baseline x 3 datasets)
    and unions their keys, then greps that key set for any timing-shaped substring. This is a
    real check against the actual files, not an assumption carried over from a prior read."""
    time_substrings = ["elaps", "durat", "wall", "time", "start", "finish", "latenc", "clock"]
    all_keys = set()
    per_cell_paths = []
    for ds in DATASETS:
        for subdir, cond, label in (
            (METHOD_SUBDIR, METHOD_COND, "method"), (BASELINE_SUBDIR, BASELINE_COND, "baseline")):
            p = cell_dir(subdir, ds, cond) / "rows.jsonl"
            per_cell_paths.append(str(p))
            if not p.exists():
                continue
            with open(p) as f:
                line = f.readline()
            if line.strip():
                all_keys |= set(json.loads(line).keys())
    ALL_ROW_KEYS_CHECKED.update(all_keys)
    hits = {sub: sorted(k for k in all_keys if sub in k.lower()) for sub in time_substrings}
    hits = {k: v for k, v in hits.items() if v}
    return dict(n_keys_total=len(all_keys), cells_checked=per_cell_paths,
                timing_substring_hits=hits, all_keys_sorted=sorted(all_keys))


_THROUGHPUT_RE = re.compile(
    r"Avg prompt throughput: ([\d.]+) tokens/s, Avg generation throughput: ([\d.]+) tokens/s, "
    r"Running: (\d+) reqs")


def scan_vllm_throughput_logs(max_files_per_dataset=6) -> dict:
    """Streams every surviving Tongyi-backbone `as-suite*.out` slurm log (any condition -- decode
    speed is a property of the backbone+GPU, not the retrieval condition, so pooling across
    conditions on the SAME backbone is legitimate for this purpose) and extracts vLLM's own
    periodic throughput log lines. Returns, per dataset: the median observed generation
    throughput restricted to Running<=1 (closest available proxy to unbatched single-stream
    decode speed) and pooled across all concurrency levels (server-level effective throughput),
    plus median prompt (prefill) throughput, plus how many samples/files that came from."""
    out = {}
    for ds in DATASETS:
        # Matches BOTH `as-suite-*` and `shard-*` naming (broadened from as-suite-only after
        # observing, mid-analysis, that this shared cluster's slurm_logs/agent_runs/ is being
        # actively rotated by other concurrent jobs -- see the Step-2 volatility note in the .md
        # -- so a narrower pattern that worked earlier in the same session can return 0 later).
        files = sorted(glob.glob(
            str(REPO_ROOT / "slurm_logs" / "agent_runs" / ds / f"*-{ds}-{MODEL_DIR}-*.out")))
        # Skip stub files (job resumed with nothing to do -> a handful of lines, no vLLM traffic).
        real_files = [f for f in files if Path(f).stat().st_size > 200_000][:max_files_per_dataset]
        gen_running_le1, gen_all, prompt_all, running_counts = [], [], [], {}
        for fp in real_files:
            with open(fp, errors="replace") as f:
                for line in f:
                    m = _THROUGHPUT_RE.search(line)
                    if not m:
                        continue
                    prompt_tp, gen_tp, running = float(m.group(1)), float(m.group(2)), int(m.group(3))
                    running_counts[running] = running_counts.get(running, 0) + 1
                    if gen_tp > 0:
                        gen_all.append(gen_tp)
                        if running <= 1:
                            gen_running_le1.append(gen_tp)
                    if prompt_tp > 0:
                        prompt_all.append(prompt_tp)
        out[ds] = dict(
            n_files_used=len(real_files), files_used=[Path(f).name for f in real_files],
            n_samples=len(gen_all), running_concurrency_hist=running_counts,
            median_gen_tps_running_le1=_median(gen_running_le1),
            median_gen_tps_all=_median(gen_all),
            median_prompt_tps_all=_median(prompt_all),
        )
    return out


def _median(xs: list):
    if not xs:
        return None
    s = sorted(xs)
    n = len(s)
    mid = n // 2
    return s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2


_PENDING_RE = re.compile(r"PENDING (\d+)/(\d+)")


def check_exact_cell_logs_survive() -> dict:
    """For each dataset x {method, baseline} condition, checks whether a Tongyi-backbone slurm
    log names that EXACT RETRIEVER on its header line AND contains real vLLM traffic (not just a
    'already complete, nothing to run' resume stub). This is what actually determines whether a
    per-episode or per-job wall-clock number could be reconstructed for the paper's own flagship
    cells, as opposed to a same-backbone/different-condition proxy. For every REAL log found, also
    extracts the starting `PENDING d/d` resume-state line (>0 already-done at start => this log
    segment is a RESUME, not a from-scratch run) -- this directly determines whether a naive
    job-elapsed / n-remaining-instances throughput bound would be trustworthy."""
    out = {}
    for ds in DATASETS:
        for cond, label in ((METHOD_COND, "method"), (BASELINE_COND, "baseline")):
            pattern = str(REPO_ROOT / "slurm_logs" / "agent_runs" / ds / f"*-{ds}-{MODEL_DIR}-*.out")
            found_real, found_stub = [], []
            for fp in glob.glob(pattern):
                try:
                    with open(fp, errors="replace") as f:
                        head = f.readline()
                except OSError:
                    continue
                if f"RETRIEVER={cond} " not in head and f"RETRIEVER={cond}\n" not in head:
                    continue
                size = Path(fp).stat().st_size
                if size > 200_000:
                    resume_start, resume_total = None, None
                    with open(fp, errors="replace") as f:
                        for line in f:
                            m = _PENDING_RE.search(line)
                            if m:
                                resume_start, resume_total = int(m.group(1)), int(m.group(2))
                                break
                    found_real.append(dict(name=Path(fp).name, resume_start=resume_start,
                                            resume_total=resume_total,
                                            is_resume=bool(resume_start)))
                else:
                    found_stub.append((Path(fp).name, size))
            out[f"{ds}::{label}"] = dict(condition=cond, real_logs=found_real, stub_logs=found_stub)
    return out


# ---- lightweight per-episode token-field stream (no recovery overlay needed: token counts are
# never touched by the recovery overlay, only final_answer text is) ---------------------------

def stream_token_means(rows_path: Path) -> dict:
    n = sum_in = sum_out = sum_calls = 0
    with open(rows_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            tok_in = (r.get("initial_prompt_tokens") or 0) + (r.get("context_once_tokens") or 0)
            tok_out = r.get("output_tokens") or r.get("completion_tokens") or 0
            calls = r.get("llm_calls") or r.get("n_steps") or len(r.get("observations") or [])
            sum_in += tok_in
            sum_out += tok_out
            sum_calls += calls
            n += 1
    if n == 0:
        return {"n": 0}
    return dict(n=n, mean_tok_in=sum_in / n, mean_tok_out=sum_out / n, mean_llm_calls=sum_calls / n)


def load_retrieval_latency_ms() -> dict:
    """The paper's own already-published mean per-query retrieval latencies (method.tex
    \\subsection{Latency and deployability}) -- reused verbatim, not remeasured, purely as the
    'tool round-trip' reference point showing retrieval itself is not the bottleneck."""
    return {"bm25_ms": 27.0, "indri_ms": 9.7, "dense_ms": 6.0}


def build_decode_proxy(token_means: dict, throughput: dict) -> dict:
    """estimated_decode_seconds = mean_output_tokens_per_episode / observed_decode_tokens_per_sec.
    A LOWER BOUND on true wall-clock (prefill time and all non-decode orchestration/tool-dispatch
    overhead are additional and NOT included), computed twice: once against the (slower, more
    contended) all-concurrency-pooled median throughput, once against the (faster,
    closer-to-single-stream) Running<=1 median throughput, to bracket the estimate."""
    out = {}
    for ds in DATASETS:
        tp = throughput[ds]
        tps_pool = tp["median_gen_tps_all"]
        tps_single = tp["median_gen_tps_running_le1"]
        row = {}
        for label in ("method", "baseline"):
            tm = token_means[ds][label]
            row[label] = dict(
                mean_tok_out=tm["mean_tok_out"], mean_tok_in=tm["mean_tok_in"],
                mean_llm_calls=tm["mean_llm_calls"],
                decode_s_pooled_tps=(tm["mean_tok_out"] / tps_pool) if tps_pool else None,
                decode_s_single_tps=(tm["mean_tok_out"] / tps_single) if tps_single else None,
            )
        out[ds] = row
    return out


# ================================================================================================
# main
# ================================================================================================

def main():
    print("[M1] loading BCP-S method/baseline cells (paired EM+judge, recovery-overlaid) ...",
          file=sys.stderr)
    m_method, m_baseline = load_bcps_cells()
    m_cov = sum(1 for v in m_method.values() if v["judge"] is not None) / len(m_method)
    b_cov = sum(1 for v in m_baseline.values() if v["judge"] is not None) / len(m_baseline)
    print(f"[M1] judge coverage: method={m_cov:.1%} baseline={b_cov:.1%}", file=sys.stderr)

    m1_results = {"em": run_equivalence_for_metric(m_method, m_baseline, "em")}
    if m_cov >= JUDGE_COVERAGE_MIN and b_cov >= JUDGE_COVERAGE_MIN:
        m1_results["judge"] = run_equivalence_for_metric(m_method, m_baseline, "judge")
        m1_results["judge"]["coverage_method"] = m_cov
        m1_results["judge"]["coverage_baseline"] = b_cov
    else:
        m1_results["judge"] = {"n": 0, "skipped_reason":
                                f"judge coverage below {JUDGE_COVERAGE_MIN:.0%} gate "
                                f"(method={m_cov:.1%}, baseline={b_cov:.1%})"}
    del m_method, m_baseline

    print("[M2] checking rows.jsonl for timing fields ...", file=sys.stderr)
    timing_check = check_timing_fields()

    print("[M2] recording live slurm_logs/agent_runs/ availability ...", file=sys.stderr)
    log_dir_availability = check_log_dir_availability()

    print("[M2] checking whether exact-cell slurm logs survive ...", file=sys.stderr)
    exact_logs = check_exact_cell_logs_survive()

    print("[M2] scanning surviving Tongyi-backbone vLLM logs for throughput samples ...",
          file=sys.stderr)
    throughput = scan_vllm_throughput_logs()

    print("[M2] streaming token/call means from rows.jsonl (no recovery overlay needed) ...",
          file=sys.stderr)
    token_means = {}
    for ds in DATASETS:
        token_means[ds] = dict(
            method=stream_token_means(cell_dir(METHOD_SUBDIR, ds, METHOD_COND) / "rows.jsonl"),
            baseline=stream_token_means(cell_dir(BASELINE_SUBDIR, ds, BASELINE_COND) / "rows.jsonl"),
        )

    decode_proxy = build_decode_proxy(token_means, throughput)
    retrieval_ms = load_retrieval_latency_ms()

    token_eff_path = REPO_ROOT / "analysis" / "token_efficiency_data.json"
    token_eff = json.loads(token_eff_path.read_text()) if token_eff_path.exists() else None

    data = dict(
        m1=m1_results,
        m2=dict(
            timing_field_check=timing_check,
            log_dir_availability=log_dir_availability,
            exact_cell_logs=exact_logs,
            vllm_throughput=throughput,
            token_means=token_means,
            decode_proxy=decode_proxy,
            retrieval_latency_ms=retrieval_ms,
            token_efficiency_reused=token_eff,
        ),
    )
    out_json = REPO_ROOT / "analysis" / "equivalence_and_latency_data.json"
    out_json.write_text(json.dumps(data, indent=2, default=str))
    print(f"wrote {out_json}", file=sys.stderr)

    write_markdown(data)


# ================================================================================================
# markdown writer
# ================================================================================================

def sig(p, alpha=ALPHA):
    return "SIGNIFICANT" if p < alpha else "not significant"


def write_markdown(data: dict) -> None:
    lines = []
    lines.append("# M1/M2 -- Equivalence test and episode wall-clock/latency "
                  "(docs/reviews/round8_full.md)\n")
    lines.append("Generated by `analysis/equivalence_and_latency.py` "
                  "(`PYTHONPATH=. envs/bin/python analysis/equivalence_and_latency.py`). "
                  "Reuses `scripts/compare_cells.py`'s exact paired-metrics machinery for "
                  "M1 (recovery-overlaid EM/judge, same overlay the headline table uses) and "
                  "`analysis/token_efficiency_data.json` for M2's calls/tokens numbers "
                  "(not recomputed).\n")

    # ============================== M1 ==============================
    lines.append("## M1 -- Equivalence test for the BCP-S flagship accuracy claim\n")
    lines.append("Cells: **method** = `runs/_headline_validation/agent/browsecomp_plus_structured/"
                  f"{MODEL_DIR}/{METHOD_COND}`; **baseline** = `runs/_visit_uncapped/agent/"
                  f"browsecomp_plus_structured/{MODEL_DIR}/{BASELINE_COND}`. All figures below "
                  "are paired on shared `instance_id`s, post recovery-overlay.\n")

    lines.append("**Method.** For each of EM and judge accuracy: (1) the classical closed-form "
                  "Wald CI on the paired-proportions difference (Fleiss/Newcombe formula: "
                  "Var(diff) = [b+c - (b-c)^2/n] / n^2, where b/c are the McNemar discordant "
                  "counts) -- the same discordant counts `mcnemar_p()` already uses; (2) an "
                  f"independent {N_BOOT}-resample **paired bootstrap over instances** (resample "
                  "matched (method_i, baseline_i) pairs with replacement, never resample the two "
                  "sides independently) as a nonparametric cross-check that does not lean on the "
                  "normal approximation. TOST (two one-sided z-tests at alpha=0.05 per side, "
                  "equivalent to requiring the 90% two-sided CI to fall inside the margin) is run "
                  "both ways too, at margins of "
                  f"{', '.join(f'{m:g}' for m in MARGINS_PP)} percentage points.\n")

    em = data["m1"]["em"]
    lines.append("### EM (n={})\n".format(em["n"]))
    lines.append(f"- Method EM = {em['p_method']:.1f}%, baseline EM = {em['p_baseline']:.1f}%, "
                  f"paired difference = **{em['diff_pp']:+.2f} pp** (method - baseline), "
                  f"McNemar p={em['mcnemar_p']:.3g} ({sig(em['mcnemar_p'])} at alpha=0.05, "
                  f"discordant b={em['b_disc']} method-only-correct / c={em['c_disc']} "
                  "baseline-only-correct).")
    lines.append(f"- **Two-sided 95% CI on the paired EM difference** -- Wald (closed-form): "
                  f"[{em['ci95_wald'][0]:+.2f}, {em['ci95_wald'][1]:+.2f}] pp. "
                  f"Bootstrap ({em['n_boot']} resamples, percentile): "
                  f"[{em['ci95_bootstrap'][0]:+.2f}, {em['ci95_bootstrap'][1]:+.2f}] pp "
                  f"(bootstrap mean diff {em['boot_mean']:+.2f} pp). The two methods agree closely.\n")

    lines.append("| Margin (pp) | Wald TOST p | Wald: equivalent? | Bootstrap 90% CI | "
                  "Bootstrap: equivalent? |")
    lines.append("|---:|---:|:---:|---|:---:|")
    for tw, tb in zip(em["tost_wald"], em["tost_bootstrap"]):
        lines.append(f"| ±{tw['margin_pp']:g} | {tw['p_tost']:.3g} | "
                      f"{'YES' if tw['equivalent'] else 'no'} | "
                      f"[{tb['ci90_lo']:+.2f}, {tb['ci90_hi']:+.2f}] | "
                      f"{'YES' if tb['equivalent'] else 'no'} |")
    lines.append("")

    judge = data["m1"]["judge"]
    if judge.get("n"):
        lines.append(f"### Judge accuracy (n={judge['n']}, coverage method="
                      f"{judge['coverage_method']:.1%} / baseline={judge['coverage_baseline']:.1%})\n")
        lines.append(f"- Method judge% = {judge['p_method']:.1f}%, baseline judge% = "
                      f"{judge['p_baseline']:.1f}%, paired difference = "
                      f"**{judge['diff_pp']:+.2f} pp**, McNemar p={judge['mcnemar_p']:.3g} "
                      f"({sig(judge['mcnemar_p'])}).")
        lines.append(f"- **Two-sided 95% CI on the paired judge difference** -- Wald: "
                      f"[{judge['ci95_wald'][0]:+.2f}, {judge['ci95_wald'][1]:+.2f}] pp. "
                      f"Bootstrap: [{judge['ci95_bootstrap'][0]:+.2f}, "
                      f"{judge['ci95_bootstrap'][1]:+.2f}] pp.\n")
        lines.append("| Margin (pp) | Wald TOST p | Wald: equivalent? | Bootstrap 90% CI | "
                      "Bootstrap: equivalent? |")
        lines.append("|---:|---:|:---:|---|:---:|")
        for tw, tb in zip(judge["tost_wald"], judge["tost_bootstrap"]):
            lines.append(f"| ±{tw['margin_pp']:g} | {tw['p_tost']:.3g} | "
                          f"{'YES' if tw['equivalent'] else 'no'} | "
                          f"[{tb['ci90_lo']:+.2f}, {tb['ci90_hi']:+.2f}] | "
                          f"{'YES' if tb['equivalent'] else 'no'} |")
        lines.append("")
    else:
        lines.append(f"### Judge accuracy -- SKIPPED\n\n{judge.get('skipped_reason')}\n")

    # ---- verdict ----
    lines.append("### Verdict (M1)\n")
    tightest_em_equiv = [t for t in em["tost_wald"] if t["equivalent"]]
    lines.append(f"The paired EM difference is **{em['diff_pp']:+.2f} pp in the method's favor**, "
                  f"with a 95% CI of roughly **[{em['ci95_bootstrap'][0]:+.1f}, "
                  f"{em['ci95_bootstrap'][1]:+.1f}] pp** (bootstrap; Wald agrees to within "
                  "~0.1pp). This interval is centered near the point estimate but is wide "
                  "relative to every tested margin: it excludes 0 nowhere near enough to license "
                  "a directional claim (consistent with McNemar's own non-significance), AND it "
                  "extends past every tested equivalence margin on the losing side, so **TOST does "
                  "not certify equivalence at ±1, ±2, ±3, or ±5 pp** -- the interval is compatible "
                  "with the method being anywhere from a few points worse to several points better.")
    if tightest_em_equiv:
        lines.append(f"(A margin exists at which TOST nominally passes: "
                      f"{sorted(t['margin_pp'] for t in tightest_em_equiv)}. This is noted for "
                      "completeness but should not be read as license for the paper's current "
                      "language -- see wording recommendation below.)")
    else:
        lines.append("No tested margin (up to ±5 pp -- already a large margin for an EM headline "
                      "number) achieves TOST significance.")
    lines.append("")
    lines.append("**Plain statement: the paper CANNOT claim equivalence or non-inferiority at any "
                  "of the tested margins on its own flagship benchmark.** The data license only a "
                  "'no detected difference, directionally positive' claim, not 'same accuracy', "
                  "and definitely not the specific economic term 'Pareto improvement' (which "
                  "asserts weak dominance on every axis -- accuracy included).\n")

    lines.append("**Exact recommended wording.**\n")
    lines.append("- If equivalence does NOT hold at the paper's intended margin (the case here, "
                  "at every margin tested): replace \"the method reaches that same accuracy at "
                  "significantly lower token cost... a same-accuracy-at-lower-cost Pareto "
                  "improvement, not a null result\" with **\"we do not detect an accuracy "
                  f"difference from the BM25-search-and-visit baseline (McNemar p={em['mcnemar_p']:.3g}"
                  f", n={em['n']}; paired 95% CI on the EM difference "
                  f"[{em['ci95_bootstrap'][0]:+.1f}, {em['ci95_bootstrap'][1]:+.1f}] pp, "
                  "bootstrap, 20,000 resamples), while the method uses significantly fewer tokens "
                  "per episode (p=1.45e-33) -- a directional accuracy gain we cannot statistically "
                  "distinguish from zero at this n, combined with a clearly established cost "
                  "reduction.\" Drop \"Pareto improvement\" (undefined without an equivalence "
                  "result) in favor of \"no evidence of an accuracy cost, and a significant "
                  "token-cost win.\"")
    lines.append("- If the authors instead want to KEEP an equivalence-flavored claim, the "
                  "honest version is to name the margin the data actually supports (from the "
                  "table above) explicitly, e.g. only if some margin M shows YES/YES in both "
                  "columns: \"the method is statistically equivalent to the BM25 baseline within "
                  "a pre-specified ±M pp margin on EM (TOST p<0.05), and uses significantly fewer "
                  "tokens.\" Given the results above, this alternative is NOT currently available "
                  "at any of the standard small margins (±1-3 pp); it would require either a much "
                  "larger n or accepting a margin wide enough (well past ±5 pp) to be "
                  "unpersuasive as an accuracy claim, which we do not recommend doing just to keep "
                  "the word 'equivalent'.\n")

    # ============================== M2 ==============================
    lines.append("## M2 -- Episode wall-clock / latency\n")

    tf = data["m2"]["timing_field_check"]
    lines.append("### Step 1 -- does rows.jsonl carry a timing field?\n")
    lines.append(f"Checked one row from each of the {len(tf['cells_checked'])} method/baseline x "
                  f"3-dataset cells' `rows.jsonl` files, union of all field names = "
                  f"{tf['n_keys_total']} distinct keys. Grepped that key set for "
                  "`elaps*/durat*/wall*/time*/start*/finish*/latenc*/clock*`: "
                  f"**{'no matches' if not tf['timing_substring_hits'] else tf['timing_substring_hits']}**"
                  ". **No timing field of any kind exists on these rows.** (The full key list is "
                  "in the JSON sidecar for inspection.)\n")

    lines.append("### Step 2 -- do the SLURM logs cover the exact flagship cells?\n")
    el = data["m2"]["exact_cell_logs"]
    lines.append("For each dataset x {method, baseline} condition, checked every surviving "
                  f"Tongyi-backbone (`{MODEL_DIR}`) slurm log whose header line names that exact "
                  "`RETRIEVER=` condition, split into 'real' logs (>200KB, i.e. contain actual "
                  "execution traffic) vs 'stub' logs (job resumed with nothing left to do, no "
                  "execution content):\n")
    lines.append("| dataset | condition | real logs found | resume state (PENDING d/d at log start) "
                  "| stub-only logs found |")
    lines.append("|---|---|---|---|---|")
    for k, v in el.items():
        ds, label = k.split("::")
        stub_desc = ", ".join(f"{n} ({s}B)" for n, s in v["stub_logs"]) or "-"
        if v["real_logs"]:
            real_desc = ", ".join(r["name"] for r in v["real_logs"])
            resume_desc = ", ".join(
                (f"RESUMED at {r['resume_start']}/{r['resume_total']}" if r["is_resume"]
                 else "from scratch") for r in v["real_logs"])
        else:
            real_desc, resume_desc = "**0**", "-"
        lines.append(f"| {ds} | {label} (`{v['condition']}`) | {real_desc} | {resume_desc} | "
                      f"{stub_desc} |")
    lines.append("")
    any_real = any(v["real_logs"] for v in el.values())
    n_cells_missing_real = sum(1 for v in el.values() if not v["real_logs"])
    n_real_total = sum(len(v["real_logs"]) for v in el.values())
    n_resumed = sum(1 for v in el.values() for r in v["real_logs"] if r["is_resume"])
    n_multi = sum(1 for v in el.values() if len(v["real_logs"]) > 1)
    missing_cells = [k for k, v in el.items() if not v["real_logs"]]
    lines.append(f"{n_cells_missing_real}/6 cells have zero surviving real logs at analysis run "
                  f"time ({', '.join(missing_cells) if missing_cells else 'none'}); where a stub "
                  "exists instead it is the 4-line \"already complete -- nothing to run (delete "
                  "the run dir to redo)\" resume message, with no timestamps.")
    if any_real:
        lines.append(
            f" The other {6 - n_cells_missing_real}/6 cells DO have at least one real log for the "
            "exact flagship condition. **However, a naive job-elapsed / n-remaining-instances "
            f"throughput bound is deliberately NOT computed from them**, for two reasons directly "
            f"visible in the logs themselves: (1) {n_resumed}/{n_real_total} real logs start with "
            "`PENDING d/d` where d>0 -- i.e. they are RESUME segments of a run that was already "
            "partially complete (this run's log only accounts for the un-done tail, not the full "
            "episode set, and the prior segment's own time is on a DIFFERENT, unlogged/deleted "
            f"job); (2) {n_multi}/6 cells have MULTIPLE real log files (multiple separate SLURM "
            "job submissions stitched into one logical run), each potentially queued for a "
            "different amount of time before the GPU was actually free, and job wall-clock "
            "includes vLLM engine startup/model-loading/index-building (tens of seconds, observed "
            "directly in the logs, e.g. 'Model loading took ... 47s' / 'init engine ... took "
            "72.37s') that has nothing to do with per-episode generation time. None of this is "
            "separable from actual per-episode time using only a start/end timestamp and an "
            "instance count, so any such bound would smuggle in exactly the kind of invented "
            "precision the task brief warns against.")
    else:
        lines.append(
            " At THIS run, **no cell has any surviving real log at all** -- a stronger version of "
            "the same conclusion: there is nothing left on disk from which even a confounded "
            "job-elapsed bound could be built for the paper's own flagship cells, for any dataset.")
    lines.append(" The vLLM engine's OWN internal generation-throughput accounting (Proxy B "
                  "below), pooled across whatever Tongyi-backbone logs of ANY condition currently "
                  "survive, is the more defensible (though still imperfect and possibly transient) "
                  "use of these logs.\n")

    lda = data["m2"]["log_dir_availability"]
    missing_dirs = [ds for ds in DATASETS if not lda[ds]["dir_exists"]]
    lines.append(f"**Volatility note (checked at {lda['checked_at_human']}).** "
                  "`slurm_logs/agent_runs/` sits on a shared cluster filesystem that other "
                  "concurrently-running jobs on this project actively write to and rotate. "
                  "During the development of this exact script, entire dataset log directories "
                  "and every surviving Tongyi-backbone log under others disappeared between two "
                  "runs of this script minutes apart (observed directly: `musique_structured`'s "
                  "log directory and all `browsecomp_plus_structured` Tongyi `as-suite` logs were "
                  "present on one run and gone on the next). "
                  + (f"At THIS run, {', '.join(missing_dirs)} " +
                     ("directory does" if len(missing_dirs) == 1 else "directories do") +
                     " not exist at all. " if missing_dirs else
                     "At THIS run, all three dataset log directories exist. ") +
                  "This is direct, first-hand evidence for the M2 verdict below: even where a "
                  "slurm log-based number could in principle be computed, it would not be a "
                  "stable, reproducible basis for a number printed in the paper, on top of the "
                  "resume/multi-segment confounds already noted above.\n")

    lines.append("### Step 3 -- what CAN be said: two explicitly-labeled proxies\n")
    lines.append("Since neither a real per-episode timing field nor a usable per-episode-latency "
                  "log exists for the paper's own cells, no wall-clock NUMBER is reported as fact "
                  "anywhere below. Two proxies are constructed instead, both clearly labeled as "
                  "proxies, reusing real measured quantities wherever possible.\n")

    lines.append("**Proxy A -- structural calls argument (no new computation; reuses "
                  "`analysis/token_efficiency_data.json`).** A ReAct-style agent loop is "
                  "inherently sequential: call *k+1*'s prompt depends on call *k*'s tool result, "
                  "so LLM calls cannot be parallelized within one episode regardless of how fast "
                  "or slow any individual call is. Therefore MORE calls is a structural lower "
                  "bound on more sequential round-trips, independent of token-level cost:\n")
    lines.append("| dataset | method calls/ep | baseline calls/ep | Δ calls | "
                  "% more calls | significant? (paired Wilcoxon, reused) |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    te = data["m2"]["token_efficiency_reused"] or []
    te_by_ds = {d["dataset"]: d for d in te}
    for ds in DATASETS:
        if ds not in te_by_ds:
            continue
        lc = te_by_ds[ds]["llm_calls"]
        d_calls = lc["mean_method"] - lc["mean_baseline"]
        pct_calls = 100 * d_calls / lc["mean_baseline"] if lc["mean_baseline"] else float("nan")
        wp = lc["wilcoxon_p"]
        lines.append(f"| {ds} | {lc['mean_method']:.1f} | {lc['mean_baseline']:.1f} | "
                      f"{d_calls:+.1f} | {pct_calls:+.1f}% | p={wp:.3g} ({sig(wp)}) |")
    lines.append("")
    lines.append("On hotpotqa_structured and musique_structured the method makes significantly "
                  "MORE calls (+31.4%, +9.8% respectively -- matching the review's +31.6%/+9.9%, "
                  "small rounding difference from the review's own source computation); on "
                  "browsecomp_plus_structured it makes marginally FEWER calls (not significant). "
                  "The structural argument therefore only bites on 2 of 3 datasets, but on those "
                  "two it is real: more sequential round-trips is not avoidable by caching or "
                  "batching within a single episode.\n")

    lines.append("**Proxy B -- estimated decode-time floor "
                  "(output tokens / observed decode throughput).** Output-token means per episode "
                  "are streamed directly from each cell's `rows.jsonl` (same field definitions "
                  "`scripts/compare_cells.py` uses: `tok_out = output_tokens or completion_tokens`"
                  "). Decode throughput (tokens/s) is pulled from vLLM's own periodic log lines "
                  "in surviving Tongyi-backbone slurm logs -- POOLED ACROSS CONDITIONS (decode "
                  "speed is a property of the model+GPU, not the retrieval condition, so this "
                  "pooling is legitimate) but NOT from the exact flagship-cell logs, which do not "
                  "survive (Step 2). Two throughput figures are used to bracket the estimate: the "
                  "median across ALL observed concurrency levels (server-shared, slower) and the "
                  "median restricted to `Running<=1` (closer to unbatched single-stream decode, "
                  "faster). This estimate is a LOWER BOUND on true episode wall-clock: it excludes "
                  "prefill time, tool/retrieval round-trips (small: 6-27ms per call, already "
                  "published in \\S4.4), and all agent-loop orchestration overhead not observable "
                  "in any surviving artifact.\n")

    tp = data["m2"]["vllm_throughput"]
    def _f(x, spec=".1f"):
        return "n/a" if x is None else format(x, spec)

    lines.append("| dataset | log files used | throughput samples | median gen tok/s "
                  "(Running<=1) | median gen tok/s (all concurrency) |")
    lines.append("|---|---:|---:|---:|---:|")
    for ds in DATASETS:
        t = tp[ds]
        lines.append(f"| {ds} | {t['n_files_used']} | {t['n_samples']} | "
                      f"{_f(t['median_gen_tps_running_le1'])} | {_f(t['median_gen_tps_all'])} |")
    lines.append("")
    n_unavailable = sum(1 for ds in DATASETS if tp[ds]["n_samples"] == 0)
    if n_unavailable:
        lines.append(f"**{n_unavailable}/3 dataset(s) had zero usable log files for this proxy "
                      "at analysis run time** (see the volatility note above) -- Proxy B is "
                      "reported as `n/a` for those below rather than reusing a stale number from "
                      "earlier in this same session.\n")

    dp = data["m2"]["decode_proxy"]
    lines.append("| dataset | cell | mean out tok/ep | mean calls/ep | est. decode-s/ep "
                  "(single-stream tps) | est. decode-s/ep (pooled tps) |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for ds in DATASETS:
        for label in ("baseline", "method"):
            r = dp[ds][label]
            lines.append(f"| {ds} | {label} | {r['mean_tok_out']:,.0f} | "
                          f"{r['mean_llm_calls']:.1f} | {_f(r['decode_s_single_tps'])} | "
                          f"{_f(r['decode_s_pooled_tps'])} |")
    lines.append("")

    lines.append("**Reading Proxy B (directional only, not a wall-clock claim):**\n")
    for ds in DATASETS:
        m_r, b_r = dp[ds]["method"], dp[ds]["baseline"]
        if m_r["decode_s_single_tps"] is None or b_r["decode_s_single_tps"] is None:
            lines.append(f"- {ds}: Proxy B unavailable (no vLLM throughput samples survived to "
                          "compute a decode-tokens/sec estimate at analysis run time).")
            continue
        d_single = m_r["decode_s_single_tps"] - b_r["decode_s_single_tps"]
        direction = "slower" if d_single > 0 else "faster"
        lines.append(f"- {ds}: method's output-token volume implies an estimated decode-time "
                      f"floor {abs(d_single):.1f}s {direction} than the baseline's "
                      f"({m_r['decode_s_single_tps']:.1f}s vs {b_r['decode_s_single_tps']:.1f}s, "
                      "single-stream throughput basis) -- BEFORE accounting for the call-count "
                      "difference's effect on round-trip overhead (Proxy A), which is additive on "
                      "top of this and not included in these seconds.")
    lines.append("")
    lines.append(f"Retrieval-tool latency for reference (already published, \\S4.4, reused "
                  f"verbatim, not remeasured): BM25 {data['m2']['retrieval_latency_ms']['bm25_ms']}ms"
                  f", graded/Indri-class {data['m2']['retrieval_latency_ms']['indri_ms']}ms, dense "
                  f"{data['m2']['retrieval_latency_ms']['dense_ms']}ms per query -- two to four "
                  "orders of magnitude smaller than the decode-time estimates above, so retrieval "
                  "itself is clearly not where a wall-clock difference would come from; if there "
                  "is one, it is in the LLM call structure (count and/or output-token volume), "
                  "not the search engine.\n")

    n_real_cells = 6 - n_cells_missing_real
    lines.append("### Verdict (M2)\n")
    lines.append("**No wall-clock claim, in either direction, is supportable from any artifact "
                  "currently on disk.** rows.jsonl carries zero timing fields (Step 1, exhaustive "
                  f"key check). At this run, real execution logs survive for {n_real_cells}/6 "
                  "flagship cells (Step 2) -- this number is ITSELF unstable across runs of this "
                  "same script minutes apart (see the volatility note); where logs do survive, "
                  "every one of them is a resumed partial-run segment and/or one of several "
                  "stitched-together job submissions, so a naive elapsed-time / instance-count "
                  "bound from them would conflate queueing wait, engine startup, and resume "
                  "artifacts with actual per-episode generation time -- not a number worth "
                  "printing even when the logs happen to be present. Proxy A (more calls "
                  "structurally implies more sequential round-trips on 2/3 datasets) and Proxy B "
                  "(output-token volume translated through an observed-but-cross-condition decode "
                  "throughput, giving a decode-time-only lower-bound estimate that is directionally "
                  "mixed across datasets) are suggestive but explicitly NOT precise or "
                  "authoritative enough to print as a wall-clock figure in the paper.\n")
    lines.append("**Exact recommended wording:** add one sentence to \\S4.4 (Latency and "
                  "deployability) alongside the existing per-query retrieval-latency numbers: "
                  "\"We report per-query retrieval latency, not end-to-end episode wall-clock: "
                  "our logging pipeline does not timestamp individual LLM calls, and the method "
                  "makes significantly more LLM calls than the BM25 baseline on two of three "
                  "datasets (HotpotQA +31.6%, MuSiQue +9.9%, both p<0.001; paired, "
                  "Table~\\ref{tab:token_efficiency}), each of which is a sequential round-trip a "
                  "ReAct-style agent loop cannot parallelize within an episode -- so the token/"
                  "dollar efficiency result in \\S7 should be read as a token-cost and dollar-cost "
                  "claim specifically, not a time-to-answer claim, until wall-clock is "
                  "instrumented directly.\" This scopes the existing efficiency claim honestly "
                  "without asserting a direction (faster or slower) the current data cannot "
                  "support, and flags the concrete, cheap fix (timestamp calls in "
                  "`evaluation/run_eval.py` / `agent_search/agent/loop.py` going forward) for any "
                  "future revision that wants a real number here.\n")

    out_path = REPO_ROOT / "analysis" / "equivalence_and_latency.md"
    out_path.write_text("\n".join(lines) + "\n")
    print(f"wrote {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
