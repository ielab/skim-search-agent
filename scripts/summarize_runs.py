#!/usr/bin/env python3
"""Compare runs, one table per directly-comparable surface (dataset x level).

Different datasets or localization levels are not comparable, so each gets its own
titled table (floors then agents; the step budget is a column, so a 10-vs-50 ablation
stays in the same table). Deep-research/document surfaces also show answer EM/F1.

Under each table, a PAIRED t-test compares every additive pair `X_bql` vs `X` (same model
& step budget) instance-by-instance, so you can tell a real +bql effect from variance.
It reads each run's rows.jsonl; the p-value is exact (Student's t via the regularized
incomplete beta function) and needs no scipy (scipy is used only as an optional fast path).
NOTE: acc@k is binary per instance, so its t-test is approximate (McNemar is the exact test);
the continuous metrics (recall/MAP/nDCG/MRR) are the ones the t-test is meant for. And this
only addresses INSTANCE variance, run multiple seeds to also bound LLM-sampling variance.

  python scripts/summarize_runs.py                 # readable tables + paired t-test
  python scripts/summarize_runs.py --no-sig        # tables only (no significance block)
  python scripts/summarize_runs.py --full          # every metric column
  python scripts/summarize_runs.py --csv > t.csv   # flat full export for the paper
  python scripts/summarize_runs.py --dir runs/agent # only agent runs
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys

# metrics that are a cost/diagnostic, not a quality score, skip them in the significance test
_SIG_SKIP = {"llm_calls", "prompt_tokens", "completion_tokens", "set_size"}

# headline quality metrics that get a run-to-run (seed) variance band shown inline as `mean±std`
_BAND_METRICS = {"acc@10", "recall@10"}


def _load_per_instance(dirpath: str) -> dict:
    """rows.jsonl -> {instance_id: {metric: value}} (numeric scalars only; skipped rows dropped).
    Needed for the PAIRED t-test, which compares the two arms instance-by-instance."""
    path = os.path.join(dirpath, "rows.jsonl")
    per: dict = {}
    if not os.path.isfile(path):
        return per
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            iid = d.get("instance_id")
            if iid is None or d.get("skipped"):
                continue
            per[iid] = {k: v for k, v in d.items()
                        if isinstance(v, (int, float)) and not isinstance(v, bool)}
    return per


def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the incomplete beta function (Numerical Recipes `betacf`,
    evaluated by the modified Lentz algorithm)."""
    MAXIT, EPS, FPMIN = 200, 3.0e-16, 1.0e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < FPMIN:
        d = FPMIN
    d = 1.0 / d
    h = d
    for m in range(1, MAXIT + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < EPS:
            break
    return h


def _betai(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta function I_x(a, b) (Numerical Recipes `betai`)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    bt = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
                  + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):             # use the faster-converging tail
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def _paired_ttest(base: list, treat: list) -> tuple:
    """Two-sided paired t-test on aligned per-instance values -> (mean_diff, p, n).

    The p-value is exact (Student's t, df = n-1): p = I_x(df/2, 1/2) with x = df/(df + t^2),
    where I_x is the regularized incomplete beta function (`_betai`). scipy.stats.t.sf is used
    only as an optional fast path when importable, it agrees with the pure-python path to <1e-9."""
    diffs = [t - b for b, t in zip(base, treat)]
    n = len(diffs)
    if n < 2:
        return (diffs[0] if diffs else 0.0, float("nan"), n)
    mean = sum(diffs) / n
    var = sum((d - mean) ** 2 for d in diffs) / (n - 1)
    if var <= 0:                                  # all diffs identical
        return (mean, (0.0 if mean != 0 else float("nan")), n)
    t = mean / math.sqrt(var / n)
    df = n - 1
    try:
        from scipy import stats                   # optional fast path
        p = float(stats.t.sf(abs(t), df=df) * 2)
    except Exception:
        p = _betai(df / 2.0, 0.5, df / (df + t * t))   # exact Student's t two-sided tail
    return (mean, p, n)


def _stars(p: float) -> str:
    if p != p:                                    # NaN
        return ""
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""


def _fmt_p(p: float) -> str:
    if p != p:
        return "  n/a"
    return "<.001" if p < 0.001 else f"{p:.3f}".lstrip("0")

# curated default columns (readable), only the high-signal, non-redundant ones;
# --full / --csv still show everything in ALL_METRICS. Dropped as redundant: acc@5
# (keep the @10 headline), hit@10 (≈acc@10), set_recall (≈recall@10), map@10 & mrr@10
# (ndcg@10 already captures ranking quality). Kept:
#   recall@10, headline localization quality
#   set_size, set_precision, what/how-clean the agent actually SUBMITS (the yield story)
#   ndcg@10, ranking quality of that set
#   llm_calls, turns. Then report token cost BY TYPE, never as one total: prompt_tokens
#   (input, cacheable/cheap) vs completion_tokens (output, not cacheable) vs read_tokens (how much
#   the agent actually READ, the "fetch a part vs read the whole" access axis). A tool carrying a
#   skill manual looks input-heavy but that input is cache-amortized; READ + output are the honest cost.
DISPLAY = ["acc@10", "recall@10", "ndcg@10",
           "set_size", "set_precision",
           "llm_calls", "total_tokens_once", "retrieved_doc_tokens", "output_tokens", "reasoning_tokens"]
# COUNT-ONCE token columns replace the raw cumulative ones: the per-step prompt_tokens re-sends
# the whole growing (cache-reused) context every turn, so its sum triple-counts the initial prompt
# and every earlier observation, making the method look expensive precisely where it is cheap.
# total_tokens_once = initial_prompt + retrieved_doc + output, each counted once (see run_eval).
_TOK_COLS = {"llm_calls", "prompt_tokens", "completion_tokens", "total_tokens_once",
             "initial_prompt_tokens", "retrieved_doc_tokens", "output_tokens", "reasoning_tokens",
             "read_tokens", "cached_input_tokens"}
ALL_METRICS = ["acc@1", "acc@3", "acc@5", "acc@10",
               "hit@1", "hit@5", "hit@10", "recall@5", "recall@10",
               "precision@5", "f1@5", "set_recall", "set_precision", "set_f1",
               "set_size", "map@10", "mrr@10", "ndcg@10",
               "answer_em", "answer_f1", "support_f1", "fix_file_ok", "timeout_rate", "llm_calls",
               "initial_prompt_tokens", "retrieved_doc_tokens", "output_tokens",
               "reasoning_tokens", "total_tokens_once",
               # raw cumulative tokens kept for reference (billing reality), not the comparison metric:
               "prompt_tokens", "cached_input_tokens", "completion_tokens", "read_tokens"]

_DATASET = {"swebench_verified": "verified", "swebench_lite": "lite",
            "loc_bench": "locbench"}


def _parse_label(rel_path: str) -> tuple[str, str, str, str, str, str]:
    """run dir -> (system, dataset, level, steps, model, seed). The dir name keeps a `model=`
    segment when the run set a model; we surface it (a model swap is a different
    experiment, not a duplicate) rather than dropping it as before. A `seed=<N>` segment
    (multi-seed variance runs) is parsed out too so the same condition's seed dirs can be
    aggregated together, '' means an unseeded (single) run. k=/dense= stay noise."""
    name = os.path.basename(rel_path.rstrip("/"))
    parts = name.split("__")
    dataset = _DATASET.get(parts[0], parts[0]) if parts else "?"
    system = parts[1] if len(parts) > 1 else "?"
    level = "func" if "function" in parts else ("file" if "file" in parts else "-")
    steps = next((p.split("=", 1)[1] for p in parts if p.startswith("steps=")), None)
    if steps is None:                                  # legacy agent runs predate
        steps = "6" if system.startswith("agent") else "-"   # the explicit suffix
    model = next((p.split("=", 1)[1] for p in parts if p.startswith("model=")), "")
    seed = next((p.split("=", 1)[1] for p in parts if p.startswith("seed=")), "")
    return system, dataset, level, steps, model, seed


def _short_model(m: str) -> str:
    """Compact, distinguishing model tag (last 3 name segments): a full id like
    Alibaba-NLP/Tongyi-DeepResearch-30B-A3B -> DeepResearch-30B-A3B."""
    return "-".join(m.replace("/", "-").split("-")[-3:]) if m else "-"


def _mean(xs: list) -> float:
    return sum(xs) / len(xs)


def _std(xs: list) -> float:
    """Sample (n-1) standard deviation; 0.0 for a single value."""
    if len(xs) < 2:
        return 0.0
    mu = _mean(xs)
    return math.sqrt(sum((x - mu) ** 2 for x in xs) / (len(xs) - 1))


def _load_result_head(path: str) -> dict:
    """Read n / n_skipped / level / metrics from results.json WITHOUT parsing the (multi-GB)
    embedded `rows` list, results.json redundantly stores a full copy of rows.jsonl, so a plain
    json.load would parse gigabytes just to reach `metrics`. `metrics` is emitted before `rows`
    (run_eval), so we stop at the `"rows":` line and close the object."""
    buf, hit_rows = [], False
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.lstrip().startswith('"rows"'):
                hit_rows = True
                break
            buf.append(line)
    txt = "".join(buf).rstrip()
    if hit_rows:                            # bloated file: stopped before rows -> close the outer object
        txt = txt.rstrip(",") + "}"
    return json.loads(txt)                  # slim file: buf is already the whole valid object


def collect(runs_dir: str, need_per: bool = True) -> list[dict]:
    # need_per=False skips reading the (multi-GB) rows.jsonl per condition, the per-instance
    # values are only used by the paired t-test, so a --no-sig run reads results.json alone (fast).
    raw = []
    for dirpath, _d, files in os.walk(runs_dir):
        if "results.json" not in files:
            continue
        try:
            res = _load_result_head(os.path.join(dirpath, "results.json"))
        except (OSError, json.JSONDecodeError, ValueError):
            continue
        # config.json is authoritative for all run metadata (the nested dir path only carries
        # dataset/model/retriever; level/steps/seed live in config.json). Fall back to parsing
        # the old flat dir name for legacy runs that predate this convention.
        cfg = {}
        try:
            cfg = json.load(open(os.path.join(dirpath, "config.json")))
        except (OSError, json.JSONDecodeError):
            pass
        if cfg.get("retriever"):
            system = cfg["retriever"]
            dataset = _DATASET.get(cfg.get("dataset", ""), cfg.get("dataset", "?"))
            level = {"function": "func", "file": "file"}.get(cfg.get("level"), "-")
            steps = str(cfg["max_steps"]) if cfg.get("max_steps") is not None else "-"
            model = cfg.get("model") or ""
            seed = str(cfg["seed"]) if cfg.get("seed") is not None else ""
        else:
            system, dataset, level, steps, model, seed = _parse_label(
                os.path.relpath(dirpath, runs_dir))
        m = res.get("metrics", {})
        kind = "agent" if system.startswith("agent") else "retr"
        row = {"kind": kind, "system": system, "dataset": dataset, "lvl": level,
               "steps": steps, "model": model or "", "seed": seed, "n": res.get("n"),
               "skip": res.get("n_skipped")}
        row.update({k: m.get(k) for k in ALL_METRICS})
        # per-instance values are only for the paired t-test, skip the multi-GB rows.jsonl read
        # entirely when significance is off (the common case; also when no _bql pairs exist).
        row["_per"] = _load_per_instance(dirpath) if need_per else {}
        raw.append(row)
    out = _aggregate_seeds(raw)
    # Order within a surface: floors first, then agents grouped BY STEP BUDGET then MODEL
    # (so same-steps/same-model systems pair up, the additive agent_tools/agent_tools_bql
    # sit together, separated from a different step budget or a model-swap ablation).
    out.sort(key=lambda r: (r["dataset"], r["kind"] == "agent",
                            _steps_key(r["steps"]), r["model"], r["system"]))
    return out


def _aggregate_seeds(raw: list[dict]) -> list[dict]:
    """Collapse the seed dirs of one condition into a single displayed row.

    A condition is (dataset, lvl, system, steps, model), i.e. everything but `seed`. The
    displayed metric is the MEAN over seeds; for the headline metrics we also keep the
    std-over-seeds (the run-to-run variance band). For the paired t-test we keep every seed's
    per-instance table in `_per_seeds`, so the test can first average each instance across
    that condition's seeds and only then pair the two arms. Single-seed conditions come out
    with n_seeds=1, no std, and `_per` unchanged, identical to the pre-aggregation behavior."""
    groups: dict = {}
    for r in raw:
        key = (r["dataset"], r["lvl"], r["system"], r["steps"], r["model"])
        groups.setdefault(key, []).append(r)
    out = []
    for members in groups.values():
        members.sort(key=lambda r: r["seed"])
        base = dict(members[0])
        base["n_seeds"] = len(members)
        base["_std"] = {}
        if len(members) > 1:
            for k in ALL_METRICS:                       # mean (and std for headline) over seeds
                vals = [r[k] for r in members if isinstance(r.get(k), (int, float))]
                base[k] = _mean(vals) if vals else None
                if k in _BAND_METRICS and len(vals) > 1:
                    base["_std"][k] = _std(vals)
            base["n"] = max((r["n"] for r in members if r["n"] is not None), default=base["n"])
        base["_per_seeds"] = [r.get("_per") or {} for r in members]
        out.append(base)
    return out


def _steps_key(steps: str) -> int:
    """Numeric sort for the step budget; non-numeric (floors, '-') sort first."""
    return int(steps) if steps.isdigit() else -1


def _fmt(v, col: str, std: float | None = None) -> str:
    if v is None:
        return "-"
    if col == "set_size":
        return f"{v:.1f}"
    if col == "timeout_rate":
        return f"{100*v:.0f}%"
    if col in _TOK_COLS:
        return f"{v/1000:.1f}k" if v >= 1000 else f"{v:.1f}"
    if std and round(std, 3) > 0:                        # seed variance band on headline metrics
        return f"{v:.3f}±{std:.3f}"                       # (hide a band that rounds to ±.000)
    return f"{v:.3f}"


# One table per directly-comparable SURFACE = (dataset, level). Different datasets or
# levels are not comparable, so they get their own table rather than one mixed dump.
# Within a surface, rows differ only by system (floors then agents) and the step budget
# (shown as a column, a 10-vs-50 ablation lives in the same table, not a separate one).
_DS_ORDER = {"verified": 0, "lite": 1, "locbench": 2}     # code splits first, then the rest
_LVL_ORDER = {"func": 0, "file": 1, "-": 2}
_LVL_LABEL = {"func": "function-level", "file": "file-level", "-": "document retrieval"}


def _group_sort_key(g: tuple) -> tuple:
    ds, lvl = g
    return (_DS_ORDER.get(ds, 9), ds, _LVL_ORDER.get(lvl, 9), lvl)


def _per_seed_avg(row: dict, metric: str) -> dict:
    """{instance_id: mean-of-`metric`-across-this-condition's-seeds}.

    With one seed this is just that seed's per-instance value; with several it averages each
    instance's score over the seeds it appears in, so the paired t-test below tests INSTANCE
    variance on seed-averaged scores (run-to-run variance is reported separately as the band)."""
    seeds = row.get("_per_seeds") or [row.get("_per") or {}]
    acc: dict = {}
    for per in seeds:
        for iid, mvals in per.items():
            v = mvals.get(metric)
            if isinstance(v, (int, float)):
                acc.setdefault(iid, []).append(v)
    return {iid: _mean(vs) for iid, vs in acc.items()}


def _print_sig_block(grp: list, metrics: list) -> None:
    """For each ADDITIVE pair in this surface, a system `X_bql` and its base `X` at the same
    model+step budget, print a PAIRED t-test (per-instance, same instances both arms) so you
    can see whether the +bql delta is significant or just variance. With multiple seeds, each
    instance's score is first averaged across that arm's seeds, then the two arms are paired."""
    by_key = {(r["system"], r["steps"], r["model"]): r for r in grp}
    test_metrics = [c for c in metrics if c not in _SIG_SKIP]
    header_printed = False
    for r in grp:
        if not r["system"].endswith("_bql"):
            continue
        base = by_key.get((r["system"][:-4], r["steps"], r["model"]))
        if base is None:
            continue
        n_seeds = max(r.get("n_seeds", 1), base.get("n_seeds", 1))
        cells = []
        n_common = 0
        for c in test_metrics:
            avg_b, avg_t = _per_seed_avg(base, c), _per_seed_avg(r, c)
            common = [i for i in avg_t if i in avg_b]
            if len(common) < 2:
                continue
            md, p, _ = _paired_ttest([avg_b[i] for i in common], [avg_t[i] for i in common])
            n_common = max(n_common, len(common))
            cells.append(f"{c} {md:+.3f} p={_fmt_p(p)}{_stars(p)}")
        if not cells:
            continue
        if not header_printed:
            print("  paired t-test (treatment − base, same model & step budget; "
                  "* p<.05  ** p<.01  *** p<.001)")
            header_printed = True
        mdl = f", {_short_model(r['model'])}" if r["model"] else ""
        seedtag = f", n_seeds={n_seeds}" if n_seeds > 1 else ""
        print(f"    {r['system']} − {base['system']}  "
              f"[steps={r['steps']}{mdl}, n={n_common}{seedtag}]")
        for j in range(0, len(cells), 4):                # wrap 4 metrics per line
            print("        " + "    ".join(cells[j:j + 4]))


_METHOD = {"agent_research": "doc", "agent_codefix": "code"}   # THE method per arm
_DOC_BASE = ["agent_research", "agent_research_bm25", "agent_research_bm25_fetch",
             "agent_research_bm25_dci", "agent_research_dci"]
_DOC_LBL = {"agent_research": "METHOD", "agent_research_bm25": "bm25",
            "agent_research_bm25_fetch": "bm25_fetch", "agent_research_bm25_dci": "bm25_dci",
            "agent_research_dci": "dci"}


def _base_dataset(ds: str) -> tuple[str, str]:
    """('2wiki_structured') -> ('2wiki', 'structured'); ('verified') -> ('verified','-')."""
    for suf in ("_structured", "_flat"):
        if ds.endswith(suf):
            return ds[: -len(suf)], suf[1:]
    return ds, "-"


def _print_comparison(rows: list) -> None:
    """The headline SETUP comparisons: (1) method vs each baseline per doc dataset (cover-EM &
    token cost + the method's savings), (2) structured vs flat for the method, (3) code arm."""
    by = {(r["dataset"], r["system"]): r for r in rows}
    def g(ds, s, m):
        r = by.get((ds, s)); v = r.get(m) if r else None
        return v if isinstance(v, (int, float)) else None
    def f3(v): return f"{v:.3f}" if isinstance(v, (int, float)) else "  -  "
    def tk(v): return f"{v/1000:.1f}k" if isinstance(v, (int, float)) else " - "

    doc_ds = sorted({d for (d, s) in by if s == "agent_research"})
    if doc_ds:
        print("\n\n════════ COMPARISON: doc method vs baselines (EM / tok-per-query) ════════")
        print(f"  {'dataset':22s} " + " ".join(f"{_DOC_LBL[s]:>10s}" for s in _DOC_BASE) + "   | vs-bm25")
        for kind, mfn in (("EM", lambda ds, s: g(ds, s, "answer_em")),
                          ("tok/q", lambda ds, s: g(ds, s, "total_tokens_once"))):
            print(f"  -- {kind} --")
            for ds in doc_ds:
                cells = " ".join(f"{(f3(mfn(ds,s)) if kind=='EM' else tk(mfn(ds,s))):>10s}" for s in _DOC_BASE)
                m, b = mfn(ds, "agent_research"), mfn(ds, "agent_research_bm25")
                if kind == "tok/q" and m and b:
                    rel = f"{b/m:.1f}x cheaper"
                elif kind == "EM" and m is not None and b is not None:
                    rel = f"{m-b:+.3f}"
                else:
                    rel = "-"
                print(f"  {ds:22s} {cells}   | {rel}")

    bases = sorted({_base_dataset(d)[0] for (d, s) in by if s == "agent_research"})
    struct_flat = [b for b in bases if (f"{b}_structured", "agent_research") in by
                   and (f"{b}_flat", "agent_research") in by]
    if struct_flat:
        print("\n════════ COMPARISON: structured vs flat (method, agent_research) ════════")
        print(f"  {'dataset':12s} {'em_struct':>10s} {'em_flat':>9s} {'Δem':>7s}  {'tok_struct':>10s} {'tok_flat':>9s}")
        for b in struct_flat:
            cs, cf = g(f"{b}_structured", "agent_research", "answer_em"), g(f"{b}_flat", "agent_research", "answer_em")
            ts, tf = g(f"{b}_structured", "agent_research", "total_tokens_once"), g(f"{b}_flat", "agent_research", "total_tokens_once")
            d = f"{cs-cf:+.3f}" if (cs is not None and cf is not None) else "  -  "
            print(f"  {b:12s} {f3(cs):>10s} {f3(cf):>9s} {d:>7s}  {tk(ts):>10s} {tk(tf):>9s}")

    code_ds = sorted({d for (d, s) in by if s == "agent_codefix"})
    if code_ds:
        print("\n════════ COMPARISON: code method vs grep (fix_file_ok / tok) ════════")
        print(f"  {'dataset':12s} {'method_fix':>10s} {'grep_fix':>9s} {'Δ':>7s}  {'method_tok':>10s} {'grep_tok':>9s}")
        for ds in code_ds:
            mf, gf = g(ds, "agent_codefix", "fix_file_ok"), g(ds, "agent_codefix_grep", "fix_file_ok")
            mt, gt = g(ds, "agent_codefix", "total_tokens_once"), g(ds, "agent_codefix_grep", "total_tokens_once")
            d = f"{mf-gf:+.3f}" if (mf is not None and gf is not None) else "  -  "
            print(f"  {ds:12s} {f3(mf):>10s} {f3(gf):>9s} {d:>7s}  {tk(mt):>10s} {tk(gt):>9s}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="runs")
    ap.add_argument("--compare", action="store_true",
                    help="append cross-cutting comparisons: method vs baselines, structured vs flat")
    ap.add_argument("--full", action="store_true", help="all metric columns")
    ap.add_argument("--csv", action="store_true")
    ap.add_argument("--no-sig", action="store_true",
                    help="suppress the paired t-test (X_bql vs X) under each table")
    a = ap.parse_args()
    if not os.path.isdir(a.dir):
        print(f"no such dir: {a.dir}", file=sys.stderr)
        return 1
    rows = collect(a.dir, need_per=not a.no_sig)
    if not rows:
        print(f"no */results.json under {a.dir}", file=sys.stderr)
        return 1

    if a.csv:        # flat export: one row per CONDITION (seed-aggregated), with the seed band
        std_cols = [f"{c}_seed_std" for c in sorted(_BAND_METRICS)]
        cols = (["kind", "system", "model", "dataset", "lvl", "steps", "n_seeds", "n", "skip"]
                + ALL_METRICS + std_cols)
        w = csv.DictWriter(sys.stdout, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            out = dict(r, n_seeds=r.get("n_seeds", 1))
            for c in sorted(_BAND_METRICS):             # seed-level std for the headline metrics
                out[f"{c}_seed_std"] = r.get("_std", {}).get(c)
            w.writerow(out)
        return 0

    base_metrics = ALL_METRICS if a.full else DISPLAY
    sys_w = max(len("system"), max(len(r["system"]) for r in rows))   # aligned across tables
    # Show a model column whenever any agent run carries a model, you need to know which backbone
    # produced a row even with a single model present (and it's what tells two same-system/same-steps
    # rows apart when several models ran, e.g. Tongyi vs AgentWorld). Hidden only for floors-only
    # output (no agent rows), where model is always blank and would be pure noise.
    agent_models = {r["model"] for r in rows if r["kind"] == "agent" and r["model"]}
    show_model = len(agent_models) >= 1
    mdl_w = max([len("model")] + [len(_short_model(r["model"])) for r in rows]) if show_model else 0
    # Show a `seeds` column (and the inline ±std band) only when some condition aggregated more
    # than one seed, otherwise single-seed output is byte-for-byte the pre-aggregation layout.
    show_seeds = any(r.get("n_seeds", 1) > 1 for r in rows)
    groups = sorted({(r["dataset"], r["lvl"]) for r in rows}, key=_group_sort_key)

    for gi, (ds, lvl) in enumerate(groups):
        grp = [r for r in rows if r["dataset"] == ds and r["lvl"] == lvl]
        # answer EM/F1 only matter for the deep-research (document) surfaces, show those
        # columns only when a row in this surface actually carries them (keeps code tables lean).
        # Columns ordered by what you COMPARE on: TASK QUALITY -> timeout -> EFFICIENCY (tokens).
        # The retrieval @k columns (acc@k/recall@k/ndcg@k/set_*) are structurally 0 for the agentic
        # arms (they declare an answer/fix, not a ranked set), so they are dropped here and only
        # re-added for a surface that actually has a retrieval floor. --full still shows everything.
        if a.full:
            metrics = list(ALL_METRICS)
        else:
            metrics = []
            if any(r.get("answer_em") is not None for r in grp):             # DOC surface
                metrics += ["answer_em", "answer_f1"]                         # canonical QA (HotpotQA/2Wiki/MuSiQue)
                if any(r.get("support_f1") is not None for r in grp):        # MuSiQue paired metric
                    metrics += ["support_f1"]
            if any(r.get("fix_file_ok") is not None for r in grp):            # CODE surface
                metrics += ["fix_file_ok"]
            if any(r.get("timeout_rate") is not None for r in grp):
                metrics += ["timeout_rate"]
            metrics += ["llm_calls", "total_tokens_once", "retrieved_doc_tokens", "output_tokens"]
            if any(r["kind"] != "agent" and (r.get("recall@10") or 0) > 0 for r in grp):
                metrics = ["acc@10", "recall@10", "ndcg@10"] + metrics        # real retrieval floor present
        # Per-column width: a metric carrying a `mean±std` band needs a wider cell than 10        # size it to the widest rendered cell in this surface (so columns stay aligned).
        cw = {c: max(10, max((len(_fmt(r[c], c, r.get("_std", {}).get(c)))
                              for r in grp), default=10)) for c in metrics}
        mcol = f"  {'model':<{mdl_w}}" if show_model else ""
        scol = f"  {'seeds':>5}" if show_seeds else ""
        head = (f"{'kind':<5}  {'system':<{sys_w}}{mcol}  steps   n  skip{scol}  "
                + "  ".join(f"{c:>{cw[c]}}" for c in metrics))
        if gi:
            print()
        blocked_by = "step budget" + (" then model" if show_model else "")
        print(f"━━━ {ds} · {_LVL_LABEL.get(lvl, lvl)} "
              f"━━━ (floors then agents; rows blocked by {blocked_by})")
        print(head)
        print("-" * len(head))
        divider = "·" * len(head)                       # thin rule between blocks
        prev_block = None
        for r in grp:                                  # grp is pre-sorted: retr, then agents by steps,model
            block = (r["kind"], r["steps"], r["model"])  # one block per (floors | each step×model)
            if prev_block is not None and block != prev_block:
                print(divider)                          # separate floors / each step / each model
            prev_block = block
            mcell = f"  {_short_model(r['model']):<{mdl_w}}" if show_model else ""
            scell = f"  {r.get('n_seeds', 1):>5}" if show_seeds else ""
            std = r.get("_std", {})
            print(f"{r['kind']:<5}  {r['system']:<{sys_w}}{mcell}  {r['steps']:>4}  "
                  f"{str(r['n'] if r['n'] is not None else '-'):>4} "
                  f"{str(r['skip'] if r['skip'] is not None else '-'):>4}{scell}  "
                  + "  ".join(f"{_fmt(r[c], c, std.get(c)):>{cw[c]}}" for c in metrics))
        if not a.no_sig:
            _print_sig_block(grp, metrics)
    if a.compare:
        _print_comparison(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
