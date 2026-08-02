"""ONE monitoring cycle for the running SLURM sweep (idempotent — safe to re-run every ~20min via
monitor_sweep.sh). Does NOT touch make_results_table.py / judge_browsecomp.py / agent_search / eval:
it only imports and calls their public entry points.

Per cycle:
  a) judge pass  — judge_dir() every runs/agent/browsecomp_plus_*/*/*/ dir that has results.json but
     no judge_summary.json yet (never re-judges, never touches partial dirs — costs real API $).
  b) regenerate docs/results.md via scripts.make_results_table.main()
  c) splice a "## Live status (auto-updated)" section at the top (after the H1 + intro paragraph),
     delimited by LIVE-STATUS-BEGIN/END html comments so re-running replaces it instead of stacking.
  d) if the sweep queue is empty AND every known condition dir has results.json: stamp SWEEP COMPLETE
     and drop the runs/_monitor/DONE sentinel (monitor_sweep.sh watches for this to stop looping).

State (previous row counts, cached ablation EM) lives in runs/_monitor/state.json so cycles are cheap
and stalled-detection works across restarts of the daemon.
"""
import collections
import datetime
import glob
import json
import os
import re
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO)
if REPO not in sys.path:
    sys.path.insert(0, REPO)

MONITOR_DIR = os.path.join(REPO, "runs", "_monitor")
STATE_PATH = os.path.join(MONITOR_DIR, "state.json")
DONE_PATH = os.path.join(MONITOR_DIR, "DONE")
RESULTS_MD = os.path.join(REPO, "docs", "results.md")

TARGETS = {"browsecomp_plus": 830, "musique": 2409, "hotpotqa": 7405, "2wiki": 12576}
LIVE_BEGIN = "<!-- LIVE-STATUS-BEGIN -->"
LIVE_END = "<!-- LIVE-STATUS-END -->"


# ---------------------------------------------------------------------------
# state
# ---------------------------------------------------------------------------
def load_state():
    if os.path.exists(STATE_PATH):
        try:
            return json.load(open(STATE_PATH))
        except Exception:
            pass
    return {}


def save_state(state):
    os.makedirs(MONITOR_DIR, exist_ok=True)
    tmp = STATE_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp, STATE_PATH)


# ---------------------------------------------------------------------------
# a) judge pass
# ---------------------------------------------------------------------------
def run_judge_pass():
    candidates = sorted({
        os.path.dirname(f) for f in glob.glob("runs/agent/browsecomp_plus_*/*/*/rows.jsonl")
        if os.path.exists(os.path.join(os.path.dirname(f), "results.json"))
        and not os.path.exists(os.path.join(os.path.dirname(f), "judge_summary.json"))
    })
    if not candidates:
        print("[judge] no eligible dirs (complete + unjudged) — skipping")
        return []
    try:
        from scripts.judge_browsecomp import judge_dir
        from evaluation.llm_judge import make_judge
        gen = make_judge("gpt-4o-mini")
    except Exception as e:
        print(f"[judge] could not set up judge (import/env issue), skipping this cycle: {e}")
        return []
    out = []
    for rd in candidates:
        try:
            s = judge_dir(rd, gen)
            out.append((rd, s))
            print(f"[judge] {rd}: {100 * s['judge_accuracy']:.1f}% ({s['n_correct']}/{s['n_judged']})")
        except Exception as e:
            print(f"[judge] FAILED {rd}: {e}")
    return out


# ---------------------------------------------------------------------------
# b) regenerate table
# ---------------------------------------------------------------------------
def regen_table():
    from scripts.make_results_table import main as make_table_main
    make_table_main()


# ---------------------------------------------------------------------------
# squeue
# ---------------------------------------------------------------------------
def squeue_summary():
    try:
        out = subprocess.run(
            ["squeue", "-u", os.environ.get("USER", ""), "-r", "-h", "-o", "%j|%t|%q|%r"],
            capture_output=True, text=True, timeout=30,
        ).stdout
    except Exception as e:
        return {"running": 0, "pending": 0, "throttled": 0, "error": str(e)}
    running = pending = throttled = 0
    for line in out.splitlines():
        parts = line.split("|")
        if len(parts) < 4:
            continue
        name, st, _qos, reason = parts[0], parts[1], parts[2], parts[3]
        if "as-suite" not in name:      # excludes the unrelated qwen25 job and anything else
            continue
        if st == "R":
            running += 1
        elif st == "PD":
            pending += 1
        if "maxcpu" in reason.lower():
            throttled += 1
    return {"running": running, "pending": pending, "throttled": throttled}


# ---------------------------------------------------------------------------
# condition scan (single pass per rows.jsonl: full line count + last-200 parsed rows)
# ---------------------------------------------------------------------------
def read_condition(path):
    dq = collections.deque(maxlen=200)
    n = 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            n += 1
            dq.append(line)
    rows200 = []
    for line in dq:
        try:
            rows200.append(json.loads(line))
        except Exception:
            pass
    return n, rows200


def compute_health(cond, rows200, raw_fn, extract_fn):
    if not rows200:
        return {}
    empty = 0
    search_total = 0
    search_zero = 0
    is_research_family = "bm25" not in cond   # BQL field-tagged `search` action (vs bm25_search)
    for r in rows200:
        raw = raw_fn(r)
        ans = extract_fn(raw or (r.get("final_answer") or ""))
        if not (ans or "").strip():
            empty += 1
        if is_research_family:
            for s in (r.get("trajectory") or []):
                if isinstance(s, dict) and s.get("action") == "search":
                    search_total += 1
                    if "0 exact matches" in (s.get("observation") or ""):
                        search_zero += 1
    n = len(rows200)
    flags = {"empty_rate": empty / n, "empty_flag": (empty / n) > 0.02}
    if search_total:
        flags["fallback_share"] = search_zero / search_total
    return flags


def scan_conditions(prev_counts, sweep_active):
    from scripts.make_results_table import _raw
    from evaluation.doc_scoring import extract_answer_span

    conditions = {}
    new_counts = {}
    for f in sorted(glob.glob("runs/agent/*/*/*/rows.jsonl")):
        parts = f.split("/")
        ds, cond = parts[2], parts[4].replace("agent_", "")
        rd = os.path.dirname(f)
        n, rows200 = read_condition(f)
        base = ds.replace("_structured", "").replace("_flat", "")
        tgt = TARGETS.get(base, n)
        done = os.path.exists(os.path.join(rd, "results.json"))
        key = f"{ds}/{cond}"
        new_counts[key] = n
        stalled = (not done) and sweep_active and prev_counts.get(key) == n
        flags = compute_health(cond, rows200, _raw, extract_answer_span)
        conditions.setdefault(ds, {})[cond] = dict(
            n=n, tgt=tgt, done=done, pct=min(100, 100 * n / max(tgt, 1)),
            stalled=stalled, flags=flags,
        )
    return conditions, new_counts


# ---------------------------------------------------------------------------
# ablation note (cheap: cached in state once a dir's row count stops changing)
# ---------------------------------------------------------------------------
def build_ablation(cache):
    tops = sorted(glob.glob("runs_old/ablation_50steps_*"))
    if not tops:
        return []
    lines = []
    for top in tops:
        for sub in sorted(glob.glob(os.path.join(top, "*"))):
            if not os.path.isdir(sub):
                continue
            parts = os.path.basename(sub).split("__")
            if len(parts) < 3:
                continue
            ds, _model, cond_raw = parts[0], parts[1], parts[2]
            cond = cond_raw.replace("agent_", "")
            base = ds.replace("_structured", "").replace("_flat", "")
            if base not in ("browsecomp_plus", "musique") or cond not in ("research", "research_bm25"):
                continue
            rows_path = os.path.join(sub, "rows.jsonl")
            if not os.path.exists(rows_path):
                continue

            js_path = os.path.join(sub, "judge_summary.json")
            if os.path.exists(js_path):
                try:
                    js = json.load(open(js_path))
                    lines.append(f"- `{ds}/{cond}`: n={js.get('n_judged', 0)}, "
                                 f"judge={100 * js.get('judge_accuracy', 0):.1f}%")
                    continue
                except Exception:
                    pass  # fall through to EM path below

            cur_n = sum(1 for l in open(rows_path) if l.strip())
            cached = cache.get(sub)
            if cached and cached.get("n") == cur_n:
                lines.append(f"- `{ds}/{cond}`: n={cached['n']}, EM={cached['em']:.1f}%")
                continue
            try:
                from evaluation.doc_scoring import extract_answer_span
                from evaluation.metrics import answer_em
                from scripts.make_results_table import _raw
                rows = [json.loads(l) for l in open(rows_path) if l.strip()]
                preds = [extract_answer_span(_raw(r) or (r.get("final_answer") or "")) for r in rows]
                golds = [r.get("gold_answer") or "" for r in rows]
                em = 100 * sum(answer_em(p, g) for p, g in zip(preds, golds)) / len(rows) if rows else 0.0
                cache[sub] = {"n": len(rows), "em": em}
                lines.append(f"- `{ds}/{cond}`: n={len(rows)}, EM={em:.1f}%")
            except Exception as e:
                lines.append(f"- `{ds}/{cond}`: (error computing EM: {e})")
    return lines


# ---------------------------------------------------------------------------
# render + splice live-status section
# ---------------------------------------------------------------------------
def bar(pct, width=10):
    filled = int(round(pct / 100 * width))
    filled = max(0, min(width, filled))
    return "█" * filled + "░" * (width - filled)


def build_live_status(queue, conditions, ablation_lines, now_str, sweep_done):
    out = [LIVE_BEGIN, "## Live status (auto-updated)", ""]
    header = "**SWEEP COMPLETE** — " if sweep_done else ""
    out.append(f"{header}Last refreshed: {now_str} (auto-refreshes ~20min while the sweep runs)")
    out.append("")
    qline = f"Queue: {queue.get('running', 0)} running / {queue.get('pending', 0)} pending `as-suite` tasks"
    if queue.get("throttled"):
        qline += f", {queue['throttled']} throttled (MaxCpu)"
    if queue.get("error"):
        qline += f" (squeue error: {queue['error']})"
    out.append(qline)
    out.append("")
    out.append("**Progress:**")
    out.append("")
    flagged_empty = []
    flagged_fallback = []
    for ds in sorted(conditions):
        for cond in sorted(conditions[ds]):
            c = conditions[ds][cond]
            mark = ""
            if c["done"]:
                mark = " ✅ done"
            elif c["stalled"]:
                mark = " ⚠ stalled?"
            out.append(f"- `{ds}/{cond}`: {bar(c['pct'])} {c['n']}/{c['tgt']} ({c['pct']:.0f}%){mark}")
            fl = c.get("flags") or {}
            if fl.get("empty_flag"):
                flagged_empty.append((ds, cond, fl["empty_rate"]))
            if "fallback_share" in fl:
                flagged_fallback.append((ds, cond, fl["fallback_share"]))
    out.append("")
    if flagged_empty:
        out.append("**Health flags:**")
        for ds, cond, rate in flagged_empty:
            out.append(f"- ⚠ `{ds}/{cond}`: empty-answer rate {100 * rate:.1f}% (last 200 rows, >2% threshold)")
        out.append("")
    if flagged_fallback:
        out.append("**Soft-fallback firing (informational, research arms, last 200 rows):**")
        for ds, cond, share in flagged_fallback:
            out.append(f"- `{ds}/{cond}`: {100 * share:.0f}% of `search` calls returned 0 exact matches")
        out.append("")
    if ablation_lines:
        out.append("**Ablation (@50, pre-fix):**")
        out.extend(ablation_lines)
        out.append("")
    out.append(LIVE_END)
    return "\n".join(out)


def splice_live_status(md_text, live_block):
    # strip any previous block first (idempotent re-run)
    stripped = re.sub(re.escape(LIVE_BEGIN) + r".*?" + re.escape(LIVE_END) + r"\n?", "",
                      md_text, flags=re.DOTALL)
    lines = stripped.split("\n")
    try:
        i1 = lines.index("", 1)          # blank line ending the H1
    except ValueError:
        i1 = min(1, len(lines))
    try:
        i2 = lines.index("", i1 + 1)     # blank line ending the intro paragraph
    except ValueError:
        i2 = i1 + 1
    insert_at = i2 + 1
    new_lines = lines[:insert_at] + [live_block, ""] + lines[insert_at:]
    return "\n".join(new_lines)


# ---------------------------------------------------------------------------
def is_sweep_done(queue, conditions):
    if (queue.get("running", 0) + queue.get("pending", 0)) > 0:
        return False
    if not conditions:
        return False
    return all(c["done"] for ds in conditions for c in conditions[ds].values())


def main():
    os.makedirs(MONITOR_DIR, exist_ok=True)
    state = load_state()
    prev_counts = state.get("prev_counts", {})
    ablation_cache = state.get("ablation_cache", {})

    run_judge_pass()
    regen_table()

    queue = squeue_summary()
    sweep_active = (queue.get("running", 0) + queue.get("pending", 0)) > 0

    conditions, new_counts = scan_conditions(prev_counts, sweep_active)
    ablation_lines = build_ablation(ablation_cache)

    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    sweep_done = is_sweep_done(queue, conditions)
    live_block = build_live_status(queue, conditions, ablation_lines, now_str, sweep_done)

    md = open(RESULTS_MD).read()
    md2 = splice_live_status(md, live_block)
    tmp = RESULTS_MD + ".tmp"
    with open(tmp, "w") as f:
        f.write(md2)
    os.replace(tmp, RESULTS_MD)

    state["prev_counts"] = new_counts
    state["ablation_cache"] = ablation_cache
    state["last_cycle"] = now_str
    save_state(state)

    if sweep_done:
        with open(DONE_PATH, "w") as f:
            f.write(f"sweep complete at {now_str}\n")

    ndone = sum(1 for ds in conditions for c in conditions[ds].values() if c["done"])
    ntot = sum(len(v) for v in conditions.values())
    total_rows = sum(c["n"] for ds in conditions for c in conditions[ds].values())
    tail = " SWEEP COMPLETE" if sweep_done else ""
    print(f"[{now_str}] cycle done: {ndone}/{ntot} conditions complete, {total_rows} rows total.{tail}")


if __name__ == "__main__":
    main()
