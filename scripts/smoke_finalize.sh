#!/usr/bin/env bash
# Finalizer: waits for both smokes (main + extra) to finish, then (1) runs the BrowseComp
# LLM-judge pass over its run dirs (its PAPER metric), and (2) writes a consolidated table to
# runs/_rebuild_smoke/SUMMARY.txt so the go/no-go read is ready without manual assembly.
set -uo pipefail
cd "$(dirname "$0")/.."

RUNS_DIR="${RUNS_DIR:-runs/_rebuild_smoke}"
LOGDIR="$RUNS_DIR/logs"
PY="${PY:-envs/bin/python}"
EXTRA_PID="${EXTRA_PID:-}"
OUT="$RUNS_DIR/SUMMARY.txt"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export PYTHONWARNINGS="${PYTHONWARNINGS:-ignore}"
ts() { date +%H:%M:%S; }

echo "[$(ts)] finalizer waiting for extra-smoke (pid=${EXTRA_PID:-n/a}) ..."
while true; do
  if [ -n "$EXTRA_PID" ]; then kill -0 "$EXTRA_PID" 2>/dev/null || break
  else grep -q "EXTRA SMOKE COMPLETE" "$LOGDIR/DRIVER_extra.log" 2>/dev/null && break; fi
  sleep 30
done
echo "[$(ts)] both smokes done — running BrowseComp judge + building summary"

# --- (1) BrowseComp LLM-judge over each browsecomp run dir --------------------------------
for f in $(find "$RUNS_DIR/agent/browsecomp_plus" -name rows.jsonl 2>/dev/null); do
  rd=$(dirname "$f")
  echo "[$(ts)] judge -> $rd"
  "$PY" -m evaluation.llm_judge --results-dir "$rd" --judge-model gpt-4o-mini \
    --dataset browsecomp_plus >"$LOGDIR/judge__$(basename "$(dirname "$rd")")_$(basename "$rd").log" 2>&1 || true
done

# --- (2) consolidated table ---------------------------------------------------------------
"$PY" - "$RUNS_DIR" > "$OUT" 2>&1 <<'PY'
import json, os, sys, statistics, glob
runs = sys.argv[1]
def med(a): return int(statistics.median(a)) if a else 0
rows_out = []
for f in sorted(glob.glob(os.path.join(runs, "agent", "*", "*", "*", "results.json"))):
    r = json.load(open(f)); m = r.get("metrics", {})
    p = f.split("/"); ds = p[p.index("agent")+1]; cond = p[-2]
    d = os.path.dirname(f)
    rr = [json.loads(l) for l in open(os.path.join(d, "rows.jsonl"))] if os.path.exists(os.path.join(d,"rows.jsonl")) else []
    toks = [x.get("total_tokens_once") for x in rr if x.get("total_tokens_once")]
    # support_f1 (MuSiQue paper metric): inline on future runs; computed post-hoc here from the
    # gold_ids + surfaced_docs already in every row, in case rows predate the run_eval wiring fix.
    sup_posthoc = None
    if m.get("support_f1") is None:
        from evaluation.metrics import support_f1 as _sf1
        sv = [_sf1(x.get("surfaced_docs") or [], set(x.get("gold_ids") or []))
              for x in rr if x.get("gold_ids")]
        if sv: sup_posthoc = round(sum(sv)/len(sv), 3)
    # judge (browsecomp)
    js = os.path.join(d, "judge_summary.json")
    judge = ""
    if os.path.exists(js):
        try: judge = f"{json.load(open(js)).get('judge_accuracy', json.load(open(js)).get('accuracy','')):.3f}"
        except Exception: judge = "?"
    # resolve (code)
    resolve = ""
    for rp in ("resolve_summary.json","apptainer_report.json"):
        pth = os.path.join(d, rp)
        if os.path.exists(pth):
            try:
                jr = json.load(open(pth))
                resolve = f"{jr.get('resolved_rate', jr.get('resolve_rate','')):.3f}" if isinstance(jr.get('resolved_rate',jr.get('resolve_rate')),(int,float)) else str(jr.get('resolved',''))
            except Exception: resolve = "?"
    def g(k,n=3):
        v=m.get(k); return round(v,n) if isinstance(v,(int,float)) else "-"
    sup = g("support_f1") if m.get("support_f1") is not None else (sup_posthoc if sup_posthoc is not None else "-")
    rows_out.append((ds,cond,r.get("n"),r.get("n_errors"),g("answer_em"),g("answer_f1"),
                     g("grounded_em"),sup,int(m.get("total_tokens_once",0)),
                     round(m.get("n_steps",0),1),judge,resolve))
hdr = ("dataset","method","n","err","EM","F1","gEM","supF1","tok_once","steps","judge","resolve")
w = [16,24,3,3,5,5,5,5,8,5,6,7]
def line(t): return " ".join(str(x).ljust(w[i]) for i,x in enumerate(t))
print("=== REBUILD SMOKE SUMMARY ===")
print(line(hdr))
last=None
for row in rows_out:
    if last and row[0]!=last: print()
    print(line(row)); last=row[0]
PY
echo "[$(ts)] FINALIZE COMPLETE -> $OUT"
cat "$OUT"
