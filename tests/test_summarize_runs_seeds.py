"""Cross-seed aggregation in scripts/summarize_runs.py.

Multi-seed variance runs land as several run dirs that differ only by a `seed=<N>` segment.
summarize_runs must collapse those into ONE displayed row (seed-averaged metric + a run-to-run
±std band) and pool the seeds correctly for the paired t-test, while leaving single-seed runs
rendering exactly as before.
"""
import importlib.util
import json
import os

_SPEC = importlib.util.spec_from_file_location(
    "summarize_runs",
    os.path.join(os.path.dirname(__file__), "..", "scripts", "summarize_runs.py"),
)
sr = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(sr)


def _write_run(root, name, metrics, rows):
    d = os.path.join(root, name)
    os.makedirs(d)
    json.dump({"n": len(rows), "n_skipped": 0, "metrics": metrics},
              open(os.path.join(d, "results.json"), "w"))
    with open(os.path.join(d, "rows.jsonl"), "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


def test_parse_label_extracts_seed():
    sysname, ds, lvl, steps, model, seed = sr._parse_label(
        "swebench_verified__agent_tools__function__steps=6__seed=2")
    assert (sysname, ds, lvl, steps, seed) == ("agent_tools", "verified", "func", "6", "2")
    # unseeded run -> empty seed string (single run, no aggregation)
    assert sr._parse_label("swebench_verified__agent_tools__function__steps=6")[5] == ""


def test_three_seeds_collapse_to_one_row_with_band(tmp_path):
    root = str(tmp_path)
    recalls = [0.30, 0.40, 0.50]                  # mean 0.40, nonzero spread -> a band
    for seed, rec in enumerate(recalls):
        rows = [{"instance_id": f"i{i}", "recall@10": rec, "acc@10": 1.0 if rec > 0.35 else 0.0}
                for i in range(20)]
        _write_run(root, f"swebench_verified__agent_tools__function__steps=6__seed={seed}",
                   {"recall@10": rec, "acc@10": 1.0 if rec > 0.35 else 0.0}, rows)
    rows = sr.collect(root)
    assert len(rows) == 1                         # three seed dirs -> ONE aggregated condition
    r = rows[0]
    assert r["n_seeds"] == 3
    assert abs(r["recall@10"] - 0.40) < 1e-9      # seed mean
    assert abs(r["_std"]["recall@10"] - sr._std(recalls)) < 1e-9
    assert "±" in sr._fmt(r["recall@10"], "recall@10", r["_std"].get("recall@10"))


def test_single_seed_unchanged(tmp_path):
    root = str(tmp_path)
    rows = [{"instance_id": f"i{i}", "recall@10": 0.4} for i in range(10)]
    _write_run(root, "swebench_verified__agent_tools__function__steps=6",
               {"recall@10": 0.4}, rows)
    (r,) = sr.collect(root)
    assert r["n_seeds"] == 1
    assert r["_std"] == {}
    assert "±" not in sr._fmt(r["recall@10"], "recall@10", None)


def test_paired_ttest_pools_seeds(tmp_path):
    """The t-test averages each instance across its arm's seeds, then pairs the two arms;
    n in the header is the instance count (not seeds*instances)."""
    root = str(tmp_path)
    for seed in range(3):
        for system, val in [("agent_tools", 0.3), ("agent_tools_bql", 0.5)]:
            rows = [{"instance_id": f"i{i}", "recall@10": val} for i in range(15)]
            _write_run(root, f"swebench_verified__{system}__function__steps=6__seed={seed}",
                       {"recall@10": val}, rows)
    rows = sr.collect(root)
    grp = [r for r in rows if r["dataset"] == "verified"]
    bql = next(r for r in grp if r["system"] == "agent_tools_bql")
    base = next(r for r in grp if r["system"] == "agent_tools")
    avg_t = sr._per_seed_avg(bql, "recall@10")
    avg_b = sr._per_seed_avg(base, "recall@10")
    assert len(avg_t) == 15 and len(avg_b) == 15        # instance-level, seeds collapsed
    common = [i for i in avg_t if i in avg_b]
    md, _p, n = sr._paired_ttest([avg_b[i] for i in common], [avg_t[i] for i in common])
    assert n == 15 and abs(md - 0.2) < 1e-9
