#!/usr/bin/env python
"""Adversarial validity audit of the structured/flat TWIN design and the corpus-vs-interface
decomposition it feeds (`analysis/flat_vs_structured.py`).

The reviewer's allegation: the "corpus step" of the decomposition compares the plain-BM25
baseline on `<base>_flat` against the SAME baseline on `<base>_structured`, but that baseline
never touches any structured field — so the two cells are the same experiment twice and their
delta is a null replicate (run-to-run variance), not a corpus effect.

This script answers that with evidence, from the code path and from the runs, not by assumption:

  Q1  What does the BASELINE condition (`agent_research_bm25`) actually read?
      - the lexical blob that reaches Lucene (units_from_documents -> BM25Pyserini.index)
      - the listing rendering and the visit rendering (Bm25Visit.search / .visit)
      Reconstructed EXACTLY (same code, imported — not reimplemented) from both corpora and
      hashed; plus md5 of the on-disk `indexes/bm25_pyserini/<key>/corpus/docs*.jsonl` shards
      that were actually indexed.
  Q2  Does `sections_from_body` (the REAL function, imported) recover sections on the FLAT
      corpus? Full-scale section-count distribution on both sides, plus heading agreement
      against the structured side's explicit `sections`.
  Q3  Exactly what do the twins differ in, at full corpus scale (all docs, all pairs)?
  Q4  The measured flat-vs-structured delta, and whether the runs were even CONFIG-MATCHED.
      Paired EM + exact McNemar via scripts.compare_cells primitives (no EM reimplementation),
      plus a behavioural identity probe: for shared instances whose FIRST issued query string
      is identical on both sides, is the FIRST search observation byte-identical?
  Q5  Config parity of the twin cells (the decomposition's hidden assumption).

    PYTHONPATH=. envs/bin/python analysis/flat_twin_validity.py
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from agent_search.agent.tools.doc_research import sections_from_body  # noqa: E402
from agent_search.corpus.units import units_from_documents  # noqa: E402
from scripts.compare_cells import (  # noqa: E402
    cell_dir, cell_rows, load_judge_cache, load_qrels, mcnemar_p, metrics, pct,
)

PAIRS = [
    ("browsecomp_plus", "_visit_uncapped"),
    ("hotpotqa", "_visit_uncapped"),
    ("musique", "_visit_uncapped"),
    ("2wiki", "_visit_uncapped"),
]
BASELINE_COND = "agent_research_bm25"
DATA = REPO_ROOT / "data"

# The headline effects the paper reports, for the noise-floor comparison (latex/sections/results.tex,
# "Corpus versus interface" paragraph): interface-step EM deltas.
HEADLINE_INTERFACE_PP = {
    "browsecomp_plus": 2.5,
    "hotpotqa": 1.6,
    "musique": 2.7,
}


def sha(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def read_corpus(ds: str) -> list[dict]:
    p = DATA / ds / "corpus.jsonl"
    if not p.exists():
        return []
    out = []
    with open(p, encoding="utf-8") as fh:
        for ln in fh:
            ln = ln.strip()
            if ln:
                out.append(json.loads(ln))
    return out


# ---------------------------------------------------------------------------- Q3 + Q1 (corpus)

def compare_corpora(base: str) -> dict:
    flat_ds, str_ds = f"{base}_flat", f"{base}_structured"
    flat, struct = read_corpus(flat_ds), read_corpus(str_ds)
    res = {"base": base, "flat_dataset": flat_ds, "structured_dataset": str_ds,
           "n_flat": len(flat), "n_structured": len(struct)}
    if not flat or not struct:
        res["status"] = "missing corpus"
        return res

    def idof(d):
        return str(d.get("_id") or d.get("id") or d.get("doc_id"))

    res["flat_keys"] = sorted({k for d in flat for k in d})
    res["structured_keys"] = sorted({k for d in struct for k in d})
    res["keys_only_in_structured"] = sorted(set(res["structured_keys"]) - set(res["flat_keys"]))
    res["keys_only_in_flat"] = sorted(set(res["flat_keys"]) - set(res["structured_keys"]))

    fids, sids = [idof(d) for d in flat], [idof(d) for d in struct]
    res["id_sets_equal"] = set(fids) == set(sids)
    res["id_order_identical"] = fids == sids

    sbyid = {idof(d): d for d in struct}
    n_text_diff = n_title_diff = n_missing = 0
    n_flat_has_hash_headings = n_struct_has_hash_headings = 0
    for d in flat:
        i = idof(d)
        s = sbyid.get(i)
        if s is None:
            n_missing += 1
            continue
        ft, st = str(d.get("text") or ""), str(s.get("text") or "")
        if ft != st:
            n_text_diff += 1
        if str(d.get("title") or "") != str(s.get("title") or ""):
            n_title_diff += 1
        if "\n## " in ft or ft.startswith("## "):
            n_flat_has_hash_headings += 1
        if "\n## " in st or st.startswith("## "):
            n_struct_has_hash_headings += 1
    res.update(n_missing_in_structured=n_missing, n_text_differs=n_text_diff,
               n_title_differs=n_title_diff,
               n_flat_docs_with_md_headings=n_flat_has_hash_headings,
               n_structured_docs_with_md_headings=n_struct_has_hash_headings)

    # --- Q1: the lexical blob that actually reaches Lucene, reconstructed with the REAL code.
    # BM25Pyserini.index() writes {"id": u.doc_id, "contents": f"{u.qualname} {u.code}"}.
    def blob_hash(docs):
        h = hashlib.sha256()
        n = 0
        for u in units_from_documents(docs):
            h.update(json.dumps({"id": u.doc_id, "contents": f"{u.qualname} {u.code}"},
                                sort_keys=True).encode("utf-8"))
            h.update(b"\n")
            n += 1
        return h.hexdigest(), n

    fh_, fn_ = blob_hash(flat)
    sh_, sn_ = blob_hash(struct)
    res["indexed_blob_sha_flat"] = fh_
    res["indexed_blob_sha_structured"] = sh_
    res["indexed_blob_identical"] = (fh_ == sh_)
    res["n_units_flat"], res["n_units_structured"] = fn_, sn_

    # --- Q1: what Bm25Visit renders. search() snippet = " ".join((u.body or u.code)[:120].split());
    # visit() = u.body (capped). Both are functions of title/body only -> hash them too.
    def render_hash(docs):
        h = hashlib.sha256()
        for u in units_from_documents(docs):
            snip = " ".join((u.body or u.code or "")[:120].split())
            h.update(f"{u.doc_id}\x00{u.title or u.qualname or ''}\x00{snip}\x00"
                     f"{u.body or u.code or ''}\n".encode("utf-8"))
        return h.hexdigest()

    res["baseline_rendering_sha_flat"] = render_hash(flat)
    res["baseline_rendering_sha_structured"] = render_hash(struct)
    res["baseline_rendering_identical"] = (
        res["baseline_rendering_sha_flat"] == res["baseline_rendering_sha_structured"])

    # --- on-disk indexed shards actually used by the runs
    def shard_md5(key):
        d = REPO_ROOT / "indexes" / "bm25_pyserini" / key / "corpus"
        files = sorted(glob.glob(str(d / "docs*.jsonl")))
        if not files:
            return None
        out = subprocess.run(["md5sum", *files], capture_output=True, text=True)
        return sha("".join(ln.split()[0] for ln in out.stdout.splitlines()))

    res["ondisk_index_corpus_md5_flat"] = shard_md5(flat_ds)
    res["ondisk_index_corpus_md5_structured"] = shard_md5(str_ds)
    res["ondisk_index_corpus_identical"] = (
        res["ondisk_index_corpus_md5_flat"] is not None
        and res["ondisk_index_corpus_md5_flat"] == res["ondisk_index_corpus_md5_structured"])

    # --- Q2: sections_from_body on BOTH sides (the REAL function)
    def sec_stats(docs, use_field=False):
        counts = Counter()
        multi = 0
        for d in docs:
            if use_field and isinstance(d.get("sections"), list) and d["sections"]:
                n = len(d["sections"])
            else:
                n = len(sections_from_body(str(d.get("text") or "")))
            counts[min(n, 50)] += 1
            if n > 1:
                multi += 1
        tot = sum(counts.values()) or 1
        mean = sum(k * v for k, v in counts.items()) / tot
        return {"n": tot, "n_multi_section": multi, "pct_multi_section": 100.0 * multi / tot,
                "mean_sections": mean}

    res["sections_from_body_on_flat"] = sec_stats(flat)
    res["sections_from_body_on_structured_body"] = sec_stats(struct)
    res["explicit_sections_field_on_structured"] = sec_stats(struct, use_field=True)

    # heading agreement: derived-from-flat-body headings vs the structured side's explicit headings
    agree = compared = 0
    for d in flat[:20000]:
        s = sbyid.get(idof(d))
        if not s or not isinstance(s.get("sections"), list) or not s["sections"]:
            continue
        derived = list(sections_from_body(str(d.get("text") or "")).keys())
        explicit = [str(x.get("heading") or "") for x in s["sections"]]
        compared += 1
        if derived == explicit:
            agree += 1
    res["heading_agreement"] = {"compared": compared, "identical": agree,
                                "pct": (100.0 * agree / compared) if compared else float("nan")}
    res["status"] = "ok"
    return res


# ---------------------------------------------------------------------------- Q4/Q5 (runs)

CONFIG_KEYS_TO_COMPARE = [
    "retriever", "model", "dense_model", "domain", "policy", "check_complete", "max_steps",
    "prompt_profile", "temperature", "seed", "backend", "level", "limit", "corpus_limit",
    "index_root", "rebuild", "runs_dir", "resolved_domain", "prompt_task", "prompt_toolset",
    "prompt_sha256", "workers",
]


def config_parity(subdir: str, base: str) -> dict:
    out = {}
    cfgs = {}
    for side in ("flat", "structured"):
        p = cell_dir(subdir, f"{base}_{side}", BASELINE_COND) / "config.json"
        cfgs[side] = json.loads(p.read_text()) if p.exists() else None
    if not cfgs["flat"] or not cfgs["structured"]:
        return {"status": "missing config"}
    diffs = {}
    for k in CONFIG_KEYS_TO_COMPARE:
        a, b = cfgs["flat"].get(k), cfgs["structured"].get(k)
        if a != b:
            diffs[k] = {"flat": a, "structured": b}
    ea, eb = cfgs["flat"].get("env_knobs") or {}, cfgs["structured"].get("env_knobs") or {}
    env_diffs = {k: {"flat": ea.get(k, "<absent>"), "structured": eb.get(k, "<absent>")}
                 for k in sorted(set(ea) | set(eb)) if ea.get(k, "<absent>") != eb.get(k, "<absent>")}
    out.update(status="ok", config_diffs=diffs, env_knob_diffs=env_diffs,
               started_at={"flat": cfgs["flat"].get("started_at"),
                           "structured": cfgs["structured"].get("started_at")},
               config_matched=(not diffs and not env_diffs))
    return out


def behavioural_probe(subdir: str, base: str, max_probe: int = 100000) -> dict:
    """If the two cells are the same experiment, an IDENTICAL first query must return an
    IDENTICAL first search observation (same index, same ranking, same rendering)."""
    rf = cell_rows(subdir, f"{base}_flat", BASELINE_COND)
    rs = cell_rows(subdir, f"{base}_structured", BASELINE_COND)
    if rf is None or rs is None:
        return {"status": "missing rows"}

    def key(iid, ds):
        return iid.split("__", 1)[1] if "__" in iid else iid

    fb = {key(r.get("instance_id", ""), None): r for r in rf}
    sb = {key(r.get("instance_id", ""), None): r for r in rs}
    shared = sorted(set(fb) & set(sb))
    same_q = same_obs = diff_obs = 0
    identical_full_query_seq = 0
    examples = []
    for q in shared[:max_probe]:
        a, b = fb[q], sb[q]
        qa, qb = (a.get("queries") or []), (b.get("queries") or [])
        oa, ob = (a.get("observations") or []), (b.get("observations") or [])
        if qa and qb and qa == qb:
            identical_full_query_seq += 1
        if qa and qb and qa[0] and qa[0] == qb[0] and oa and ob:
            same_q += 1
            if oa[0] == ob[0]:
                same_obs += 1
            else:
                diff_obs += 1
                if len(examples) < 3:
                    examples.append({"instance": q, "query": qa[0],
                                     "flat_obs": oa[0][:400], "structured_obs": ob[0][:400]})
    return {"status": "ok", "n_shared": len(shared),
            "n_first_query_identical": same_q,
            "n_first_obs_identical": same_obs,
            "n_first_obs_differs": diff_obs,
            "pct_first_obs_identical": (100.0 * same_obs / same_q) if same_q else float("nan"),
            "n_full_query_sequence_identical": identical_full_query_seq,
            "diff_examples": examples}


def paired_delta(subdir: str, base: str) -> dict:
    flat_ds, str_ds = f"{base}_flat", f"{base}_structured"
    rf, rs = cell_rows(subdir, flat_ds, BASELINE_COND), cell_rows(subdir, str_ds, BASELINE_COND)
    if rf is None or rs is None:
        return {"status": "missing rows"}
    mf = metrics(rf, load_qrels(flat_ds), flat_ds, load_judge_cache(cell_dir(subdir, flat_ds, BASELINE_COND)))
    ms = metrics(rs, load_qrels(str_ds), str_ds, load_judge_cache(cell_dir(subdir, str_ds, BASELINE_COND)))
    strip = lambda iid, ds: iid[len(ds) + 2:] if iid.startswith(ds + "__") else iid  # noqa: E731
    mf = {strip(k, flat_ds): v for k, v in mf.items()}
    ms = {strip(k, str_ds): v for k, v in ms.items()}
    shared = sorted(set(mf) & set(ms))
    if not shared:
        return {"status": "no shared instances"}
    b = sum(1 for q in shared if mf[q]["em"] and not ms[q]["em"])   # flat-only correct
    c = sum(1 for q in shared if ms[q]["em"] and not mf[q]["em"])   # structured-only correct
    em_f, em_s = pct([mf[q]["em"] for q in shared]), pct([ms[q]["em"] for q in shared])
    return {"status": "ok", "n_shared": len(shared),
            "em_flat_pct": em_f, "em_structured_pct": em_s,
            "delta_em_pp": em_s - em_f, "b_flat_only": b, "c_structured_only": c,
            "mcnemar_p": mcnemar_p(b, c),
            "mean_llm_calls_flat": sum(mf[q]["llm_calls"] for q in shared) / len(shared),
            "mean_llm_calls_structured": sum(ms[q]["llm_calls"] for q in shared) / len(shared),
            "mean_tok_flat": sum(mf[q]["tok"] for q in shared) / len(shared),
            "mean_tok_structured": sum(ms[q]["tok"] for q in shared) / len(shared),
            "recall_flat_pct": pct([mf[q]["recall"] for q in shared]),
            "recall_structured_pct": pct([ms[q]["recall"] for q in shared])}


def budget_matched_subset(subdir: str, base: str, cap: int = 50) -> dict:
    """Sensitivity analysis for the max_steps mismatch on hotpotqa/musique.

    The flat cells were run with max_steps=100, the structured cells with max_steps=50, so the
    flat agent had twice the tool budget. Restrict to instances where BOTH sides terminated
    VOLUNTARILY (`stopped == "answer"`) within `cap` steps: on that subset neither budget could
    have bound, so the budget mismatch cannot mechanically drive the delta. This is conditioning
    on a post-treatment variable and is therefore NOT an unbiased causal estimate -- it is a
    sensitivity probe: if the delta survives unchanged the budget is not the driver, if it
    collapses the budget is implicated."""
    flat_ds, str_ds = f"{base}_flat", f"{base}_structured"
    rf, rs = cell_rows(subdir, flat_ds, BASELINE_COND), cell_rows(subdir, str_ds, BASELINE_COND)
    if rf is None or rs is None:
        return {"status": "missing rows"}
    strip = lambda iid, ds: iid[len(ds) + 2:] if iid.startswith(ds + "__") else iid  # noqa: E731
    fb = {strip(r.get("instance_id", ""), flat_ds): r for r in rf}
    sb = {strip(r.get("instance_id", ""), str_ds): r for r in rs}
    mf = {strip(k, flat_ds): v for k, v in
          metrics(rf, load_qrels(flat_ds), flat_ds, {}).items()}
    ms = {strip(k, str_ds): v for k, v in
          metrics(rs, load_qrels(str_ds), str_ds, {}).items()}
    keep = [q for q in sorted(set(fb) & set(sb))
            if fb[q].get("stopped") == "answer" and sb[q].get("stopped") == "answer"
            and (fb[q].get("n_steps") or 0) < cap and (sb[q].get("n_steps") or 0) < cap]
    if not keep:
        return {"status": "empty subset"}
    b = sum(1 for q in keep if mf[q]["em"] and not ms[q]["em"])
    c = sum(1 for q in keep if ms[q]["em"] and not mf[q]["em"])
    em_f, em_s = pct([mf[q]["em"] for q in keep]), pct([ms[q]["em"] for q in keep])
    return {"status": "ok", "cap": cap, "n": len(keep),
            "em_flat_pct": em_f, "em_structured_pct": em_s, "delta_em_pp": em_s - em_f,
            "b_flat_only": b, "c_structured_only": c, "mcnemar_p": mcnemar_p(b, c),
            "mean_llm_calls_flat": sum(fb[q].get("llm_calls") or 0 for q in keep) / len(keep),
            "mean_llm_calls_structured": sum(sb[q].get("llm_calls") or 0 for q in keep) / len(keep)}


def step_cap_stats(subdir: str, base: str) -> dict:
    """How often each side ran to its step budget — the mechanism by which a max_steps mismatch
    (if any) turns into an accuracy/cost difference."""
    out = {}
    for side in ("flat", "structured"):
        rows = cell_rows(subdir, f"{base}_{side}", BASELINE_COND)
        if rows is None:
            out[side] = None
            continue
        caps = Counter(r.get("max_steps") for r in rows)
        stopped = Counter(r.get("stopped") for r in rows)
        n = len(rows) or 1
        at_cap = sum(1 for r in rows
                     if r.get("max_steps") and (r.get("n_steps") or 0) >= r["max_steps"] - 1)
        out[side] = {"n": len(rows), "max_steps_values": dict(caps),
                     "stopped": dict(stopped),
                     "pct_at_step_cap": 100.0 * at_cap / n,
                     "mean_n_steps": sum((r.get("n_steps") or 0) for r in rows) / n}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO_ROOT / "analysis" / "flat_twin_validity.md"))
    ap.add_argument("--json", default=str(REPO_ROOT / "analysis" / "flat_twin_validity_data.json"))
    args = ap.parse_args()
    os.chdir(REPO_ROOT)

    data = {"pairs": []}
    for base, subdir in PAIRS:
        entry = {"base": base, "run_group": f"runs/{subdir}"}
        entry["corpus"] = compare_corpora(base)
        entry["config_parity"] = config_parity(subdir, base)
        if entry["config_parity"].get("status") == "ok":
            entry["behavioural_probe"] = behavioural_probe(subdir, base)
            entry["paired_delta"] = paired_delta(subdir, base)
            entry["step_caps"] = step_cap_stats(subdir, base)
            entry["budget_matched_subset"] = budget_matched_subset(subdir, base)
            pd = entry["paired_delta"]
            if pd.get("status") == "ok":
                n, disc = pd["n_shared"], pd["b_flat_only"] + pd["c_structured_only"]
                entry["instability"] = {
                    "n_discordant_pairs": disc,
                    "pct_instances_flipping_em": 100.0 * disc / n,
                    # SE of the paired % difference (McNemar): sqrt(b+c)/n
                    "se_delta_em_pp": 100.0 * (disc ** 0.5) / n,
                    "delta_in_se_units": (pd["delta_em_pp"] / (100.0 * (disc ** 0.5) / n))
                    if disc else float("nan"),
                }
        data["pairs"].append(entry)
        print(f"[done] {base}", file=sys.stderr)

    data["headline_interface_pp"] = HEADLINE_INTERFACE_PP
    with open(args.json, "w") as fh:
        json.dump(data, fh, indent=2, default=str)
    print(f"wrote {args.json}", file=sys.stderr)


if __name__ == "__main__":
    main()
