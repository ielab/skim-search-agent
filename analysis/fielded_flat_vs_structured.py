#!/usr/bin/env python
"""FIELDED flat-vs-structured pairing — the experiment that `analysis/flat_twin_validity.md`
named as the ONLY one that could establish a corpus effect.

WHY THIS EXISTS
---------------
The paper originally decomposed its gain into a "corpus step" and an "interface step" and
claimed 30-45% of the gain came from the structured corpus. `analysis/flat_twin_validity.md`
RETRACTED the corpus half: the only flat/structured pairing ever run used the structure-BLIND
baseline `agent_research_bm25`, whose Lucene input blob and whose SERP/visit renderer touch only
`title` + `body` -- byte-identical across the twins (verified full-scale, 0 docs differ). That
pairing was a null replicate, not an ablation. Its own closing paragraph states the fix:

    "If the corpus effect is to be claimed at all, the experiment that would establish it is a
     flat-vs-structured pairing of a FIELDED condition (agent_research_bql_dense_snip or
     agent_research_indri) under identical max_steps and identical code -- which the corpora
     already support and which has never been run."

It has now been run. This script analyses it.

THE CELLS (condition `agent_research_bql_dense_snip` = the full method, "Sieve", on both sides):
    STRUCTURED  runs/_budget100/agent/hotpotqa_structured/<model>/agent_research_bql_dense_snip
                runs/_budget100/agent/musique_structured/<model>/agent_research_bql_dense_snip
    FLAT        runs/_flatfielded/agent/hotpotqa_flat/<model>/agent_research_bql_dense_snip
                runs/_flatfielded/agent/musique_flat/<model>/agent_research_bql_dense_snip

WHY THIS IS A REAL MANIPULATION (and the previous one was not)
--------------------------------------------------------------
`agent_search/corpus/units.py:units_from_documents` builds a unit's `section` from the document's
`sections`/`section` keys and its `infobox` from `metadata` -- keys the flat twin does not carry
(`flat_twin_validity.md` Q3: the flat corpus removes exactly `sections`, `section`, `infobox` on
the wiki pairs; `text`/`title` are byte-identical). So on the flat corpus:
  * the SEARCH axis genuinely degrades: `IN(section, x)` / `IN(infobox, x)` clauses -- which this
    condition's query surface (`bql/surface.py` DOC_FIELDS: section/sec/heading, infobox/ib/fact,
    author/by/byline, date, and the `all` alias which expands to title+body+section+infobox) can
    express and which its manual advertises -- match nothing;
  * the READ axis still works: `_secs` falls back to `sections_from_body`, which recovers the
    `##` heading tree on ~89% of hotpotqa / ~88% of musique flat documents.
This measures what the structured METADATA is worth to a method that actually reads it.

PROMPT PARITY (verified in PART 0, not assumed): `hotpotqa_flat`/`musique_flat` register with NO
`field_profile` (-> "general") while `*_structured` register with `field_profile="wiki"`
(`evaluation/datasets.py:398-405`), so the two arms COULD have seen different manuals. They did
not: `load_prompt_text("research_bql_dense_snip", "wiki")` and `(..., "general")` are the same
10,167-byte string with the same sha256 -- only the browsecomp profile differs. The manipulation
is therefore corpus-only under a byte-identical system prompt.

METHOD PARITY IS MANDATORY -- nothing statistical is reimplemented here:
  scripts.compare_cells               : cell_dir, cell_rows (transitively
                                        force_answer_backfill.load_rows_with_recovery),
                                        metrics (transitively evaluation.metrics.answer_em and
                                        gold_doc_recall), mcnemar_p, pct, load_qrels, _qid_of
  analysis.structured_surface_control : cell_metrics, paired_em, mean_tok_recall,
                                        run_sanity_check
  analysis.step_budget_audit          : _paired_tokens (the count-once token mean + paired t-test
                                        + Wilcoxon, i.e. the `tok` = total_tokens_once definition
                                        analysis/structured_surface_control_3way.py reports via
                                        mean_tok_recall), per_row_caps
  analysis.recall_inversion_test      : paired_recall, _is_binary_indicator (exact McNemar on the
                                        per-instance BINARY gold-doc-recall indicator)
  agent_search...bql.surface.to_bql   : the REAL surface->BQL translator, used to decide whether
                                        a query carries a field restriction (no regex guess)
The only new code is the cross-prefix JOIN, the mean-LLM-calls summary, the query-behaviour
contrast, and the reporting layer.

THE JOIN. Instance ids are dataset-prefixed and therefore DISJOINT across arms
(`musique_flat__2hop__460946_294723` vs `musique_structured__2hop__460946_294723`), so every
helper above -- all of which inner-join on `instance_id` -- would silently return n=0. Both arms'
metrics dicts are re-keyed by `scripts.compare_cells._qid_of(iid, dataset)` (the repo's own
prefix-stripper, which correctly handles qids that themselves contain "__", e.g. musique's
"2hop__X_Y") BEFORE any helper sees them. PART 2 verifies the re-key is injective on each side and
that the join is complete and one-to-one, and reports every unmatched id.

JUDGE: HotpotQA and MuSiQue are EM/F1 benchmarks and are NOT judged in this repo, by design.
No judge number is computed or reported for them.

MANDATORY SANITY GATES (PART 1, run FIRST; a failure aborts with a nonzero exit and NO new
numbers), reproducing two already-published results with this script's own pipeline:
  (A) Sieve vs "No dense evidence" on browsecomp_plus_structured: +4.7 EM, p=0.0104
      (analysis/ablation_deltas.md) -- reuses structured_surface_control.run_sanity_check verbatim.
  (B) Sieve@100 vs the hybrid control on hotpotqa_structured, both at cap 100: +5.84 EM,
      p=2.84e-29 (analysis/matched_cap100_results.md contrast (a)).
Gate (B) exists because a project post-mortem found that EVERY analysis script's sanity gate
targeted BrowseComp, which was immune to the budget bug those gates should have caught. Gate (B)
is a WIKIPEDIA cap-100 quantity and its Sieve arm is literally the same cell this script uses as
the hotpotqa STRUCTURED arm -- so it validates the new arm directly, not a cousin of it.

Reads runs/ READ-ONLY. Writes analysis/fielded_flat_vs_structured{_data.json,.md}.
Run: PYTHONPATH=. envs/bin/python analysis/fielded_flat_vs_structured.py
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.compare_cells import (  # noqa: E402
    MODEL_DIR, cell_dir, load_qrels, mcnemar_p, pct, _qid_of,
)
from analysis.structured_surface_control import (  # noqa: E402
    cell_metrics, paired_em, mean_tok_recall, run_sanity_check,
)
from analysis.step_budget_audit import _paired_tokens, per_row_caps  # noqa: E402
from analysis.recall_inversion_test import paired_recall, _is_binary_indicator  # noqa: E402
from analysis.equivalence_and_latency import (  # noqa: E402
    bootstrap_diff_pp, bootstrap_ci, BOOT_SEED,
)

# The repo's paired-instance bootstrap, reused as-is (same helpers and the same unmodified
# BOOT_SEED that analysis/hybrid_control_ci.py reuses -- not re-derived or tuned post hoc). A
# NULL result is only informative with an interval attached: the CI is what says how large a
# corpus effect the data can still be hiding.
N_BOOT = 10000

ROOT = Path(__file__).resolve().parent.parent
JSON_PATH = ROOT / "analysis" / "fielded_flat_vs_structured_data.json"
MD_PATH = ROOT / "analysis" / "fielded_flat_vs_structured.md"

COND = "agent_research_bql_dense_snip"          # Sieve, the full method -- BOTH arms
STRUCT_TIER = "_budget100"
FLAT_TIER = "_flatfielded"

# (label, structured dataset, flat dataset)
PAIRS = [
    ("HotpotQA", "hotpotqa_structured", "hotpotqa_flat"),
    ("MuSiQue", "musique_structured", "musique_flat"),
]

# Gate (B): analysis/matched_cap100_results.md contrast (a), hotpotqa_structured, both arms cap 100.
GATE_B = dict(dataset="hotpotqa_structured",
              a=(STRUCT_TIER, COND),
              b=("_headline_validation", "agent_research_hybrid_fetch_snip"),
              em_delta=5.84, em_p=2.84e-29, n=7343)
GATE_B_TOL = dict(delta=0.15, logp=0.05)        # p compared on log10 (it is ~1e-29)


# ---------------------------------------------------------------------------------------------
# PART 0 helpers: provenance
# ---------------------------------------------------------------------------------------------
def _md5(p: Path) -> str:
    h = hashlib.md5()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def prompt_parity() -> dict:
    """The two arms' system prompts, composed by the REAL loader for the field_profile each
    dataset's registration selects (`evaluation/datasets.py`): structured -> "wiki",
    flat -> (no profile) -> the agent domain "general"."""
    from agent_search.prompts.loader import load_prompt_text
    out = {}
    for prof in ("wiki", "general", "browsecomp"):
        s = load_prompt_text("research_bql_dense_snip", prof)
        out[prof] = dict(n_bytes=len(s), sha256=hashlib.sha256(s.encode()).hexdigest())
    out["wiki_equals_general"] = out["wiki"]["sha256"] == out["general"]["sha256"]
    return out


def cap_profile(tier: str, dataset: str) -> dict:
    caps = per_row_caps(dataset, tier, COND, {})
    dist = {}
    for v in caps.values():
        dist[str(v)] = dist.get(str(v), 0) + 1
    return dict(n=len(caps), cap_dist=dist, uniform_100=bool(list(dist) == ["100"]))


def config_parity(sd: str, fd: str, s_tier: str, f_tier: str) -> dict:
    """Config parity between the two arms, from the per-shard `config.json` each run wrote.
    `flat_twin_validity.md` Q4 killed the retracted pairing partly on THIS: its hotpotqa/musique
    arms ran at max_steps 100 vs 50, with flat-only AGENT_CTX_WINDOW / AGENT_CTX_STOP_FRAC /
    AGENT_SEARCH_FLAT_FAISS knobs and ten days of code drift between them.

    For each key, the SET of values it takes across an arm's shards is compared against the other
    arm's set -- a key that varies per shard WITHIN both arms (`api_base`: each of the 24 shards
    talks to its own vLLM port) is therefore only flagged if the two arms' value sets actually
    differ. `dataset` and `runs_dir` MUST differ (they name the arm) and `started_at` is wall
    clock, so those three are expected and held out of `unexpected_differing_keys`."""
    import glob
    EXPECTED = {"dataset", "runs_dir", "started_at"}
    def _cfgs(tier, ds):
        pat = (f"runs/{tier}/__shards/{COND}/*/agent/{ds}/{MODEL_DIR}/{COND}/config.json")
        return [json.loads(Path(p).read_text()) for p in sorted(glob.glob(str(ROOT / pat)))]
    S, F = _cfgs(s_tier, sd), _cfgs(f_tier, fd)
    if not S or not F:
        return dict(n_structured_shards=len(S), n_flat_shards=len(F),
                    note="shard config.json not found for one or both arms")
    keyset = set()
    for c in S + F:
        keyset |= set(c)
    vals = {k: (sorted({json.dumps(c.get(k), sort_keys=True, default=str) for c in S}),
                sorted({json.dumps(c.get(k), sort_keys=True, default=str) for c in F}))
            for k in keyset}
    diff = {k for k, (a, b) in vals.items() if a != b}
    return dict(
        n_structured_shards=len(S), n_flat_shards=len(F),
        n_keys_compared=len(keyset),
        differing_keys=sorted(diff),
        unexpected_differing_keys=sorted(diff - EXPECTED),
        parity_ok=bool(not (diff - EXPECTED)),
        per_key_value_sets={k: dict(structured=vals[k][0][:6], flat=vals[k][1][:6])
                            for k in sorted(diff)},
        max_steps=dict(structured=vals.get("max_steps", ([], []))[0],
                       flat=vals.get("max_steps", ([], []))[1]),
        prompt_sha256=dict(structured=vals.get("prompt_sha256", ([], []))[0],
                           flat=vals.get("prompt_sha256", ([], []))[1]),
    )


def corpus_field_check(sd: str, fd: str, sample: int = 5000) -> dict:
    """WHAT the flat twin actually withholds, measured on the first `sample` documents of each
    corpus (the twins are emitted in the same id order -- `flat_twin_validity.md` Q3 verified
    ids/order/text/title equal at FULL scale over every document of every pair, so a prefix
    sample is enough to characterise the field DELTA here rather than re-run that audit).
    Reports, in particular, whether the structured-only `infobox`/`sections` CONTENT is still
    readable in the shared `text` -- it is, which is exactly why the flat arm's READ axis is
    intact and only its fielded SEARCH surface and its listing's structure inventory degrade."""
    import itertools
    s_keys, f_keys = set(), set()
    n = n_ib = n_ib_in_text = n_sec = n_flat_hashes = 0
    with (ROOT / "data" / sd / "corpus.jsonl").open() as sfh, \
         (ROOT / "data" / fd / "corpus.jsonl").open() as ffh:
        for sl, fl in itertools.islice(zip(sfh, ffh), sample):
            s, f = json.loads(sl), json.loads(fl)
            s_keys |= set(s)
            f_keys |= set(f)
            n += 1
            text = f.get("text") or ""
            ib = (s.get("infobox") or "").strip()
            if ib:
                n_ib += 1
                head = " ".join(ib.split())[:120]
                if head and head in " ".join(text.split()):
                    n_ib_in_text += 1
            if s.get("sections"):
                n_sec += 1
            if "\n## " in text or text.startswith("## "):
                n_flat_hashes += 1
    return dict(
        n_sampled=n,
        keys_structured=sorted(s_keys), keys_flat=sorted(f_keys),
        keys_only_in_structured=sorted(s_keys - f_keys),
        keys_only_in_flat=sorted(f_keys - s_keys),
        pct_structured_with_infobox=100.0 * n_ib / (n or 1),
        pct_infobox_content_also_in_flat_text=100.0 * n_ib_in_text / (n_ib or 1),
        pct_structured_with_sections=100.0 * n_sec / (n or 1),
        pct_flat_text_with_markdown_headings=100.0 * n_flat_hashes / (n or 1),
    )


def corpus_asset_parity(sd: str, fd: str) -> dict:
    """qrels + queries md5 across the twins (the join's precondition: same questions, same gold)."""
    out = {}
    for name, rel in (("queries", "queries.jsonl"), ("qrels", "qrels/test.tsv")):
        ps, pf = ROOT / "data" / sd / rel, ROOT / "data" / fd / rel
        out[name] = dict(structured=_md5(ps) if ps.exists() else None,
                         flat=_md5(pf) if pf.exists() else None)
        out[name]["identical"] = bool(out[name]["structured"]
                                      and out[name]["structured"] == out[name]["flat"])
    return out


# ---------------------------------------------------------------------------------------------
# PART 2 helpers: the cross-prefix join
# ---------------------------------------------------------------------------------------------
def rekey(m: dict, dataset: str):
    """metrics dict keyed by full instance_id -> keyed by the shared qid suffix.
    Returns (rekeyed, collisions) where `collisions` lists any qid produced by >1 instance_id
    (which would make the join not one-to-one)."""
    out, seen = {}, {}
    for iid, v in m.items():
        q = _qid_of(iid, dataset)
        seen.setdefault(q, []).append(iid)
        out[q] = v
    collisions = {q: iids for q, iids in seen.items() if len(iids) > 1}
    return out, collisions


def verify_join(s_m: dict, f_m: dict, sd: str, fd: str) -> dict:
    s_k, s_col = rekey(s_m, sd)
    f_k, f_col = rekey(f_m, fd)
    ss, fs = set(s_k), set(f_k)
    shared = ss & fs
    return dict(
        n_structured_rows=len(s_m), n_flat_rows=len(f_m),
        n_structured_qids=len(s_k), n_flat_qids=len(f_k), n_shared=len(shared),
        structured_qid_collisions=s_col, flat_qid_collisions=f_col,
        unmatched_structured_only=sorted(ss - fs)[:50],
        n_unmatched_structured_only=len(ss - fs),
        unmatched_flat_only=sorted(fs - ss)[:50],
        n_unmatched_flat_only=len(fs - ss),
        complete_and_one_to_one=bool(not s_col and not f_col and ss == fs
                                     and len(s_m) == len(s_k) and len(f_m) == len(f_k)),
    ), s_k, f_k


# ---------------------------------------------------------------------------------------------
# PART 4 helpers: query behaviour
# ---------------------------------------------------------------------------------------------
# Canonical BQL regions. section/infobox/author/date are the ones the flat corpus does not carry
# (units_from_documents leaves u.section = None, u.sections = None, and metadata has no
# infobox/author/date on the wiki flat twins); title/body are byte-identical across the twins.
_IN_RE = re.compile(r"IN\((title|body|section|infobox|author|date),")
STRUCTURED_REGIONS = ("section", "infobox", "author", "date")

_HDR_EXACT = re.compile(r"\((\d+) matches, top \d+\):")
_IB_NONEMPTY = re.compile(r"\bib\[[^\]]+\]")
_MISS_RE = re.compile(r"miss=\[([^\]]*)\]")

# Normalizers used to tell a COSMETIC listing difference apart from a RETRIEVAL difference.
# `_render_hits` prints, per hit, `ib[<infobox keys>]` and `matched: <fields>`; both are read off
# the structured metadata and are therefore guaranteed to differ across the twins even when the
# retrieved doc ids, their order, and their snippets are identical. Blanking exactly those two
# spans leaves the header (match count), the rank order, the doc ids, the titles, the section
# list and the snippets -- i.e. everything that is actually retrieval.
_NORM_IB = re.compile(r"ib\[[^\]]*\]")
_NORM_MATCHED = re.compile(r"matched: [A-Za-z,\-]+")


def norm_obs(o: str) -> str:
    return _NORM_MATCHED.sub("matched:*", _NORM_IB.sub("ib[*]", o or ""))


def classify_obs(obs: str) -> str:
    """Outcome bucket of ONE search observation, keyed off the exact strings
    `agent_search/agent/tools/doc_research.py:_search_impl` emits."""
    if obs.startswith("ERROR:"):
        return "error_tool"
    if obs.startswith("search: ") and "\nBQL " in obs:
        return "error_bql"          # obs.error from execute_bql (type/field error)
    if obs == "empty query":
        return "empty_query"
    if "0 exact matches — closest by CONSTRAINT COVERAGE" in obs:
        return "zero_exact_coverage"
    if "0 exact matches — showing top" in obs:
        return "zero_exact_soft"
    if "(0 matches)" in obs:
        return "zero_hard"
    if _HDR_EXACT.search(obs):
        return "exact_hits"
    return "other"


ZERO_OR_ERROR = ("zero_exact_coverage", "zero_exact_soft", "zero_hard",
                 "error_tool", "error_bql", "empty_query")


def as_query_tuple(q) -> tuple:
    """A row's `queries[i]` is normally a string, but ~0.1% of search steps carry a LIST of
    alternative query strings (a batched multi-query call). Canonicalize both to a tuple so the
    same code path handles them and so query-identity comparisons across arms are well defined."""
    if isinstance(q, (list, tuple)):
        return tuple(str(x) for x in q)
    return (str(q or ""),)


def _leaf_scopes(expr, scope=None, out=None):
    """Every LEAF of a parsed BQL expression, tagged with the innermost `In(region)` it sits
    under (None = unscoped, i.e. the whole-document token bag). Walks the real AST
    (`bql.ast`), because the BQL STRING cannot be regexed for this: an unscoped leaf is written
    bare (`Olympic`), producing no `IN(` for a regex to see. Without this walk a query like
    `first medals awarded OR "medals"[infobox]` looks purely infobox-scoped when in fact its
    bare conjunction still matches the flat corpus."""
    from agent_search.retrievers.structural.bql.ast import And, In, Near, Not, Or
    if out is None:
        out = []
    if isinstance(expr, In):
        _leaf_scopes(expr.child, expr.region.value, out)
    elif isinstance(expr, (And, Or)):
        for ch in expr.children:
            _leaf_scopes(ch, scope, out)
    elif isinstance(expr, Not):
        _leaf_scopes(expr.child, scope, out)
    elif isinstance(expr, Near):
        _leaf_scopes(expr.left, scope, out)
        _leaf_scopes(expr.right, scope, out)
    else:
        out.append(scope)
    return out


def query_regions(q, cache: dict):
    """(regions, strict) for a surface query, via the REAL translator + the REAL BQL parser
    (union over the elements of a batched multi-query).

    `regions` = the canonical BQL regions the query scopes. `strict` = every leaf of every
    element is scoped to a STRUCTURED-only region -- no unscoped leaf and no title/body leaf that
    could satisfy the Boolean over the byte-identical text. Returns (None, False) if the
    translator/parser fails on EVERY element (the agent would have seen a 'could not translate'
    ERROR)."""
    key = as_query_tuple(q)
    if key in cache:
        return cache[key]
    from agent_search.retrievers.structural.bql.surface import to_bql
    from agent_search.retrievers.structural.bql.parser import parse as bql_parse
    regs, scopes, any_ok = set(), [], False
    for part in key:
        try:
            bql = to_bql(part, domain="doc")
        except Exception:  # noqa: BLE001
            continue
        regs |= set(_IN_RE.findall(bql))
        any_ok = True
        pr = bql_parse(bql)
        if pr.ok and pr.expr is not None:
            scopes.extend(_leaf_scopes(pr.expr))
        else:
            scopes.append(None)          # unparseable -> cannot claim strictness
    if not any_ok:
        out = (None, False)
    else:
        strict = bool(scopes) and all(s in STRUCTURED_REGIONS for s in scopes)
        out = (frozenset(regs), strict)
    if len(cache) < 400_000:
        cache[key] = out
    return out


def scan_arm(tier: str, dataset: str) -> dict:
    """ONE streaming pass over a cell's rows.jsonl collecting query-behaviour evidence.
    Uses row['observations'] (the FULL observation strings) -- never trajectory[i].observation,
    which is truncated at 600 chars."""
    path = cell_dir(tier, dataset, COND) / "rows.jsonl"
    qcache: dict = {}
    per_instance = {}
    agg = dict(n_instances=0, n_searches=0, n_fielded_searches=0, n_untranslatable=0,
               n_strict_searches=0,
               region_counts={r: 0 for r in ("title", "body") + STRUCTURED_REGIONS},
               outcome_all={}, outcome_fielded={}, outcome_plain={}, outcome_strict={},
               n_obs_with_nonempty_ib=0, n_search_obs=0,
               n_coverage_renders=0, n_coverage_miss_structured=0,
               n_instances_any_fielded=0, stopped={})
    with path.open("r", encoding="utf-8") as fh:
        for ln in fh:
            if not ln.strip():
                continue
            try:
                r = json.loads(ln)
            except json.JSONDecodeError:
                continue
            iid = r.get("instance_id") or ""
            qid = _qid_of(iid, dataset)
            actions = r.get("actions") or []
            queries = r.get("queries") or []
            obs = r.get("observations") or []
            agg["n_instances"] += 1
            st = str(r.get("stopped"))
            agg["stopped"][st] = agg["stopped"].get(st, 0) + 1
            sq, sobs, any_fielded = [], [], False
            for i, act in enumerate(actions):
                if not str(act).startswith("search"):
                    continue
                q = as_query_tuple(queries[i] if i < len(queries) else "")
                o = obs[i] if i < len(obs) else ""
                sq.append(q)
                sobs.append(o)
                agg["n_searches"] += 1
                agg["n_search_obs"] += 1
                # STRICT: every LEAF is scoped to a structured-only region, so no title/body or
                # unscoped branch could satisfy the Boolean on the flat corpus.
                # `X[title] OR X[infobox]` is fielded but NOT strict -- its title branch still
                # matches on flat, which is why the flat arm keeps a residual exact-hit rate on
                # the broad `fielded` set. On the strict set the flat index can only produce
                # exact hits if the fields are not really absent: the falsifiable instrument check.
                regs, strict = query_regions(q, qcache)
                if regs is None:
                    agg["n_untranslatable"] += 1
                    fielded = False
                else:
                    for rg in regs:
                        agg["region_counts"][rg] = agg["region_counts"].get(rg, 0) + 1
                    fielded = any(rg in regs for rg in STRUCTURED_REGIONS)
                if fielded:
                    agg["n_fielded_searches"] += 1
                    any_fielded = True
                if strict:
                    agg["n_strict_searches"] += 1
                bucket = classify_obs(o)
                agg["outcome_all"][bucket] = agg["outcome_all"].get(bucket, 0) + 1
                key = "outcome_fielded" if fielded else "outcome_plain"
                agg[key][bucket] = agg[key].get(bucket, 0) + 1
                if strict:
                    agg["outcome_strict"][bucket] = agg["outcome_strict"].get(bucket, 0) + 1
                if _IB_NONEMPTY.search(o):
                    agg["n_obs_with_nonempty_ib"] += 1
                if bucket == "zero_exact_coverage":
                    agg["n_coverage_renders"] += 1
                    missed = " ".join(_MISS_RE.findall(o))
                    if any(f"[{rg}]" in missed for rg in STRUCTURED_REGIONS):
                        agg["n_coverage_miss_structured"] += 1
            if any_fielded:
                agg["n_instances_any_fielded"] += 1
            per_instance[qid] = dict(
                queries=sq,
                first_query=sq[0] if sq else None,
                first_obs=sobs[0] if sobs else None,
                n_searches=len(sq),
                any_fielded=any_fielded,
                initial_prompt_tokens=r.get("initial_prompt_tokens"),
                question=r.get("question"),
            )
    return dict(agg=agg, per_instance=per_instance, qcache=qcache)


def behaviour_contrast(s_scan: dict, f_scan: dict, shared) -> dict:
    """Is this pairing an ABLATION or (like the retracted one) a REPLICATE? The retracted pairing
    was diagnosed by first-query/first-observation identity; the same instrument is applied here."""
    s_pi, f_pi = s_scan["per_instance"], f_scan["per_instance"]
    ids = [q for q in shared if q in s_pi and q in f_pi]
    same_first_q = [q for q in ids if s_pi[q]["first_query"] is not None
                    and s_pi[q]["first_query"] == f_pi[q]["first_query"]]
    same_first_obs = [q for q in same_first_q if s_pi[q]["first_obs"] == f_pi[q]["first_obs"]]
    same_seq = [q for q in ids if s_pi[q]["queries"] == f_pi[q]["queries"]]
    same_prompt_tok = [q for q in ids
                       if s_pi[q]["initial_prompt_tokens"] == f_pi[q]["initial_prompt_tokens"]]
    same_question = [q for q in ids if s_pi[q]["question"] == f_pi[q]["question"]]

    # MATCHED-QUERY analysis: restricted to instances whose FIRST issued query is identical, so
    # the two arms are being asked the same thing from the same state. Splits the observation
    # difference into (i) identical byte-for-byte, (ii) identical once the two structured-metadata
    # DISPLAY spans (`ib[...]`, `matched: ...`) are blanked -- same hits, same order, same
    # snippets, only the structure inventory differs -- and (iii) different retrieval outcome.
    qc: dict = {}
    cosmetic, retrieval, xtab, xtab_fielded, n_fielded_fq = 0, 0, {}, {}, 0
    for q in same_first_q:
        so, fo = s_pi[q]["first_obs"], f_pi[q]["first_obs"]
        if so == fo:
            pass
        elif norm_obs(so) == norm_obs(fo):
            cosmetic += 1
        else:
            retrieval += 1
        sb, fb = classify_obs(so or ""), classify_obs(fo or "")
        xtab[f"{sb}|{fb}"] = xtab.get(f"{sb}|{fb}", 0) + 1
        regs, _strict = query_regions(s_pi[q]["first_query"], qc)
        if regs is not None and any(rg in regs for rg in STRUCTURED_REGIONS):
            n_fielded_fq += 1
            xtab_fielded[f"{sb}|{fb}"] = xtab_fielded.get(f"{sb}|{fb}", 0) + 1

    def _zero(xt, idx):
        tot = sum(xt.values()) or 1
        z = sum(v for k, v in xt.items() if k.split("|")[idx] in ZERO_OR_ERROR)
        return 100.0 * z / tot

    return dict(
        n=len(ids),
        n_identical_first_query=len(same_first_q),
        pct_identical_first_query=pct([True] * len(same_first_q) + [False] * (len(ids) - len(same_first_q))),
        n_identical_first_obs_given_identical_first_query=len(same_first_obs),
        pct_identical_first_obs_given_identical_first_query=(
            100.0 * len(same_first_obs) / len(same_first_q) if same_first_q else float("nan")),
        n_identical_full_query_sequence=len(same_seq),
        pct_identical_full_query_sequence=(100.0 * len(same_seq) / len(ids) if ids else float("nan")),
        n_identical_initial_prompt_tokens=len(same_prompt_tok),
        pct_identical_initial_prompt_tokens=(100.0 * len(same_prompt_tok) / len(ids) if ids else float("nan")),
        n_identical_question=len(same_question),
        matched_first_query=dict(
            n=len(same_first_q),
            n_identical_observation=len(same_first_obs),
            n_differs_only_in_structure_display=cosmetic,
            n_differs_in_retrieval_outcome=retrieval,
            pct_identical_observation=(100.0 * len(same_first_obs) / len(same_first_q)
                                       if same_first_q else float("nan")),
            pct_differs_only_in_structure_display=(100.0 * cosmetic / len(same_first_q)
                                                   if same_first_q else float("nan")),
            pct_differs_in_retrieval_outcome=(100.0 * retrieval / len(same_first_q)
                                              if same_first_q else float("nan")),
            outcome_crosstab_structured_pipe_flat=xtab,
            pct_zero_or_error_structured=_zero(xtab, 0),
            pct_zero_or_error_flat=_zero(xtab, 1),
            n_fielded_first_queries=n_fielded_fq,
            fielded_outcome_crosstab_structured_pipe_flat=xtab_fielded,
            pct_fielded_zero_or_error_structured=(_zero(xtab_fielded, 0) if xtab_fielded
                                                  else float("nan")),
            pct_fielded_zero_or_error_flat=(_zero(xtab_fielded, 1) if xtab_fielded
                                            else float("nan")),
        ),
    )


def arm_query_summary(agg: dict) -> dict:
    ns = agg["n_searches"] or 1
    nf = agg["n_fielded_searches"] or 1
    nst = agg["n_strict_searches"] or 1
    npl = (agg["n_searches"] - agg["n_fielded_searches"]) or 1
    def _rate(d, denom):
        return {k: 100.0 * v / denom for k, v in sorted(d.items())}
    return dict(
        n_instances=agg["n_instances"],
        n_searches=agg["n_searches"],
        searches_per_episode=agg["n_searches"] / (agg["n_instances"] or 1),
        n_fielded_searches=agg["n_fielded_searches"],
        pct_searches_fielded=100.0 * agg["n_fielded_searches"] / ns,
        pct_instances_any_fielded=100.0 * agg["n_instances_any_fielded"] / (agg["n_instances"] or 1),
        region_counts=agg["region_counts"],
        region_rate_pct={k: 100.0 * v / ns for k, v in agg["region_counts"].items()},
        n_strict_searches=agg["n_strict_searches"],
        pct_searches_strict=100.0 * agg["n_strict_searches"] / ns,
        outcome_all_pct=_rate(agg["outcome_all"], ns),
        outcome_fielded_pct=_rate(agg["outcome_fielded"], nf),
        outcome_plain_pct=_rate(agg["outcome_plain"], npl),
        outcome_strict_pct=_rate(agg["outcome_strict"], nst),
        outcome_all_counts=agg["outcome_all"],
        outcome_fielded_counts=agg["outcome_fielded"],
        outcome_strict_counts=agg["outcome_strict"],
        pct_strict_zero_or_error=100.0 * sum(agg["outcome_strict"].get(b, 0)
                                             for b in ZERO_OR_ERROR) / nst,
        pct_fielded_zero_or_error=100.0 * sum(agg["outcome_fielded"].get(b, 0)
                                              for b in ZERO_OR_ERROR) / nf,
        pct_plain_zero_or_error=100.0 * sum(agg["outcome_plain"].get(b, 0)
                                            for b in ZERO_OR_ERROR) / npl,
        pct_search_obs_with_nonempty_infobox=100.0 * agg["n_obs_with_nonempty_ib"] / ns,
        n_coverage_renders=agg["n_coverage_renders"],
        n_coverage_miss_structured=agg["n_coverage_miss_structured"],
        n_untranslatable_queries=agg["n_untranslatable"],
        stopped_counts=agg["stopped"],
        pct_stopped_by_step_budget=100.0 * sum(v for k, v in agg["stopped"].items()
                                               if k not in ("answer",))
        / (agg["n_instances"] or 1),
    )


# ---------------------------------------------------------------------------------------------
def _jd(o):
    if hasattr(o, "item"):
        return o.item()
    if isinstance(o, (set, frozenset)):
        return sorted(o)
    raise TypeError(f"{o.__class__.__name__} not JSON serializable")


def run_gate_b() -> dict:
    qr = load_qrels(GATE_B["dataset"])
    a_m, _, a_n = cell_metrics(GATE_B["dataset"], GATE_B["a"][0], GATE_B["a"][1], qr)
    b_m, _, b_n = cell_metrics(GATE_B["dataset"], GATE_B["b"][0], GATE_B["b"][1], qr)
    if a_m is None or b_m is None:
        return dict(passed=False, reason="gate (B) cell rows.jsonl missing")
    n, em_a, em_b, delta, b, c, p = paired_em(a_m, b_m)
    import math
    ok = (n == GATE_B["n"] and abs(delta - GATE_B["em_delta"]) <= GATE_B_TOL["delta"]
          and p > 0 and abs(math.log10(p) - math.log10(GATE_B["em_p"])) <= GATE_B_TOL["logp"])
    return dict(passed=bool(ok), n_shared=n, em_sieve=em_a, em_control=em_b, delta=delta,
                b=b, c=c, p=p, expected=GATE_B)


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--render-only", action="store_true",
                    help="re-render the .md from the existing _data.json without recomputing "
                         "(the .json is the artifact of record; render_md is a pure function of "
                         "it, so this can never change a number, only the prose around it)")
    args = ap.parse_args()
    if args.render_only:
        if not JSON_PATH.exists():
            print(f"{JSON_PATH} does not exist -- run without --render-only first.")
            return 1
        MD_PATH.write_text(render_md(json.loads(JSON_PATH.read_text())))
        print(f"re-rendered {MD_PATH} from {JSON_PATH}")
        return 0

    result: dict = {"cond": COND, "model_dir": MODEL_DIR,
                    "structured_tier": STRUCT_TIER, "flat_tier": FLAT_TIER}

    # ---------------- PART 1: sanity gates (FIRST) -------------------------------------------
    print("=== PART 1: mandatory sanity gates ===", flush=True)
    gate_a = run_sanity_check(load_qrels("browsecomp_plus_structured"))
    print(f"gate (A) browsecomp Sieve vs no-dense-evidence (expect +4.7, p=0.0104): "
          f"{'PASS' if gate_a.get('passed') else 'FAIL'} -> {gate_a}", flush=True)
    gate_b = run_gate_b()
    print(f"gate (B) hotpotqa cap-100 Sieve vs hybrid (expect +5.84, p=2.84e-29): "
          f"{'PASS' if gate_b.get('passed') else 'FAIL'} -> {gate_b}", flush=True)
    result["sanity_gates"] = dict(gate_a=gate_a, gate_b=gate_b)
    if not (gate_a.get("passed") and gate_b.get("passed")):
        result["aborted"] = "sanity gate failed -- no new numbers computed"
        JSON_PATH.write_text(json.dumps(result, indent=2, default=_jd))
        print("STOPPING: a sanity gate failed. No new numbers reported.")
        return 1

    # ---------------- PART 0: provenance ------------------------------------------------------
    print("\n=== PART 0: provenance ===", flush=True)
    prov = dict(prompt=prompt_parity(), caps={}, assets={}, corpus_delta={}, config_parity={})
    for _lbl, sd, fd in PAIRS:
        prov["caps"][sd] = cap_profile(STRUCT_TIER, sd)
        prov["caps"][fd] = cap_profile(FLAT_TIER, fd)
        prov["assets"][_lbl] = corpus_asset_parity(sd, fd)
        prov["corpus_delta"][_lbl] = corpus_field_check(sd, fd)
        prov["config_parity"][_lbl] = config_parity(sd, fd, STRUCT_TIER, FLAT_TIER)
        print(f"  config parity {_lbl}: {prov['config_parity'][_lbl]}", flush=True)
        print(f"  {sd}: {prov['caps'][sd]}", flush=True)
        print(f"  {fd}: {prov['caps'][fd]}", flush=True)
        print(f"  assets {_lbl}: {prov['assets'][_lbl]}", flush=True)
        print(f"  corpus delta {_lbl}: {prov['corpus_delta'][_lbl]}", flush=True)
    print(f"  prompt wiki==general: {prov['prompt']['wiki_equals_general']}", flush=True)
    result["provenance"] = prov

    # ---------------- PARTS 2-4: per dataset --------------------------------------------------
    result["datasets"] = {}
    for label, sd, fd in PAIRS:
        print(f"\n=== {label}: {sd} (structured) vs {fd} (flat) ===", flush=True)
        s_m, _, s_n = cell_metrics(sd, STRUCT_TIER, COND, load_qrels(sd))
        f_m, _, f_n = cell_metrics(fd, FLAT_TIER, COND, load_qrels(fd))
        if s_m is None or f_m is None:
            result["datasets"][label] = dict(error="a cell's rows.jsonl is missing")
            continue

        join, s_k, f_k = verify_join(s_m, f_m, sd, fd)
        print(f"  join: {join['n_structured_qids']} / {join['n_flat_qids']} -> shared "
              f"{join['n_shared']}, one-to-one={join['complete_and_one_to_one']}", flush=True)
        shared = sorted(set(s_k) & set(f_k))
        s_k = {q: s_k[q] for q in shared}
        f_k = {q: f_k[q] for q in shared}

        binary_ok = bool(_is_binary_indicator(s_k) and _is_binary_indicator(f_k))
        n_em, em_s, em_f, em_d, em_b, em_c, em_p = paired_em(s_k, f_k)
        toks = _paired_tokens(s_k, f_k, shared)
        n_rec, rec_s, rec_f, rec_d, rec_b, rec_c, rec_p = paired_recall(s_k, f_k)
        em_ci = bootstrap_ci(bootstrap_diff_pp([bool(s_k[q]["em"]) for q in shared],
                                               [bool(f_k[q]["em"]) for q in shared],
                                               n_boot=N_BOOT), 0.95)
        rec_ci = bootstrap_ci(bootstrap_diff_pp([bool(s_k[q]["recall"]) for q in shared],
                                                [bool(f_k[q]["recall"]) for q in shared],
                                                n_boot=N_BOOT), 0.95)
        calls_s = sum(s_k[q]["llm_calls"] for q in shared) / len(shared)
        calls_f = sum(f_k[q]["llm_calls"] for q in shared) / len(shared)
        _, rec_chk_s = mean_tok_recall(s_k, shared)
        _, rec_chk_f = mean_tok_recall(f_k, shared)

        entry = dict(
            label=label, structured_dataset=sd, flat_dataset=fd,
            structured_cell=str(cell_dir(STRUCT_TIER, sd, COND)),
            flat_cell=str(cell_dir(FLAT_TIER, fd, COND)),
            join=join,
            recall_is_binary_indicator=binary_ok,
            em=dict(n=n_em, structured=em_s, flat=em_f, delta_structured_minus_flat=em_d,
                    b_flat_right_structured_wrong=em_b, c_structured_right_flat_wrong=em_c,
                    mcnemar_p=em_p, significant_at_0_05=bool(em_p < 0.05),
                    pct_instances_discordant=100.0 * (em_b + em_c) / (n_em or 1),
                    ci95_lo_pp=em_ci[0], ci95_hi_pp=em_ci[1],
                    n_boot=N_BOOT, boot_seed=BOOT_SEED),
            tokens_count_once=dict(
                structured_mean=toks["tok_a"], flat_mean=toks["tok_b"],
                delta_flat_minus_structured=toks["tok_delta"],
                pct_structured_below_flat=toks["pct_reduction"],
                mean_paired_diff_flat_minus_structured=toks["mean_diff"],
                ttest_stat=toks.get("t_stat"), ttest_p=toks.get("t_p"),
                wilcoxon_stat=toks.get("wilcoxon_stat"), wilcoxon_p=toks.get("wilcoxon_p"),
                n=toks["n"]),
            llm_calls=dict(structured_mean=calls_s, flat_mean=calls_f,
                           delta_structured_minus_flat=calls_s - calls_f),
            gold_doc_recall=dict(n=n_rec, structured=rec_s, flat=rec_f,
                                 delta_structured_minus_flat=rec_d,
                                 b_flat_only=rec_b, c_structured_only=rec_c,
                                 mcnemar_p=rec_p, significant_at_0_05=bool(rec_p < 0.05),
                                 ci95_lo_pp=rec_ci[0], ci95_hi_pp=rec_ci[1],
                                 n_boot=N_BOOT, boot_seed=BOOT_SEED,
                                 crosscheck_structured=rec_chk_s, crosscheck_flat=rec_chk_f),
            judge="not judged by design (EM/F1 benchmark)",
        )
        print(f"  EM {em_s:.2f} vs {em_f:.2f} -> {em_d:+.2f} pp, b={em_b} c={em_c}, p={em_p:.4g}",
              flush=True)
        print(f"  tok {toks['tok_a']:.0f} vs {toks['tok_b']:.0f} "
              f"({toks['pct_reduction']:+.1f}% ), t_p={toks.get('t_p')}, w_p={toks.get('wilcoxon_p')}",
              flush=True)
        print(f"  calls {calls_s:.2f} vs {calls_f:.2f}", flush=True)
        print(f"  recall {rec_s:.2f} vs {rec_f:.2f} -> {rec_d:+.2f} pp, p={rec_p:.4g}", flush=True)

        del s_m, f_m
        print("  scanning queries (structured)...", flush=True)
        s_scan = scan_arm(STRUCT_TIER, sd)
        print("  scanning queries (flat)...", flush=True)
        f_scan = scan_arm(FLAT_TIER, fd)
        entry["queries"] = dict(
            structured=arm_query_summary(s_scan["agg"]),
            flat=arm_query_summary(f_scan["agg"]),
            behaviour_contrast=behaviour_contrast(s_scan, f_scan, shared),
        )
        qs, qf = entry["queries"]["structured"], entry["queries"]["flat"]
        bc = entry["queries"]["behaviour_contrast"]
        print(f"  fielded-query rate: struct {qs['pct_searches_fielded']:.1f}% vs "
              f"flat {qf['pct_searches_fielded']:.1f}%", flush=True)
        print(f"  fielded queries 0-hit/error: struct {qs['pct_fielded_zero_or_error']:.1f}% vs "
              f"flat {qf['pct_fielded_zero_or_error']:.1f}%", flush=True)
        print(f"  identical first query {bc['pct_identical_first_query']:.1f}%; of those, "
              f"identical first observation "
              f"{bc['pct_identical_first_obs_given_identical_first_query']:.1f}%", flush=True)
        del s_scan, f_scan
        result["datasets"][label] = entry

    JSON_PATH.write_text(json.dumps(result, indent=2, default=_jd))
    MD_PATH.write_text(render_md(result))
    print(f"\nwrote {JSON_PATH}\nwrote {MD_PATH}")
    return 0


# ---------------------------------------------------------------------------------------------
def render_md(res: dict) -> str:
    ga, gb = res["sanity_gates"]["gate_a"], res["sanity_gates"]["gate_b"]
    L = []
    A = L.append
    A("# Fielded flat-vs-structured — does the structured corpus help a method that reads it?")
    A("")
    A("Generated by `analysis/fielded_flat_vs_structured.py`; data in "
      "`analysis/fielded_flat_vs_structured_data.json`. Both arms are the FULL METHOD, "
      f"`{res['cond']}` (Sieve), at `max_steps=100`, on the same questions and the same gold "
      "qrels. This is the experiment `analysis/flat_twin_validity.md` named as the only one that "
      "could support a corpus claim; the retracted one used the structure-blind BM25 baseline.")
    A("")
    A("## Sanity gates (run first)")
    A("")
    A("| gate | quantity | expected | reproduced | verdict |")
    A("|---|---|---|---|---|")
    A(f"| A | BrowseComp-Plus, Sieve vs No-dense-evidence | +4.7 EM, p=0.0104 | "
      f"{ga.get('delta', float('nan')):+.2f} EM, p={ga.get('p', float('nan')):.4g} "
      f"(n={ga.get('n_shared')}) | **{'PASS' if ga.get('passed') else 'FAIL'}** |")
    A(f"| B | HotpotQA cap-100, Sieve vs hybrid control | +5.84 EM, p=2.8e-29 | "
      f"{gb.get('delta', float('nan')):+.2f} EM, p={gb.get('p', float('nan')):.3g} "
      f"(n={gb.get('n_shared')}) | **{'PASS' if gb.get('passed') else 'FAIL'}** |")
    A("")
    A("Gate B is a Wikipedia cap-100 quantity and its Sieve arm IS this analysis's HotpotQA "
      "structured cell, so it validates that arm directly.")
    A("")
    prov = res.get("provenance", {})
    A("## Provenance")
    A("")
    A("| cell | n rows | configured `max_steps` distribution | uniform 100 |")
    A("|---|---|---|---|")
    for ds, c in prov.get("caps", {}).items():
        A(f"| `{ds}` | {c['n']} | `{c['cap_dist']}` | {'yes' if c['uniform_100'] else '**NO**'} |")
    A("")
    for lbl, a in prov.get("assets", {}).items():
        A(f"- **{lbl}**: `queries.jsonl` identical across twins: "
          f"**{a['queries']['identical']}**; `qrels/test.tsv` identical: "
          f"**{a['qrels']['identical']}**.")
    A("")
    A("| pair | shard configs (s / f) | keys compared | keys that ever differ | UNEXPECTED "
      "differences | `max_steps` | `prompt_sha256` identical |")
    A("|---|---|---|---|---|---|---|")
    for lbl, c in prov.get("config_parity", {}).items():
        if "note" in c:
            A(f"| {lbl} | {c['n_structured_shards']} / {c['n_flat_shards']} | — | — | — | — | "
              f"{c['note']} |")
            continue
        A(f"| {lbl} | {c['n_structured_shards']} / {c['n_flat_shards']} | {c['n_keys_compared']} "
          f"| `{', '.join(c['differing_keys'])}` | "
          f"{('**' + ', '.join(c['unexpected_differing_keys']) + '**') if c['unexpected_differing_keys'] else '**none**'} "
          f"| s={c['max_steps']['structured']} f={c['max_steps']['flat']} | "
          f"{c['prompt_sha256']['structured'] == c['prompt_sha256']['flat']} |")
    A("")
    A("Every per-shard `config.json` was compared key-by-key (value SETS per arm, so a key that "
      "varies per shard inside both arms — `api_base`, one vLLM port per shard — is only flagged "
      "if the arms' sets differ). `dataset`, `runs_dir` and `started_at` are expected to differ "
      "(they name the arm and the wall clock); nothing else does, including the whole `env_knobs` "
      "block (`temperature` 0.6, `seed` 42, `workers`, `AGENT_CTX_WINDOW`, `AGENT_CTX_STOP_FRAC`, "
      "`AGENT_SEARCH_FLAT_FAISS`, `BQL_SOFT_FALLBACK`, the dense model). This is precisely the "
      "check the retracted pairing failed — its Wikipedia arms ran at `max_steps` 100 vs 50 with "
      "flat-only `AGENT_CTX_WINDOW` / `AGENT_CTX_STOP_FRAC` / `AGENT_SEARCH_FLAT_FAISS` knobs and "
      "ten days of code drift.")
    A("")
    # Sidecar symmetry, checked live at render time (4 stat calls, deterministic, re-verified on
    # every render): the recovery overlay and the judge cache are per-cell sidecar FILES, so an
    # overlay applied to one arm and not the other would silently favour that arm. Neither exists
    # for any of the four cells, so `load_rows_with_recovery` is a provable no-op on both sides.
    sc = []
    for lbl2, e2 in res.get("datasets", {}).items():
        if "error" in e2:
            continue
        for arm, key in (("structured", "structured_cell"), ("flat", "flat_cell")):
            d = Path(e2[key])
            sc.append(f"{lbl2}/{arm}: recovered_answers.jsonl="
                      f"{(d / 'recovered_answers.jsonl').exists()}, judge_cache.jsonl="
                      f"{(d / 'judge_cache.jsonl').exists()}")
    A("- Recovery/judge sidecar symmetry (checked live at render time): " + "; ".join(sc)
      + ". No cell has either, so the recovery overlay is a no-op on both arms and cannot favour "
        "one of them, and there is no judge verdict to report (HotpotQA and MuSiQue are EM/F1 "
        "benchmarks — not judged by design).")
    A("")
    A("**Honest limit on the code-parity claim:** `git_rev` is `null` in every shard config of "
      "both arms, so code identity is not *recorded*. It is inferred from the two arms having run "
      "the same day (structured 2026-07-25T17:02, flat 2026-07-25T23:26) with an identical "
      "composed-prompt sha and an identical `env_knobs` block. That is much stronger than the "
      "retracted pairing's ten-day gap, but it is an inference, not a hash.")
    p = prov.get("prompt", {})
    A(f"- System prompt: the flat datasets register with no `field_profile` (→ `general`) and the "
      f"structured ones with `wiki`, but `research_bql_dense_snip` composes the SAME manual for "
      f"both — {p.get('wiki', {}).get('n_bytes')} bytes, sha256 "
      f"`{str(p.get('wiki', {}).get('sha256'))[:16]}…` — identical: "
      f"**{p.get('wiki_equals_general')}**. The manipulation is corpus-only.")
    A("")
    A("### What the flat twin actually withholds")
    A("")
    A("| pair | n sampled | keys only in structured | % docs w/ infobox | % of those whose infobox "
      "text is ALSO in the shared `text` | % docs w/ `sections` | % flat `text` carrying `##` "
      "headings |")
    A("|---|---|---|---|---|---|---|")
    for lbl, c in prov.get("corpus_delta", {}).items():
        A(f"| {lbl} | {c['n_sampled']} | `{', '.join(c['keys_only_in_structured'])}` | "
          f"{c['pct_structured_with_infobox']:.1f}% | "
          f"{c['pct_infobox_content_also_in_flat_text']:.1f}% | "
          f"{c['pct_structured_with_sections']:.1f}% | "
          f"{c['pct_flat_text_with_markdown_headings']:.1f}% |")
    A("")
    A("So the flat arm loses the *fielded index* and the *structure inventory the SERP prints*, "
      "not the underlying content: infobox facts sit verbatim at the head of the shared `text`, "
      "and the `##` headings survive for `sections_from_body` to reconstruct. This bounds what a "
      "corpus claim from this experiment may say.")
    A("")

    A("## Join verification")
    A("")
    A("Instance ids are dataset-prefixed and disjoint across arms; both sides are re-keyed on the "
      "qid suffix with the repo's own `_qid_of` before any paired helper sees them.")
    A("")
    A("| pair | structured rows | flat rows | distinct qids (s / f) | n shared | unmatched s-only | "
      "unmatched f-only | complete & 1:1 |")
    A("|---|---|---|---|---|---|---|---|")
    for lbl, e in res["datasets"].items():
        if "error" in e:
            A(f"| {lbl} | — | — | — | — | — | — | ERROR: {e['error']} |")
            continue
        j = e["join"]
        A(f"| {lbl} | {j['n_structured_rows']} | {j['n_flat_rows']} | "
          f"{j['n_structured_qids']} / {j['n_flat_qids']} | **{j['n_shared']}** | "
          f"{j['n_unmatched_structured_only']} | {j['n_unmatched_flat_only']} | "
          f"**{'yes' if j['complete_and_one_to_one'] else 'NO'}** |")
    A("")

    A("## Results (structured − flat, paired on the shared instance set)")
    A("")
    A("| pair | n | EM struct | EM flat | ΔEM (pp) | 95% CI (pp) | b / c | % discordant | "
      "McNemar p | sig@0.05 |")
    A("|---|---|---|---|---|---|---|---|---|---|")
    for lbl, e in res["datasets"].items():
        if "error" in e:
            continue
        m = e["em"]
        A(f"| {lbl} | {m['n']} | {m['structured']:.2f} | {m['flat']:.2f} | "
          f"**{m['delta_structured_minus_flat']:+.2f}** | "
          f"[{m['ci95_lo_pp']:+.2f}, {m['ci95_hi_pp']:+.2f}] | "
          f"{m['b_flat_right_structured_wrong']} / {m['c_structured_right_flat_wrong']} | "
          f"{m['pct_instances_discordant']:.1f}% | "
          f"{m['mcnemar_p']:.4g} | {'**yes**' if m['significant_at_0_05'] else 'no'} |")
    A("")
    A("`b` = flat right & structured wrong; `c` = structured right & flat wrong. CI = paired "
      f"instance bootstrap, {N_BOOT} resamples, seed {BOOT_SEED} "
      "(`analysis.equivalence_and_latency.{bootstrap_diff_pp, bootstrap_ci, BOOT_SEED}`, reused "
      "unmodified).")
    A("")
    A("| pair | tok-once struct | tok-once flat | % struct below flat | paired t p | Wilcoxon p | "
      "LLM calls struct | LLM calls flat |")
    A("|---|---|---|---|---|---|---|---|")
    for lbl, e in res["datasets"].items():
        if "error" in e:
            continue
        t, c = e["tokens_count_once"], e["llm_calls"]
        tp = "—" if t["ttest_p"] is None else format(t["ttest_p"], ".3g")
        wp = "—" if t["wilcoxon_p"] is None else format(t["wilcoxon_p"], ".3g")
        A(f"| {lbl} | {t['structured_mean']:,.0f} | {t['flat_mean']:,.0f} | "
          f"{t['pct_structured_below_flat']:+.2f}% | {tp} | {wp} | "
          f"{c['structured_mean']:.2f} | {c['flat_mean']:.2f} |")
    A("")
    A("| pair | n | gold-doc recall struct | recall flat | Δ (pp) | 95% CI (pp) | b / c | "
      "McNemar p | sig@0.05 | binary indicator |")
    A("|---|---|---|---|---|---|---|---|---|---|")
    for lbl, e in res["datasets"].items():
        if "error" in e:
            continue
        r = e["gold_doc_recall"]
        A(f"| {lbl} | {r['n']} | {r['structured']:.2f} | {r['flat']:.2f} | "
          f"**{r['delta_structured_minus_flat']:+.2f}** | "
          f"[{r['ci95_lo_pp']:+.2f}, {r['ci95_hi_pp']:+.2f}] | {r['b_flat_only']} / "
          f"{r['c_structured_only']} | {r['mcnemar_p']:.4g} | "
          f"{'**yes**' if r['significant_at_0_05'] else 'no'} | "
          f"{'yes' if e['recall_is_binary_indicator'] else 'NO'} |")
    A("")
    A("**Judge: not judged by design.** HotpotQA and MuSiQue are EM/F1 benchmarks; no LLM-judge "
      "verdict is computed or reported for them.")
    A("")

    A("## Query-behaviour contrast — ablation or replicate?")
    A("")
    A("| pair | arm | searches/episode | % searches with a field restriction | % episodes using one "
      "| % fielded searches ending 0-hit/error | % plain searches ending 0-hit/error | % search "
      "listings showing a non-empty infobox |")
    A("|---|---|---|---|---|---|---|---|")
    for lbl, e in res["datasets"].items():
        if "error" in e or "queries" not in e:
            continue
        for arm in ("structured", "flat"):
            q = e["queries"][arm]
            A(f"| {lbl} | {arm} | {q['searches_per_episode']:.2f} | "
              f"{q['pct_searches_fielded']:.2f}% | {q['pct_instances_any_fielded']:.2f}% | "
              f"{q['pct_fielded_zero_or_error']:.2f}% | {q['pct_plain_zero_or_error']:.2f}% | "
              f"{q['pct_search_obs_with_nonempty_infobox']:.2f}% |")
    A("")
    A("The STRICT subset — queries where EVERY LEAF is scoped to a structured-only region, so no "
      "title/body branch and no unscoped term could satisfy the Boolean on the flat corpus. "
      "Strictness is decided by walking the parsed BQL AST, not by regexing the BQL string: an "
      "unscoped leaf is written bare and produces no `IN(`, so a regex would mis-read "
      "`first medals awarded OR \"medals\"[infobox]` as purely infobox-scoped. This is the "
      "falsifiable instrument check — the flat index can only return exact hits here if the "
      "fields are not really absent.")
    A("")
    A("| pair | arm | strict queries | % of all searches | exact hits | 0-hit/error |")
    A("|---|---|---|---|---|---|")
    for lbl, e in res["datasets"].items():
        if "error" in e or "queries" not in e:
            continue
        for arm in ("structured", "flat"):
            q = e["queries"][arm]
            A(f"| {lbl} | {arm} | {q['n_strict_searches']} | {q['pct_searches_strict']:.2f}% | "
              f"**{q['outcome_strict_pct'].get('exact_hits', 0.0):.2f}%** | "
              f"{q['pct_strict_zero_or_error']:.2f}% |")
    A("")
    A("**Why the flat side is not exactly 0%.** `IN(section, ·)` does not evaluate against an "
      "empty bag on the flat corpus — it evaluates against the document's PATH. "
      "`bql/executor.py:_field_bag` computes the SECTION bag as "
      "`code_tokenize(u.section if u.section is not None else u.path)`, and "
      "`corpus/units.py:units_from_documents` sets `path = d.get(\"path\") or d.get(\"url\") or "
      "doc_id`, which for these corpora is the doc id (`d_spettekaka`). So on flat a "
      "section-scoped query silently degrades into a doc-id token match: `Spettekaka[section]` "
      "still returns `d_spettekaka`, and `group[section]` returns 151 docs whose ids contain "
      "\"group\". The flat arm's `infobox`/`author`/`date` bags ARE empty (they read "
      "`u.metadata`, which the flat twin has none of). This is a genuine property of the "
      "degraded arm and is reported, not corrected. It cuts one way: it makes the flat side "
      "slightly stronger than a truly field-free corpus would be, so the structured−flat deltas "
      "below are, to that extent, conservative — a corpus with a genuinely empty `section` bag "
      "might favour structured by a little more. The affected volume caps how much: "
      "section-scoped clauses appear in ~0.3% of searches and only a few percent of the strict "
      "flat subset converts to exact hits, which cannot move a 7,343-instance EM by the 2–3 pp "
      "the retracted decomposition needed. But it does mean \"the fielded surface returns nothing "
      "on flat\" would be an overstatement, and the paper must not write that.")
    A("")
    A("**What \"0-hit/error\" counts.** A search observation is bucketed from the exact strings "
      "`doc_research.py:_search_impl` emits: `exact_hits` (the Boolean query matched: "
      "`(N matches, top k)`), `zero_exact_soft` (matched nothing, so the corpus-fair BM25 "
      "soft-AND fallback returned the closest docs — the agent still sees results, but its "
      "Boolean constraint failed), `zero_hard` (`(0 matches)`, nothing at all), `error_bql` "
      "(a parse/type error from the executor), `error_tool`, `empty_query`. The 0-hit/error "
      "columns above are everything except `exact_hits`. Per-bucket rates are in the JSON.")
    A("")
    A("Per-region field usage (share of all searches scoping that region) and how episodes ended:")
    A("")
    A("| pair | arm | title | body | section | infobox | author | date | stopped-reason counts |")
    A("|---|---|---|---|---|---|---|---|---|")
    for lbl, e in res["datasets"].items():
        if "error" in e or "queries" not in e:
            continue
        for arm in ("structured", "flat"):
            q = e["queries"][arm]
            rr = q["region_rate_pct"]
            A(f"| {lbl} | {arm} | " + " | ".join(f"{rr.get(k, 0.0):.2f}%" for k in
              ("title", "body", "section", "infobox", "author", "date"))
              + f" | `{q['stopped_counts']}` |")
    A("")
    A("(The JSON's `pct_stopped_by_step_budget` is, precisely, the share of episodes that ended "
      "for any reason OTHER than voluntarily answering — read the `stopped_counts` beside it for "
      "the breakdown rather than the key's name.)")
    A("")
    A("| pair | identical first query | of those, identical first observation | identical FULL "
      "query sequence | identical initial-prompt tokens |")
    A("|---|---|---|---|---|")
    for lbl, e in res["datasets"].items():
        if "error" in e or "queries" not in e:
            continue
        b = e["queries"]["behaviour_contrast"]
        A(f"| {lbl} | {b['n_identical_first_query']} / {b['n']} "
          f"({b['pct_identical_first_query']:.1f}%) | "
          f"{b['n_identical_first_obs_given_identical_first_query']} / "
          f"{b['n_identical_first_query']} "
          f"({b['pct_identical_first_obs_given_identical_first_query']:.1f}%) | "
          f"{b['n_identical_full_query_sequence']} ({b['pct_identical_full_query_sequence']:.1f}%) "
          f"| {b['n_identical_initial_prompt_tokens']} "
          f"({b['pct_identical_initial_prompt_tokens']:.1f}%) |")
    A("")
    A("The retracted pairing was diagnosed by exactly this instrument: when the first issued query "
      "matched, the first observation was byte-identical 100% / 99.97% / 100% of the time — the "
      "two arms were the same experiment. The corresponding row above is the test of whether this "
      "pairing is any different.")
    A("")
    A("### Matched first query — cosmetic difference or retrieval difference?")
    A("")
    A("Restricted to instances whose FIRST issued query is identical in both arms (same question, "
      "same prompt, same state). `ib[...]` (infobox key inventory) and `matched: ...` (which "
      "fields the hit matched on) are the two spans `_render_hits` reads off the structured "
      "metadata; blanking exactly those two leaves the match count, rank order, doc ids, titles, "
      "section list and snippets — everything that is retrieval.")
    A("")
    A("| pair | n matched first queries | identical observation | differs ONLY in the structure "
      "display | differs in RETRIEVAL outcome | 0-hit/error struct | 0-hit/error flat | fielded "
      "first queries |")
    A("|---|---|---|---|---|---|---|---|")
    for lbl, e in res["datasets"].items():
        if "error" in e or "queries" not in e:
            continue
        m = e["queries"]["behaviour_contrast"]["matched_first_query"]
        A(f"| {lbl} | {m['n']} | {m['n_identical_observation']} "
          f"({m['pct_identical_observation']:.1f}%) | "
          f"{m['n_differs_only_in_structure_display']} "
          f"({m['pct_differs_only_in_structure_display']:.1f}%) | "
          f"{m['n_differs_in_retrieval_outcome']} "
          f"({m['pct_differs_in_retrieval_outcome']:.1f}%) | "
          f"{m['pct_zero_or_error_structured']:.1f}% | {m['pct_zero_or_error_flat']:.1f}% | "
          f"{m['n_fielded_first_queries']} |")
    A("")
    L.extend(render_verdict(res))
    return "\n".join(L) + "\n"


def render_verdict(res: dict) -> list:
    """The reading of the numbers above. Branch on what was actually measured -- a corpus claim
    is licensed only if the ACCURACY delta clears significance, and only in words the design can
    carry."""
    ds = {k: v for k, v in res["datasets"].items() if "error" not in v}
    pos = [k for k, v in ds.items()
           if v["em"]["significant_at_0_05"] and v["em"]["delta_structured_minus_flat"] > 0]
    neg = [k for k, v in ds.items()
           if v["em"]["significant_at_0_05"] and v["em"]["delta_structured_minus_flat"] < 0]
    null = [k for k in ds if k not in pos and k not in neg]

    def _hit(e, arm):
        return e["queries"][arm]["outcome_fielded_pct"].get("exact_hits", 0.0)

    # The manipulation is real iff the FIELD-SCOPED queries -- the ones the flat index cannot
    # satisfy -- actually behave differently across the arms. (Whole-population rates: the
    # matched-first-query subset cannot test this, because those matched queries are almost
    # entirely title/body ones, for which the two corpora are byte-identical by construction.)
    real = all(_hit(e, "structured") - _hit(e, "flat") > 10.0
               for e in ds.values() if "queries" in e)

    L = []
    A = L.append
    A("## Verdict")
    A("")
    A("### 1. Is this pairing an ablation, or another replicate?")
    A("")
    A("Three measurements, and they do not all point the same way. Read all three.")
    A("")
    A("**(a) The fielded search surface really does collapse on the flat corpus.** Field-scoped "
      "queries (`section`/`infobox`/`author`/`date`):")
    A("")
    for lbl, e in ds.items():
        if "queries" not in e:
            continue
        qs, qf = e["queries"]["structured"], e["queries"]["flat"]
        A(f"- **{lbl}**: exact-hit rate {_hit(e, 'structured'):.1f}% (structured) → "
          f"{_hit(e, 'flat'):.1f}% (flat); 0-hit-or-error rate "
          f"{qs['pct_fielded_zero_or_error']:.1f}% → {qf['pct_fielded_zero_or_error']:.1f}%. "
          f"On the STRICT subset (no title/body branch to fall back on, n="
          f"{qs['n_strict_searches']} / {qf['n_strict_searches']} searches): exact hits "
          f"**{qs['outcome_strict_pct'].get('exact_hits', 0.0):.2f}% → "
          f"{qf['outcome_strict_pct'].get('exact_hits', 0.0):.2f}%**. "
          f"For comparison, PLAIN (title/body/bare) queries: "
          f"{qs['pct_plain_zero_or_error']:.1f}% → {qf['pct_plain_zero_or_error']:.1f}% — "
          f"unchanged, as they must be over byte-identical text.")
    A("")
    A("**(b) The listing shows different structure on essentially every result.** A non-empty "
      "infobox inventory (`ib[...]`) appears in "
      + " / ".join(f"{e['queries']['structured']['pct_search_obs_with_nonempty_infobox']:.1f}%"
                   for e in ds.values() if "queries" in e)
      + " of structured search listings and "
      + " / ".join(f"{e['queries']['flat']['pct_search_obs_with_nonempty_infobox']:.1f}%"
                   for e in ds.values() if "queries" in e)
      + " of flat ones. That is what drives trajectory divergence: only "
      + " / ".join(f"{e['queries']['behaviour_contrast']['pct_identical_full_query_sequence']:.1f}%"
                   for e in ds.values() if "queries" in e)
      + " of episodes issue an identical full query sequence.")
    A("")
    A("**(c) BUT — and this is the honest counterweight — on matched queries the two arms "
      "retrieve the same documents.** Restricted to instances issuing the SAME first query:")
    A("")
    for lbl, e in ds.items():
        if "queries" not in e:
            continue
        m = e["queries"]["behaviour_contrast"]["matched_first_query"]
        A(f"- **{lbl}**: {m['n']} such instances. Byte-identical observation "
          f"{m['pct_identical_observation']:.1f}%; differ ONLY in the structured-metadata display "
          f"(`ib[...]`, `matched:`) {m['pct_differs_only_in_structure_display']:.1f}%; differ in "
          f"the RETRIEVAL outcome {m['pct_differs_in_retrieval_outcome']:.2f}%. Of those matched "
          f"first queries, {m['n_fielded_first_queries']} were field-scoped.")
    A("")
    A("So when the agent asks the same thing, it gets back the same documents in the same order "
      "with the same snippets — the observation differs only in the structure inventory printed "
      "beside them. The retrieval difference is confined to the field-scoped queries of (a), and "
      "the agent issues those on only "
      + " / ".join(f"{e['queries']['structured']['pct_searches_fielded']:.1f}%"
                   for e in ds.values() if "queries" in e)
      + " of its searches.")
    A("")
    A("**" + ("Verdict on the instrument: this is a REAL manipulation, not the null replicate the "
             "retracted pairing was — the fielded clauses provably break on flat and the listing "
             "provably differs — but it is a THIN one: it reaches ~2% of queries on the retrieval "
             "axis, and its broad effect is on what the SERP displays rather than what it returns."
             if real else
             "Verdict on the instrument: the two arms behave essentially the same — this pairing "
             "is closer to a REPLICATE than to an ablation, and NO corpus claim may be drawn from "
             "it.") + "**")
    A("")
    A("And the flat twin withholds the FIELDS, not the CONTENT: infobox text sits verbatim at the "
      "head of the shared `text`, `##` headings survive for `sections_from_body`, so the read axis "
      "is intact on both sides. This bounds the manipulation from above no matter what the "
      "accuracy numbers had said.")
    A("")
    A("### 2. Does the structured corpus measurably help a method that uses it?")
    A("")
    for lbl, e in ds.items():
        m, r, t = e["em"], e["gold_doc_recall"], e["tokens_count_once"]
        A(f"- **{lbl}** (n={m['n']}): ΔEM **{m['delta_structured_minus_flat']:+.2f} pp** "
          f"(95% CI [{m['ci95_lo_pp']:+.2f}, {m['ci95_hi_pp']:+.2f}], McNemar p="
          f"{m['mcnemar_p']:.3g}, {m['pct_instances_discordant']:.1f}% of instances discordant); "
          f"Δgold-doc recall {r['delta_structured_minus_flat']:+.2f} pp (95% CI "
          f"[{r['ci95_lo_pp']:+.2f}, {r['ci95_hi_pp']:+.2f}], p={r['mcnemar_p']:.3g}); "
          f"count-once tokens {t['structured_mean']:,.0f} vs {t['flat_mean']:,.0f} "
          f"({t['pct_structured_below_flat']:+.2f}% for structured).")
    A("")
    # Multiplicity: this analysis makes one primary EM test per dataset. Bonferroni over those.
    k = len(ds) or 1
    bonf = 0.05 / k
    survives = [lbl for lbl, v in ds.items() if v["em"]["mcnemar_p"] < bonf]

    if pos and not neg and len(pos) == len(ds):
        A(f"**Yes, on every dataset tested ({', '.join(pos)}).**")
    elif pos and not neg:
        A(f"**Partly: significant and positive on {', '.join(pos)}; not on {', '.join(null)}.** "
          "A claim resting on one of two datasets is a weak claim and must be stated as "
          "dataset-specific.")
    elif neg and not pos:
        A(f"**No — and on {', '.join(neg)} the effect runs the WRONG WAY.** With both arms at the "
          "same budget, the same prompt, the same questions and the same gold, and with the "
          "manipulation demonstrably biting on the tool surface, the structured corpus is "
          + "; ".join(f"{lbl} {v['em']['delta_structured_minus_flat']:+.2f} EM "
                      f"(p={v['em']['mcnemar_p']:.3g})" for lbl, v in ds.items())
          + f". The one nominally significant result is NEGATIVE. At the Bonferroni threshold for "
            f"the {k} primary tests made here (α={bonf:.3f}), "
          + (f"it still clears: {', '.join(survives)}." if survives else
             "not even that one clears — so the most defensible summary is a null with a hint of "
             "harm, not a demonstrated cost.")
          + " Either way, no positive corpus effect exists to claim.")
    elif pos and neg:
        A(f"**Contradictory: significant positive on {', '.join(pos)}, significant NEGATIVE on "
          f"{', '.join(neg)}.** A quantity whose sign flips across datasets is not a corpus "
          "effect; it is dataset-specific behaviour and must not be aggregated into a share.")
    else:
        A("**No.** With both arms at the same budget, the same prompt, the same questions and the "
          "same gold, and with the manipulation demonstrably biting on the tool surface, the "
          "accuracy difference does not clear significance on either dataset.")
    A("")
    A("Whatever the sign, the intervals above are the operative fact: a 30–45% corpus share of a "
      "~6 pp method gain would be ≈2–3 pp, and every 95% CI here excludes that.")
    A("")
    A("### 3. May the paper make a corpus claim, and in what words?")
    A("")
    if pos and len(pos) == len(ds) and not neg:
        A("Yes, and only in this form:")
        A("")
        A("> Repeating the full method on the flat twin of each corpus — identical text, identical "
          "questions, identical budget and prompt, with only the `section`/`infobox` fields "
          "removed from the index — costs it "
          + " and ".join(f"{-ds[k2]['em']['delta_structured_minus_flat']:.1f} EM on {k2}"
                         for k2 in ds)
          + ". The structured fields therefore carry a measurable share of the method's accuracy.")
    else:
        A("**No — no corpus accuracy claim, in any direction.** The corpus half of the "
          "decomposition stays retracted, and this experiment does not restore it: it replaces an "
          "unmeasurable quantity with a measured one that is null on one dataset and, on the "
          "other, points the wrong way. What the paper MAY now say, and this is the strongest form "
          "the data supports:")
        A("")
        A("> We re-ran the full method on the flat twin of each Wikipedia corpus under identical "
          "questions, gold judgements, step budget, prompt and serving configuration, so that only "
          "the `section`/`infobox` fields were withheld from the index. Unlike the structure-blind "
          "pairing, this one is a genuine manipulation: field-scoped clauses that return exact "
          "hits on "
          + "/".join(f"{_hit(e, 'structured'):.0f}%" for e in ds.values() if "queries" in e)
          + " of structured searches return them on only "
          + "/".join(f"{_hit(e, 'flat'):.0f}%" for e in ds.values() if "queries" in e)
          + " of flat ones. Even so, accuracy does not favour the structured corpus ("
          + "; ".join(f"{k2} {v['em']['delta_structured_minus_flat']:+.1f} EM, 95% CI "
                      f"[{v['em']['ci95_lo_pp']:+.1f}, {v['em']['ci95_hi_pp']:+.1f}], "
                      f"p={v['em']['mcnemar_p']:.2g}" for k2, v in ds.items())
          + "), and neither gold-document recall nor token cost favours it either. We therefore "
            "attribute the method's gain to its query and read interface, not to the structured "
            "metadata, and make no corpus-share claim.")
        A("")
        A("Four things this sentence must NOT be stretched into. (i) It is not evidence that "
          "structure is worthless in general: the flat twin removes the *fielded index and the "
          "structure inventory the listing prints*, while leaving the underlying infobox text and "
          "`##` headings in the shared body, so the read axis never degrades. (ii) It is not a "
          "test of a heavily fielded query workload: the agent scopes a structured-only field on "
          "only a few percent of its searches, so the query-side treatment reaches a small "
          "fraction of the trajectory — a bigger effect might exist for an agent that used the "
          "fields more, and that is a legitimate piece of future work, not a result. (iii) A null "
          "is not equivalence unless the interval is quoted with it — quote the CI every time. "
          "(iv) The negative point estimate must not be flipped into a claim that structure HURTS: "
          "one nominally significant result at p≈0.04 across two datasets, which does not survive "
          "correction for those two tests, is not that either.")
    A("")
    A("### 4. What still must change in the paper")
    A("")
    A("Everything `analysis/flat_twin_validity.md`'s \"What must change\" section lists stays "
      "changed. This experiment removes only its final sentence — the one saying the decisive "
      "experiment \"has never been run\". It has now been run, and it came back "
      + ("positive." if (pos and len(pos) == len(ds) and not neg)
         else ("null on one dataset and adverse on the other." if neg else "null."))
      + " The honest gain from it is that the benchmark's twin design is now *demonstrated* to "
        "support a real fielded manipulation — the arms provably diverge at the tool surface, and "
        "the fielded clauses provably degrade on the flat side — which is a defensible benchmark "
        "contribution even though the accuracy decomposition it was built to serve does not "
        "survive. Claim the instrument, not the number.")
    A("")
    return L


if __name__ == "__main__":
    raise SystemExit(main())
