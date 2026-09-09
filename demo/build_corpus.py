"""One-time OFFLINE curation of the demo's BrowseComp-Plus subsample -> corpus_data.json.

Needs internet (Hugging Face) once; needs NO OpenAI key (the published structured corpus is
already sectioned by the paper's own batch pass). Two sources, joined on docid (see
demo/README.md, "Data provenance"):

  - Tevatron/browsecomp-plus (streamed): query text + answer + gold/negative/evidence docids
    for QUERY_IDS, de-obfuscated via corpus_build.browsecomp_plus.build.transform_decrypt.
  - wshuai190/browsecomp-plus-structured-full structured/corpus.jsonl (stream-filtered over
    HTTP, early-stopped): the matching docs' title/author/date/sections.

    python demo/build_corpus.py                       # full run (~10-50 min: streams
                                                               #   through a 6.1GB remote file)
    python demo/build_corpus.py --cache /tmp/bcp_subsample_structured.jsonl
                                                               # reuse an already-filtered dump
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

# Doc pools are pulled for ALL of these queries (gold + negatives + evidence become the
# collection). 798/792 were the original curated pair; their pools stay as distractors and
# 798's gold feeds the warm-up. The five 2026-08-07 additions are REAL wins picked from the
# author's own pairwise gain/loss analysis on browsecomp_plus_structured_full: on the full
# corpus, sieve_bm25 answered all five correctly while the Search-Visit baseline missed all
# five at 5-18x the tokens (see PAPER_RESULTS below).
QUERY_IDS = ("798", "792", "78", "159", "579", "401", "1148")
# Only these become example questions on the page (798/792 defeat every affordable model, so
# they made a discouraging first click; their docs remain in the collection).
DEMO_QUESTION_IDS = ("78", "159", "579", "401", "1148")
GOLD_DOCIDS = ("37133", "39666", "41817", "51481")
# Collection sizing: gold and evidence docs are always kept for EVERY pooled query (they
# make the questions answerable, including intermediate hops); NEGATIVES are capped per
# query at the HARDEST 30 — ranked by BM25 score against the query itself (the repo's own
# BM25Local over the row's decrypted doc texts), not taken in dataset order. Hard negatives
# are the distractors that actually compete with the gold docs in a search, so the demo's
# difficulty survives the downsampling. ~30/query keeps the 7-query union around 250-300
# docs and the JSON near 10MB.
NEG_CAP = 30

# Measured on the paper's full-corpus runs (author-supplied, 2026-08-07), for the two arms
# this demo ships: sieve_bm25 (the demo's "Sieve") and visit (the demo's "Search-Visit").
# Shown on the example cards so a visitor sees the real stakes before spending a cent.
PAPER_RESULTS = {
    "78":   {"sieve": {"correct": True,  "tokens": 13226,  "calls": 13},
             "visit": {"correct": False, "tokens": 239501, "calls": 41}},
    "159":  {"sieve": {"correct": True,  "tokens": 13485,  "calls": 9},
             "visit": {"correct": False, "tokens": 95011,  "calls": 100}},
    "579":  {"sieve": {"correct": True,  "tokens": 23995,  "calls": 72},
             "visit": {"correct": False, "tokens": 110167, "calls": 100}},
    "401":  {"sieve": {"correct": True,  "tokens": 19542,  "calls": 25},
             "visit": {"correct": False, "tokens": 124406, "calls": 100}},
    "1148": {"sieve": {"correct": True,  "tokens": 20130,  "calls": 31},
             "visit": {"correct": False, "tokens": 128578, "calls": 100}},
}
STRUCTURED_URL = ("https://huggingface.co/datasets/wshuai190/browsecomp-plus-structured-full"
                  "/resolve/main/structured/corpus.jsonl")
OUT = HERE / "corpus_data.json"

# Outlier cap: purely demo-UX/cost motivated (repo size does NOT require it — the whole
# subsample is ~3.5MB). The 40K floor keeps every known gold section intact (largest:
# 37,225 chars in doc 51481).
CAP_CHARS = 80_000
CAP_FLOOR = 40_000
CAP_MARK = "…(truncated for demo)"

# Two DEMO-ONLY data repairs. The published corpus is used verbatim otherwise; both entries
# below are recorded here (rather than hand-edited into the JSON) so the build stays reproducible.
#
# 1. Titles the published corpus ships EMPTY. Without a title a document renders as its bare
#    docid in the result cards, so the agent cannot recognise it — fatal for 51481, which is
#    the gold document for the creatine question. Each title below is copied from that
#    document's OWN body text, not invented (51481 cites itself: "Oral creatine
#    supplementation: A potential adjunct therapy for rheumatoid arthritis patients. World J
#    Rheumatol 2014"). Verify with: grep -o 'Oral creatine[^.]*' on the doc's body.
TITLE_FIXUPS = {
    "51481": "Oral creatine supplementation: A potential adjunct therapy for "
             "rheumatoid arthritis patients",
    # distractors that ship untitled; titles taken from each doc's own opening lines
    "28645": "Deciphering Urban Breccia: Hidden and Visible Layers of Istanbul "
             "(13th Space Syntax Symposium)",
    "8844": "AvidPlay FAQ: Terminologies and Definitions",
    "9170": "Hacı Ömer Sabancı Holding A.Ş. Annual Report 2016",
}
# 2. Documents whose crawl FAILED upstream — the whole body is a scraper error page, so they
#    are noise rather than distractors. 30615's entire text is "Request unsuccessful. Incapsula
#    incident ID: ...".
DROP_DOCIDS = {"30615"}

# A DEMO-AUTHORED warm-up question (not a BrowseComp query). The two real benchmark questions
# below are deliberately adversarial multi-hop puzzles that gpt-4o-mini reliably fails; a
# visitor's first click should show the mechanism working end to end, so this asks for the same
# fact as query 798 in a directly searchable way. Its gold is that query's own answer.
WARMUP = {"query_id": "demo-warmup", "label": "warm-up · answerable in a few steps",
          "question": "Which women's college in Delhi was established in 1956, and what is it "
                      "called today?",
          # gold is the SHORT form: both strategies answer at least this much, and the
          # page's check is a substring test, so the longer official name still matches.
          "answer": "Lady Shri Ram College", "gold_docids": [], "paper": None}


def cap_body(body: str, cap: int = CAP_CHARS) -> str:
    assert cap >= CAP_FLOOR
    if len(body) <= cap:
        return body
    return body[:cap] + CAP_MARK


def cap_sections(sections: list, cap: int = CAP_CHARS) -> list:
    """Truncate the SECTION LIST so the derived `## heading\\ntext` body stays under `cap`
    characters: sections are kept whole until the budget runs out, the section that crosses
    it is cut with a marker, and the rest are dropped. The loader derives the body from
    these sections, so capping here caps everything downstream."""
    out, used = [], 0
    for h, t in sections:
        cost = len(h) + len(t) + 4                     # "## " + heading + newline + text
        if used + cost <= cap:
            out.append([h, t]); used += cost
        else:
            room = max(cap - used - len(h) - 4, 0)
            out.append([h, t[:room] + CAP_MARK])
            break
    return out


def record_to_doc(rec: dict) -> dict:
    """A structured/corpus.jsonl record -> the JSON doc shape corpus.py loads. ONLY the
    sections are stored — the loader derives the `## heading` body from them, so the
    collection is not written to disk twice (body + sections doubled the old file)."""
    doc_id = str(rec["_id"])
    sections = cap_sections([[s.get("heading") or "(untitled)", s.get("text") or ""]
                             for s in (rec.get("sections") or [])])
    title = rec.get("title") or TITLE_FIXUPS.get(doc_id) or doc_id
    return {"_id": doc_id, "title": title, "sections": sections,
            "metadata": {"author": rec.get("author") or "", "date": rec.get("date") or ""}}


def hardest_negatives(query: str, negatives: list, cap: int = NEG_CAP) -> list:
    """The `cap` negatives that BM25 ranks HIGHEST against the query — the distractors that
    genuinely compete with the gold docs in a lexical search. `negatives` are the row's
    decrypted {docid,url,text} entries; scoring uses the repo's own BM25Local so "hard"
    here means hard for the demo's actual search engine, not for some other ranker."""
    if len(negatives) <= cap:
        return [str(d["docid"]) for d in negatives]
    from agent_search.corpus.units import units_from_documents
    from agent_search.retrievers.lexical.bm25 import BM25Local
    units = units_from_documents(
        [{"_id": str(d["docid"]), "text": str(d.get("text") or "")} for d in negatives])
    ranked = BM25Local().index(units).search(query, k=cap)
    keep = list(ranked)
    if len(keep) < cap:                      # BM25 can return < k on zero-overlap docs
        seen = set(keep)
        keep += [str(d["docid"]) for d in negatives if str(d["docid"]) not in seen]
    return keep[:cap]


def pull_queries() -> tuple[list[dict], set[str], dict]:
    """Stream Tevatron/browsecomp-plus for QUERY_IDS -> (question rows, union of target
    docids, per-query {gold, evidence} pools for the answerability assertions)."""
    from datasets import load_dataset
    from corpus_build.browsecomp_plus.build import transform_decrypt
    # NOT streaming: `hf download Tevatron/browsecomp-plus` puts the shards in the local HF
    # cache, and a plain load reads them from disk. Streaming mode re-fetches from the CDN
    # every run regardless of the cache, which is what made rebuilds hang on a flaky VPN.
    ds = load_dataset("Tevatron/browsecomp-plus", split="test")
    questions, targets, pools = [], set(), {}
    for row in ds:
        if str(row.get("query_id")) not in QUERY_IDS:
            continue
        row = transform_decrypt(row)
        qid = str(row["query_id"])
        query = str(row.get("query") or "")
        gold = [str(d["docid"]) for d in (row.get("gold_docs") or [])]
        evidence = [str(d["docid"]) for d in (row.get("evidence_docs") or [])]
        negatives = hardest_negatives(query, row.get("negative_docs") or [])
        pools[qid] = {"gold": gold, "evidence": evidence}
        if qid in DEMO_QUESTION_IDS:
            questions.append({"query_id": qid, "question": query,
                              "label": "real BrowseComp-Plus question",
                              "answer": str(row.get("answer") or ""), "gold_docids": gold,
                              "paper": PAPER_RESULTS.get(qid)})
        targets |= set(gold) | set(evidence) | set(negatives)
        if len(pools) == len(QUERY_IDS):
            break
    assert len(questions) == len(DEMO_QUESTION_IDS), \
        f"found only {[q['query_id'] for q in questions]}"
    return questions, targets, pools


def local_corpus_path() -> Path | None:
    """The full structured corpus, if `hf download wshuai190/browsecomp-plus-structured-full`
    has been run (resumable, one-time, ~6.1GB) — then every rebuild is a local scan measured
    in seconds instead of an hour of re-streaming."""
    try:
        from huggingface_hub import hf_hub_download
        return Path(hf_hub_download("wshuai190/browsecomp-plus-structured-full",
                                    "structured/corpus.jsonl", repo_type="dataset",
                                    local_files_only=True))
    except Exception:
        return None


def pull_structured(targets: set[str], cache: Path | None) -> list[dict]:
    """The structured records for `targets`: cache dump first, then the locally downloaded
    full corpus (see `local_corpus_path`), then — last resort — stream-filtering the 6.1GB
    remote file (early-stop once every target is found)."""
    found = {}
    if cache and cache.exists():
        recs = [json.loads(l) for l in cache.open() if l.strip()]
        found = {str(r["_id"]): r for r in recs if str(r["_id"]) in targets}
        if set(found) == targets:
            print(f"cache hit: all {len(targets)} docs from {cache}")
            return list(found.values())
        print(f"cache covers {len(found)}/{len(targets)}; streaming the remote file "
              f"for the {len(targets) - len(found)} missing")
    missing = targets - set(found)
    local = local_corpus_path()
    if local and missing:
        print(f"scanning local corpus {local} for {len(missing)} docs")
        with local.open("rb") as fh:
            for line in fh:
                if not line.strip():
                    continue
                rec = json.loads(line)
                rid = str(rec.get("_id"))
                if rid in missing and rid not in found:
                    found[rid] = rec
                    if len(found) == len(targets):
                        break
        missing = targets - set(found)
    if not missing:
        if cache:
            with cache.open("w") as fh:
                for rec in found.values():
                    fh.write(json.dumps(rec) + "\n")
            print(f"cache refreshed: {len(found)} docs -> {cache}")
        return list(found.values())
    print(f"local scan left {len(missing)} unresolved; falling back to the remote stream")
    import requests
    with requests.get(STRUCTURED_URL, stream=True, timeout=600) as r:
        r.raise_for_status()
        buf = b""
        for chunk in r.iter_content(chunk_size=1 << 20):
            buf += chunk
            *complete, buf = buf.split(b"\n")
            for line in complete:
                if not line:
                    continue
                rec = json.loads(line)
                rid = str(rec.get("_id"))
                if rid in missing and rid not in found:
                    found[rid] = rec
            if len(found) == len(targets):
                break
    still_missing = targets - set(found)
    assert not still_missing, f"structured corpus missing docids: {sorted(still_missing)[:10]}"
    if cache:
        with cache.open("w") as fh:
            for rec in found.values():
                fh.write(json.dumps(rec) + "\n")
        print(f"cache refreshed: {len(found)} docs -> {cache}")
    return list(found.values())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", type=Path, default=None,
                    help="an already stream-filtered structured/corpus.jsonl subset")
    args = ap.parse_args()
    questions, targets, pools = pull_queries()
    print(f"queries {QUERY_IDS} -> {len(targets)} unique target docids")
    docs = sorted((record_to_doc(r) for r in pull_structured(targets, args.cache)
                   if str(r["_id"]) not in DROP_DOCIDS),
                  key=lambda d: d["_id"])
    have = {d["_id"] for d in docs}
    for gold in GOLD_DOCIDS:
        assert gold in have, f"gold doc {gold} missing"
    # EVERY pooled query (the five examples AND 798/792) must remain fully answerable:
    # all gold and all evidence docs present, minus documents dropped as broken crawls.
    for qid, pool in pools.items():
        for kind in ("gold", "evidence"):
            missing = [d for d in pool[kind] if d not in have and d not in DROP_DOCIDS]
            assert not missing, f"query {qid} missing {kind} docs: {missing}"
    untitled = [d["_id"] for d in docs if d["title"] == d["_id"]]
    if untitled:                       # a bare-docid title is unrecognisable in a result card
        print(f"NOTE: {len(untitled)} doc(s) still have no title: {untitled}")
    OUT.write_text(json.dumps({"docs": docs, "questions": [WARMUP] + questions},
                              ensure_ascii=False))
    print(f"wrote {OUT} ({len(docs)} docs, {OUT.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
