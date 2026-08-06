"""One-time OFFLINE curation of the demo's BrowseComp-Plus subsample -> browsecomp_data.json.

Needs internet (Hugging Face) once; needs NO OpenAI key (the published structured corpus is
already sectioned by the paper's own batch pass). Two sources, joined on docid (see
docs/superpowers/specs/2026-08-06-live-demo-design.md §2):

  - Tevatron/browsecomp-plus (streamed): query text + answer + gold/negative/evidence docids
    for QUERY_IDS, de-obfuscated via corpus_build.browsecomp_plus.build.transform_decrypt.
  - wshuai190/browsecomp-plus-structured-full structured/corpus.jsonl (stream-filtered over
    HTTP, early-stopped): the matching docs' title/author/date/sections.

    python demo/recorder/build_corpus.py                       # full run (~10-50 min: streams
                                                               #   through a 6.1GB remote file)
    python demo/recorder/build_corpus.py --cache /tmp/bcp_subsample_structured.jsonl
                                                               # reuse an already-filtered dump
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))

# The two curated queries (verified against the live dataset — spec §2):
#   798 -> Lady Shri Ram College for Women (3 gold docs)
#   792 -> "Oral creatine supplementation ... rheumatoid arthritis" (1 gold doc)
QUERY_IDS = ("798", "792")
GOLD_DOCIDS = ("37133", "39666", "41817", "51481")
STRUCTURED_URL = ("https://huggingface.co/datasets/wshuai190/browsecomp-plus-structured-full"
                  "/resolve/main/structured/corpus.jsonl")
OUT = HERE / "browsecomp_data.json"

# Outlier cap: purely demo-UX/cost motivated (repo size does NOT require it — the whole
# subsample is ~3.5MB). The 40K floor keeps every known gold section intact (largest:
# 37,225 chars in doc 51481).
CAP_CHARS = 80_000
CAP_FLOOR = 40_000
CAP_MARK = "…(truncated for demo)"


def cap_body(body: str, cap: int = CAP_CHARS) -> str:
    assert cap >= CAP_FLOOR
    if len(body) <= cap:
        return body
    return body[:cap] + CAP_MARK


def record_to_doc(rec: dict) -> dict:
    """A structured/corpus.jsonl record -> the JSON doc shape browsecomp_corpus.py loads.
    Body is built from `sections` (## heading markers — how DocSearchFetch/Bm25Visit and the
    demo UIs expect section boundaries), not the redundant flat `text` field."""
    sections = [[s.get("heading") or "(untitled)", s.get("text") or ""]
                for s in (rec.get("sections") or [])]
    body = cap_body("\n".join(f"## {h}\n{t}" for h, t in sections))
    return {"_id": str(rec["_id"]), "title": rec.get("title") or str(rec["_id"]),
            "body": body, "sections": sections,
            "metadata": {"author": rec.get("author") or "", "date": rec.get("date") or ""}}


def pull_queries() -> tuple[list[dict], set[str]]:
    """Stream Tevatron/browsecomp-plus for QUERY_IDS -> ([question rows], union of target docids)."""
    from datasets import load_dataset
    from corpus_build.browsecomp_plus.build import transform_decrypt
    ds = load_dataset("Tevatron/browsecomp-plus", split="test", streaming=True)
    questions, targets = [], set()
    for row in ds:
        if str(row.get("query_id")) not in QUERY_IDS:
            continue
        row = transform_decrypt(row)
        gold = [str(d["docid"]) for d in (row.get("gold_docs") or [])]
        docids = {str(d["docid"])
                  for field in ("gold_docs", "negative_docs", "evidence_docs")
                  for d in (row.get(field) or [])}
        questions.append({"query_id": str(row["query_id"]), "question": str(row["query"]),
                          "answer": str(row.get("answer") or ""), "gold_docids": gold})
        targets |= docids
        if len(questions) == len(QUERY_IDS):
            break
    assert len(questions) == len(QUERY_IDS), f"found only {[q['query_id'] for q in questions]}"
    return questions, targets


def pull_structured(targets: set[str], cache: Path | None) -> list[dict]:
    """The structured records for `targets`: from a local cache dump when it covers them all,
    else by stream-filtering the 6.1GB remote file (early-stop once every target is found)."""
    if cache and cache.exists():
        recs = [json.loads(l) for l in cache.open() if l.strip()]
        found = {str(r["_id"]): r for r in recs if str(r["_id"]) in targets}
        if set(found) == targets:
            print(f"cache hit: all {len(targets)} docs from {cache}")
            return list(found.values())
        print(f"cache covers {len(found)}/{len(targets)} — falling back to the remote stream")
    import requests
    found = {}
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
                if rid in targets and rid not in found:
                    found[rid] = rec
            if len(found) == len(targets):
                break
    missing = targets - set(found)
    assert not missing, f"structured corpus missing docids: {sorted(missing)[:10]}"
    return list(found.values())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", type=Path, default=None,
                    help="an already stream-filtered structured/corpus.jsonl subset")
    args = ap.parse_args()
    questions, targets = pull_queries()
    print(f"queries {QUERY_IDS} -> {len(targets)} unique target docids")
    docs = sorted((record_to_doc(r) for r in pull_structured(targets, args.cache)),
                  key=lambda d: d["_id"])
    for gold in GOLD_DOCIDS:
        assert any(d["_id"] == gold for d in docs), f"gold doc {gold} missing"
    OUT.write_text(json.dumps({"docs": docs, "questions": questions}, ensure_ascii=False))
    print(f"wrote {OUT} ({len(docs)} docs, {OUT.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
