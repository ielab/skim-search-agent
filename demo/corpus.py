"""The demo's curated BrowseComp-Plus collection: ~250 real docs (7 queries' gold, evidence and
distractor pools from the paper's published structured corpus) + the 6 curated questions.

Data lives in corpus_data.json (generated once, offline, by build_corpus.py, checked in so
no demo run ever needs Hugging Face access). demo/server.py imports from here."""
from __future__ import annotations

import json
from pathlib import Path

from agent_search.corpus.units import CodeUnit

_DATA = json.loads((Path(__file__).resolve().parent / "corpus_data.json").read_text())


def _unit(d: dict) -> CodeUnit:
    # the body is derived from the sections (## markers are what DocSearchFetch/Bm25Visit
    # and the demo UIs use as section boundaries), the JSON stores each doc only once
    body = d.get("body") or "\n".join(f"## {h}\n{t}" for h, t in d["sections"])
    return CodeUnit(doc_id=d["_id"], path=f"{d['_id']}.md", qualname=d["_id"],
                    start_line=1, end_line=1, code=body, body=body,
                    title=d["title"],
                    sections=tuple((h, t) for h, t in d["sections"]),
                    metadata=dict(d["metadata"]))


CORPUS = [_unit(d) for d in _DATA["docs"]]
QUESTIONS = [(q["question"], q["answer"], q.get("label", "example question"),
              q.get("paper"))
             for q in _DATA["questions"]]
