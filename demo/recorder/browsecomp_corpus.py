"""The demo's curated BrowseComp-Plus collection: ~102 real docs (2 queries' gold + distractor
pools from the paper's published structured corpus) + the 2 curated questions.

Data lives in browsecomp_data.json (generated once, offline, by build_corpus.py — checked in so
no demo run ever needs Hugging Face access). Same CORPUS/QUESTIONS surface the old hand-written
corpus.py exposed; record.py, build_web.py, and demo/live/server.py all import from here."""
from __future__ import annotations

import json
from pathlib import Path

from agent_search.corpus.units import CodeUnit

_DATA = json.loads((Path(__file__).resolve().parent / "browsecomp_data.json").read_text())


def _unit(d: dict) -> CodeUnit:
    return CodeUnit(doc_id=d["_id"], path=f"{d['_id']}.md", qualname=d["_id"],
                    start_line=1, end_line=1, code=d["body"], body=d["body"],
                    title=d["title"],
                    sections=tuple((h, t) for h, t in d["sections"]),
                    metadata=dict(d["metadata"]))


CORPUS = [_unit(d) for d in _DATA["docs"]]
QUESTIONS = [(q["question"], q["answer"]) for q in _DATA["questions"]]
