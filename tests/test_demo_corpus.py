"""The curated BrowseComp-Plus demo corpus: the record->CodeUnit mapping and the outlier cap
(demo/build_corpus.py helpers), plus integrity checks on the checked-in corpus
(gold docs present, GitHub-friendly size)."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from demo.build_corpus import (DEMO_QUESTION_IDS, GOLD_DOCIDS, PAPER_RESULTS,
                               QUERY_IDS, record_to_doc)

REC = {"_id": "42", "title": "T", "author": "A", "date": "2020-01-01",
       "sections": [{"heading": "(intro)", "text": "intro text"},
                    {"heading": "History", "text": "history text"}],
       "text": "intro text\nhistory text"}


def test_record_to_doc_stores_sections_only():
    d = record_to_doc(REC)
    assert d["_id"] == "42" and d["title"] == "T"
    assert "body" not in d                       # the loader derives it; no double storage
    assert d["sections"] == [["(intro)", "intro text"], ["History", "history text"]]
    assert d["metadata"] == {"author": "A", "date": "2020-01-01"}


def test_cap_sections_bounds_the_derived_body():
    from demo.build_corpus import cap_sections
    small = cap_sections([["A", "short"]])
    assert small == [["A", "short"]]
    capped = cap_sections([["A", "x" * 100_000], ["B", "dropped entirely"]])
    assert len(capped) == 1 and capped[0][1].endswith("…(truncated for demo)")
    body = "\n".join(f"## {h}\n{t}" for h, t in capped)
    assert len(body) < 81_000


DATA = Path(__file__).resolve().parent.parent / "demo" / "corpus_data.json"


@pytest.mark.skipif(not DATA.exists(), reason="generated corpus not built yet")
def test_generated_corpus_integrity():
    d = json.loads(DATA.read_text())
    ids = {doc["_id"] for doc in d["docs"]}
    assert len(ids) == len(d["docs"])
    for gold in GOLD_DOCIDS:
        assert gold in ids, f"gold doc {gold} missing"
    # the demo-authored warm-up leads, then the five real wins from the pairwise analysis
    qids = [q["query_id"] for q in d["questions"]]
    assert qids[0] == "demo-warmup"
    assert set(DEMO_QUESTION_IDS) <= set(qids)
    assert all(q.get("label") for q in d["questions"])
    for q in d["questions"]:
        if q["query_id"] in PAPER_RESULTS:      # measured paper-run numbers ride along
            assert q["paper"]["sieve"]["tokens"] > 0
            assert q["paper"]["visit"]["tokens"] > q["paper"]["sieve"]["tokens"]
    for q in d["questions"]:
        assert q["question"] and q["answer"]
    assert DATA.stat().st_size < 15_000_000     # target ~10MB; hard stop well past it


@pytest.mark.skipif(not DATA.exists(), reason="generated corpus not built yet")
def test_loader_exposes_corpus_and_questions():
    from demo.corpus import CORPUS, QUESTIONS
    assert len(CORPUS) > 80
    assert all(u.body and u.title for u in CORPUS)
    assert all("## " in u.body for u in CORPUS)
    assert len(QUESTIONS) == 1 + len(DEMO_QUESTION_IDS)   # warm-up + the five real wins
    assert all(q and gold and label for q, gold, label, _paper in QUESTIONS)
