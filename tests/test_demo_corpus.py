"""The curated BrowseComp-Plus demo corpus: the record->CodeUnit mapping and the outlier cap
(demo/build_corpus.py helpers), plus integrity checks on the checked-in corpus
(gold docs present, GitHub-friendly size)."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from demo.build_corpus import GOLD_DOCIDS, QUERY_IDS, cap_body, record_to_doc

REC = {"_id": "42", "title": "T", "author": "A", "date": "2020-01-01",
       "sections": [{"heading": "(intro)", "text": "intro text"},
                    {"heading": "History", "text": "history text"}],
       "text": "intro text\nhistory text"}


def test_record_to_doc_builds_heading_marked_body():
    d = record_to_doc(REC)
    assert d["_id"] == "42" and d["title"] == "T"
    assert "## (intro)\nintro text" in d["body"]
    assert "## History\nhistory text" in d["body"]
    assert d["metadata"] == {"author": "A", "date": "2020-01-01"}


def test_cap_body_truncates_only_past_80k_with_marker():
    assert cap_body("## A\nshort") == "## A\nshort"
    capped = cap_body("## A\n" + "x" * 100_000)
    assert len(capped) < 81_000 and capped.endswith("…(truncated for demo)")


DATA = Path(__file__).resolve().parent.parent / "demo" / "corpus_data.json"


@pytest.mark.skipif(not DATA.exists(), reason="generated corpus not built yet")
def test_generated_corpus_integrity():
    d = json.loads(DATA.read_text())
    ids = {doc["_id"] for doc in d["docs"]}
    assert len(ids) == len(d["docs"])
    for gold in GOLD_DOCIDS:
        assert gold in ids, f"gold doc {gold} missing"
    assert {q["query_id"] for q in d["questions"]} == set(QUERY_IDS)
    for q in d["questions"]:
        assert q["question"] and q["answer"]
    assert DATA.stat().st_size < 8_000_000


@pytest.mark.skipif(not DATA.exists(), reason="generated corpus not built yet")
def test_loader_exposes_corpus_and_questions():
    from demo.corpus import CORPUS, QUESTIONS
    assert len(CORPUS) > 80
    assert all(u.body and u.title for u in CORPUS)
    assert all("## " in u.body for u in CORPUS)
    assert len(QUESTIONS) == 2
    assert all(q and gold for q, gold in QUESTIONS)
