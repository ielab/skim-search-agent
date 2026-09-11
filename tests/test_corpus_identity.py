"""Flat-vs-structured corpus fairness: a structured doc and its flat twin must give bm25/dense
the same searchable input; they differ only in the scopeable fields that BQL uses. This guards
the headline control (the flat/structured pair isolates BQL's value, not a content difference)."""
from agent_search.corpus.units import units_from_documents
from agent_search.retrievers.bql.executor import StructuralExecutor
from agent_search.retrievers.bql.parser import parse


def test_flat_and_structured_units_share_identical_bm25_blob():
    # the pair corpus_build emits: SAME _id, SAME text; structured adds section/infobox fields.
    text = "country: Yemen\n\n## History\nSanaa was founded long ago.\n## Geography\nA highland capital."
    flat = {"_id": "Sanaa", "title": "Sanaa", "text": text}
    structured = {"_id": "Sanaa", "title": "Sanaa", "section": "History Geography",
                  "infobox": "country: Yemen", "text": text}

    fu = units_from_documents([flat])[0]
    su = units_from_documents([structured])[0]

    # the bm25/dense blob (u.code) is byte-identical -> no lexical/dense confound across arms
    assert fu.code == su.code, "structured arm must not inflate the bm25/dense blob"
    assert fu.doc_id == su.doc_id
    # heading tokens appear ONCE (from body), not duplicated by a standalone section field
    assert su.code.lower().count("history") == 1 and su.code.lower().count("geography") == 1

    # the structure lives where ONLY BQL reads it: u.section + metadata, not the blob
    assert su.section == "History Geography" and fu.section is None
    assert su.metadata.get("infobox") == "country: Yemen"
    assert "history" not in (fu.section or "").lower()


def test_section_still_scopeable_by_bql_after_blob_change():
    su = units_from_documents([{"_id": "Sanaa", "title": "Sanaa",
                                "section": "History Geography", "text": "founded long ago"}])
    ex = StructuralExecutor(su)
    assert [d for d, _ in ex.run(parse("IN(section, history)").expr)] == ["Sanaa"]
    # body-only token is NOT in the section region (proves section isn't just the whole blob)
    assert [d for d, _ in ex.run(parse("IN(section, founded)").expr)] == []
