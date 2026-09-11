"""agent_search.corpus.flat_export: materializing units to a flat file tree (the DCI arm's
corpus shape). One .txt per doc_id; a shared corpus (given a `key`) is exported once per
process and reused; an un-keyed export always gets a fresh directory."""
from agent_search.corpus.flat_export import export_flat_corpus
from agent_search.corpus.units import units_from_documents

DOCS = [
    {"_id": "alpha", "title": "Alpha Doc", "text": "Alpha body text."},
    {"_id": "beta", "title": "Beta Doc", "text": "Beta body text."},
]


def test_export_writes_one_file_per_doc():
    units = units_from_documents(DOCS)
    export_dir, doc_to_rel = export_flat_corpus(units)
    assert set(doc_to_rel) == {"alpha", "beta"}
    for doc_id, rel in doc_to_rel.items():
        assert (export_dir / rel).is_file()


def test_exported_file_contains_title_and_body():
    units = units_from_documents(DOCS)
    export_dir, doc_to_rel = export_flat_corpus(units)
    text = (export_dir / doc_to_rel["alpha"]).read_text()
    assert "Alpha Doc" in text and "Alpha body text" in text


def test_filename_is_filesystem_safe():
    units = units_from_documents([{"_id": "a/b c", "title": "T", "text": "x"}])
    _, doc_to_rel = export_flat_corpus(units)
    rel = doc_to_rel["a/b c"]
    assert "/" not in rel and " " not in rel


def test_keyed_export_is_reused_across_calls():
    units = units_from_documents(DOCS)
    d1, _ = export_flat_corpus(units, key="__test_flat_export_key__")
    d2, _ = export_flat_corpus(units, key="__test_flat_export_key__")
    assert d1 == d2


def test_unkeyed_export_gets_a_fresh_dir_each_call():
    units = units_from_documents(DOCS)
    d1, _ = export_flat_corpus(units)
    d2, _ = export_flat_corpus(units)
    assert d1 != d2


def test_rebuild_forces_a_fresh_write():
    units = units_from_documents(DOCS)
    d1, _ = export_flat_corpus(units, key="__test_flat_export_rebuild__")
    d2, _ = export_flat_corpus(units, key="__test_flat_export_rebuild__", rebuild=True)
    assert d1 == d2                       # same directory, re-populated
