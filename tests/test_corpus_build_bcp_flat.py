"""The browsecomp_plus `corpus` stage: flat's text must be the plain original body, not a copy
of the structured (sectioned) text. This guards the fix: before it, flat carried the structured
arm's leading date line and `## heading` lines too, so the twins were not actually differentiated.

Builds two tiny synthetic documents (frontmatter + body, one with an author, one whose frontmatter
has no title) and a matching sections.jsonl, runs `cmd_corpus` directly with `iter_rows`
monkeypatched to yield them (no hub, no network), then checks the two corpus.jsonl files."""
import argparse
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BUILD_PY = ROOT / "corpus_build" / "browsecomp_plus" / "build.py"


def _load_build():
    spec = importlib.util.spec_from_file_location("bcp_build", BUILD_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _read_jsonl(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(ln) for ln in fh if ln.strip()]


RAW_A = (
    "---\n"
    "title: Sanaa\n"
    "author: Jane Doe\n"
    "date: 2009-07-27\n"
    "---\n"
    "Sanaa is a city.\n"
    "It has long history.\n"
)
SECTIONS_A = [
    {"heading": "(intro)", "text": "Sanaa is a city."},
    {"heading": "History", "text": "It has long history."},
]

# no `title` in the frontmatter: the structured record's title resolves to "", and the flat
# twin must carry that SAME title, not compute its own.
RAW_B = (
    "---\n"
    "date: 2010-01-01\n"
    "---\n"
    "Doc B has no title.\n"
    "Second line of doc B.\n"
)
SECTIONS_B = [
    {"heading": "(intro)", "text": "Doc B has no title."},
    {"heading": "Details", "text": "Second line of doc B."},
]


def _fake_iter_rows(limit=None):
    docs = [
        {"docid": "A", "url": "", "text": RAW_A},
        {"docid": "B", "url": "", "text": RAW_B},
    ]
    yield "q1", "What city is this?", ["A"], docs, "Sanaa"


def _build(monkeypatch, tmp_path):
    build = _load_build()
    monkeypatch.setattr(build, "iter_rows", _fake_iter_rows)

    sections_path = tmp_path / "sections.jsonl"
    with open(sections_path, "w", encoding="utf-8") as fh:
        fh.write(json.dumps({"_id": "A", "sections": SECTIONS_A}) + "\n")
        fh.write(json.dumps({"_id": "B", "sections": SECTIONS_B}) + "\n")

    out_dir = tmp_path / "out"
    args = argparse.Namespace(out_dir=str(out_dir), limit=None, sections=str(sections_path))
    rc = build.cmd_corpus(args)
    assert rc == 0
    return out_dir


def test_flat_text_is_plain_original_body(monkeypatch, tmp_path):
    out_dir = _build(monkeypatch, tmp_path)
    flat = {r["_id"]: r for r in _read_jsonl(out_dir / "browsecomp_plus_flat" / "corpus.jsonl")}

    assert flat["A"]["text"] == "Sanaa is a city.\nIt has long history."
    assert flat["B"]["text"] == "Doc B has no title.\nSecond line of doc B."
    for rec in flat.values():
        assert "## " not in rec["text"]
        assert not rec["text"].startswith("By ")
        assert not rec["text"].startswith("2010-01-01")
        assert rec["text"] == rec["text"].strip("\n")


def test_structured_text_has_date_line_and_headings(monkeypatch, tmp_path):
    out_dir = _build(monkeypatch, tmp_path)
    structured = {r["_id"]: r
                  for r in _read_jsonl(out_dir / "browsecomp_plus_structured" / "corpus.jsonl")}

    a = structured["A"]
    assert a["text"].startswith("By Jane Doe 2009-07-27")
    assert "## History" in a["text"]

    b = structured["B"]
    assert b["text"].startswith("2010-01-01")
    assert "## Details" in b["text"]


def test_ids_and_titles_match_across_twins(monkeypatch, tmp_path):
    out_dir = _build(monkeypatch, tmp_path)
    flat = {r["_id"]: r for r in _read_jsonl(out_dir / "browsecomp_plus_flat" / "corpus.jsonl")}
    structured = {r["_id"]: r
                  for r in _read_jsonl(out_dir / "browsecomp_plus_structured" / "corpus.jsonl")}

    assert set(flat) == set(structured) == {"A", "B"}
    for did in flat:
        assert flat[did]["title"] == structured[did]["title"]

    assert structured["A"]["title"] == "Sanaa"
    # doc B's frontmatter has no title: the flat twin still gets the (empty) structured title,
    # not some independently-recomputed value.
    assert structured["B"]["title"] == ""
    assert flat["B"]["title"] == structured["B"]["title"] == ""


def test_queries_and_qrels_identical_between_twins(monkeypatch, tmp_path):
    out_dir = _build(monkeypatch, tmp_path)
    fq = _read_jsonl(out_dir / "browsecomp_plus_flat" / "queries.jsonl")
    sq = _read_jsonl(out_dir / "browsecomp_plus_structured" / "queries.jsonl")
    assert fq == sq

    fqrels = (out_dir / "browsecomp_plus_flat" / "qrels" / "test.tsv").read_text()
    sqrels = (out_dir / "browsecomp_plus_structured" / "qrels" / "test.tsv").read_text()
    assert fqrels == sqrels
    assert "q1\tA\t1" in fqrels
