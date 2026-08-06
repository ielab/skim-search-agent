"""Shared demo observation parsers (demo/recorder/parse.py): the exact renderings of
DocSearchFetch.search/.fetch and Bm25Visit.search/.visit -> the card dicts the live SSE server (demo/server.py) sends the player."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from demo.parse import parse_bm25_search, parse_fetch, parse_search, parse_visit


def test_parse_search_structured_hit_line():
    obs = ("search: creatine AND arthritis -> AND(creatine, arthritis)   (2 matches)\n"
           "  1  51481  'Oral creatine supplementation'  §[Abstract·Conclusions] ib[date:2013]"
           "  matched: body  » creatine supplementation in rheumatoid arthritis")
    out = parse_search(obs)
    assert out["compiled"] == "AND(creatine, arthritis)"
    assert out["hits"][0]["id"] == "51481"
    assert out["hits"][0]["sections"] == ["Abstract", "Conclusions"]


def test_parse_fetch_named_section():
    obs = "fetch:\n  [51481 §Conclusions]  Creatine may be a useful adjunct therapy."
    out = parse_fetch(obs)
    assert out["doc"] == "51481" and out["section"] == "Conclusions"
    assert not out["error"]


def test_parse_bm25_search_listing():
    obs = ("search: lady shri ram college   (2 matches):\n"
           "  1  39666  'Lady Shri Ram College'  The college was established in 1956 in…\n"
           "  2  37133  'Delhi University'  One of the constituent colleges of the…")
    out = parse_bm25_search(obs)
    assert [h["id"] for h in out["hits"]] == ["39666", "37133"]
    assert out["hits"][0]["title"] == "Lady Shri Ram College"
    assert out["hits"][0]["sections"] == [] and out["hits"][0]["matched"] == ""


def test_parse_visit_whole_doc_and_quote_styles():
    out = parse_visit("39666  'Lady Shri Ram College':\nThe college was established in 1956.")
    assert out["doc"] == "39666" and out["title"] == "Lady Shri Ram College"
    out2 = parse_visit("7  \"The College's History\":\nBody text.")
    assert out2["title"] == "The College's History" and out2["doc"] == "7"


def test_parse_visit_error_observation():
    out = parse_visit("ERROR: no such doc 'zorp' — use a rank from the last search or a doc_id.")
    assert out["error"] and "no such doc" in out["text"]

