"""The fetch tool's flat call shape and its recoveries (agent_search/tools/fetch/tool.py).

The paper's `{"specs": [[rank, section]]}` list of pairs is still accepted; the declared shape
is `{"rank": 1, "section": "Career"}`. A request that is not a section name on the referenced
document is resolved instead of refused: whole-document words, facts words, fuzzy names,
pasted lists, and a section that sits on another listed document."""
from agent_search.corpus.units import units_from_documents
from agent_search.tasks.render import render_tool_rules
from agent_search.tools.base import EpisodeState, ToolBox
from agent_search.tools.fetch.tool import Fetch

DOCS = [
    {"_id": "d1", "title": "Ann Example", "author": "Bob Byline", "date": "2014-05-02",
     "text": "Ann Example is a swimmer. History Ann started early. Career Ann won twice.",
     "sections": [{"heading": "(intro)", "text": "Ann Example is a swimmer."},
                  {"heading": "History", "text": "Ann started early."},
                  {"heading": "Career", "text": "Ann won twice."}]},
    {"_id": "d2", "title": "2014 Final", "author": "", "date": "2014-06-01",
     "text": "The 2014 final. Results Ann beat Cy 5-3. Prize fund was large.",
     "sections": [{"heading": "(intro)", "text": "The 2014 final."},
                  {"heading": "Results", "text": "Ann beat Cy 5-3."},
                  {"heading": "Prize fund", "text": "Prize fund was large."}]},
    {"_id": "d3", "title": "2015 Final", "author": "", "date": "2015-06-01",
     "text": "The 2015 final. Results Cy beat Ann 6-4.",
     "sections": [{"heading": "(intro)", "text": "The 2015 final."},
                  {"heading": "Results", "text": "Cy beat Ann 6-4."}]},
    {"_id": "d4", "title": "Flat Note", "text": "One flat paragraph about Ann."},
    {"_id": "d5", "title": "رحلة علمية", "text": "x", "author": "", "date": "2022-11-03",
     "sections": [{"heading": "(intro)", "text": "مقدمة"},
                  {"heading": "تفاصيل الرحلة العلمية", "text": "تفاصيل الرحلة"},
                  {"heading": "Details", "text": "the details section"}]},
    {"_id": "d9", "title": "Old Listing Doc", "text": "x",
     "sections": [{"heading": "(intro)", "text": "old"},
                  {"heading": "Unique Nine", "text": "only here"}]},
]


def _box(last_hits):
    units = list(units_from_documents(DOCS))
    ubyid = {u.doc_id: u for u in units}
    state = EpisodeState(question="q")
    state.last_hits = list(last_hits)
    fetch = Fetch(name="fetch").bind(state, units, ubyid, {})
    return ToolBox([fetch], state), state, fetch


def test_declaration_is_flat():
    f = Fetch()
    assert set(f.parameters["properties"]) == {"rank", "section"}
    assert f.parameters["properties"]["rank"]["type"] == "integer"
    assert "array" not in str(f.parameters)
    assert render_tool_rules([f.declaration()]) == '- fetch: {"rank": <integer>, "section": <string>}'


def test_flat_call_reads_one_section():
    box, _, _ = _box(["d1", "d2"])
    out = box.run("fetch", {"rank": 1, "section": "Career"})
    assert "[d1 §Career]" in out and "Ann won twice." in out
    assert "History" not in out.split("§Career]")[1]


def test_paper_specs_shape_still_accepted():
    box, _, _ = _box(["d1", "d2"])
    assert "Ann won twice." in box.run("fetch", {"specs": [[1, "Career"]]})
    assert "Ann won twice." in box.run("fetch", {"specs": [1, "Career"]})
    two = box.run("fetch", {"specs": [[1, "History"], [2, "Results"]]})
    assert "Ann started early." in two and "Ann beat Cy 5-3." in two
    assert "Ann won twice." in box.run("fetch", {"specs": [{"rank": 1, "section": "Career"}]})


def test_rank_as_string_and_doc_id():
    box, _, _ = _box(["d1", "d2"])
    assert "Ann won twice." in box.run("fetch", {"rank": "1", "section": "Career"})
    assert "Ann beat Cy" in box.run("fetch", {"rank": "d2", "section": "Results"})


def test_whole_document_word_returns_opening_and_names_sections():
    box, _, _ = _box(["d1"])
    for word in ("body", "full text", "*", "L1-200"):
        out = box.run("fetch", {"rank": 1, "section": word})
        assert "[d1 §(intro)]" in out and "Ann Example is a swimmer." in out, word
        assert "Named sections on this document: History·Career" in out, word
        assert "ERROR" not in out, word


def test_empty_section_reads_opening():
    box, _, _ = _box(["d1"])
    out = box.run("fetch", {"rank": 1, "section": ""})
    assert "[d1 §(intro)]" in out and "swimmer" in out and "Named sections" not in out


def test_facts_words_return_title_author_date():
    box, _, _ = _box(["d1", "d2"])
    out = box.run("fetch", {"rank": 1, "section": "infobox"})
    assert "[d1 §infobox]" in out
    assert "title=Ann Example" in out and "author=Bob Byline" in out and "date=2014-05-02" in out
    assert "author=Bob Byline" in box.run("fetch", {"rank": 1, "section": "author"})
    # an empty byline is left out, the date stays
    out2 = box.run("fetch", {"rank": 2, "section": "date"})
    assert "date=2014-06-01" in out2 and "author=" not in out2


def test_fuzzy_section_names():
    box, _, _ = _box(["d1"])
    assert "Ann won twice." in box.run("fetch", {"rank": 1, "section": "career"})
    assert "Ann won twice." in box.run("fetch", {"rank": 1, "section": "Career section"})
    assert "Ann won twice." in box.run("fetch", {"rank": 1, "section": "the career"})
    assert "Ann started early." in box.run("fetch", {"rank": 1, "section": "§History"})


def test_pasted_list_reads_its_first_existing_section():
    box, _, _ = _box(["d1"])
    out = box.run("fetch", {"rank": 1, "section": "§[History·Career]"})
    assert "[d1 §History]" in out and "Ann started early." in out


def test_pasted_whole_list_is_a_whole_document_request():
    box, _, _ = _box(["d2"])
    out = box.run("fetch", {"rank": 1, "section": "§[(intro)·Results·Prize fund]"})
    assert "[d2 §(intro)]" in out and "Named sections on this document" in out


def test_section_on_another_listed_document_is_read_from_there():
    box, _, _ = _box(["d1", "d2"])
    out = box.run("fetch", {"rank": 1, "section": "Results"})
    assert "[d2 §Results (read from rank 2, which has this section)]" in out
    assert "Ann beat Cy 5-3." in out and "ERROR" not in out


def test_section_on_several_listed_documents_names_them():
    box, _, _ = _box(["d1", "d2", "d3"])
    out = box.run("fetch", {"rank": 1, "section": "Results"})
    assert "ERROR: no section 'Results' on d1; rank 2 and rank 3 have it." in out
    assert "Sections here: (intro)·History·Career" in out


def test_section_from_an_earlier_listing():
    box, state, fetch = _box(["d9"])
    box.run("fetch", {"rank": 1, "section": "Unique Nine"})     # d9 was listed and read earlier
    state.last_hits = ["d1"]
    out = box.run("fetch", {"rank": 1, "section": "Unique Nine"})
    assert "[d9 §Unique Nine (read from an earlier listing, doc d9)]" in out and "only here" in out


def test_unknown_section_error_lists_every_section():
    box, _, _ = _box(["d1"])
    out = box.run("fetch", {"rank": 1, "section": "Zzz"})
    assert "ERROR: no section 'Zzz' on d1. Sections here: (intro)·History·Career" in out


def test_flat_document_returns_its_text_for_any_name():
    box, _, _ = _box(["d4"])
    assert "One flat paragraph" in box.run("fetch", {"rank": 1, "section": "History"})
    assert "One flat paragraph" in box.run("fetch", {"rank": 1, "section": "body"})


def test_missing_rank_and_bad_rank():
    box, _, _ = _box(["d1"])
    assert "rank" in box.run("fetch", {"section": "Career"})
    assert "out of range" in box.run("fetch", {"rank": 7, "section": "Career"})
    assert "needs a rank" in box.run("fetch", {})


def test_fetch_marks_the_document_seen():
    box, state, _ = _box(["d1", "d2"])
    box.run("fetch", {"rank": 2, "section": "Results"})
    assert "d2" in state.seen



def test_non_latin_section_names_match():
    box, _, _ = _box(["d5"])
    out = box.run("fetch", {"rank": 1, "section": "تفاصيل الرحلة العلمية"})
    assert "[d5 §تفاصيل الرحلة العلمية]" in out and "تفاصيل الرحلة" in out and "ERROR" not in out


def test_a_real_section_named_like_a_special_word_wins():
    box, _, _ = _box(["d5"])
    out = box.run("fetch", {"rank": 1, "section": "Details"})
    assert "[d5 §Details]" in out and "the details section" in out
    assert "date=2022-11-03" in box.run("fetch", {"rank": 1, "section": "infobox"})
