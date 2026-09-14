"""`x[infobox]` is a real field (Lucene schema 3). The wiki manuals advertise it as the reverse
link (whose facts name x); schema 2 had no infobox field and raised a compile error, which the
Sieve paper's appendix reports as an instruction-index mismatch."""
from agent_search.corpus.units import units_from_documents
from agent_search.retrievers.indri.fields import FIELDS, field_text
from agent_search.retrievers.lucene.schema import SCHEMA_VERSION, SCORED_FIELD_MAP
from agent_search.tools.base import EpisodeState, ToolBox
from agent_search.tools.fetch.tool import Fetch
from agent_search.tools.search_bql.tool import SearchBql
from tests import lucene_support

lucene_support.require_jvm()

DOCS = [
    {"_id": "harbor", "title": "Harbor Festival",
     "text": "Harbor Festival is an annual event.\n\n## History\nFounded long ago.",
     "infobox": "Founded: 1897; Founder: Alma Smith; Location: Portville"},
    {"_id": "smith", "title": "Alma Smith",
     "text": "Alma Smith was an organiser who lived in Portville and founded a festival."},
    {"_id": "other", "title": "Quantum Chromodynamics",
     "text": "Quarks and gluons interact via the strong force.", "infobox": "Field: physics"},
]


def _box():
    units = list(units_from_documents(DOCS))
    ubyid = {u.doc_id: u for u in units}
    ex = lucene_support.build_lucene_bql(units)
    state = EpisodeState(question="q")
    search = SearchBql(name="search").bind(state, units, ubyid, {"bql": ex})
    fetch = Fetch(name="fetch").bind(state, units, ubyid, {})
    return ToolBox([search, fetch], state), state


def test_schema_carries_the_infobox_field():
    assert SCHEMA_VERSION >= 3
    assert "infobox" in SCORED_FIELD_MAP and "infobox" in FIELDS
    u = next(u for u in units_from_documents(DOCS) if u.doc_id == "harbor")
    assert field_text(u, "infobox") == "Founded: 1897; Founder: Alma Smith; Location: Portville"


def test_infobox_scope_is_the_reverse_link():
    box, state = _box()
    out = box.run("search", {"query": "smith[infobox]"})
    assert state.last_hits == ["harbor"], out          # whose facts name Smith: the festival, not her page
    assert "ERROR" not in out and "0 matches" not in out


def test_infobox_scope_misses_documents_without_the_fact():
    box, state = _box()
    out = box.run("search", {"query": "quarks[infobox]"})
    assert "0 exact matches" in out or "0 matches" in out   # quarks is in the body, not the infobox
    out = box.run("search", {"query": "physics[infobox]"})
    assert state.last_hits == ["other"] and "0 " not in out.splitlines()[0]
