"""Direct BQL retriever condition.

`bql` is the non-agent path: callers pass an already-formulated BQL query and get
ranked unit ids back. Natural-language-to-BQL remains the agent's job.
"""

import pytest

from agent_search.corpus.units import CodeUnit
from agent_search.retrievers.structural.bql.retriever import BQLRetriever


def _units():
    return [
        CodeUnit(
            "auth/session.py::create_session_token",
            "auth/session.py",
            "create_session_token",
            1,
            4,
            "def create_session_token(user):\n"
            "    token = make_token(user)\n"
            "    return token",
        ),
        CodeUnit(
            "auth/session.py::make_token",
            "auth/session.py",
            "make_token",
            6,
            7,
            "def make_token(user):\n"
            "    return str(user)",
        ),
        CodeUnit(
            "billing/invoice.py::create_invoice",
            "billing/invoice.py",
            "create_invoice",
            1,
            2,
            "def create_invoice(order):\n"
            "    return order.total",
        ),
    ]


def test_direct_bql_retriever_searches_structural_query():
    retriever = BQLRetriever().index(_units())

    got = retriever.search("IN(def, NEAR/func(session, token))", k=10)

    assert got == ["auth/session.py::create_session_token"]
    assert retriever.last_error is None


def test_direct_bql_retriever_reports_untruncated_count():
    retriever = BQLRetriever().index(_units())

    ranked, count = retriever.search_with_count("IN(file, session)", k=1)

    assert count == 2
    assert len(ranked) == 1
    assert ranked[0][0].startswith("auth/session.py::")


def test_direct_bql_retriever_parse_error_returns_no_hits():
    retriever = BQLRetriever().index(_units())

    # genuinely malformed (stray unbalanced paren) -> structured parse error.
    assert retriever.search("fix the session ( token", k=10) == []
    assert retriever.last_error.startswith("parse error:")


def test_direct_bql_retriever_bare_multiword_is_phrase_not_error():
    """Forgiveness: a run of bare words is read as an implicit PHRASE, not a parse
    error. It simply 0-hits when no unit contains that contiguous phrase
    (recoverable), instead of erroring out and wasting an agent step."""
    retriever = BQLRetriever().index(_units())

    assert retriever.search("fix the session token expiry bug", k=10) == []
    assert retriever.last_error is None        # parsed cleanly, just no match


def test_direct_bql_retriever_type_error_returns_no_hits():
    retriever = BQLRetriever().index(_units())

    assert retriever.search("NOT(token)", k=10) == []
    assert retriever.last_error.startswith("type error:")


def test_direct_bql_retriever_requires_index_first():
    with pytest.raises(RuntimeError, match="index"):
        BQLRetriever().search("token", k=10)
