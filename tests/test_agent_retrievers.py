"""Retrievers trained for search agents: the query each one reads and how its checkpoint is
recognised. No model is loaded."""
import json
import os

from agent_search.retrievers.dense.base import family_for
from agent_search.retrievers.dense.e5 import AgenticRRetriever, LratE5Retriever
from agent_search.retrievers.learned.modern_colbert import AGENT_QUERY_PREFIX, agent_query, pylate_config
from agent_search.training.queries import canonical_style, default_style_for, render_query


def test_agentic_r_query_is_question_sep_query():
    assert render_query("question_sep", "Who wrote it?", "old man and the sea", []) == \
        "Who wrote it? [SEP] old man and the sea"
    assert canonical_style("question_sep") == "question_sep"
    assert default_style_for("liuwenhan/Agentic-R_e5") == "question_sep"
    assert default_style_for("Yuqi-Zhou/LRAT-multilingual-e5-large") == "plain"


def test_e5_families():
    assert family_for("liuwenhan/Agentic-R_e5") is AgenticRRetriever
    assert family_for("Yuqi-Zhou/LRAT-multilingual-e5-large") is LratE5Retriever
    assert family_for("Yuqi-Zhou/LRAT-Qwen3-Embedding-0.6B") is not LratE5Retriever
    assert AgenticRRetriever.query_prefix == "query: "
    assert AgenticRRetriever.document_template.format(title="T", body="B") == "passage: T\nB"
    assert LratE5Retriever.query_prefix.startswith("Instruct: ") and LratE5Retriever.query_prefix.endswith("\nQuery: ")


def test_agent_query_matches_the_model_card():
    # the prefix ends with a backslash and an n, not a newline
    assert AGENT_QUERY_PREFIX.endswith("\\nQuery:") and "\n" not in AGENT_QUERY_PREFIX
    assert agent_query("think", "q") == AGENT_QUERY_PREFIX + "Reasoning: think\n\nQuery: q"
    assert agent_query("", "q") == AGENT_QUERY_PREFIX + "Reasoning: Empty\n\nQuery: q"


def test_pylate_checkpoint_is_recognised(tmp_path):
    d = tmp_path / "ckpt"
    d.mkdir()
    assert pylate_config(str(d)) is None
    (d / "modules.json").write_text(json.dumps([{"type": "sentence_transformers.models.Transformer"},
                                                {"type": "pylate.models.Dense.Dense"}]))
    (d / "config_sentence_transformers.json").write_text(json.dumps({"query_length": 8192, "document_length": 4096}))
    assert pylate_config(str(d)) == {"query_length": 8192, "document_length": 4096}
    (d / "modules.json").write_text(json.dumps([{"type": "sentence_transformers.models.Transformer"}]))
    assert pylate_config(str(d)) is None


def test_colbert_routes_a_pylate_checkpoint(tmp_path):
    from agent_search.retrievers.learned.colbert import ColbertRetriever
    from agent_search.retrievers.learned.modern_colbert import ModernColbertRetriever
    d = tmp_path / "Agent-ModernColBERT"
    d.mkdir()
    (d / "modules.json").write_text(json.dumps([{"type": "pylate.models.Dense.Dense"}]))
    (d / "config_sentence_transformers.json").write_text(json.dumps({"query_length": 8192, "document_length": 4096}))
    r = ColbertRetriever(str(d), doc_length=512, query_length=9000)
    assert isinstance(r, ModernColbertRetriever)
    assert (r.doc_length, r.query_length, r.agent_queries) == (512, 8192, True)
    assert r.query_text("q") == agent_query("", "q")          # outside an episode: no reasoning
    plain = tmp_path / "bert"
    plain.mkdir()
    assert type(ColbertRetriever(str(plain))) is ColbertRetriever
