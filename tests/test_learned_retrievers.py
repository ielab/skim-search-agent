"""SPLADE and ColBERT (`agent_search/retrievers/learned/`) on tiny random BERTs built here, on the
CPU: the index is built, persisted and reloaded, a changed corpus rebuilds it, and the ranking is
the exact score (SPLADE's dot product, ColBERT's MaxSim) computed by hand. The released
checkpoints are exercised on the cluster."""
from __future__ import annotations

import os

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from agent_search.corpus.units import CodeUnit
from agent_search.errors import SetupError
from agent_search.retrievers.learned import ColbertRetriever, SpladeRetriever
from agent_search.retrievers.learned import colbert as colbert_mod

WORDS = ["treaty", "war", "mexican", "american", "paris", "spanish", "ended", "signed", "harbor", "festival",
         "annual", "event", "history", "the", "of", "in", "1848", "1898"]


def _tokenizer(path):
    from transformers import AutoTokenizer, BertTokenizer
    vocab = ["[PAD]", "[unused0]", "[unused1]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", ".", ",", "!", "?", ":", ";", "'",
             '"', "(", ")", "-"] + WORDS
    os.makedirs(path, exist_ok=True)
    # transformers 5 builds a BERT tokenizer from a vocab mapping, not a vocab file
    BertTokenizer(vocab={w: i for i, w in enumerate(vocab)}, do_lower_case=True).save_pretrained(path)
    return AutoTokenizer.from_pretrained(path)


def _config(tok):
    from transformers import BertConfig
    return BertConfig(vocab_size=len(tok), hidden_size=32, num_hidden_layers=1, num_attention_heads=2,
                      intermediate_size=64, max_position_embeddings=128)


@pytest.fixture(scope="module")
def splade_dir(tmp_path_factory):
    from transformers import BertForMaskedLM
    path = str(tmp_path_factory.mktemp("splade"))
    tok = _tokenizer(path)
    torch.manual_seed(0)
    BertForMaskedLM(_config(tok)).save_pretrained(path)
    return path


@pytest.fixture(scope="module")
def colbert_dir(tmp_path_factory):
    from safetensors.torch import load_file, save_file
    from transformers import BertModel
    path = str(tmp_path_factory.mktemp("colbert"))
    tok = _tokenizer(path)
    torch.manual_seed(0)
    BertModel(_config(tok), add_pooling_layer=False).save_pretrained(path)
    f = os.path.join(path, "model.safetensors")
    weights = load_file(f)
    weights["linear.weight"] = torch.randn(colbert_mod.DIM, 32)
    save_file(weights, f)
    return path


def _units(texts):
    return [CodeUnit(doc_id=f"d{i}", path=f"d{i}.txt", qualname=t.split()[0], start_line=1, end_line=1,
                     code=t, body=t, title=t.split()[0]) for i, t in enumerate(texts)]


DOCS = ["treaty of 1848 ended the mexican american war", "the treaty of paris signed in 1898 ended the spanish war",
        "annual harbor festival event history", "history of the harbor", "the war in 1898"]


def test_splade_builds_reloads_and_ranks_by_the_exact_dot_product(splade_dir, tmp_path):
    units = _units(DOCS)
    r = SpladeRetriever(model=splade_dir, doc_length=32, query_length=16, index_root=str(tmp_path), device="cpu")
    r.index(units, key="tiny")
    assert r.is_cached("tiny") and os.path.exists(os.path.join(r._cache_dir("tiny"), "weights.npz"))
    q = "treaty that ended the war"
    ranked = r.search_scored(q, 5)
    assert sorted(d for d, _ in ranked) == sorted(u.doc_id for u in units)
    qw = r._weights([q], 16)[0]
    dw = r._weights([f"{u.qualname}\n{u.code}" for u in units], 32)
    by_hand = (dw @ qw).tolist()
    for d, s in ranked:
        assert s == pytest.approx(by_hand[int(d[1:])], rel=1e-4, abs=1e-5)
    again = SpladeRetriever(model=splade_dir, doc_length=32, query_length=16, index_root=str(tmp_path), device="cpu")
    again.index(units, key="tiny")
    assert [d for d, _ in again.search_scored(q, 5)] == [d for d, _ in ranked]     # loaded, not rebuilt


def test_a_changed_corpus_rebuilds_the_cached_index(splade_dir, tmp_path):
    r = SpladeRetriever(model=splade_dir, doc_length=32, index_root=str(tmp_path), device="cpu")
    r.index(_units(DOCS), key="tiny")
    changed = _units(DOCS[:-1] + ["the harbor festival"])
    r2 = SpladeRetriever(model=splade_dir, doc_length=32, index_root=str(tmp_path), device="cpu").index(changed, key="tiny")
    assert len(r2.doc_ids) == len(changed)
    assert r2.search("harbor festival", 1)


def test_colbert_scores_are_the_exact_maxsim_and_chunking_does_not_change_them(colbert_dir, tmp_path, monkeypatch):
    units = _units(DOCS)
    r = ColbertRetriever(model=colbert_dir, doc_length=24, query_length=8, index_root=str(tmp_path), device="cpu",
                         store_device="cpu")
    r.index(units, key="tiny")
    q = "treaty that ended the war"
    whole = r.scores(q).tolist()
    qv = r.query_vectors(q).half().float()
    store, off = r._store.float(), r._offsets
    by_hand = [float((store[off[i]:off[i + 1]] @ qv.T).max(dim=0).values.sum()) for i in range(len(units))]
    assert whole == pytest.approx(by_hand, rel=1e-3, abs=1e-3)
    monkeypatch.setattr(colbert_mod, "CHUNK_TOKENS", 7)          # several pages per chunk boundary
    assert r.scores(q).tolist() == pytest.approx(whole, rel=1e-5, abs=1e-5)
    assert len(r.search(q, 3)) == 3


def test_colbert_drops_punctuation_and_pads_queries_with_mask(colbert_dir, tmp_path):
    r = ColbertRetriever(model=colbert_dir, doc_length=24, query_length=8, index_root=str(tmp_path), device="cpu",
                         store_device="cpu")
    r.index(_units(["treaty , war . paris !", "harbor festival"]), key="punct")
    off = r._offsets
    # the page is indexed as its title line and body, "treaty\ntreaty , war . paris !":
    # [CLS] [D] treaty treaty war paris [SEP] are stored, the three punctuation marks are not
    assert int(off[1] - off[0]) == 7
    assert r.query_vectors("treaty").shape == (8, colbert_mod.DIM)


def test_engines_refuse_a_run_without_a_built_index(tmp_path, monkeypatch):
    from agent_search.retrievers.engines import Engines
    monkeypatch.setenv("SPLADE_MODEL", "nonexistent/model")
    e = Engines(_units(DOCS), "nothing-built", index_root=str(tmp_path))
    with pytest.raises(SetupError, match="--retriever splade"):
        e.get("splade")


def test_the_learned_conditions_render_the_dense_cells_prompt():
    from agent_search.strategies import CONDITIONS
    dense = CONDITIONS["research_iter_dense"].system_sha256(None)
    for name in ("research_iter_splade", "research_iter_colbert"):
        c = CONDITIONS[name]
        assert c.tool_names == ("search", "get_document")
        assert c.system_sha256(None) == dense


def test_the_index_builder_gives_learned_retrievers_their_own_model(monkeypatch, tmp_path):
    """`build_indexes --retriever splade|colbert` without --model must not pass the dataset's dense
    model (bge-base once loaded as SPLADE with a random head and as ColBERT without its projection)."""
    import sys
    import agent_search.evaluation.build_indexes as B
    seen = {}

    def fake_build(instances, **kw):
        seen.update(kw)
        return {"corpora": 0, "built": 0, "skipped": 0, "failed": 0, "missing_repos": []}

    monkeypatch.setattr(B, "build", fake_build)
    monkeypatch.setattr(B, "load_dataset", lambda *a, **k: [], raising=False)
    for retriever in ("splade", "colbert"):
        monkeypatch.setattr(sys, "argv", ["build_indexes", "--dataset", "doc_fixture", "--retriever", retriever,
                                          "--index-root", str(tmp_path)])
        try:
            B.main()
        except SystemExit:
            pass
        assert seen.get("retriever") == retriever and seen.get("model") is None, seen

