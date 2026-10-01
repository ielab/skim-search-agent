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


# --- DiffRetriever: the library side against a fake encoding server ---------------------------

class _FakeEncoder:
    """Stands in for scripts/serve_diffretriever.py: deterministic vectors per text, 4 per query and
    16 per passage, so the library's index, cache and MaxSim can be checked by hand."""

    def __init__(self, hidden=8):
        import json, threading, base64
        from http.server import BaseHTTPRequestHandler, HTTPServer
        import numpy as np
        outer = self
        self.hidden, self.calls = hidden, []

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):
                req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.calls.append(req)
                v = np.stack([outer.vec(t, req["is_query"]) for t in req["texts"]]).astype(np.float16)
                reply = {"shape": list(v.shape), "dtype": "float16", "data": base64.b64encode(v.tobytes()).decode()}
                if req.get("sparse"):
                    reply["sparse"] = [outer.terms(t) for t in req["texts"]]
                body = json.dumps(reply).encode()
                self.send_response(200); self.send_header("Content-Length", str(len(body))); self.end_headers()
                self.wfile.write(body)

        self.server = HTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"

    def terms(self, text):
        """A sparse vector over the text's own words: term id = a hash of the word, weight = its count x 100."""
        import collections, zlib
        c = collections.Counter(w for w in text.lower().split() if w.isalpha())
        return {"ids": [zlib.crc32(w.encode()) % 997 for w in c], "vals": [100.0 * n for n in c.values()]}

    def vec(self, text, is_query):
        import numpy as np, zlib
        rng = np.random.default_rng(zlib.crc32(text.encode()) + (1 if is_query else 0))
        return rng.standard_normal((4 if is_query else 16, self.hidden))


def test_diffretriever_builds_over_http_and_scores_with_the_cards_maxsim(tmp_path, monkeypatch):
    from agent_search.retrievers.learned import DiffRetrieverRetriever
    from agent_search.retrievers.learned import diffretriever as dr
    fake = _FakeEncoder()
    units = _units(DOCS)
    r = DiffRetrieverRetriever(model="fake/diffretriever", doc_length=512, url=fake.url,
                               index_root=str(tmp_path), store_device="cpu")
    r.index(units, key="tiny")
    assert all(c["max_text_tokens"] == 512 for c in fake.calls if not c["is_query"])      # pages at 512
    q = "treaty that ended the war"
    got = r.scores(q).tolist()
    qv = r.vectors([q], True)[0]
    for i, u in enumerate(units):
        pv = r.vectors([f"{u.qualname}\n{u.code}"], False)[0]
        want = float((qv @ pv.T).max(dim=-1).values.clamp(min=0).sum())            # the card's formula
        assert got[i] == pytest.approx(want, rel=2e-3, abs=2e-3)
    monkeypatch.setattr(dr, "CHUNK_PAGES", 2)
    assert r.scores(q).tolist() == pytest.approx(got, rel=1e-5, abs=1e-5)
    n = len(fake.calls)
    again = DiffRetrieverRetriever(model="fake/diffretriever", doc_length=512, url=fake.url,
                                   index_root=str(tmp_path), store_device="cpu").index(units, key="tiny")
    assert len(fake.calls) == n                                                      # loaded, not re-encoded
    assert again.search(q, 2) == r.search(q, 2)
    fake.server.shutdown()


def test_diffretriever_without_a_server_says_how_to_start_one(tmp_path, monkeypatch):
    from agent_search.retrievers.learned import DiffRetrieverRetriever
    monkeypatch.delenv("DIFFRETRIEVER_URL", raising=False)
    r = DiffRetrieverRetriever(model="fake/diffretriever", index_root=str(tmp_path), store_device="cpu")
    with pytest.raises(SetupError, match="serve_diffretriever.py"):
        r.vectors(["q"], True)


def test_diffretriever_sparse_and_hybrid_modes(tmp_path, monkeypatch):
    """Sparse: the dot product of the server's term weights. Hybrid: the authors' fusion (each
    list's top FUSION_DEPTH min-max normalised, 0.5 + 0.5, 0 where a page is missing). A dense
    index built first gets its sparse part added in its own pass, without re-encoding the vectors."""
    import numpy as np
    from agent_search.retrievers.learned import DiffRetrieverRetriever
    from agent_search.retrievers.learned import diffretriever as dr
    fake = _FakeEncoder()
    units = _units(DOCS)
    kw = dict(model="fake/diffretriever", doc_length=64, url=fake.url, index_root=str(tmp_path), store_device="cpu")
    DiffRetrieverRetriever(mode="dense", **kw).index(units, key="tiny")
    vectors = os.path.join(DiffRetrieverRetriever(mode="dense", **kw)._cache_dir("tiny"), "vectors.f16")
    stamp, before = os.path.getmtime(vectors), len(fake.calls)
    sparse_r = DiffRetrieverRetriever(mode="sparse", **kw)
    assert not sparse_r.is_cached("tiny")                                  # the dense index alone is not enough
    sparse_r.index(units, key="tiny")
    assert sparse_r.is_cached("tiny")
    added = fake.calls[before:]
    assert added and all(c.get("sparse") and not c["is_query"] for c in added)   # one sparse pass over the pages
    assert os.path.getmtime(vectors) == stamp                             # the dense vectors were not re-encoded
    q = "treaty war 1848"
    qt = fake.terms(q)
    got = sparse_r.scores(q)
    for i, u in enumerate(units):
        pt = fake.terms(f"{u.qualname}\n{u.code}")
        p = dict(zip(pt["ids"], pt["vals"]))
        assert got[i] == pytest.approx(sum(v * p.get(t, 0.0) for t, v in zip(qt["ids"], qt["vals"])))
    monkeypatch.setattr(dr, "FUSION_DEPTH", 3)
    hyb = DiffRetrieverRetriever(mode="hybrid", **kw).index(units, key="tiny")
    dense = DiffRetrieverRetriever(mode="dense", **kw).index(units, key="tiny").scores(q).float().cpu().numpy()
    fused = hyb.scores(q)
    def top_minmax(s):
        top = np.argsort(-s)[:3]; v = s[top]
        return dict(zip(top.tolist(), ((v - v.min()) / max(v.max() - v.min(), 1e-9)).tolist()))
    a, b = top_minmax(dense), top_minmax(got)
    for i in range(len(units)):
        want = 0.5 * a.get(i, 0.0) + 0.5 * b.get(i, 0.0) if (i in a or i in b) else -1.0
        assert fused[i] == pytest.approx(want, abs=1e-5)
    fake.server.shutdown()
