"""Dense code retriever via sentence-transformers (cluster: needs torch + GPU).

Defaults to CodeRankEmbed (137M, strong/compact). Swap `model` for a larger anchor
(e.g. Alibaba-NLP/gte-Qwen2-7B-instruct). CodeRankEmbed expects a query prefix.

Protocol notes vs the CoRNStack paper's own SWE-bench eval (eval_swebench.py):
- max_seq_length=1024 matches theirs (model default is 8192).
- Document text is `qualname\ncode`, NOT bare code as in their eval: every
  BoolAgent condition (structural executor, BM25, grep) sees the qualname, so
  dense gets it too — cross-condition uniformity outranks external protocol
  fidelity for the four-way comparison. Flagged when citing their numbers.
"""
from __future__ import annotations

import os
import re
import sys
import threading
from typing import Optional, Sequence

from agent_search.corpus.units import CodeUnit
from agent_search.core.interfaces import Retriever
from agent_search.retrievers.dense.vector_index import (
    build_index, index_exists, load_index, save_index)

# Per-model query instruction prefixes (empty if none required).
#
# Qwen3-Embedding-{0.6B,4B,8B}: per the model card (Qwen/Qwen3-Embedding-0.6B README,
# `get_detailed_instruct`), queries get `f'Instruct: {task}\nQuery:{query}'` — NOTE no space
# between "Query:" and the query text, matching the model's own training format and its
# sentence-transformers `config_sentence_transformers.json` `prompts.query` string verbatim.
# Documents get NO prefix (`prompts.document == ""`), same as every other model here.
_QWEN3_EMBED_INSTRUCT = (
    "Instruct: Given a web search query, retrieve relevant passages that answer the query\n"
    "Query:"
)
_QUERY_PREFIX = {
    "nomic-ai/CodeRankEmbed": "Represent this query for searching relevant code: ",
    "cornstack/CodeRankEmbed": "Represent this query for searching relevant code: ",
    "Qwen/Qwen3-Embedding-0.6B": _QWEN3_EMBED_INSTRUCT,
    "Qwen/Qwen3-Embedding-4B": _QWEN3_EMBED_INSTRUCT,
    "Qwen/Qwen3-Embedding-8B": _QWEN3_EMBED_INSTRUCT,
}


_ENCODER_CACHE: dict = {}
_ENCODER_LOCK = threading.Lock()


def _model_max_positions(enc) -> Optional[int]:
    """The model's hard position-embedding limit, or None if it can't be read."""
    try:
        cfg = enc[0].auto_model.config
    except Exception:
        return None
    for attr in ("max_position_embeddings", "n_positions", "model_max_length"):
        v = getattr(cfg, attr, None)
        if isinstance(v, int) and 0 < v < 1_000_000:    # ignore the int32 sentinel
            return v
    return None


def _shared_encoder(model_id: str, device, max_seq_length: int):
    """One SentenceTransformer per (model, device, seqlen), shared across threads."""
    key = (model_id, device or "auto", max_seq_length)
    with _ENCODER_LOCK:
        enc = _ENCODER_CACHE.get(key)
        if enc is None:
            from sentence_transformers import SentenceTransformer  # heavy, cluster-only
            enc = SentenceTransformer(model_id, trust_remote_code=True, device=device)
            # Clamp the requested length to the model's ACTUAL position capacity.
            # Forcing 1024 (the CoRNStack code protocol) onto a 512-position model
            # (e.g. bge-base for documents) overruns the position-embedding table ->
            # the CUDA forward stalls/asserts on the first long batch. Long-context
            # models (CodeRankEmbed) keep the requested length.
            cap = _model_max_positions(enc)
            enc.max_seq_length = min(max_seq_length, cap) if cap else max_seq_length
            enc._agent_search_lock = threading.Lock()   # serialize encode on the shared model
            _ENCODER_CACHE[key] = enc
        return enc


class DenseRetriever(Retriever):
    name = "dense"

    def __init__(self, model: str = "nomic-ai/CodeRankEmbed", batch_size: int = 64,
                 index_root: str = "indexes", rebuild: bool = False, encoder=None,
                 max_seq_length: int = 1024, device: str | None = None):
        self.model_id = model
        self.max_seq_length = max_seq_length
        self._model = encoder
        self._device = device
        self._batch = batch_size
        self.index_root = index_root
        self.rebuild = rebuild
        self._doc_ids: list[str] = []
        self._index = None

    def _encoder(self):
        if self._model is None:
            # Device: AGENT_SEARCH_DENSE_DEVICE wins (agent scripts force "cpu" when a
            # co-resident vLLM server owns the GPU). Unset -> ST auto (GPU floors).
            device = self._device or os.environ.get("AGENT_SEARCH_DENSE_DEVICE") or None
            # SHARED across all worker threads: a fresh load per instance meant up
            # to WORKERS copies of the model in (host) RAM -> OOM on agent_dense.
            self._model = _shared_encoder(self.model_id, device, self.max_seq_length)
        return self._model

    def index(self, units: Sequence[CodeUnit], key: Optional[str] = None) -> "DenseRetriever":
        cache_dir = self._cache_dir(key)
        # The embeddings of a fixed corpus are a one-time artifact: load the persisted
        # vector index (any backend: flat / hnsw / ivfpq) and never re-encode.
        if not self.rebuild:
            try:
                idx = load_index(cache_dir)
                if idx is not None:
                    expected_ids = [u.doc_id for u in units]
                    if list(idx.doc_ids) != expected_ids:
                        # Corpus congruence check -- mirrors StructuralExecutor.attach_units's
                        # doc-id-order validation for the BQL pickle path (bql/executor.py):
                        # a wrong-key collision or a stale cache from a since-changed corpus
                        # would otherwise silently hand back embeddings for the WRONG doc
                        # identities (row i's vector no longer means what `expected_ids[i]`
                        # says it means) -- every downstream score/rank would be corrupted
                        # with no error anywhere. Fail LOUD and rebuild rather than trust it.
                        print(
                            f"  [dense] WARNING: cached index at {cache_dir!r} is "
                            f"INCONGRUENT with the current corpus (cached {len(idx.doc_ids)} "
                            f"doc_ids, expected {len(expected_ids)} for key={key!r}) -- "
                            f"discarding the stale/mismatched cache and rebuilding from "
                            f"scratch (this is a correctness guard, not a perf hint: a "
                            f"silently-wrong cache would return embeddings for the wrong "
                            f"documents)", file=sys.stderr, flush=True)
                    else:
                        self._index = idx
                        self._doc_ids = idx.doc_ids
                        return self
            except Exception:
                pass            # truncated/corrupt cache (killed writer): rebuild

        self._doc_ids = [u.doc_id for u in units]
        # Cap input CHARS before tokenizing. The encoder truncates to max_seq_length
        # TOKENS anyway, so a multi-MB web document (BrowseComp corpus) is tokenized in
        # full and then thrown away — single-threaded, that stalls the encode with the
        # GPU idle. ~16 chars/token is a safe upper bound for normal text/code (typical
        # is 3-5), so the kept prefix still contains the first max_seq_length tokens =>
        # identical embeddings for any realistic doc; the tokenizer never processes more
        # than ~tens of KB. (A pathological <1-token-per-16-char blob could in theory
        # lose a token at the boundary — not reachable for real qualname\ncode units.)
        char_cap = max(2048, self.max_seq_length * 16)
        texts = [f"{u.qualname}\n{u.code}"[:char_cap] for u in units]
        model = self._encoder()
        # A large shared corpus (e.g. BrowseComp-Plus ~100k docs) takes minutes to
        # embed; building it silently at eval time looks like a hang. Announce it and
        # show the bar so progress is visible — and pre-build via build_indexes.py to
        # skip this at eval time entirely. CPU encoding of 100k docs is effectively a
        # hang; build on the GPU (do not pin AGENT_SEARCH_DENSE_DEVICE=cpu here).
        big = len(texts) >= 5000
        if big:
            dev = getattr(model, "device", "?")
            print(f"  [dense] encoding {len(texts)} units for key={key!r} on {dev} "
                  f"(pre-build with build_indexes.py to avoid this at eval time)",
                  file=sys.stderr, flush=True)
        lock = getattr(model, "_agent_search_lock", None)
        if lock is not None:
            lock.acquire()
        try:
            emb = model.encode(
                texts, batch_size=self._batch, convert_to_numpy=True,
                normalize_embeddings=True, show_progress_bar=big,
            )
        finally:
            if lock is not None:
                lock.release()
        # Backend auto-selected by corpus size (flat default; FAISS ANN at scale).
        self._index = build_index(emb, self._doc_ids)
        if big:
            print(f"  [dense] built {self._index.backend} index for {len(texts)} units",
                  file=sys.stderr, flush=True)
        try:
            save_index(self._index, cache_dir)
        except Exception:
            pass                # best-effort persistence; the in-memory index still serves
        return self

    def search(self, query: str, k: int) -> list[str]:
        q = _QUERY_PREFIX.get(self.model_id, "") + query
        model = self._encoder()
        lock = getattr(model, "_agent_search_lock", None)
        if lock is not None:
            lock.acquire()
        try:
            qv = model.encode([q], convert_to_numpy=True, normalize_embeddings=True)[0]
        finally:
            if lock is not None:
                lock.release()
        return self._index.search(qv, k)              # backend-agnostic NN search

    def _cache_dir(self, key: Optional[str]) -> str:
        model_key = re.sub(r"[^A-Za-z0-9_.@-]+", "__", self.model_id)
        model_key += f"-sl{self.max_seq_length}"   # seq len changes the embeddings
        corpus_key = re.sub(r"[^A-Za-z0-9_.@-]+", "__", key or "default")
        return os.path.join(self.index_root, self.name, model_key, corpus_key)

    def is_cached(self, key: Optional[str] = None) -> bool:
        """True if a complete persisted index for `key` is already on disk (no load),
        so a pre-builder can skip re-parsing the corpus's units for it."""
        return index_exists(self._cache_dir(key))


# --- registry ---------------------------------------------------------------
from agent_search.retrievers.registry import RetrieverConfig, register  # noqa: E402


@register("dense")
def _build_dense(cfg: RetrieverConfig, name: str):
    return lambda: DenseRetriever(
        cfg.dense_model or cfg.model or "nomic-ai/CodeRankEmbed",
        index_root=cfg.index_root, rebuild=cfg.rebuild)
