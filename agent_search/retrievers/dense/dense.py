"""Dense code retriever via sentence-transformers (cluster: needs torch + GPU).

Defaults to CodeRankEmbed (137M, strong/compact). Swap `model` for a larger anchor
(e.g. Alibaba-NLP/gte-Qwen2-7B-instruct). CodeRankEmbed expects a query prefix.

Protocol notes vs the CoRNStack paper's own SWE-bench eval (eval_swebench.py):
- max_seq_length=1024 matches theirs (model default is 8192).
- Document text is `qualname\ncode`, NOT bare code as in their eval: every
  SkimSearchAgent condition (structural executor, BM25, grep) sees the qualname, so
  dense gets it too — cross-condition uniformity outranks external protocol
  fidelity for the four-way comparison. Flagged when citing their numbers.
"""
from __future__ import annotations

import json
import os
import re
import sys
import threading
from typing import Optional, Sequence

from agent_search.corpus.fingerprint import corpus_fingerprint
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

SERVING_NOTE = "skimsearchagent_dense.json"


def _local_snapshot(model_id: str) -> Optional[str]:
    """The directory holding `model_id`'s files: the path itself for a local checkpoint, the
    cached hub snapshot for a hub id (no network), None when neither is available."""
    if os.path.isdir(model_id):
        return model_id
    try:
        from huggingface_hub import snapshot_download
        return snapshot_download(model_id, local_files_only=True)
    except Exception:  # noqa: BLE001 — not cached, or no hub library: let the caller decide
        return None


def serving_note(model_id: str) -> dict:
    """The note a checkpoint trained by `agent_search.training` carries (query instruction,
    pooling, normalisation, lengths); {} for a model without one."""
    try:
        d = _local_snapshot(model_id)
        p = os.path.join(d, SERVING_NOTE) if d else None
        if p and os.path.exists(p):
            with open(p) as fh:
                return json.load(fh) or {}
    except Exception:  # noqa: BLE001 — a bad note must not break loading; the table applies
        pass
    return {}


def resolve_pooling(model_id: str, note: Optional[dict] = None) -> Optional[str]:
    """Pooling for a local checkpoint directory: `DENSE_POOLING` (last_token | mean | cls) wins;
    then the serving note; then `last_token` for a decoder (Qwen) checkpoint whose config.json
    says so, which covers ITER's and LRAT's released retrievers. None means "let
    sentence-transformers decide" (hub models and directories with modules.json)."""
    env = (os.environ.get("DENSE_POOLING") or "").strip().lower()
    if env and env != "auto":
        return env
    note = note if note is not None else serving_note(model_id)
    if note.get("pooling"):
        return str(note["pooling"])
    try:
        d = _local_snapshot(model_id)
        cfg_path = os.path.join(d, "config.json") if d else ""
        if d and os.path.exists(cfg_path):
            with open(cfg_path) as fh:
                mtype = str((json.load(fh) or {}).get("model_type", "")).lower()
            if mtype.startswith(("qwen", "llama", "mistral", "gemma")):
                return "last_token"
    except Exception:  # noqa: BLE001
        pass
    return None


def external_index_path() -> Optional[str]:
    """`DENSE_INDEX_PATH`: a prebuilt vector index to serve instead of the per-corpus cache."""
    p = (os.environ.get("DENSE_INDEX_PATH") or "").strip()
    return p or None


_DTYPES = {"float32": "float32", "fp32": "float32", "float16": "float16", "fp16": "float16", "half": "float16",
           "bfloat16": "bfloat16", "bf16": "bfloat16"}


def resolve_dtype(model_id: str, note: Optional[dict] = None, device: Optional[str] = None) -> str:
    """The precision the encoder runs in: `DENSE_DTYPE` wins; then the checkpoint's serving note
    (`dtype`, written by the trainer: a model trained with bf16 autocast is served in bf16); else
    float32. float16 on a CPU becomes bfloat16 (CPUs have no fp16 matmul)."""
    env = (os.environ.get("DENSE_DTYPE") or "").strip().lower()
    chosen = None
    if env and env != "auto":
        if env not in _DTYPES:
            raise ValueError(f"unknown DENSE_DTYPE={env!r}; choose float32, float16 or bfloat16")
        chosen = _DTYPES[env]
    else:
        note = note if note is not None else serving_note(model_id)
        v = str(note.get("dtype") or "").lower()
        chosen = _DTYPES.get(v, "float32") if v else "float32"
    if chosen == "float16" and (device or "").lower() == "cpu":
        chosen = "bfloat16"
    return chosen


def _torch_dtype(name: str):
    import torch
    return {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[name]


def _to_numpy(embeddings):
    """sentence-transformers output to float32 numpy (bf16/fp16 tensors cannot convert directly)."""
    import numpy as np
    try:
        import torch
        if isinstance(embeddings, torch.Tensor):
            return embeddings.detach().float().cpu().numpy()
    except ImportError:
        pass
    return np.asarray(embeddings, dtype="float32")


def query_seq_length(model_id: str, default: int) -> int:
    """The token length queries are encoded with: the checkpoint's `query_max_len` (history-
    conditioned styles train with long queries) when its serving note has one, else `default`."""
    try:
        v = serving_note(model_id).get("query_max_len")
        return int(v) if v else int(default)
    except (TypeError, ValueError):
        return int(default)


def _encode_query(model, text: str, query_len: int):
    """Encode one query under the shared lock with the query-side length, restoring the
    document length afterwards (one encoder serves both sides)."""
    lock = getattr(model, "_agent_search_lock", None)
    if lock is not None:
        lock.acquire()
    try:
        doc_len = getattr(model, "max_seq_length", None)
        if doc_len is not None and query_len and query_len != doc_len:
            try:
                model.max_seq_length = query_len
                return _to_numpy(model.encode([text], convert_to_tensor=True, normalize_embeddings=True))[0]
            finally:
                model.max_seq_length = doc_len
        return _to_numpy(model.encode([text], convert_to_tensor=True, normalize_embeddings=True))[0]
    finally:
        if lock is not None:
            lock.release()


def query_prefix_for(model_id: str) -> str:
    """The query-side prefix for `model_id`, resolved ONCE for every dense arm: the
    `DENSE_QUERY_INSTRUCTION` env knob wins; then a trained checkpoint's serving note; then
    the built-in table for known hub models; else nothing."""
    instr = os.environ.get("DENSE_QUERY_INSTRUCTION")
    if instr:
        return f"Instruct: {instr}\nQuery: "
    note = serving_note(model_id)
    if note.get("query_instruction"):
        return f"Instruct: {note['query_instruction']}\nQuery: "
    return _QUERY_PREFIX.get(model_id, "")


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


def _shared_encoder(model_id: str, device, max_seq_length: int, dtype: Optional[str] = None):
    """One SentenceTransformer per (model, device, seqlen, dtype), shared across threads. `dtype`
    is the precision the weights are loaded in (see `resolve_dtype`)."""
    dtype = dtype or resolve_dtype(model_id, device=device)
    key = (model_id, device or "auto", max_seq_length, dtype)
    with _ENCODER_LOCK:
        enc = _ENCODER_CACHE.get(key)
        if enc is None:
            from sentence_transformers import SentenceTransformer  # heavy, cluster-only
            torch_dtype = _torch_dtype(dtype)
            note = serving_note(model_id)
            # a hub id resolves to its cached snapshot, so a released decoder checkpoint without a
            # sentence-transformers config (ITER, LRAT) gets the same pooling rebuild as a local one
            snap = _local_snapshot(model_id)
            has_st_config = os.path.exists(os.path.join(snap, "modules.json")) if snap else True
            pooling = resolve_pooling(model_id, note)
            if pooling and not has_st_config:
                note = dict(note, pooling=pooling)
                # a checkpoint from agent_search.training: a plain HF encoder dir; rebuild the
                # sentence-transformers pipeline it was trained with (pooling + normalisation)
                modes = {"last_token": "lasttoken", "mean": "mean", "cls": "cls"}
                if note["pooling"] not in modes:
                    raise ValueError(f"unknown pooling {note['pooling']!r} for {model_id}; choose one of "
                                     f"{sorted(modes)} (DENSE_POOLING / the serving note)")
                mode = modes[note["pooling"]]
                from sentence_transformers import models as st_models
                word = st_models.Transformer(snap or model_id, max_seq_length=int(note.get("max_seq_length") or max_seq_length),
                                             model_args={"trust_remote_code": True, "torch_dtype": torch_dtype})
                pool = st_models.Pooling(word.get_word_embedding_dimension(), pooling_mode=mode)
                mods = [word, pool] + ([st_models.Normalize()] if note.get("normalize", True) else [])
                enc = SentenceTransformer(modules=mods, device=device)
            else:
                enc = SentenceTransformer(model_id, trust_remote_code=True, device=device,
                                          model_kwargs={"torch_dtype": torch_dtype})
            # Clamp the requested length to the model's ACTUAL position capacity.
            # Forcing 1024 (the CoRNStack code protocol) onto a 512-position model
            # (e.g. bge-base for documents) overruns the position-embedding table ->
            # the CUDA forward stalls/asserts on the first long batch. Long-context
            # models (CodeRankEmbed) keep the requested length.
            cap = _model_max_positions(enc)
            enc.max_seq_length = min(max_seq_length, cap) if cap else max_seq_length
            enc._agent_search_lock = threading.Lock()   # serialize encode on the shared model
            enc._agent_search_dtype = dtype
            _ENCODER_CACHE[key] = enc
        return enc


class DenseRetriever(Retriever):
    name = "dense"

    def __init__(self, model: str = "nomic-ai/CodeRankEmbed", batch_size: int = 64,
                 index_root: str = "indexes", rebuild: bool = False, encoder=None,
                 max_seq_length: int = 1024, device: str | None = None, dtype: str | None = None):
        self.model_id = model
        self.max_seq_length = max_seq_length
        self._model = encoder
        self._device = device
        self.dtype = dtype or resolve_dtype(model, device=device or os.environ.get("AGENT_SEARCH_DENSE_DEVICE"))
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
            self._model = _shared_encoder(self.model_id, device, self.max_seq_length, self.dtype)
        return self._model

    def index(self, units: Sequence[CodeUnit], key: Optional[str] = None) -> "DenseRetriever":
        external = external_index_path()
        if external:
            # a prebuilt index on disk (this library's cache layout or ITER's index.faiss +
            # index.lookup.pkl): open it, never touch the units
            from agent_search.retrievers.dense.vector_index import load_external_index
            self._index = load_external_index(external)
            built_dtype = (self._index.meta or {}).get("dense_dtype")
            if built_dtype and built_dtype != self.dtype:
                print(f"  [dense] WARNING: {external} was built in {built_dtype} but queries are encoded in "
                      f"{self.dtype}; set retrieval.dense_dtype={built_dtype} to match", file=sys.stderr, flush=True)
            built_with = (self._index.meta or {}).get("dense_model")
            if built_with and built_with != self.model_id:
                raise ValueError(
                    f"DENSE_INDEX_PATH={external} was built with {built_with!r} but this run encodes "
                    f"queries with {self.model_id!r}; point retrieval.dense_model at the model the index "
                    f"was built with")
            if not built_with:
                print(f"  [dense] serving {external} (no record of the model that built it; make sure it "
                      f"matches {self.model_id!r})", file=sys.stderr, flush=True)
            self._doc_ids = self._index.doc_ids
            return self
        if getattr(units, "lazy", False):
            from agent_search.core.errors import SetupError
            raise SetupError(
                f"corpus key {key!r} is an on-disk document store; dense retrieval over it needs a "
                f"prebuilt index: set DENSE_INDEX_PATH (retrieval.dense_index) to its directory")
        cache_dir = self._cache_dir(key)
        # The embeddings of a fixed corpus are a one-time artifact: load the persisted
        # vector index (any backend: flat / hnsw / ivfpq) and never re-encode.
        fp = corpus_fingerprint(units)
        if not self.rebuild:
            try:
                idx = load_index(cache_dir)
                if idx is not None:
                    expected_ids = [u.doc_id for u in units]
                    cached_fp = (idx.meta or {}).get("corpus_fingerprint")
                    fp_stale = cached_fp is not None and cached_fp != fp
                    if list(idx.doc_ids) != expected_ids or fp_stale:
                        # Corpus congruence check -- mirrors StructuralExecutor.attach_units's
                        # doc-id-order validation for the BQL pickle path (bql/executor.py):
                        # a wrong-key collision or a stale cache from a since-changed corpus
                        # would otherwise silently hand back embeddings for the WRONG doc
                        # identities (row i's vector no longer means what `expected_ids[i]`
                        # says it means) -- every downstream score/rank would be corrupted
                        # with no error anywhere. Fail LOUD and rebuild rather than trust it.
                        # A present-but-different `corpus_fingerprint` catches the same-doc-
                        # ids-different-CONTENT case (a unit edited in place) that the doc-id
                        # list alone can't see; an OLD cache with no fingerprint key at all is
                        # trusted as before (nothing to compare against).
                        reason = ("content changed (corpus_fingerprint mismatch)" if fp_stale
                                  else f"cached {len(idx.doc_ids)} doc_ids, expected "
                                       f"{len(expected_ids)}")
                        print(
                            f"  [dense] WARNING: cached index at {cache_dir!r} is "
                            f"INCONGRUENT with the current corpus ({reason} for key={key!r}) "
                            f"-- discarding the stale/mismatched cache and rebuilding from "
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
        # No character pre-truncation: the encoder's own `max_seq_length` truncates in
        # TOKENS, which is the only length limit that applies here.
        texts = [f"{u.qualname}\n{u.code}" for u in units]
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
            emb = _to_numpy(model.encode(
                texts, batch_size=self._batch, convert_to_tensor=True,
                normalize_embeddings=True, show_progress_bar=big,
            ))
        finally:
            if lock is not None:
                lock.release()
        # Backend auto-selected by corpus size (flat default; FAISS ANN at scale).
        self._index = build_index(emb, self._doc_ids)
        if big:
            print(f"  [dense] built {self._index.backend} index for {len(texts)} units",
                  file=sys.stderr, flush=True)
        try:
            save_index(self._index, cache_dir, extra_meta={"corpus_fingerprint": fp, "dense_model": self.model_id,
                                                           "dense_dtype": self.dtype})
        except Exception:
            pass                # best-effort persistence; the in-memory index still serves
        return self

    def search(self, query: str, k: int) -> list[str]:
        q = query_prefix_for(self.model_id) + query
        qv = _encode_query(self._encoder(), q, query_seq_length(self.model_id, self.max_seq_length))
        return self._index.search(qv, k)              # backend-agnostic NN search

    def _cache_dir(self, key: Optional[str]) -> str:
        model_key = re.sub(r"[^A-Za-z0-9_.@-]+", "__", self.model_id)
        model_key += f"-sl{self.max_seq_length}"   # seq len changes the embeddings
        if self.dtype != "float32":
            model_key += f"-{self.dtype}"           # so do the weights' precision
        corpus_key = re.sub(r"[^A-Za-z0-9_.@-]+", "__", key or "default")
        return os.path.join(self.index_root, self.name, model_key, corpus_key)

    def is_cached(self, key: Optional[str] = None) -> bool:
        """True if a complete persisted index for `key` is already on disk (no load),
        so a pre-builder can skip re-parsing the corpus's units for it."""
        external = external_index_path()
        if external:
            return os.path.exists(external)
        return index_exists(self._cache_dir(key))


# --- registry ---------------------------------------------------------------
from agent_search.retrievers.registry import RetrieverConfig, register  # noqa: E402


@register("dense")
def _build_dense(cfg: RetrieverConfig, name: str):
    return lambda: DenseRetriever(
        cfg.dense_model or cfg.model or "nomic-ai/CodeRankEmbed",
        index_root=cfg.index_root, rebuild=cfg.rebuild)
