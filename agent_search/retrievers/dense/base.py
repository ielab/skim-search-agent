"""The dense retriever base class.

A dense retriever embeds every unit once, keeps the vectors in a persisted index, and answers a
query with the nearest neighbours. What differs between encoder families is small and explicit:
the prefix put in front of a query, the pooling, the precision the weights load in, the sequence
lengths. Each family is one file in this package (`bge.py`, `coderank.py`,
`qwen3_embedding.py`, `trained.py`) that sets those attributes on a subclass; `DenseRetriever(model)`
returns the matching subclass (see `__new__` and `family_for`).

Environment knobs override a family's choices for one run: `DENSE_QUERY_INSTRUCTION` (query
prefix), `DENSE_POOLING`, `DENSE_DTYPE`, `DENSE_INDEX_PATH` (a prebuilt index), and
`AGENT_SEARCH_DENSE_DEVICE`.
"""
from __future__ import annotations

import os
import re
import sys
import threading
from typing import Optional, Sequence

from agent_search.corpus.fingerprint import corpus_fingerprint
from agent_search.corpus.units import CodeUnit
from agent_search.retrievers.base import Retriever
from agent_search.retrievers.dense.vector_index import build_index, index_exists, load_index, save_index

SERVING_NOTE = "skimsearchagent_dense.json"

_ENCODER_CACHE: dict = {}
_ENCODER_LOCK = threading.Lock()

# the families, most specific first; `register_family` adds one (plugins included)
FAMILIES: list = []

_DTYPES = {"float32": "float32", "fp32": "float32", "float16": "float16", "fp16": "float16", "half": "float16",
           "bfloat16": "bfloat16", "bf16": "bfloat16"}
_POOLINGS = {"last_token": "lasttoken", "mean": "mean", "cls": "cls"}


def register_family(cls):
    """Register a DenseRetriever subclass; `DenseRetriever(model)` asks each family's `matches`
    in registration order and uses the first that says yes."""
    if cls not in FAMILIES:
        FAMILIES.append(cls)
    return cls


def family_for(model_id: str):
    for cls in FAMILIES:
        try:
            if cls.matches(model_id):
                return cls
        except Exception:  # noqa: BLE001: a family's probe must never break resolution
            continue
    return DenseRetriever


def local_snapshot(model_id: str) -> Optional[str]:
    """The directory holding `model_id`'s files: the path itself for a local checkpoint, the
    cached hub snapshot for a hub id (no network), None when neither is available."""
    if os.path.isdir(model_id):
        return model_id
    path = _hub_snapshot(model_id)
    return _complete_snapshot(path) if path else None


def _hub_snapshot(model_id: str) -> Optional[str]:
    """The cached hub snapshot directory for `model_id` (no network), or None."""
    try:
        from huggingface_hub import snapshot_download
        return snapshot_download(model_id, local_files_only=True)
    except Exception:  # noqa: BLE001
        return None


def _snapshot_is_complete(path: str) -> bool:
    """A snapshot directory that can be loaded: it has config.json and either weights or a
    sentence-transformers configuration."""
    if not os.path.exists(os.path.join(path, "config.json")):
        return False
    names = set(os.listdir(path))
    return "modules.json" in names or any(n.endswith((".safetensors", ".bin")) for n in names)


def _complete_snapshot(path: str) -> str:
    """The hub cache keeps one directory per revision, and a refresh that only fetched the
    README leaves the newest revision with nothing to load. When the resolved snapshot is not
    loadable, serve the newest sibling snapshot that is."""
    if _snapshot_is_complete(path):
        return path
    parent = os.path.dirname(path)
    try:
        siblings = [os.path.join(parent, n) for n in os.listdir(parent)]
    except OSError:
        return path
    complete = [d for d in siblings if os.path.isdir(d) and _snapshot_is_complete(d)]
    if not complete:
        return path
    return max(complete, key=os.path.getmtime)


def external_index_path() -> Optional[str]:
    """`DENSE_INDEX_PATH`: a prebuilt vector index to serve instead of the per-corpus cache."""
    p = (os.environ.get("DENSE_INDEX_PATH") or "").strip()
    return p or None


def _torch_dtype(name: str):
    import torch
    return {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[name]


def to_numpy(embeddings):
    """sentence-transformers output to float32 numpy (bf16/fp16 tensors cannot convert directly)."""
    import numpy as np
    try:
        import torch
        if isinstance(embeddings, torch.Tensor):
            return embeddings.detach().float().cpu().numpy()
    except ImportError:
        pass
    return np.asarray(embeddings, dtype="float32")


def encode_with_retry(model, texts, **kw):
    """`model.encode` with one retry after a CUDA out-of-memory error: the cache is released and
    the call repeated, so a search does not fail on a transient allocation next to the served
    backbone. A second failure raises."""
    try:
        return model.encode(texts, **kw)
    except Exception as e:  # noqa: BLE001: torch raises RuntimeError or OutOfMemoryError
        if "out of memory" not in str(e).lower():
            raise
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass
        return model.encode(texts, **kw)


def encode_query(model, text: str, query_len: int):
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
                return to_numpy(encode_with_retry(model, [text], convert_to_tensor=True, normalize_embeddings=True))[0]
            finally:
                model.max_seq_length = doc_len
        return to_numpy(encode_with_retry(model, [text], convert_to_tensor=True, normalize_embeddings=True))[0]
    finally:
        if lock is not None:
            lock.release()


def _model_max_positions(enc) -> Optional[int]:
    """The model's hard position-embedding limit, or None if it can't be read."""
    try:
        cfg = enc[0].auto_model.config
    except Exception:  # noqa: BLE001
        return None
    for attr in ("max_position_embeddings", "n_positions", "model_max_length"):
        v = getattr(cfg, attr, None)
        if isinstance(v, int) and 0 < v < 1_000_000:    # ignore the int32 sentinel
            return v
    return None


class DenseRetriever(Retriever):
    """Embed units once, search by cosine similarity. Subclass per encoder family; override the
    class attributes below (or `build_encoder` for anything unusual)."""

    name = "dense"

    # --- what a family declares -----------------------------------------------------------
    query_prefix: str = ""            # put in front of every query (documents get no prefix)
    pooling: Optional[str] = None     # last_token | mean | cls; None: the model's own config
    default_dtype: str = "float32"    # precision the weights load in
    normalize: bool = True            # L2-normalise embeddings (cosine retrieval)
    query_max_len: Optional[int] = None   # tokens for queries; None: same as documents
    encoder_seq_length: Optional[int] = None   # documents; None: the constructor's max_seq_length

    @classmethod
    def matches(cls, model_id: str) -> bool:
        """Does this family serve `model_id`? The base class serves anything."""
        return True

    def __new__(cls, model: str = "nomic-ai/CodeRankEmbed", *args, **kwargs):
        # `DenseRetriever("Qwen/Qwen3-Embedding-0.6B")` returns the Qwen3 family, and so on
        if cls is DenseRetriever:
            cls = family_for(model)
        return object.__new__(cls)

    def __init__(self, model: str = "nomic-ai/CodeRankEmbed", batch_size: int = 64,
                 index_root: str = "indexes", rebuild: bool = False, encoder=None,
                 max_seq_length: int | None = None, device: str | None = None, dtype: str | None = None):
        self.model_id = model
        # the length documents and queries are cut to, in the encoder's tokens: the constructor's
        # value, else `DENSE_SEQ_LENGTH` (retrieval.dense_seq_length), else 1024. ITER encodes at
        # 512, the length its checkpoints were trained on.
        self.max_seq_length = int(max_seq_length or os.environ.get("DENSE_SEQ_LENGTH") or 1024)
        self._model = encoder
        self._device = device
        self._batch = batch_size
        self.index_root = index_root
        self.rebuild = rebuild
        self._doc_ids: list[str] = []
        self._index = None
        self.dtype = dtype or self.resolve_dtype(device or os.environ.get("AGENT_SEARCH_DENSE_DEVICE"))

    # --- the knobs a run can override ------------------------------------------------------
    def query_prefix_for(self) -> str:
        """The query-side prefix: `DENSE_QUERY_INSTRUCTION` wins, else the family's."""
        instr = os.environ.get("DENSE_QUERY_INSTRUCTION")
        if instr:
            return f"Instruct: {instr}\nQuery: "
        return self.query_prefix

    def resolve_pooling(self) -> Optional[str]:
        """`DENSE_POOLING` wins (validated when the encoder is built), else the family's."""
        env = (os.environ.get("DENSE_POOLING") or "").strip().lower()
        if env and env != "auto":
            return env
        return self.pooling

    def resolve_dtype(self, device: Optional[str] = None) -> str:
        """`DENSE_DTYPE` wins, else the family's; float16 on a CPU becomes bfloat16."""
        env = (os.environ.get("DENSE_DTYPE") or "").strip().lower()
        if env and env != "auto":
            if env not in _DTYPES:
                raise ValueError(f"unknown DENSE_DTYPE={env!r}; choose float32, float16 or bfloat16")
            chosen = _DTYPES[env]
        else:
            chosen = _DTYPES.get(str(self.default_dtype).lower(), "float32")
        if chosen == "float16" and (device or "").lower() == "cpu":
            chosen = "bfloat16"
        return chosen

    def query_seq_length(self) -> int:
        return int(self.query_max_len or self.max_seq_length)

    # --- the encoder --------------------------------------------------------------------
    def build_encoder(self, device, max_seq_length: int, dtype: str):
        """A sentence-transformers model for this family. Families with a pooling of their own
        (a decoder checkpoint without a sentence-transformers config) get the pipeline rebuilt:
        transformer, pooling, optional normalisation."""
        from sentence_transformers import SentenceTransformer  # heavy, cluster-only
        torch_dtype = _torch_dtype(dtype)
        pooling = self.resolve_pooling()
        snap = local_snapshot(self.model_id)
        has_st_config = os.path.exists(os.path.join(snap, "modules.json")) if snap else True
        if pooling and not has_st_config:
            if pooling not in _POOLINGS:
                raise ValueError(f"unknown pooling {pooling!r} for {self.model_id}; choose one of "
                                 f"{sorted(_POOLINGS)} (DENSE_POOLING / the serving note)")
            from sentence_transformers import models as st_models
            word = st_models.Transformer(snap or self.model_id,
                                         max_seq_length=int(self.encoder_seq_length or max_seq_length),
                                         model_args={"trust_remote_code": True, "torch_dtype": torch_dtype})
            pool = st_models.Pooling(word.get_word_embedding_dimension(), pooling_mode=_POOLINGS[pooling])
            mods = [word, pool] + ([st_models.Normalize()] if self.normalize else [])
            return SentenceTransformer(modules=mods, device=device)
        return SentenceTransformer(self.model_id, trust_remote_code=True, device=device,
                                   model_kwargs={"torch_dtype": torch_dtype})

    def _encoder(self):
        if self._model is None:
            device = self._device or os.environ.get("AGENT_SEARCH_DENSE_DEVICE") or None
            self._model = self._shared_encoder(device)
        return self._model

    def _shared_encoder(self, device):
        """One encoder per (model, device, sequence length, dtype), shared across threads."""
        key = (self.model_id, device or "auto", self.max_seq_length, self.dtype)
        with _ENCODER_LOCK:
            enc = _ENCODER_CACHE.get(key)
            if enc is None:
                enc = self.build_encoder(device, self.max_seq_length, self.dtype)
                # clamp to the model's real position capacity (forcing 1024 on a 512-position
                # model overruns its position table on the first long batch)
                cap = _model_max_positions(enc)
                enc.max_seq_length = min(self.max_seq_length, cap) if cap else self.max_seq_length
                enc._agent_search_lock = threading.Lock()   # serialize encode on the shared model
                enc._agent_search_dtype = self.dtype
                _ENCODER_CACHE[key] = enc
            return enc

    # --- index and search -------------------------------------------------------------------
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
            from agent_search.errors import SetupError
            raise SetupError(
                f"corpus key {key!r} is an on-disk document store; dense retrieval over it needs a "
                f"prebuilt index: set DENSE_INDEX_PATH (retrieval.dense_index) to its directory")
        cache_dir = self._cache_dir(key)
        # the embeddings of a fixed corpus are a one-time artifact: load the persisted index and
        # never re-encode; a cache that does not match the corpus is discarded, never trusted
        fp = corpus_fingerprint(units)
        if not self.rebuild:
            try:
                idx = load_index(cache_dir)
                if idx is not None:
                    expected_ids = [u.doc_id for u in units]
                    cached_fp = (idx.meta or {}).get("corpus_fingerprint")
                    fp_stale = cached_fp is not None and cached_fp != fp
                    if list(idx.doc_ids) != expected_ids or fp_stale:
                        reason = ("content changed (corpus_fingerprint mismatch)" if fp_stale
                                  else f"cached {len(idx.doc_ids)} doc_ids, expected {len(expected_ids)}")
                        print(f"  [dense] WARNING: cached index at {cache_dir!r} does not match the current "
                              f"corpus ({reason} for key={key!r}); rebuilding it", file=sys.stderr, flush=True)
                    else:
                        self._index = idx
                        self._doc_ids = idx.doc_ids
                        return self
            except Exception:  # noqa: BLE001: a truncated cache (killed writer) is rebuilt
                pass

        self._doc_ids = [u.doc_id for u in units]
        texts = [f"{u.qualname}\n{u.code}" for u in units]     # the encoder truncates in tokens
        model = self._encoder()
        big = len(texts) >= 5000
        if big:
            print(f"  [dense] encoding {len(texts)} units for key={key!r} on {getattr(model, 'device', '?')} "
                  f"(pre-build with build_indexes.py to avoid this at eval time)", file=sys.stderr, flush=True)
        lock = getattr(model, "_agent_search_lock", None)
        if lock is not None:
            lock.acquire()
        try:
            emb = to_numpy(encode_with_retry(model, texts, batch_size=self._batch, convert_to_tensor=True,
                                              normalize_embeddings=True, show_progress_bar=big))
        finally:
            if lock is not None:
                lock.release()
        self._index = build_index(emb, self._doc_ids)      # flat by default, FAISS ANN at scale
        if big:
            print(f"  [dense] built {self._index.backend} index for {len(texts)} units", file=sys.stderr, flush=True)
        try:
            save_index(self._index, cache_dir, extra_meta={"corpus_fingerprint": fp, "dense_model": self.model_id,
                                                           "dense_dtype": self.dtype})
        except Exception:  # noqa: BLE001: best-effort persistence; the in-memory index still serves
            pass
        return self

    def search(self, query: str, k: int) -> list[str]:
        qv = self.encode_query(query)
        return self._index.search(qv, k)

    def encode_query(self, query: str):
        """The query vector: the family's prefix, the shared encoder, the query-side length."""
        return encode_query(self._encoder(), self.query_prefix_for() + query, self.query_seq_length())

    def _cache_dir(self, key: Optional[str]) -> str:
        model_key = re.sub(r"[^A-Za-z0-9_.@-]+", "__", self.model_id)
        model_key += f"-sl{self.max_seq_length}"   # seq len changes the embeddings
        if self.dtype != "float32":
            model_key += f"-{self.dtype}"           # so do the weights' precision
        corpus_key = re.sub(r"[^A-Za-z0-9_.@-]+", "__", key or "default")
        return os.path.join(self.index_root, self.name, model_key, corpus_key)

    def is_cached(self, key: Optional[str] = None) -> bool:
        """True if a complete persisted index for `key` is on disk (no load), or a prebuilt one
        is named by `DENSE_INDEX_PATH`."""
        external = external_index_path()
        if external:
            return os.path.exists(external)
        return index_exists(self._cache_dir(key))


__all__ = ["DenseRetriever", "register_family", "family_for", "FAMILIES", "local_snapshot",
           "external_index_path", "encode_query", "to_numpy", "SERVING_NOTE"]
