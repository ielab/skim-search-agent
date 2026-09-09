"""On-disk document store for corpora too large to hold in memory.

`JsonlDocStore` opens a `corpus.jsonl` (one JSON document per line: `docid`/`_id`/`id`, `text`,
optional `title`) and builds a byte-offset index next to it on first use
(`<corpus>.offsets.npy`, `<corpus>.docids.*`). After that a document costs one seek and one
line parse, and the store never holds more than the ids in memory (ints as a numpy array, so
the 11.2M-chunk wiki corpus ITER used takes ~200 MB of ids, not 18 GB of text).

`LazyUnits` presents the store as the sequence of `CodeUnit`s the harness and the workspaces
expect, without materialising it: `len()` and positional access work, iteration streams, and
`by_id` is a mapping that builds a unit on demand (with a small LRU cache). Retrieval over such
a corpus goes through prebuilt indexes on disk (`DENSE_INDEX_PATH`, `BM25_INDEX_PATH`); the
in-memory engines refuse it loudly rather than trying to load everything.

Two title conventions are recognised: an explicit `title` field; a `---\\ntitle: ...\\n---`
front matter (BrowseComp-Plus chunks); or, for plain wiki chunks, the first line of `text`
when it is short.
"""
from __future__ import annotations

import bisect
import functools
import json
import os
import sys
from collections.abc import Mapping, Sequence
from typing import Iterator, Optional

from agent_search.corpus.units import CodeUnit, units_from_documents

_FRONT = "---\ntitle:"


def normalise_document(d: dict) -> dict:
    """{"doc_id", "title", "text"} from any of the supported record shapes."""
    doc_id = str(d.get("docid") if d.get("docid") is not None else d.get("_id") or d.get("id") or d.get("doc_id"))
    text = str(d.get("text") or d.get("contents") or d.get("body") or "")
    title = str(d.get("title") or "")
    if not title and text.startswith(_FRONT):
        head, _, rest = text[len("---\n"):].partition("\n---\n")
        title = head.split("\n", 1)[0].replace("title:", "", 1).strip().strip('"')
        text = rest.lstrip("\n") if rest else text
    elif not title:
        first, nl, _ = text.partition("\n")
        if nl and 0 < len(first) <= 200:
            title = first.strip()
    return {"doc_id": doc_id, "title": title, "text": text}


class JsonlDocStore:
    def __init__(self, path: str, *, build: bool = True):
        self.path = path
        import threading
        self._tl = threading.local()
        self._offsets = None
        self._ids_int = None      # sorted int ids + positions when every id is an integer
        self._ids_pos = None
        self._ids_map: Optional[dict] = None
        self._n = 0
        if build:
            self._load_or_build()

    # --- offsets index ------------------------------------------------------------
    @property
    def _offsets_path(self) -> str:
        return self.path + ".offsets.npy"

    def _load_or_build(self) -> None:
        import numpy as np
        st = os.stat(self.path)
        meta_path = self.path + ".docstore.json"
        ok = False
        if os.path.exists(self._offsets_path) and os.path.exists(meta_path):
            try:
                with open(meta_path) as fh:
                    meta = json.load(fh)
                ok = meta.get("size") == st.st_size and meta.get("mtime") == int(st.st_mtime)
            except Exception:  # noqa: BLE001
                ok = False
        if not ok:
            self._build_index(st)
        self._offsets = np.load(self._offsets_path, mmap_mode="r")
        self._n = int(len(self._offsets))
        if os.path.exists(self.path + ".docids.npy"):
            ids = np.load(self.path + ".docids.npy")
            order = np.argsort(ids, kind="stable")
            self._ids_int = ids[order]
            self._ids_pos = order
        else:
            with open(self.path + ".docids.json") as fh:
                self._ids_map = {str(d): i for i, d in enumerate(json.load(fh))}

    def _build_index(self, st) -> None:
        import numpy as np
        print(f"  [docstore] indexing {self.path} ({st.st_size / 1e9:.1f} GB) once ...",
              file=sys.stderr, flush=True)
        offsets, ids, all_int = [], [], True
        pos = 0
        with open(self.path, "rb") as fh:
            for line in fh:
                if line.strip():
                    offsets.append(pos)
                    d = json.loads(line)
                    doc_id = normalise_document(d)["doc_id"]
                    ids.append(doc_id)
                    if all_int and not doc_id.lstrip("-").isdigit():
                        all_int = False
                pos += len(line)
        tmp = self._offsets_path + f".tmp.{os.getpid()}"
        np.save(tmp, np.asarray(offsets, dtype=np.int64))
        os.replace(tmp + ".npy", self._offsets_path)
        if all_int:
            t = self.path + f".docids.tmp.{os.getpid()}"
            np.save(t, np.asarray([int(x) for x in ids], dtype=np.int64))
            os.replace(t + ".npy", self.path + ".docids.npy")
            if os.path.exists(self.path + ".docids.json"):
                os.remove(self.path + ".docids.json")
        else:
            t = self.path + f".docids.json.tmp.{os.getpid()}"
            with open(t, "w") as fh:
                json.dump(ids, fh)
            os.replace(t, self.path + ".docids.json")
            if os.path.exists(self.path + ".docids.npy"):
                os.remove(self.path + ".docids.npy")
        with open(self.path + ".docstore.json", "w") as fh:
            json.dump({"size": st.st_size, "mtime": int(st.st_mtime), "n": len(offsets)}, fh)
        print(f"  [docstore] {len(offsets)} documents indexed", file=sys.stderr, flush=True)

    # --- lookups ---------------------------------------------------------------------
    def __len__(self) -> int:
        return self._n

    def position(self, doc_id) -> Optional[int]:
        s = str(doc_id).strip()
        if self._ids_map is not None:
            return self._ids_map.get(s)
        if not s.lstrip("-").isdigit():
            return None
        import numpy as np
        v = int(s)
        i = int(np.searchsorted(self._ids_int, v))
        if i < len(self._ids_int) and int(self._ids_int[i]) == v:
            return int(self._ids_pos[i])
        return None

    def __contains__(self, doc_id) -> bool:
        return self.position(doc_id) is not None

    def id_at(self, pos: int) -> str:
        if self._ids_map is not None:
            if not hasattr(self, "_ids_list"):
                self._ids_list = [None] * len(self._ids_map)
                for k, v in self._ids_map.items():
                    self._ids_list[v] = k
            return self._ids_list[pos]
        import numpy as np
        # positions are a permutation; invert lazily once
        if not hasattr(self, "_pos_to_id"):
            inv = np.empty(len(self._ids_pos), dtype=np.int64)
            inv[self._ids_pos] = self._ids_int
            self._pos_to_id = inv
        return str(int(self._pos_to_id[pos]))

    def ids(self) -> list[str]:
        return [self.id_at(i) for i in range(self._n)]

    def raw_at(self, pos: int) -> dict:
        # one handle per thread: concurrent episodes share a store, and seek+readline on a
        # shared handle interleave
        fh = getattr(self._tl, "fh", None)
        if fh is None:
            fh = self._tl.fh = open(self.path, "rb")
        fh.seek(int(self._offsets[pos]))
        return json.loads(fh.readline())

    def get(self, doc_id) -> Optional[dict]:
        pos = self.position(doc_id)
        if pos is None:
            return None
        return normalise_document(self.raw_at(pos))

    def unit_at(self, pos: int) -> CodeUnit:
        d = normalise_document(self.raw_at(pos))
        return units_from_documents([d])[0]

    def unit(self, doc_id) -> Optional[CodeUnit]:
        pos = self.position(doc_id)
        return None if pos is None else self.unit_at(pos)

    def iter_units(self, limit: Optional[int] = None) -> Iterator[CodeUnit]:
        n = self._n if limit is None else min(limit, self._n)
        with open(self.path, "rb") as fh:
            for pos in range(n):
                fh.seek(int(self._offsets[pos]))
                d = normalise_document(json.loads(fh.readline()))
                yield units_from_documents([d])[0]

    def fingerprint(self) -> str:
        st = os.stat(self.path)
        return f"docstore:{os.path.abspath(self.path)}:{st.st_size}:{int(st.st_mtime)}"


class LazyUnitMap(Mapping):
    """doc_id -> CodeUnit, built on demand from a JsonlDocStore (small LRU cache)."""

    def __init__(self, store: JsonlDocStore, cache: int = 8192):
        self.store = store
        self._unit = functools.lru_cache(maxsize=cache)(store.unit)

    def __getitem__(self, doc_id):
        u = self._unit(str(doc_id))
        if u is None:
            raise KeyError(doc_id)
        return u

    def get(self, doc_id, default=None):
        u = self._unit(str(doc_id))
        return default if u is None else u

    def __contains__(self, doc_id) -> bool:
        return str(doc_id) in self.store

    def __iter__(self):
        return iter(self.store.ids())

    def __len__(self) -> int:
        return len(self.store)


class LazyUnits(Sequence):
    """The corpus as a sequence of units, streamed from disk; `by_id` for lookups."""
    lazy = True

    def __init__(self, store: JsonlDocStore):
        self.store = store
        self.by_id = LazyUnitMap(store)

    def __len__(self) -> int:
        return len(self.store)

    def __getitem__(self, i):
        if isinstance(i, slice):
            return [self.store.unit_at(p) for p in range(*i.indices(len(self)))]
        if i < 0:
            i += len(self)
        return self.store.unit_at(i)

    def __iter__(self):
        return self.store.iter_units()

    def __bool__(self) -> bool:
        return len(self) > 0


def is_lazy(units) -> bool:
    return bool(getattr(units, "lazy", False))


def refuse_lazy(units, engine: str, knob: str) -> None:
    """Raise SetupError when `units` is an on-disk corpus an in-memory engine would have to
    load whole. `engine` names the engine, `knob` the setting that serves it from a prebuilt
    index instead."""
    if is_lazy(units):
        from agent_search.core.errors import SetupError
        raise SetupError(
            f"the corpus is an on-disk document store ({len(units)} documents); {engine} would "
            f"have to load all of it into memory. Use a prebuilt index instead ({knob}), or a "
            f"strategy whose engines support prebuilt indexes (dedup_bm25, dedup_dense, search_visit "
            f"with bm25_backend: pyserini).")


__all__ = ["JsonlDocStore", "LazyUnitMap", "LazyUnits", "normalise_document", "is_lazy", "refuse_lazy"]
