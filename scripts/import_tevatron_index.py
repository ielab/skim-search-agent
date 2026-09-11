"""Turn Tevatron encode output into a prebuilt vector index the library serves.

Tevatron (and ITER's `encode.py`) writes one pickle per shard: `(embeddings, doc_ids)`, a
float32 array of shape (n, dim) and a list of ids. This script concatenates the shards into
`<out>/index.faiss` (an exact inner-product index) and `<out>/index.lookup.pkl` (the ids in
index order), the layout `retrieval.dense_index` / `DENSE_INDEX_PATH` opens
(`agent_search/retrievers/dense/vector_index.py`). Use it for a corpus served from disk, which
the library never embeds during a run.

    python scripts/import_tevatron_index.py --shards /path/to/index-*.pkl --out indexes/external/iter06b_bcp_chunks
    python scripts/import_tevatron_index.py ... --corpus data/browsecomp_plus_chunks/corpus.jsonl   # check the ids exist
"""
from __future__ import annotations

import argparse
import glob
import os
import pickle

import numpy as np


def load_shards(patterns: list[str]) -> tuple[np.ndarray, list[str]]:
    paths = sorted(p for pat in patterns for p in glob.glob(pat))
    if not paths:
        raise SystemExit(f"no shard matches {patterns}")
    embs, ids = [], []
    for p in paths:
        with open(p, "rb") as fh:
            emb, docids = pickle.load(fh)
        embs.append(np.asarray(emb, dtype=np.float32))
        ids.extend(str(d) for d in docids)
        print(f"  {p}: {embs[-1].shape}")
    return np.concatenate(embs, axis=0), ids


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shards", nargs="+", required=True, help="Tevatron pickle shards, globs allowed")
    ap.add_argument("--out", required=True, help="directory for index.faiss and index.lookup.pkl")
    ap.add_argument("--corpus", default=None, help="a corpus.jsonl whose docid set the ids must be in")
    args = ap.parse_args()
    import faiss
    emb, ids = load_shards(args.shards)
    if len(ids) != emb.shape[0]:
        raise SystemExit(f"{emb.shape[0]} vectors but {len(ids)} ids")
    if len(set(ids)) != len(ids):
        raise SystemExit("duplicate ids across shards")
    if args.corpus:
        from agent_search.corpus.docstore import JsonlDocStore
        store = JsonlDocStore(args.corpus)
        missing = [d for d in ids if d not in store]
        if missing:
            raise SystemExit(f"{len(missing)} ids are not in {args.corpus}, e.g. {missing[:5]}")
        print(f"  every id is in {args.corpus} ({len(store)} documents)")
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)
    os.makedirs(args.out, exist_ok=True)
    faiss.write_index(index, os.path.join(args.out, "index.faiss"))
    with open(os.path.join(args.out, "index.lookup.pkl"), "wb") as fh:
        pickle.dump(ids, fh)
    print(f"wrote {args.out}: {index.ntotal} vectors, dim {emb.shape[1]}")


if __name__ == "__main__":
    main()
