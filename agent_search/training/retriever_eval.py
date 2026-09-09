"""`skimsearchagent-eval-retriever`: score a retriever on trajectory triples without an agent.

    skimsearchagent-eval-retriever --triples train_data/infoseek_i2.jsonl --dataset infoseek_train \\
        --dense-model models/my-retriever --k 1,5,10 --subset 5000

For every triple the retriever is asked the triple's `query` (already rendered in the training
style) and two things are measured at each cutoff k:

* **recall@k**: is the positive among the top k;
* **novelty@k**: the fraction of the top k the agent had not already read at that point (ids
  outside `neg_diversity_id` and `neg_hard_id`). ITER's retriever is trained to return new
  documents, so this is the number its objective moves.

Corpus: `--dense-model` serves the dataset's persisted embedding cache, or `DENSE_INDEX_PATH`
when set (a prebuilt index, e.g. ITER's wiki index), or with `--subset N` encodes a small corpus
made of every document the triples mention plus the first N documents of the collection (the
quick check after a smoke training run: minutes, not the hours a full re-index takes).
`--bm25` scores the in-memory BM25 instead (or `BM25_INDEX_PATH` with the pyserini backend).
Results go to stdout as JSON and next to the triples as `<triples>.eval.json`.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Callable, Iterable, Optional, Sequence


def evaluate(triples: Iterable[dict], search: Callable[[str, int], Sequence[str]],
             ks: Sequence[int] = (1, 5, 10)) -> dict:
    ks = sorted(set(int(k) for k in ks))
    kmax = max(ks)
    n = 0
    recall = {k: 0.0 for k in ks}
    novelty = {k: 0.0 for k in ks}
    for t in triples:
        pos = set(t.get("pos_id") or [])
        if not pos:
            continue
        seen = set(t.get("neg_diversity_id") or []) | set(t.get("neg_hard_id") or [])
        ranked = [str(d) for d in search(t["query"], kmax)]
        n += 1
        for k in ks:
            top = ranked[:k]
            recall[k] += 1.0 if pos & set(top) else 0.0
            novelty[k] += (sum(1 for d in top if d not in seen) / len(top)) if top else 0.0
    out = {"n": n}
    for k in ks:
        out[f"recall@{k}"] = recall[k] / n if n else 0.0
        out[f"novelty@{k}"] = novelty[k] / n if n else 0.0
    return out


def _triple_doc_ids(triples: Sequence[dict]) -> list[str]:
    ids: list[str] = []
    seen: set = set()
    for t in triples:
        for key in ("pos_id", "neg_diversity_id", "neg_hard_id", "neg_random_id"):
            for d in t.get(key) or []:
                if str(d) not in seen:
                    seen.add(str(d))
                    ids.append(str(d))
    return ids


def corpus_units(dataset: str, triples: Optional[Sequence[dict]] = None, subset: Optional[int] = None):
    """The units to index: the dataset's corpus (lazy when it lives on disk), or a subset of it."""
    from agent_search.corpus.docstore import LazyUnits
    from agent_search.corpus.units import units_from_documents
    from agent_search.evaluation.datasets import load_dataset_by_name
    inst = load_dataset_by_name(dataset)
    if not inst:
        raise SystemExit(f"dataset {dataset!r} is empty")
    first = inst[0]
    if first.docstore is not None:
        units = LazyUnits(first.docstore)
    elif first.docs is not None:
        units = units_from_documents(first.docs)
    else:
        raise SystemExit(f"dataset {dataset!r} has no shared corpus")
    if subset is None:
        return units
    by_id = getattr(units, "by_id", None) or {u.doc_id: u for u in units}
    out, seen = [], set()
    for d in _triple_doc_ids(triples or []):
        u = by_id.get(d)
        if u is not None and d not in seen:
            seen.add(d)
            out.append(u)
    for i in range(min(subset, len(units))):
        u = units[i]
        if u.doc_id not in seen:
            seen.add(u.doc_id)
            out.append(u)
    return out


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="skimsearchagent-eval-retriever", description=__doc__.split("\n\n")[0])
    ap.add_argument("--triples", required=True)
    ap.add_argument("--dataset", required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dense-model", default=None)
    g.add_argument("--bm25", action="store_true")
    ap.add_argument("--k", default="1,5,10")
    ap.add_argument("--index-root", default="indexes")
    ap.add_argument("--subset", type=int, default=None,
                    help="encode only the triples' documents plus the first N of the corpus")
    ap.add_argument("--out", default=None, help="where to write the JSON (default <triples>.eval.json)")
    a = ap.parse_args(argv)
    triples = [json.loads(l) for l in Path(a.triples).read_text(encoding="utf-8").splitlines() if l.strip()]
    units = corpus_units(a.dataset, triples, a.subset)
    key = a.dataset if a.subset is None else f"{a.dataset}-subset{a.subset}"
    if a.bm25:
        from agent_search.retrievers.lexical import build_bm25_engine
        engine = build_bm25_engine(units, index_root=a.index_root, key=key)
        search = lambda q, k: engine.search(q, k=k)  # noqa: E731
    else:
        from agent_search.retrievers.dense.dense import DenseRetriever
        if a.subset is not None:
            os.environ.pop("DENSE_INDEX_PATH", None)   # a subset is always encoded fresh
        r = DenseRetriever(a.dense_model, index_root=a.index_root).index(units, key=key)
        search = lambda q, k: r.search(q, k)  # noqa: E731
    res = evaluate(triples, search, ks=[int(x) for x in a.k.split(",")])
    res.update({"triples": a.triples, "dataset": a.dataset, "retriever": a.dense_model or "bm25",
                "n_docs": len(units), "subset": a.subset,
                "query_instruction": os.environ.get("DENSE_QUERY_INSTRUCTION")})
    out = Path(a.out) if a.out else Path(a.triples).with_suffix(".eval.json")
    out.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
