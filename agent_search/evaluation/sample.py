"""`skimsearchagent-sample-dataset`: a small dataset cut from a big one, in the topics layout.

    skimsearchagent-sample-dataset --dataset browsecomp_plus_chunks --out browsecomp_plus_chunks_sample \\
        --n-topics 20 --n-docs 20000 --seed 0
    skimsearchagent-sample-dataset --dataset infoseek_eval --out infoseek_eval_sample --n-topics 20 \\
        --n-docs 20000 --pool-bm25-index indexes/external/wiki25_512_lucene --pool-k 100

The sample keeps `n_topics` questions (their answers and qrels included), every gold document of
those questions, an optional retrieval pool per question from a prebuilt Lucene index (so an
answer-only set still has the documents an agent could find), and `n_docs` documents drawn at
random from the rest of the corpus. It is written to `data/<out>/` as `topics.tsv`, `qrels.txt`
(when the source has them) and `corpus.jsonl`, the same layout the loader reads, so the sample is
a dataset like any other: index it (`skimsearchagent-build-indexes --dataset <out> ...`) and run
it. Use it to try a retriever or a backbone on a paper setting in minutes instead of hours.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from typing import Optional


def _topic_rows(inst_list):
    for inst in inst_list:
        qid = inst.instance_id.split("__", 1)[-1]
        yield qid, inst.problem_statement, inst.answer, inst.gold_doc_ids


def sample_dataset(dataset: str, out_dir: str, n_topics: int, n_docs: int, seed: int = 0,
                   pool_search=None, pool_k: int = 100) -> dict:
    from agent_search.evaluation.datasets import load_dataset_by_name
    rng = random.Random(seed)
    instances = load_dataset_by_name(dataset)
    if not instances:
        raise SystemExit(f"dataset {dataset!r} is empty")
    first = instances[0]
    picked = rng.sample(instances, min(n_topics, len(instances)))
    keep: dict[str, None] = {}
    for _, question, _, gold in _topic_rows(picked):
        for d in sorted(gold or []):
            keep.setdefault(str(d))
        if pool_search is not None:
            for d in pool_search(question, pool_k):
                keep.setdefault(str(d))
    # random documents from the rest of the corpus
    if first.docstore is not None:
        store = first.docstore
        n_total = len(store)
        positions = rng.sample(range(n_total), min(n_docs, n_total))
        random_ids = [store.id_at(p) for p in positions]
        get = lambda d: store.get(d)  # noqa: E731
    else:
        docs = first.docs or []
        by_id = {str(d.get("_id") or d.get("doc_id") or d.get("id")): d for d in docs}
        random_ids = rng.sample(list(by_id), min(n_docs, len(by_id)))
        get = lambda d: (lambda x: {"doc_id": d, "title": x.get("title", ""), "text": x.get("text", "")} if x else None)(by_id.get(d))  # noqa: E731
    for d in random_ids:
        keep.setdefault(str(d))
    os.makedirs(out_dir, exist_ok=True)
    n_written = 0
    with open(os.path.join(out_dir, "corpus.jsonl"), "w", encoding="utf-8") as fh:
        for d in keep:
            doc = get(d)
            if doc is None:
                continue
            fh.write(json.dumps({"docid": doc["doc_id"], "title": doc.get("title", ""), "text": doc["text"]},
                                ensure_ascii=False) + "\n")
            n_written += 1
    has_qrels = any(gold for _, _, _, gold in _topic_rows(picked))
    with open(os.path.join(out_dir, "topics.tsv"), "w", encoding="utf-8") as ft, \
            open(os.path.join(out_dir, "qrels.txt"), "w", encoding="utf-8") as fq:
        for qid, question, answer, gold in _topic_rows(picked):
            ft.write(f"{qid}\t{question}\t{answer or ''}\n")
            for d in sorted(gold or []):
                fq.write(f"{qid} Q0 {d} 1\n")
    if not has_qrels:
        os.remove(os.path.join(out_dir, "qrels.txt"))
    meta = {"source": dataset, "n_topics": len(picked), "n_docs": n_written, "seed": seed,
            "pool_k": pool_k if pool_search is not None else 0, "qrels": has_qrels}
    with open(os.path.join(out_dir, "sample.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    return meta


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="skimsearchagent-sample-dataset", description=__doc__.split("\n\n")[0])
    ap.add_argument("--dataset", required=True, help="source dataset (registered name)")
    ap.add_argument("--out", required=True, help="sample name; written to data/<out>/ (or a path)")
    ap.add_argument("--n-topics", type=int, default=20)
    ap.add_argument("--n-docs", type=int, default=20000, help="random documents added to the gold ones and the pool")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pool-bm25-index", default=None, help="a Lucene index over the source corpus: adds the top --pool-k BM25 hits per question")
    ap.add_argument("--pool-k", type=int, default=100)
    a = ap.parse_args(argv)
    from agent_search.evaluation.datasets import DATA_DIR
    out_dir = a.out if os.sep in a.out else os.path.join(DATA_DIR, a.out)
    pool_search = None
    if a.pool_bm25_index:
        os.environ["BM25_INDEX_PATH"] = a.pool_bm25_index
        from agent_search.retrievers.lexical.pyserini import BM25Pyserini
        engine = BM25Pyserini().index([], key="pool")
        pool_search = lambda q, k: engine.search(q, k=k)  # noqa: E731
    meta = sample_dataset(a.dataset, out_dir, a.n_topics, a.n_docs, a.seed, pool_search, a.pool_k)
    meta["out"] = out_dir
    print(json.dumps(meta, indent=2))
    print(f"register it, or rely on discovery (data/<name>/topics.tsv), then:\n"
          f"  skimsearchagent-build-indexes --dataset {os.path.basename(out_dir)} --retriever dense --model <model>",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
