"""ITER / DIVER topics+qrels layout: `topics.tsv` (+ optional TREC qrels, optional
answers.tsv) over a `corpus.jsonl` that may be served from disk instead of loaded into
memory. Covers InfoSeek and the browsecomp_plus_chunks sets, plus any `data/<name>/
topics.tsv` folder discovered at runtime (see `base.discover_topics_datasets`).
"""
from __future__ import annotations

import os

from agent_search.evaluation.datasets.base import (
    Instance,
    _find_existing,
    _jsonl,
    data_dir,
    register_dataset,
)

DOCSTORE_MIN_BYTES = int(os.environ.get("AGENT_SEARCH_DOCSTORE_MIN_BYTES", str(1 << 30)))


def _load_topics_qrels(root: str, name: str, limit: int | None = None,
                       corpus_limit: int | None = None, corpus_dir: str | None = None,
                       corpus_id: str | None = None) -> list[Instance]:
    """ITER / DIVER layout: `corpus.jsonl` ({"docid", "text"}; a `title` field, a leading
    `---\ntitle: ...\n---` front matter, or a short first line carries the title),
    `topics.tsv` (id<TAB>question[<TAB>answer]), optional TREC `qrels.txt` (qid Q0 docid rel)
    and optional `answers.tsv` (id<TAB>answer). A set with answers but no qrels is answer-only:
    every topic is kept and scored on its answer. A corpus above `DOCSTORE_MIN_BYTES`
    (default 1 GiB; `AGENT_SEARCH_DOCSTORE=1` forces it) is served from disk through
    `agent_search.corpus.docstore` instead of being loaded into memory."""
    cdir = corpus_dir or root
    corpus_path = _find_existing(os.path.join(cdir, "corpus.jsonl"), os.path.join(root, "corpus.jsonl"))
    topics_path = _find_existing(os.path.join(root, "topics.tsv"), os.path.join(root, f"{name}.tsv"))
    qrels_path = _find_existing(os.path.join(root, "qrels.txt"), os.path.join(root, "qrels.tsv"),
                                os.path.join(root, "qrel_evidence.txt"))
    ans_path = _find_existing(os.path.join(root, "answers.tsv"))
    if not (corpus_path and topics_path):
        raise FileNotFoundError(
            f"{name} expects topics.tsv under {root} and corpus.jsonl under {cdir} "
            f"(see corpus_build/README.md, 'ITER layout').")
    docs: list[dict] | None = None
    store = None
    big = os.path.getsize(corpus_path) >= DOCSTORE_MIN_BYTES or \
        os.environ.get("AGENT_SEARCH_DOCSTORE", "") in ("1", "true", "yes")
    if big and corpus_limit is None:
        from agent_search.corpus.docstore import JsonlDocStore
        store = JsonlDocStore(corpus_path)
        has_doc = store.__contains__
    else:
        from agent_search.corpus.docstore import normalise_document
        docs = []
        for d in _jsonl(corpus_path, limit=corpus_limit):
            nd = normalise_document(d)
            docs.append({"_id": nd["doc_id"], "title": nd["title"], "text": nd["text"]})
        doc_ids = {d["_id"] for d in docs}
        has_doc = doc_ids.__contains__
    answers: dict[str, str] = {}
    if ans_path:
        with open(ans_path, encoding="utf-8") as fh:
            for line in fh:
                parts = line.rstrip("\n").split("\t")
                if len(parts) >= 2:
                    answers[parts[0]] = parts[-1]
    qrels: dict[str, set] | None = None
    if qrels_path:
        qrels = {}
        with open(qrels_path, encoding="utf-8") as fh:
            for line in fh:
                parts = line.split()
                if parts and parts[0].lower() in ("query-id", "qid", "query_id", "id"):
                    continue                                   # a header row
                if len(parts) >= 4 and parts[3] != "0":
                    qrels.setdefault(parts[0], set()).add(parts[2])
                elif len(parts) == 3:
                    qrels.setdefault(parts[0], set()).add(parts[1])
    instances: list[Instance] = []
    n_dropped = 0
    with open(topics_path, encoding="utf-8") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2 or parts[0].lower() in ("id", "qid", "query-id"):
                continue
            qid, question = parts[0], parts[1]
            ans = parts[2] if len(parts) > 2 and parts[2] else answers.get(qid)
            gold: set | None = None
            if qrels is not None:
                gold = {d for d in qrels.get(qid, set()) if has_doc(d)}
                if not gold and ans is None:
                    n_dropped += 1
                    continue
                if not gold:
                    gold = None            # answer-only topic inside a qrels set
            elif ans is None:
                n_dropped += 1
                continue
            instances.append(Instance(
                instance_id=f"{name}__{qid}", repo=f"local/{name}", base_commit="0" * 40,
                problem_statement=question, patch="", docs=docs, docstore=store,
                gold_doc_ids=gold, answer=ans, corpus_id=corpus_id or name))
            if limit is not None and len(instances) >= limit:
                break
    if n_dropped:
        import sys
        print(f"  [{name}] WARNING: {n_dropped} topics dropped (neither a gold document in the corpus "
              f"nor an answer).", file=sys.stderr, flush=True)
    return instances


def _topics_qrels_loader(name: str, root: str | None = None, corpus: str | None = None):
    """A loader for `register_dataset`: topics under data/<name>/, the corpus under
    data/corpora/<corpus>/ when `corpus` is given (so several topic sets share one corpus and
    its indexes), else next to the topics."""
    def load(limit=None, corpus_limit=None):
        cdir = os.path.join(data_dir(), "corpora", corpus) if corpus else None
        return _load_topics_qrels(root or os.path.join(data_dir(), name), name, limit=limit,
                                  corpus_limit=corpus_limit, corpus_dir=cdir, corpus_id=corpus)
    return load


# ITER's sets: topics (+ answers, + TREC qrels when they exist) over a chunked corpus. Stage
# them as data/<name>/topics.tsv [qrels.txt] and data/corpora/<corpus>/corpus.jsonl — see
# corpus_build/README.md. InfoSeek has answers but no document labels (answer-only).
register_dataset("infoseek_eval", domain="general")(_topics_qrels_loader("infoseek_eval", corpus="wiki25_512"))
register_dataset("infoseek_train", domain="general")(_topics_qrels_loader("infoseek_train", corpus="wiki25_512"))
register_dataset("browsecomp_plus_chunks", domain="general")(_topics_qrels_loader("browsecomp_plus_chunks"))
# samples cut by `skimsearchagent-sample-dataset` (a few topics, their gold documents, a
# retrieval pool, random chunks): the paper settings in minutes
register_dataset("browsecomp_plus_chunks_sample", domain="general")(_topics_qrels_loader("browsecomp_plus_chunks_sample"))
register_dataset("infoseek_eval_sample", domain="general")(_topics_qrels_loader("infoseek_eval_sample"))
