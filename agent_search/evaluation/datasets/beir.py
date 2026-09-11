"""Shared-corpus deep-research benchmarks in the BEIR layout (corpus.jsonl + queries.jsonl +
qrels): BrowseComp-Plus and the classic multi-hop QA sets (HotpotQA / 2WikiMultiHopQA /
MuSiQue), plus their built corpus_build/ pairs (flat vs structured, and the browsecomp full
variants). One fixed document collection per dataset; every query searches it.
"""
from __future__ import annotations

import os

from agent_search.evaluation.datasets.base import (
    Instance,
    _find_existing,
    _jsonl,
    _qrels_from_jsonl,
    _qrels_from_tsv,
    _sidecar_answers,
    data_dir,
    register_dataset,
)


def _load_beir_style(root: str, name: str, limit: int | None = None,
                     corpus_limit: int | None = None) -> list[Instance]:
    """corpus.jsonl + queries.jsonl + qrels/*.tsv|qrels.jsonl -> shared-corpus Instances."""
    corpus_path = _find_existing(os.path.join(root, "corpus.jsonl"),
                                 os.path.join(root, "corpus", "corpus.jsonl"))
    queries_path = _find_existing(os.path.join(root, "queries.jsonl"),
                                  os.path.join(root, "queries", "queries.jsonl"))
    qrels_path = _find_existing(os.path.join(root, "qrels", "test.tsv"),
                                os.path.join(root, "qrels", "dev.tsv"),
                                os.path.join(root, "qrels.tsv"),
                                os.path.join(root, "qrels.jsonl"))
    if not (corpus_path and queries_path and qrels_path):
        raise FileNotFoundError(
            f"{name} expects BEIR-style files under {root}: corpus.jsonl, queries.jsonl, "
            f"and qrels/test.tsv (or qrels.jsonl). Place those files under data/{name}/ "
            f"before running offline eval.")
    docs = _jsonl(corpus_path, limit=corpus_limit)
    from agent_search.corpus.units import units_from_documents
    doc_ids = {u.doc_id for u in units_from_documents(docs)}   # reachable after chunking
    queries = _jsonl(queries_path)
    ans_map = _sidecar_answers(root, name)            # gold answers shipped in a separate file
    qrels = (_qrels_from_jsonl(qrels_path) if qrels_path.endswith(".jsonl")
             else _qrels_from_tsv(qrels_path))
    instances: list[Instance] = []
    n_dropped = 0
    for q in queries:
        qid = str(q.get("_id") or q.get("id") or q.get("query_id"))
        gold = qrels.get(qid, set()) & doc_ids        # keep only reachable gold
        if not gold:
            n_dropped += 1
            continue
        ans = q.get("answer") or q.get("gold_answer") or ans_map.get(qid)
        instances.append(Instance(
            instance_id=f"{name}__{qid}", repo=f"local/{name}", base_commit="0" * 40,
            problem_statement=str(q.get("text") or q.get("question") or q.get("query") or ""),
            patch="", docs=docs, gold_doc_ids=gold,
            answer=str(ans) if ans is not None else None,
            # corpus_id = the dataset name (unique per shared corpus) so the persisted index
            # dir is clean (indexes/bql/2wiki_flat-v1.pkl), not the abspath-mangled
            # `2wiki_flat:__scratch3__...__data__2wiki_flat`. Data lives at a fixed path here.
            corpus_id=name))
        if limit is not None and len(instances) >= limit:
            break
    if n_dropped:
        import sys
        print(f"  [{name}] WARNING: {n_dropped} of {len(queries)} queries dropped because none "
              f"of their gold documents are in the loaded corpus"
              + (f" (corpus_limit={corpus_limit} — the evaluated subset is NOT a random sample)"
                 if corpus_limit is not None else "") + ".", file=sys.stderr, flush=True)
    return instances


def load_browsecomp_plus(root: str | None = None, limit: int | None = None,
                         corpus_limit: int | None = None) -> list[Instance]:
    """BrowseComp-Plus: a reproducible deep-research benchmark, 830 human-authored
    multi-step questions over a fixed ~100k-document corpus (chen et al.). Stage the
    BEIR-style files under data/browsecomp_plus/ (corpus.jsonl/queries.jsonl/qrels)."""
    return _load_beir_style(root or os.path.join(data_dir(), "browsecomp_plus"),
                            "browsecomp_plus", limit=limit, corpus_limit=corpus_limit)


def _multihop_loader(name: str):
    """The classic multi-hop QA benchmarks (HotpotQA / 2WikiMultiHopQA / MuSiQue) as a
    shared paragraph corpus with supporting-paragraph qrels: the field-standard deep-
    research retrieval sets (the FrugalRAG combo, alongside browsecomp_plus). Stage them
    with scripts/stage_multihop.py (raw release -> BEIR layout under data/<name>/)."""
    def load(root: str | None = None, limit: int | None = None,
             corpus_limit: int | None = None) -> list[Instance]:
        return _load_beir_style(root or os.path.join(data_dir(), name), name,
                                limit=limit, corpus_limit=corpus_limit)
    return load


load_hotpotqa = _multihop_loader("hotpotqa")
load_2wiki = _multihop_loader("2wiki")
load_musique = _multihop_loader("musique")


def _built_corpus_loader(name: str, sub_dir: str):
    """Loader for a corpus_build/ output dir under data/<sub_dir>/ (BEIR layout). The wikipedia
    builder emits a pair over the same docs/queries: `<base>_structured/` (title/section/infobox/
    body fields) and `<base>_flat/` (the same content aggregated into title/body only). The pair
    is the paired experiment: BQL's lift should grow as the corpus gains scopeable structure."""
    def load(root: str | None = None, limit: int | None = None,
             corpus_limit: int | None = None) -> list[Instance]:
        rt = root or os.path.join(data_dir(), sub_dir)
        return _load_beir_style(rt, name, limit=limit, corpus_limit=corpus_limit)
    return load


# kept name for the browsecomp_plus structured loader below
def _structured_loader(name: str, src_root: str):
    return _built_corpus_loader(name, f"{src_root}_structured")


register_dataset("browsecomp_plus", domain="general")(load_browsecomp_plus)
# classic multi-hop QA as shared-corpus retrieval (stage via scripts/stage_multihop.py)
register_dataset("hotpotqa", domain="general")(load_hotpotqa)
register_dataset("2wiki", domain="general")(load_2wiki)
register_dataset("musique", domain="general")(load_musique)
# Wikipedia pairs (corpus_build/wikipedia emits <base>_flat + <base>_structured over the same
# docs/queries; only the section/infobox fields differ). The structured arm's field_profile picks
# the matching BQL manual (wiki = title/section/infobox/body); the flat arm has title/body only
# (default general manual). Build them before running these. The pair is the headline experiment.
for _base in ("hotpotqa", "2wiki", "musique"):
    register_dataset(f"{_base}_structured", domain="general", field_profile="wiki")(
        _built_corpus_loader(f"{_base}_structured", f"{_base}_structured"))
    register_dataset(f"{_base}_flat", domain="general")(
        _built_corpus_loader(f"{_base}_flat", f"{_base}_flat"))
# browsecomp pair (corpus_build/browsecomp_plus emits both from the hub, same docs/queries,
# only the author/date fields differ). flat = title/body; structured = +author/date (browsecomp manual).
register_dataset("browsecomp_plus_structured", domain="general", field_profile="browsecomp")(
    _built_corpus_loader("browsecomp_plus_structured", "browsecomp_plus_structured"))
register_dataset("browsecomp_plus_flat", domain="general")(
    _built_corpus_loader("browsecomp_plus_flat", "browsecomp_plus_flat"))
# Full-corpus browsecomp variants: same 830 queries/qrels as the pooled pair above, but over
# the complete BrowseComp-Plus collection (100,195 docs; the pooled 67,707-doc corpora are
# row-subsets of this). Same BEIR layout under data/<name>_full/; answers ship inline in
# queries.jsonl. Each variant gets its own dataset name so it has its own index/cache
# namespace (corpus_id = dataset name), keeping the pooled artifact separate.
register_dataset("browsecomp_plus_structured_full", domain="general", field_profile="browsecomp")(
    _built_corpus_loader("browsecomp_plus_structured_full", "browsecomp_plus_structured_full"))
register_dataset("browsecomp_plus_flat_full", domain="general")(
    _built_corpus_loader("browsecomp_plus_flat_full", "browsecomp_plus_flat_full"))
