"""Build persistent per-corpus indexes up front (separate from retrieval).

SWE-bench instances each retrieve over their repo @ base_commit, and (Verified)
those commits are nearly all distinct -> ~499 indexes. This is resumable
(skipping already-built indexes), so the retrieval phase (`run_eval`) just reuses
the persisted indexes. Shared document datasets have one corpus key, so dense/BM25
indexes are built once and reused across all queries.

Parallelism on the cluster is by SLURM array: `--shard i N` splits the work by
REPO (a repo's commits stay in one shard) so concurrent shards never check out the
same git clone at once. Run `run_eval` afterwards with the same --index-root.
"""
from __future__ import annotations

import argparse
import sys
from typing import Sequence

from agent_search.corpus.code_repo import RepoError, get_files

from .datasets import Instance, available_datasets, load_dataset_by_name
from .run_eval import _build_corpus, _build_document_corpus, _corpus_key, _progress


def _bm25_backend_is_pyserini() -> bool:
    """Resolve the BM25 backend the same way agent_search.retrievers.lexical.build_bm25_engine
    does (default 'local'), without importing pyserini eagerly. Step 0 has to answer 'does this
    run need a Lucene prebuild' from just the env var, cheaply."""
    import os
    return (os.environ.get("BM25_BACKEND") or "local").strip().lower() == "pyserini"


def _structured_backend_is_lucene() -> bool:
    """Resolve the structured backend the same way agent_search.retrievers.backend
    .structured_backend does (default 'python'), without importing the structured retrieval
    package eagerly. Same cheap-answer-from-env-var reasoning as `_bm25_backend_is_pyserini`
    above."""
    import os
    return (os.environ.get("STRUCTURED_BACKEND") or "python").strip().lower() == "lucene"


def prebuildable_for(retriever: str) -> list[str]:
    """Which persistent indexes a run of `retriever` should materialize in step 0.

    The "step 0" pre-build stage is the same for code and documents: only the corpora count
    differs. A floor maps to itself; an agent condition pre-builds `dense` if its
    toolset uses embedding search, and `search_bql` if its toolset uses BQL (its
    O(N) postings+BM25 index is otherwise built inside the first episode, on the clock).
    grep / bm25_local are in-memory, so they need no step 0 (return []). A bm25-family
    toolset (bm25_search/bm25q_search/bm25_search_snip) pre-builds `bm25_pyserini` too, but
    only when env `BM25_BACKEND=pyserini`: with the default 'local' backend the bm25 engine
    stays in-memory, needing no step 0.
    """
    if retriever in ("dense", "bm25_pyserini", "search_bql", "search_indri", "search_lucene"):
        return [retriever]
    if retriever == "bql":                       # the direct BQL floor loads the same artifact
        return ["search_lucene" if _structured_backend_is_lucene() else "search_bql"]
    if retriever == "agent" or retriever.startswith("agent_"):
        try:
            from agent_search.strategies.conditions import get_condition
            cond = get_condition(retriever)
            toolset = set(cond.tool_names)
            engines = set(cond.strategy.engines)
        except Exception:
            return []
        # the strategy's engine kinds say which artifacts a run will load: a condition using
        # dense retrieval (or BQL fused with the dense model) reads the persisted embedding
        # cache; BQL and Indri read their own artifact, or the shared Lucene index under
        # STRUCTURED_BACKEND=lucene; a condition using BM25 reads the Pyserini index only
        # under BM25_BACKEND=pyserini.
        kinds = []
        if engines & {"dense", "bql_fused", "bql_dense"}:
            kinds.append("dense")
        if engines & {"bql", "bql_fused", "bql_dense", "bql_plain"}:
            kinds.append("search_lucene" if _structured_backend_is_lucene() else "search_bql")
        if "indri" in engines:
            kinds.append("search_lucene" if _structured_backend_is_lucene() else "search_indri")
        if "bm25" in engines and _bm25_backend_is_pyserini():
            kinds.append("bm25_pyserini")
        del toolset
        return kinds
    return []


def unique_corpora(instances: Sequence[Instance]) -> list:
    """One representative Instance per distinct (repo, base_commit). Order-stable."""
    seen, out = set(), []
    for inst in instances:
        key = _corpus_key(inst)
        if key not in seen:
            seen.add(key)
            out.append((key, inst))
    return out


def shard_by_repo(corpora: list, shard: int, nshards: int) -> list:
    """Keep each repo's corpora together in one shard (avoids concurrent checkouts
    of the same git clone). Repos are round-robin assigned to shards."""
    if nshards <= 1:
        return corpora
    repos = sorted({inst.repo for _, inst in corpora})
    repo_shard = {r: i % nshards for i, r in enumerate(repos)}
    return [(k, inst) for (k, inst) in corpora if repo_shard[inst.repo] == shard]


def build(instances, index_root="indexes", rebuild=False, cache_dir="data/repos",
          shard=0, nshards=1, progress=True, retriever="bm25_pyserini",
          model: str | None = None) -> dict:
    if retriever == "bm25_pyserini":
        from agent_search.retrievers.lexical.pyserini import BM25Pyserini
        make_retriever = lambda: BM25Pyserini(index_root=index_root, rebuild=rebuild)
    elif retriever == "dense":
        from agent_search.retrievers.dense import DenseRetriever
        make_retriever = lambda: DenseRetriever(model or "nomic-ai/CodeRankEmbed",
                                                index_root=index_root, rebuild=rebuild)
    elif retriever == "search_bql":
        from agent_search.retrievers.bql.executor import BQLIndexBuilder
        make_retriever = lambda: BQLIndexBuilder(index_root=index_root, rebuild=rebuild)
    elif retriever == "search_indri":
        from agent_search.retrievers.indri.model import IndriIndexBuilder
        make_retriever = lambda: IndriIndexBuilder(index_root=index_root, rebuild=rebuild)
    elif retriever == "search_lucene":
        from agent_search.retrievers.lucene.index_builder import LuceneIndexBuilder
        make_retriever = lambda: LuceneIndexBuilder(index_root=index_root, rebuild=rebuild)
    else:
        raise ValueError(f"cannot pre-build indexes for retriever {retriever!r}")

    corpora = shard_by_repo(unique_corpora(instances), shard, nshards)
    retriever_obj = make_retriever()        # one instance, reused: the encoder loads once,
                                            # and is_cached() reads the same paths .index() writes
    built = skipped = failed = 0
    missing_repos: set = set()
    for key, inst in _progress(corpora, progress):
        try:
            # already persisted? skip the corpus's unit parse and the (re)build entirely.
            # This is what makes a re-run, and run.sh's step 0 prebuild after Phase 1, a
            # true per-corpus no-op instead of re-parsing every repo/doc to feed a cache hit.
            if not rebuild and retriever_obj.is_cached(key):
                skipped += 1
                continue
            if inst.docs is not None:
                units = _build_document_corpus(inst.docs)
            else:
                files = get_files(inst, cache_dir)
                units, _ = _build_corpus(files)
            if not units:
                continue
            retriever_obj.index(units, key=key)
            built += 1
        except RepoError:                       # repo not staged: aggregate by repo so the
            failed += 1                          # caller can report ~80 unique repos, not
            missing_repos.add(inst.repo)         # one identical line per commit (461 lines)
        except Exception as e:  # noqa: BLE001 - keep going; one bad repo isn't fatal
            failed += 1
            print(f"  [error] {key}: {type(e).__name__}: {e}")
    return {"corpora": len(corpora), "built": built, "skipped": skipped, "failed": failed,
            "missing_repos": sorted(missing_repos)}


def main() -> None:
    ap = argparse.ArgumentParser(description="Pre-build per-corpus persistent indexes")
    ap.add_argument("--dataset", default="swebench_verified",
                    choices=sorted(available_datasets()))
    ap.add_argument("--index-root", default="indexes")
    ap.add_argument("--repo-cache", default="data/repos",
                    help="pre-staged repo clones (default: data/repos)")
    ap.add_argument("--retriever", default="bm25_pyserini",
                    choices=["bm25_pyserini", "dense", "search_bql", "search_indri",
                            "search_lucene"],
                    help="which persistent index to pre-build")
    ap.add_argument("--model", default=None,
                    help="dense model id (only for --retriever dense)")
    ap.add_argument("--limit", type=int, default=None, help="cap #instances")
    ap.add_argument("--corpus-limit", type=int, default=None,
                    help="cap fixed-corpus documents for shared document datasets; "
                         "ignored by code datasets")
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--shard", type=int, default=0, help="this shard id (0-based)")
    ap.add_argument("--nshards", type=int, default=1, help="total shards (SLURM array)")
    args = ap.parse_args()

    instances = load_dataset_by_name(args.dataset, limit=args.limit,
                                     corpus_limit=args.corpus_limit)
    # The dense embedder must match what run_eval will later load (same model -> same
    # cache key), so default it from the dataset's domain exactly as the eval does.
    from agent_search.evaluation.datasets import dataset_domain, default_dense_model
    model = args.model or default_dense_model(dataset_domain(args.dataset))
    n_unique = len(unique_corpora(instances))
    print(f"{args.dataset}: {len(instances)} instances -> {n_unique} unique corpora "
          f"(shard {args.shard}/{args.nshards}); dense model: {model}")
    res = build(instances, index_root=args.index_root, rebuild=args.rebuild,
                cache_dir=args.repo_cache, shard=args.shard, nshards=args.nshards,
                retriever=args.retriever, model=model)
    print(f"done: {res['built']} built, {res.get('skipped', 0)} already-cached, "
          f"{res['failed']} failed, of {res['corpora']} in shard")
    miss = res.get("missing_repos") or []
    if miss:
        # the common cause for a code dataset: its repos were never prefetched. Report the
        # unique repos plus the exact stage command once, instead of one error line per commit.
        print(f"\n{len(miss)} repo(s) for '{args.dataset}' are NOT staged under "
              f"{args.repo_cache} — stage them on a node WITH internet, then re-run:")
        print(f"  git clone https://github.com/<owner>/<repo> {args.repo_cache}/<owner>__<repo>   "
              f"# one per missing repo (no prefetch script ships with this release)")
        print("  missing: " + ", ".join(miss[:40]) + (" ..." if len(miss) > 40 else ""))
    # fail loudly: a shard that built nothing (all corpora errored) must not look like
    # success, so callers (scripts/run.sh Phase 0) abort instead of serving with no index.
    # `skipped` (already-cached) counts as success: built==0 is fine when everything was
    # already on disk, only an empty-and-nothing-cached shard is the error.
    nothing_done = res["built"] == 0 and res.get("skipped", 0) == 0
    if res["failed"] > 0 or (res["corpora"] > 0 and nothing_done):
        sys.exit(1)


if __name__ == "__main__":
    main()
