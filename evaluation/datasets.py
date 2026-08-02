"""Dataset loading: code-localization and shared document-corpus instances.

An Instance carries the issue text and the gold patch. `files` is populated only
for the built-in fixture (inline sources, so the pipeline runs with no git/deps);
for real SWE-bench it is None and resolved by `repo.get_files` at the base commit.
Document-domain instances carry `docs`, `gold_doc_ids`, and an optional answer.
"""
from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from typing import Optional

# Datasets are pre-downloaded here (on a node with internet) so the GPU node — which
# usually has NO internet — can load them offline. See scripts/download_data.sh.
DATA_DIR = os.environ.get("AGENT_SEARCH_DATA", "data")


@dataclass
class Instance:
    instance_id: str
    repo: str
    base_commit: str
    problem_statement: str
    patch: str                                   # gold diff -> localization labels
    files: Optional[dict] = None                 # {path: source}; set for fixtures only
    docs: Optional[list[dict]] = None            # fixed-corpus retrieval docs
    gold_doc_ids: Optional[set[str]] = None      # generic retrieval labels
    answer: Optional[str] = None                 # optional QA answer for later eval
    corpus_id: Optional[str] = None              # stable cache key for shared corpora


def _local_dir(hf_name: str) -> str:
    """Where a pre-downloaded split lives, e.g. data/SWE-bench_Verified."""
    return os.path.join(DATA_DIR, hf_name.split("/")[-1])


def load_swebench(name: str = "princeton-nlp/SWE-bench_Verified",
                  split: str = "test") -> list[Instance]:
    """Load a SWE-bench split. Reads the pre-downloaded copy from `data/` if present
    (offline GPU node); otherwise downloads via HF (needs internet)."""
    local = _local_dir(name)
    if os.path.isdir(local) and os.listdir(local):
        from datasets import load_from_disk
        ds = load_from_disk(local)            # offline, from data/ — no network, no HF cache
    else:
        import sys
        print(f"[agent_search] WARNING: {local} is NOT staged — falling back to HuggingFace "
              f"({name}) via network / HF cache. This is the 'caching' you may see. "
              f"Run `bash scripts/download_data.sh` to stage it for offline use.",
              file=sys.stderr)
        from datasets import load_dataset  # lazy import: heavy dep
        ds = load_dataset(name, split=split)
    return [
        Instance(
            instance_id=row["instance_id"],
            repo=row["repo"],
            base_commit=row["base_commit"],
            problem_statement=row["problem_statement"],
            patch=row["patch"],
            files=None,
        )
        for row in ds
    ]


def fixture_instances() -> list[Instance]:
    """One tiny self-contained instance, so the eval runs anywhere. The gold patch
    edits `create_session_token`; the issue text mentions session-token expiry, so
    a lexical retriever should localize it."""
    session_py = (
        "import time\n"
        "\n"
        "\n"
        "def create_session_token(user):\n"
        "    token = make_token(user)\n"
        "    return token\n"
        "\n"
        "\n"
        "def make_token(user):\n"
        "    return str(user) + \"-static\"\n"
    )
    html_py = (
        "def render_page(title, body):\n"
        "    return \"<html>\" + title + body + \"</html>\"\n"
    )
    patch = (
        "diff --git a/auth/session.py b/auth/session.py\n"
        "--- a/auth/session.py\n"
        "+++ b/auth/session.py\n"
        "@@ -4,3 +4,4 @@ def create_session_token(user):\n"
        " def create_session_token(user):\n"
        "     token = make_token(user)\n"
        "-    return token\n"
        "+    return token  # TODO: attach expiry\n"
        "+    # expiry handling\n"
    )
    return [
        Instance(
            instance_id="fixture__session-expiry-1",
            repo="fixture/app",
            base_commit="0" * 40,
            problem_statement=(
                "Session tokens never expire. create_session_token should attach an "
                "expiry so user sessions time out."
            ),
            patch=patch,
            files={"auth/session.py": session_py, "render/html.py": html_py},
        )
    ]


# --- shared-corpus deep-research benchmarks (BrowseComp-Plus, multi-hop QA) ------
# These are the "shared corpus" regime: ONE fixed document collection, every query
# searches it. Each query -> an Instance carrying the whole corpus in `docs` (built
# ONCE and reused across queries via `corpus_id`), evidence qrels in `gold_doc_ids`,
# and an optional `answer` for EM/F1. The BEIR layout (corpus.jsonl + queries.jsonl
# + qrels) is the ecosystem convention these benchmarks ship in.

def _jsonl(path: str, limit: int | None = None) -> list[dict]:
    rows: list[dict] = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
                if limit is not None and len(rows) >= limit:
                    break
    return rows


def _find_existing(*paths: str) -> str | None:
    return next((p for p in paths if os.path.exists(p)), None)


def _qrels_from_tsv(path: str) -> dict[str, set[str]]:
    """BEIR (query-id<TAB>corpus-id<TAB>score, with header) or TREC (qid 0 docid rel)."""
    out: dict[str, set[str]] = {}
    with open(path, newline="") as fh:
        sample = fh.readline()
        fh.seek(0)
        delim = "\t" if "\t" in sample else ","
        reader = csv.DictReader(fh, delimiter=delim)
        if reader.fieldnames and {"query-id", "corpus-id"} <= set(reader.fieldnames):
            for row in reader:
                if float(row.get("score") or 1) > 0:
                    out.setdefault(row["query-id"], set()).add(row["corpus-id"])
            return out
        fh.seek(0)
        for raw in csv.reader(fh, delimiter=delim):
            row = raw[0].split() if len(raw) < 2 else raw   # tolerate space-delimited
            if not row or row[0].lower() in {"query-id", "query_id", "qid"}:
                continue
            if len(row) >= 4 and row[1] in {"0", "Q0"}:          # TREC: qid 0 docid rel
                qid, docid, score = row[0], row[2], float(row[3] or 1)
            elif len(row) >= 2:
                qid, docid, score = row[0], row[1], float(row[2] if len(row) > 2 else 1)
            else:
                continue                                          # malformed line: skip
            if score > 0:
                out.setdefault(qid, set()).add(docid)
    return out


def _qrels_from_jsonl(path: str) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for row in _jsonl(path):
        qid = str(row.get("query_id") or row.get("qid") or row.get("_id") or row.get("id"))
        raw = (row.get("gold_doc_ids") or row.get("positive_doc_ids")
               or row.get("doc_ids") or [row.get("doc_id") or row.get("corpus_id")])
        docs = {str(x) for x in raw if x}
        if docs:
            out.setdefault(qid, set()).update(docs)
    return out


def _sidecar_answers(root: str, name: str) -> dict:
    """Some BEIR-style sets (browsecomp_plus) ship gold ANSWERS in a separate file, not in
    queries.jsonl. Look for answers.jsonl in `root`, or a `<base>_decrypted.jsonl` under the base
    dataset dir (the _structured/_flat variants share the base's answers). Returns {query_id: answer}."""
    base = name
    for suf in ("_structured", "_flat"):
        if base.endswith(suf):
            base = base[: -len(suf)]
    f = _find_existing(os.path.join(root, "answers.jsonl"),
                       os.path.join(root, f"{name}_decrypted.jsonl"),
                       os.path.join(os.path.dirname(root), base, f"{base}_decrypted.jsonl"))
    if not f:
        return {}
    out: dict = {}
    for row in _jsonl(f):
        qid = str(row.get("query_id") or row.get("_id") or row.get("qid") or row.get("id") or "")
        ans = row.get("answer") or row.get("gold_answer")
        if qid and ans is not None:
            out[qid] = str(ans)
    return out


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
    for q in queries:
        qid = str(q.get("_id") or q.get("id") or q.get("query_id"))
        gold = qrels.get(qid, set()) & doc_ids        # keep only reachable gold
        if not gold:
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
    return instances


def load_browsecomp_plus(root: str | None = None, limit: int | None = None,
                         corpus_limit: int | None = None) -> list[Instance]:
    """BrowseComp-Plus: a reproducible deep-research benchmark — 830 human-authored
    multi-step questions over a fixed ~100k-document corpus (chen et al.). Stage the
    BEIR-style files under data/browsecomp_plus/ (corpus.jsonl/queries.jsonl/qrels)."""
    return _load_beir_style(root or os.path.join(DATA_DIR, "browsecomp_plus"),
                            "browsecomp_plus", limit=limit, corpus_limit=corpus_limit)


def _multihop_loader(name: str):
    """The classic multi-hop QA benchmarks (HotpotQA / 2WikiMultiHopQA / MuSiQue) as a
    shared paragraph corpus with supporting-paragraph qrels — the field-standard deep-
    research retrieval sets (the FrugalRAG combo, alongside browsecomp_plus). Stage them
    with scripts/stage_multihop.py (raw release -> BEIR layout under data/<name>/)."""
    def load(root: str | None = None, limit: int | None = None,
             corpus_limit: int | None = None) -> list[Instance]:
        return _load_beir_style(root or os.path.join(DATA_DIR, name), name,
                                limit=limit, corpus_limit=corpus_limit)
    return load


load_hotpotqa = _multihop_loader("hotpotqa")
load_2wiki = _multihop_loader("2wiki")
load_musique = _multihop_loader("musique")


def _built_corpus_loader(name: str, sub_dir: str):
    """Loader for a corpus_build/ output dir under data/<sub_dir>/ (BEIR layout). The wikipedia
    builder emits a PAIR over the SAME docs/queries: `<base>_structured/` (title/section/infobox/
    body fields) and `<base>_flat/` (the same content aggregated into title/body only). The pair
    is the paired experiment — BQL's lift should grow as the corpus gains scopeable structure."""
    def load(root: str | None = None, limit: int | None = None,
             corpus_limit: int | None = None) -> list[Instance]:
        rt = root or os.path.join(DATA_DIR, sub_dir)
        return _load_beir_style(rt, name, limit=limit, corpus_limit=corpus_limit)
    return load


# kept name for the browsecomp_plus structured loader below
def _structured_loader(name: str, src_root: str):
    return _built_corpus_loader(name, f"{src_root}_structured")


def _doc_corpus_fixture(name: str) -> list[Instance]:
    """Tiny inline shared-corpus instance (no staging) so the doc-retrieval path is
    testable anywhere — mirrors the BrowseComp/multi-hop shape: question + evidence + answer."""
    docs = [
        {"_id": "d_guadalupe", "title": "Treaty of Guadalupe Hidalgo",
         "text": "The Treaty of Guadalupe Hidalgo ended the Mexican-American War in 1848."},
        {"_id": "d_paris", "title": "Treaty of Paris (1898)",
         "text": "The 1898 Treaty of Paris ended the Spanish-American War."},
        {"_id": "d_adams", "title": "Adams-Onis Treaty",
         "text": "The Adams-Onis Treaty of 1819 concerned Florida."},
    ]
    return [Instance(
        instance_id=f"{name}__mexican_war",
        repo=f"fixture/{name}", base_commit="0" * 40,
        problem_statement="Which treaty ended the Mexican-American War, and in what year?",
        patch="", docs=docs, gold_doc_ids={"d_guadalupe"},
        answer="Treaty of Guadalupe Hidalgo, 1848", corpus_id=f"{name}_fixture")]


# --- dataset registry (the single extension point) --------------------------
# Adding a dataset = register a loader here (or anywhere that imports this module),
# nothing else: `available_datasets()` is the one source of truth the CLIs use for
# their --dataset choices, so run_eval / build_indexes never change.

_DATASETS: dict[str, "object"] = {}
_DATASET_DOMAIN: dict[str, str] = {}
_DATASET_PROFILE: dict[str, str] = {}


def register_dataset(name: str, domain: str = "code", field_profile: str | None = None):
    """Decorator: bind a name to a loader ``(limit, corpus_limit) -> [Instance]``.

    `domain` is declared BY the dataset (``code`` = per-query repo localization,
    ``general`` = shared document corpus / deep research). It drives the dense
    embedder, the prompt domain, and whether the corpus is pre-embedded — so adding
    a new dataset is a single registration, with no central list to edit.

    `field_profile` selects which BQL manual variant the agent sees — the manual must
    advertise EXACTLY the corpus's fields. A STRUCTURED corpus declares its profile
    (``wiki`` = title/section/infobox/body, ``browsecomp`` = title/author/date/section/
    body); a flat corpus omits it and inherits the domain (general = title/body)."""
    def deco(fn):
        _DATASETS[name] = fn
        _DATASET_DOMAIN[name] = domain
        if field_profile is not None:
            _DATASET_PROFILE[name] = field_profile
        return fn
    return deco


def available_datasets() -> set[str]:
    return set(_DATASETS)


def dataset_domain(name: str, override: Optional[str] = None) -> str:
    """The domain a dataset declared at registration (overridable)."""
    return override or _DATASET_DOMAIN.get(name, "code")


def dataset_field_profile(name: str, override: Optional[str] = None) -> Optional[str]:
    """The per-dataset BQL field profile, or None to inherit the domain's manual."""
    return override or _DATASET_PROFILE.get(name)


def default_dense_model(domain: str) -> str:
    """The default dense embedder for `domain`. Env `DENSE_MODEL` overrides the GENERAL-domain
    default only (e.g. `DENSE_MODEL=Qwen/Qwen3-Embedding-0.6B` to swap in Qwen3-Embedding for
    browsecomp_plus_structured/hotpotqa_structured/etc.) — the CODE-domain default
    (nomic-ai/CodeRankEmbed) is a code-trained embedder with no general-text equivalent in this
    knob's scope, so it stays fixed regardless of DENSE_MODEL (an accidental `export
    DENSE_MODEL=...` in a shell must never silently swap the SWE-bench code embedder).
    Unset -> unchanged defaults (BAAI/bge-base-en-v1.5 for general), so every existing
    run/cache is untouched by this knob's addition. The chosen model's own cache namespace
    (`indexes/dense/<model>-sl<len>/<corpus_key>/`, from DenseRetriever._cache_dir) keeps it
    from ever mixing with another model's embeddings."""
    import os
    if domain == "general":
        return os.environ.get("DENSE_MODEL") or "BAAI/bge-base-en-v1.5"
    return "nomic-ai/CodeRankEmbed"


def load_dataset_by_name(name: str, limit: int | None = None,
                         corpus_limit: int | None = None) -> list[Instance]:
    try:
        fn = _DATASETS[name]
    except KeyError:
        raise ValueError(
            f"unknown dataset {name!r}; choose from {sorted(_DATASETS)}") from None
    return fn(limit=limit, corpus_limit=corpus_limit)


def _swebench(hf_name: str):
    def load(limit=None, corpus_limit=None):
        rows = load_swebench(hf_name)
        return rows[:limit] if limit is not None else rows
    return load


# built-in datasets (loaders defined above; SWE-bench schema = code localization,
# the *_fixture / doc datasets = shared-corpus deep research)
register_dataset("fixture")(lambda limit=None, corpus_limit=None: fixture_instances())
register_dataset("swebench_verified")(_swebench("princeton-nlp/SWE-bench_Verified"))
register_dataset("swebench_lite")(_swebench("princeton-nlp/SWE-bench_Lite"))
register_dataset("loc_bench")(_swebench("czlll/Loc-Bench_V1"))  # LocAgent's own benchmark
register_dataset("browsecomp_plus", domain="general")(load_browsecomp_plus)
# classic multi-hop QA as shared-corpus retrieval (stage via scripts/stage_multihop.py)
register_dataset("hotpotqa", domain="general")(load_hotpotqa)
register_dataset("2wiki", domain="general")(load_2wiki)
register_dataset("musique", domain="general")(load_musique)
# Wikipedia PAIRS (corpus_build/wikipedia emits <base>_flat + <base>_structured over the SAME
# docs/queries — only the section/infobox FIELDS differ). The structured arm's field_profile picks
# the matching BQL manual (wiki = title/section/infobox/body); the flat arm has title/body only
# (default general manual). Build them before running these. The pair is the headline experiment.
for _base in ("hotpotqa", "2wiki", "musique"):
    register_dataset(f"{_base}_structured", domain="general", field_profile="wiki")(
        _built_corpus_loader(f"{_base}_structured", f"{_base}_structured"))
    register_dataset(f"{_base}_flat", domain="general")(
        _built_corpus_loader(f"{_base}_flat", f"{_base}_flat"))
# browsecomp PAIR (corpus_build/browsecomp_plus emits both from the hub — same docs/queries,
# only the author/date fields differ). flat = title/body; structured = +author/date (browsecomp manual).
register_dataset("browsecomp_plus_structured", domain="general", field_profile="browsecomp")(
    _built_corpus_loader("browsecomp_plus_structured", "browsecomp_plus_structured"))
register_dataset("browsecomp_plus_flat", domain="general")(
    _built_corpus_loader("browsecomp_plus_flat", "browsecomp_plus_flat"))
# FULL-corpus browsecomp variants (ADDITIVE, 2026-07-29): same 830 queries/qrels as the
# pooled pair above, but over the complete BrowseComp-Plus collection (100,195 docs — the
# pooled 67,707-doc corpora are byte-identical row-subsets). Same BEIR layout under
# data/<name>_full/; answers ship inline in queries.jsonl. New sibling dataset names ->
# new index/cache namespaces (corpus_id = dataset name), so no pooled artifact is touched.
register_dataset("browsecomp_plus_structured_full", domain="general", field_profile="browsecomp")(
    _built_corpus_loader("browsecomp_plus_structured_full", "browsecomp_plus_structured_full"))
register_dataset("browsecomp_plus_flat_full", domain="general")(
    _built_corpus_loader("browsecomp_plus_flat_full", "browsecomp_plus_flat_full"))
register_dataset("browsecomp_plus_fixture", domain="general")(
    lambda limit=None, corpus_limit=None: _doc_corpus_fixture("browsecomp_plus"))
register_dataset("hotpotqa_fixture", domain="general")(
    lambda limit=None, corpus_limit=None: _doc_corpus_fixture("hotpotqa"))
register_dataset("musique_fixture", domain="general")(
    lambda limit=None, corpus_limit=None: _doc_corpus_fixture("musique"))
