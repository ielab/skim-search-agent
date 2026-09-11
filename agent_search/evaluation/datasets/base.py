"""Shared dataset plumbing: the `Instance` record, `DATA_DIR`, the dataset registry
(`register_dataset` / `available_datasets` / `load_dataset_by_name` / ...), the default
dense-embedder pick, and small file-format helpers every loader reuses.
"""
from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from typing import Any, Optional

# Datasets are pre-downloaded here (on a node with internet) so the GPU node, which
# usually has no internet, can load them offline. See scripts/download_data.sh.
DATA_DIR = os.environ.get("AGENT_SEARCH_DATA", "data")


def data_dir() -> str:
    """The current data root, read from the package attribute (`agent_search.evaluation.
    datasets.DATA_DIR`), not this module's own copy. Tests monkeypatch the package attribute
    (`monkeypatch.setattr(DS, "DATA_DIR", ...)` where `DS` is the package), and every loader
    must see that override. Call this instead of referencing `DATA_DIR` directly."""
    import agent_search.evaluation.datasets as _pkg
    return _pkg.DATA_DIR


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
    docstore: Optional[Any] = None               # on-disk corpus (agent_search.corpus.docstore) for
                                                 # collections too large for `docs`


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
    """Some BEIR-style sets (browsecomp_plus) ship gold answers in a separate file, not in
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


# --- dataset registry (the single extension point) --------------------------
# Adding a dataset = register a loader here (or anywhere that imports this module),
# nothing else: `available_datasets()` is the one source of truth the CLIs use for
# their --dataset choices, so run_eval / build_indexes never change.

_DATASETS: dict[str, "object"] = {}
_DATASET_DOMAIN: dict[str, str] = {}
_DATASET_PROFILE: dict[str, str] = {}


def register_dataset(name: str, domain: str = "code", field_profile: str | None = None):
    """Decorator: bind a name to a loader ``(limit, corpus_limit) -> [Instance]``.

    `domain` is declared by the dataset (``code`` = per-query repo localization,
    ``general`` = shared document corpus / deep research). It drives the dense
    embedder, the prompt domain, and whether the corpus is pre-embedded, so adding
    a new dataset is a single registration, with no central list to edit.

    `field_profile` selects which BQL manual variant the agent sees: the manual must
    advertise exactly the corpus's fields. A structured corpus declares its profile
    (``wiki`` = title/section/infobox/body, ``browsecomp`` = title/author/date/section/
    body); a flat corpus omits it and inherits the domain (general = title/body)."""
    def deco(fn):
        _DATASETS[name] = fn
        _DATASET_DOMAIN[name] = domain
        if field_profile is not None:
            _DATASET_PROFILE[name] = field_profile
        return fn
    return deco


def discover_topics_datasets(root: Optional[str] = None) -> list[str]:
    """Register every `data/<name>/topics.tsv` (+ corpus.jsonl there or under data/corpora/<name>)
    that is not registered yet, so a folder dropped into `data/` (a sample written by
    `skimsearchagent-sample-dataset`, a set staged by hand) is a dataset without code."""
    root = root or data_dir()
    found = []
    try:
        names = sorted(os.listdir(root))
    except OSError:
        return found
    for name in names:
        d = os.path.join(root, name)
        if name in _DATASETS or name == "corpora" or not os.path.isfile(os.path.join(d, "topics.tsv")):
            continue
        if not (os.path.exists(os.path.join(d, "corpus.jsonl"))
                or os.path.exists(os.path.join(root, "corpora", name, "corpus.jsonl"))):
            continue
        # imported lazily to avoid a module-load-time cycle: topics.py registers datasets with
        # base.py's registry, so base.py cannot import topics.py at the top level.
        from agent_search.evaluation.datasets.topics import _topics_qrels_loader
        register_dataset(name, domain="general")(_topics_qrels_loader(name, root=d))
        found.append(name)
    return found


def available_datasets() -> set[str]:
    discover_topics_datasets()
    return set(_DATASETS)


def dataset_domain(name: str, override: Optional[str] = None) -> str:
    """The domain a dataset declared at registration (overridable)."""
    return override or _DATASET_DOMAIN.get(name, "code")


def dataset_field_profile(name: str, override: Optional[str] = None) -> Optional[str]:
    """The per-dataset BQL field profile, or None to inherit the domain's manual."""
    return override or _DATASET_PROFILE.get(name)


def default_dense_model(domain: str) -> str:
    """The default dense embedder for `domain`. Env `DENSE_MODEL` overrides the general-domain
    default only (e.g. `DENSE_MODEL=Qwen/Qwen3-Embedding-0.6B` to swap in Qwen3-Embedding for
    browsecomp_plus_structured/hotpotqa_structured/etc.). The code-domain default
    (nomic-ai/CodeRankEmbed) is a code-trained embedder with no general-text equivalent in this
    knob's scope, so it stays fixed regardless of DENSE_MODEL: an accidental `export
    DENSE_MODEL=...` in a shell must never silently swap the SWE-bench code embedder.
    Unset falls back to BAAI/bge-base-en-v1.5 for general. The chosen model's own cache
    namespace (`indexes/dense/<model>-sl<len>/<corpus_key>/`, from DenseRetriever._cache_dir)
    keeps it from ever mixing with another model's embeddings."""
    import os
    if domain == "general":
        return os.environ.get("DENSE_MODEL") or "BAAI/bge-base-en-v1.5"
    return "nomic-ai/CodeRankEmbed"


def load_dataset_by_name(name: str, limit: int | None = None,
                         corpus_limit: int | None = None) -> list[Instance]:
    if name not in _DATASETS:
        discover_topics_datasets()
    try:
        fn = _DATASETS[name]
    except KeyError:
        raise ValueError(
            f"unknown dataset {name!r}; choose from {sorted(_DATASETS)}") from None
    return fn(limit=limit, corpus_limit=corpus_limit)
