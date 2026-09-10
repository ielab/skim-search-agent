"""Dataset loading: shared document-corpus instances, plus a retained code-localization arm.

Document-domain instances carry `docs`, `gold_doc_ids`, and an optional answer, over a
shared corpus with qrels. The code-localization arm follows SweRank / LocAgent: an
Instance there carries the issue text and the gold patch. `files` is populated only
for the built-in fixture (inline sources, so the pipeline runs with no git/deps);
for real SWE-bench it is None and resolved by `repo.get_files` at the base commit.

Split across `base` (the `Instance` record, the registry, shared helpers), `swebench`,
`fixtures`, `beir` and `topics` (one file per dataset family); importing this package
imports every submodule below so their `register_dataset(...)` calls run.
"""
from __future__ import annotations

from agent_search.evaluation.datasets.base import (
    _DATASET_DOMAIN,
    _DATASET_PROFILE,
    _DATASETS,
    DATA_DIR,
    Instance,
    _find_existing,
    _jsonl,
    _qrels_from_jsonl,
    _qrels_from_tsv,
    _sidecar_answers,
    available_datasets,
    data_dir,
    dataset_domain,
    dataset_field_profile,
    default_dense_model,
    discover_topics_datasets,
    load_dataset_by_name,
    register_dataset,
)
from agent_search.evaluation.datasets.swebench import (
    _local_dir,
    _swebench,
    load_swebench,
)
from agent_search.evaluation.datasets.fixtures import (
    _doc_corpus_fixture,
    fixture_instances,
)
from agent_search.evaluation.datasets.beir import (
    _built_corpus_loader,
    _load_beir_style,
    _multihop_loader,
    _structured_loader,
    load_2wiki,
    load_browsecomp_plus,
    load_hotpotqa,
    load_musique,
)
from agent_search.evaluation.datasets.topics import (
    DOCSTORE_MIN_BYTES,
    _load_topics_qrels,
    _topics_qrels_loader,
)

__all__ = [
    "DATA_DIR",
    "DOCSTORE_MIN_BYTES",
    "Instance",
    "available_datasets",
    "data_dir",
    "dataset_domain",
    "dataset_field_profile",
    "default_dense_model",
    "discover_topics_datasets",
    "fixture_instances",
    "load_2wiki",
    "load_browsecomp_plus",
    "load_dataset_by_name",
    "load_hotpotqa",
    "load_musique",
    "load_swebench",
    "register_dataset",
]
