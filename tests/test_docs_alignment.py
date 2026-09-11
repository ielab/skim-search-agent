"""The documentation names real things: every repository path, console script, strategy,
dataset and environment knob mentioned in the docs exists in the code. This keeps the docs from
drifting when something is renamed."""
from __future__ import annotations

import re

from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
DOCS = [REPO / "README.md", REPO / "CONTRIBUTING.md",
        *(REPO / "docs").glob("*.md"), REPO / "corpus_build" / "README.md",
        REPO / "scripts" / "slurm" / "README.md", REPO / "demo" / "README.md"]
DOCS = [p for p in DOCS if p.exists()]

_PATH = re.compile(r"`((?:agent_search|configs|scripts|docs|tests|corpus_build|demo|examples)/[A-Za-z0-9_./\-*]+)`")
_SCRIPT = re.compile(r"(?<![\w#/-])(skimsearchagent(?:-[a-z]+(?:-[a-z]+)*)?)(?![\w.-])")
_KNOB_ROW = re.compile(r"^\| `([A-Z][A-Z0-9_]{3,})` \|", re.M)


def _pyproject_scripts() -> set[str]:
    text = (REPO / "pyproject.toml").read_text()
    block = text.split("[project.scripts]", 1)[1].split("\n[", 1)[0]
    return set(re.findall(r"^([A-Za-z0-9_-]+)\s*=", block, re.M))


@pytest.mark.parametrize("doc", DOCS, ids=[p.relative_to(REPO).as_posix() for p in DOCS])
def test_paths_named_in_docs_exist(doc):
    missing = []
    for m in _PATH.finditer(doc.read_text(encoding="utf-8")):
        rel = m.group(1).rstrip("/")
        if "*" in rel:
            if not list(REPO.glob(rel)):
                missing.append(rel)
        elif not (REPO / rel).exists():
            missing.append(rel)
    assert not missing, f"{doc.name} names paths that do not exist: {sorted(set(missing))}"


@pytest.mark.parametrize("doc", DOCS, ids=[p.relative_to(REPO).as_posix() for p in DOCS])
def test_console_scripts_named_in_docs_exist(doc):
    scripts = _pyproject_scripts()
    named = {m.group(1) for m in _SCRIPT.finditer(doc.read_text(encoding="utf-8"))}
    # bare `skimsearchagent` and the plugin entry-point group are not scripts
    named -= {"skimsearchagent"}
    unknown = {n for n in named if n not in scripts and not n.startswith("skimsearchagent-<")}
    assert not unknown, f"{doc.name} names console scripts that do not exist: {sorted(unknown)}"


def test_env_knobs_in_configuration_tables_are_read_by_the_code():
    doc = (REPO / "docs" / "CONFIGURATION.md").read_text(encoding="utf-8")
    src = "\n".join(p.read_text(encoding="utf-8", errors="ignore")
                    for p in (REPO / "agent_search").rglob("*.py"))
    src += "\n".join(p.read_text(encoding="utf-8", errors="ignore")
                     for p in (REPO / "scripts").rglob("*") if p.is_file())
    unknown = [k for k in _KNOB_ROW.findall(doc) if k not in src]
    assert not unknown, f"knobs documented but not read anywhere: {unknown}"


def test_strategies_and_datasets_named_in_readme_exist():
    from agent_search.evaluation.datasets import available_datasets
    from agent_search.strategies.names import STRATEGIES
    readme = (REPO / "README.md").read_text(encoding="utf-8")
    table = readme.split("## Strategies", 1)[1].split("\n## ", 1)[0]
    named = set(re.findall(r"`([a-z0-9_]+)`", table.split("|---|")[1] if "|---|" in table else table))
    retriever_names = {"bm25", "bm25_lucene", "dense", "bql", "grep"}
    unknown = {n for n in named if n not in STRATEGIES and n not in retriever_names and not n.startswith("dataset=")}
    unknown -= {"code_fixture"}
    assert not unknown, f"README strategies table names unknown strategies: {sorted(unknown)}"
    # dataset names used in shell commands must be registered (Python examples may invent names)
    for line in readme.splitlines():
        if line.strip().startswith("skimsearchagent"):
            for ds in re.findall(r"dataset=([a-z0-9_]+)", line):
                assert ds in available_datasets(), ds
