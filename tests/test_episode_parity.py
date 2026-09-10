"""One stub-policy episode per paper condition is identical through the pre-0.3 workspace agent
and the condition runner: the same tool calls, the same observations, the same ranking, the
same answer and stop reason, the same record. Conditions that need a dense embedding cache are
covered by scripts/episode_parity.py on a node with the cache built."""
from __future__ import annotations

import tempfile
import threading

import pytest

from agent_search.retrievers.registry import RetrieverConfig

DENSE = {"dense", "bql_fused", "bql_dense"}


def _conditions():
    from agent_search.strategies import CONDITIONS
    return sorted(n for n, c in CONDITIONS.items()
                  if c.strategy.loop and not (set(c.strategy.engines) & DENSE))


def _strip(meta: dict) -> dict:
    m = dict(meta)
    if isinstance(m.get("trajectory"), list):
        m["trajectory"] = [{k: v for k, v in st.items() if not k.startswith("t_")} for st in m["trajectory"]]
    return m


@pytest.mark.parametrize("name", _conditions())
def test_episode_is_identical_old_and_new(name, tmp_path, monkeypatch):
    from agent_search.agent.retriever import _build_legacy_agent
    from agent_search.evaluation.agent_runner import build_condition_agent
    from agent_search.evaluation.corpus_units import _corpus_key, _units_for_instance
    from agent_search.evaluation.datasets import load_dataset_by_name
    from agent_search.strategies import CONDITIONS

    monkeypatch.setenv("AGENT_SEARCH_DCI_CACHE", str(tmp_path / "dci"))
    cond = CONDITIONS[name]
    inst = load_dataset_by_name("code_fixture" if cond.domain == "code" else "doc_fixture")[0]
    units, _by_file, files = _units_for_instance(inst, str(tmp_path / "repos"), False, {}, threading.Lock())
    key = _corpus_key(inst)
    cfg = RetrieverConfig(policy="stub", index_root=str(tmp_path / "idx"))
    outs = []
    for build in (lambda: _build_legacy_agent(cfg, f"agent_{name}")(), lambda: build_condition_agent(cfg, name)()):
        r = build()
        if getattr(r, "needs_files", False) and files is not None:
            r.set_files(files)
        r.index(units, key=key)
        located = r.search(inst.problem_statement, 10)
        t = r.last_trajectory
        outs.append((located, [(s.name, s.args, s.observation) for s in t.steps], t.final_answer,
                     t.stopped_reason, _strip(r.last_trajectory_meta or {})))
    (l0, s0, a0, r0, m0), (l1, s1, a1, r1, m1) = outs
    assert s1 == s0
    assert l1 == l0 and a1 == a0 and r1 == r0
    if cond.domain == "code":
        # the code records now carry the real file ids a search listed and a fetch read,
        # where the workspace agent recorded [] and the rank; everything else is equal
        for st in m0["trajectory"] + m1["trajectory"]:
            st.pop("hit_ids", None); st.pop("read_ids", None)
    assert m1 == m0
