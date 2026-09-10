"""Run one stub-policy episode per condition through the pre-0.3 workspace agent and the
condition runner over the same fixture corpus, and report whether the episodes are identical.

    python scripts/episode_parity.py                      # every condition that needs no dense cache
    PARITY_INDEX_ROOT=indexes python scripts/episode_parity.py research_dense research_hybrid

With PARITY_INDEX_ROOT set, the dense conditions run too and read the doc_fixture embedding
cache under that root (build it with `skimsearchagent-build-indexes --dataset doc_fixture
--retriever dense --index-root <root>`). Kept for the release that still ships the old code.
"""
import os, sys, threading, tempfile
os.environ.setdefault("AGENT_SEARCH_DCI_CACHE", tempfile.mkdtemp(prefix="dci_parity_"))
from agent_search.retrievers.registry import RetrieverConfig
from agent_search.agent.retriever import _build_legacy_agent
from agent_search.evaluation.agent_runner import build_condition_agent
from agent_search.evaluation.datasets import load_dataset_by_name
from agent_search.evaluation.corpus_units import _units_for_instance, _corpus_key
from agent_search.strategies import CONDITIONS

DENSE = {"dense", "bql_fused", "bql_dense"}
idx = os.environ.get("PARITY_INDEX_ROOT") or tempfile.mkdtemp(prefix="idx_parity_")
names = sys.argv[1:] or [n for n, c in CONDITIONS.items() if c.strategy.loop
                         and (os.environ.get("PARITY_INDEX_ROOT") or not (set(c.strategy.engines) & DENSE))]

for name in names:
    cond = CONDITIONS[name]
    ds = "code_fixture" if cond.domain == "code" else "doc_fixture"
    insts = load_dataset_by_name(ds)
    inst = insts[0]
    units, by_file, files = _units_for_instance(inst, tempfile.mkdtemp(), False, {}, threading.Lock())
    key = _corpus_key(inst)
    cfg = RetrieverConfig(policy="stub", index_root=idx)
    outs = []
    for label, build in (("old", lambda: _build_legacy_agent(cfg, f"agent_{name}")()),
                         ("new", lambda: build_condition_agent(cfg, name)())):
        r = build()
        if getattr(r, "needs_files", False) and files is not None:
            r.set_files(files)
        r.index(units, key=key)
        located = r.search(inst.problem_statement, 10)
        t = r.last_trajectory
        steps = [(s.name, s.args, s.observation) for s in t.steps]
        meta = r.last_trajectory_meta or {}
        outs.append((located, steps, t.final_answer, t.stopped_reason, meta))
    (l0, s0, a0, r0, m0), (l1, s1, a1, r1, m1) = outs
    same = l0 == l1 and s0 == s1 and a0 == a1 and r0 == r1
    def _strip(m):
        m = dict(m)
        if isinstance(m.get("trajectory"), list):
            m["trajectory"] = [{k: v for k, v in st.items() if not k.startswith("t_")} for st in m["trajectory"]]
        return m
    m0, m1 = _strip(m0), _strip(m1)
    diffm = {k for k in set(m0) | set(m1) if m0.get(k) != m1.get(k)}
    for k in sorted(diffm):
        print("   meta", k, "\n   OLD:", str(m0.get(k))[:300], "\n   NEW:", str(m1.get(k))[:300])
    print(f"{name:28} steps={len(s0):2d} located_eq={l0==l1} steps_eq={s0==s1} answer_eq={a0==a1} stop_eq={r0==r1} meta_diff={sorted(diffm)}")
    if s0 != s1:
        for i, (a, b) in enumerate(zip(s0, s1)):
            if a != b:
                print("   first differing step", i, "\n   OLD:", str(a)[:300], "\n   NEW:", str(b)[:300]); break
