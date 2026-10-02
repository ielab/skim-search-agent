"""Laya (`convaiinnovations/laya`): a non-autoregressive "System 1" decision model (ModernBERT-large,
421M) that answers typed questions about a state with calibrated probabilities in one forward
pass, in TypeSafe Jev's request shape. As a reranker it follows TypeSafe's re-ranking recipe: the
state holds everything to be judged, `{"question": the episode's original question, "search": the
current sub-query, "candidate": the page}`, and one yes/no (`noul`) question with explicit true and
false criteria asks whether the candidate helps answer the question for what the search looks
for. A sub-query alone ("December 2011" "rocks, minerals") does not say what the agent needs; the
original question does. The score is P(yes). The question comes from the episode's QueryContext
(`agent_search.training.history`); outside an episode the state carries the search and the page.

The model's inference code (`rl_common.py`, `rl_agent_api.py`) ships in the main repo,
`convaiinnovations/laya`; this file loads it from there rather than copying it, and loads the
weights, config and tokenizer from the checkpoint named in `model=`: the English checkpoint
(`convaiinnovations/laya`, ModernBERT-large, 512 tokens), `convaiinnovations/laya-multilingual`
(mmBERT-base, 1,024) or `convaiinnovations/laya-typed-decisions` (ModernBERT-large, 1,024). One sequence per candidate
is built with its `build_sequence`, the batch is scored by the model, and P(yes) is computed the way
its `RLAgent.system_one` computes it: the yes/no logits over the calibrated temperature for a
two-option `noul` question, then a softmax. So a score equals what Laya's own API returns for that
page; the candidates are only batched.

Each checkpoint reads its own trained length unless `max_length` says otherwise: the question
and its options first, then the page, which `build_sequence` cuts to what is left. The state's fields are in that order, so a cut trims
the end of the page, never the question (the page is cut at a whitespace boundary first, so the
tokenizer never reads a megabyte tail). `question=` and `criteria=` set the yes/no question.
"""
from __future__ import annotations

import importlib.util
import os
import sys
import threading
from typing import Optional, Sequence

from agent_search.retrievers.dense.base import encoder_prefix
from agent_search.retrievers.rerankers.base import Candidate, Reranker, register_reranker

DEFAULT_MODEL = "convaiinnovations/laya"
DEFAULT_QUESTION = "Does the candidate contain information that helps answer the question, for what the search is looking for?"
DEFAULT_CRITERIA = {"true": "The candidate states facts that help answer the question, of the kind the search is looking for.",
                    "false": "The candidate is only on a related topic, or is unrelated."}
CODE_REPO = "convaiinnovations/laya"
CODE_FILES = ["rl_common.py", "rl_agent_api.py"]
LAYA_FILES = ["rl_agent_config.json", "model.safetensors", "encoder/*", "tokenizer/*"]

_MODELS: dict = {}
_MODELS_LOCK = threading.Lock()


def _snapshot(repo: str, files: list) -> str:
    if os.path.isdir(repo):
        return repo
    from huggingface_hub import snapshot_download
    # only what inference reads (the repos also carry images), so an offline node accepts a
    # snapshot that holds just these
    return snapshot_download(repo, allow_patterns=files)


def _shared_agent(model_id: str, device: Optional[str]):
    """Laya's own RLAgent (weights, tokenizer, calibrated temperatures) and its helper module."""
    key = (model_id, device or "auto")
    with _MODELS_LOCK:
        m = _MODELS.get(key)
        if m is None:
            code = _snapshot(CODE_REPO, CODE_FILES)
            snap = _snapshot(model_id, LAYA_FILES)
            if code not in sys.path:
                sys.path.insert(0, code)                 # rl_agent_api imports `rl_common` by name
            mods = {}
            for name in ("rl_common", "rl_agent_api"):
                spec = importlib.util.spec_from_file_location(name, os.path.join(code, f"{name}.py"))
                mods[name] = importlib.util.module_from_spec(spec)
                sys.modules[name] = mods[name]
                spec.loader.exec_module(mods[name])
            agent = mods["rl_agent_api"].RLAgent(snap, device=device)
            m = (agent, mods["rl_common"], threading.Lock())
            _MODELS[key] = m
    return m


@register_reranker
class LayaReranker(Reranker):
    name = "laya"
    default_model = DEFAULT_MODEL
    default_max_length = 0            # 0 = the checkpoint's own trained length (512 or 1,024)

    def __init__(self, model: str = DEFAULT_MODEL, batch_size: int = 32, max_length: int = 0,
                 device: Optional[str] = None, question: str = DEFAULT_QUESTION, criteria: Optional[dict] = None):
        self.model_id = model
        self.batch_size = int(batch_size)
        self.max_length = int(max_length)
        self.device = device or os.environ.get("AGENT_SEARCH_DENSE_DEVICE") or None
        self.question = question
        self.criteria = dict(criteria or DEFAULT_CRITERIA)
        self._loaded = None

    def _agent(self):
        if self._loaded is None:
            self._loaded = _shared_agent(self.model_id, self.device)
        return self._loaded

    def scores(self, query: str, texts: Sequence[str]) -> list[float]:
        import numpy as np
        import torch
        agent, rc, lock = self._agent()
        q = agent._to_internal({"type": "noul", "instructions": self.question, "criteria": self.criteria})
        from agent_search.training.history import CURRENT
        ctx = CURRENT.get()
        head = {"question": ctx.question} if ctx is not None and ctx.question else {}
        head["search"] = query
        max_len = self.max_length or int(agent.cfg["max_len"])
        temp = agent.temperature_by_options.get(rc.temp_bucket(rc.QTYPES["noul"], 2),
                                                agent.temperature[rc.QTYPES["noul"]])
        out: list[float] = []
        with lock, torch.no_grad():
            for i in range(0, len(texts), self.batch_size):
                items = []
                for t in texts[i:i + self.batch_size]:
                    state = {**head, "candidate": encoder_prefix(t or "", agent.tok, max_len)}
                    seq, markers = rc.build_sequence(agent.tok, state, q, max_len, agent.cfg["head_max_len"])
                    items.append({"ids": seq, "markers": markers, "qtype": rc.QTYPES["noul"], "target": [0.0] * len(markers),
                                  "label": -1, "episode": 0, "ep_step": 0, "ep_len": 1, "src": "api"})
                b = rc.collate_items([items], agent.tok.pad_token_id)
                dev = agent.device
                with torch.autocast(device_type=dev.type, dtype=agent.dtype, enabled=dev.type == "cuda"):
                    logits, _ = agent.model(b["input_ids"].to(dev), b["attention_mask"].to(dev), b["marker_pos"].to(dev),
                                            b["marker_mask"].to(dev), b["qtype"].to(dev))
                z = logits.float().cpu().numpy()[:, :2] / temp
                p = np.exp(z - z.max(axis=1, keepdims=True))
                out.extend((p[:, 1] / p.sum(axis=1)).tolist())
        return out

    def rerank(self, query: str, candidates: Sequence[Candidate],
               k: Optional[int] = None) -> list[tuple[str, float]]:
        cands = [(d, t or "") for d, t in candidates]
        if not cands:
            return []
        s = self.scores(query, [t for _, t in cands])
        ranked = sorted(zip((d for d, _ in cands), s), key=lambda x: (-x[1], x[0]))
        return ranked[:k] if k is not None else ranked

    def describe(self) -> dict:
        return {"reranker": self.name, "model": self.model_id, "max_length": self.max_length,
                "question": self.question, "criteria": self.criteria}


__all__ = ["LayaReranker", "DEFAULT_MODEL", "DEFAULT_QUESTION"]
