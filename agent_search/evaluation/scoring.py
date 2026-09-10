"""Scoring one instance: build/index a retriever, run it, and grade the result.

`_score_instance` is the entry point `runner.py` calls per instance. It builds
corpus units (via `corpus_units.py`), gets an indexed retriever, runs the
query, and scores the ranking (or the answer, for QA datasets). It also
carries the token-cost decomposition (initial prompt / retrieved-context /
output, each counted once) used for the LocAgent-style cost axis.
"""
from __future__ import annotations

import re
import threading
from typing import Callable, Optional, Sequence

from agent_search.corpus.code_repo import get_files
from agent_search.core.errors import SetupError
from agent_search.core.interfaces import Retriever
from agent_search.core.tokens import count_tokens

from .rows import observations_of

from . import metrics as M
from .datasets import Instance
from .ground_truth import changed_line_ranges, gold_files, gold_units
from .corpus_units import _corpus_key, _to_file_ranking, _units_for_instance
from agent_search.corpus.units import CodeUnit, is_test_path

RetrieverFactory = Callable[[], Retriever]


def _retriever_meta(retriever: Retriever) -> dict:
    out = {"retriever": getattr(retriever, "name", type(retriever).__name__)}
    for attr, key in (
        ("tool", "tool_condition"),
        ("tool_description", "tool_description"),
        ("domain", "domain"),
        ("prompt_path", "prompt_profile_path"),
        ("max_steps", "max_steps"),
    ):
        val = getattr(retriever, attr, None)
        if val is not None:
            out[key] = val
    return out


def _trajectory_meta(retriever: Retriever) -> dict:
    """The episode metadata an agent retriever pre-serializes (rows.jsonl shape).

    `AgentRetriever` builds this in `last_trajectory_meta` (see agent/retriever.py);
    non-agent retrievers have none, so this is empty for them. The numeric cost fields
    (llm_calls, n_steps, tokens) are mean-aggregated into results.json — the
    LocAgent-style cost axis (calls + tokens to reach the result)."""
    meta = getattr(retriever, "last_trajectory_meta", None)
    return meta if isinstance(meta, dict) else {}


# pool size for cutoff-free set metrics on one-shot floors (the agent ignores
# this — it returns its own accumulated set). A tool returning >this is not
# usefully 'retrieving'; set_precision will correctly read ~0 for it.
_SET_K = 1000


_SETUP_MARKERS = ("needs a persisted", "prebuild", "build it with", "no lucene_structured index",
                  "not installed", "cannot start")


def _build_indexed(retriever_factory: RetrieverFactory, units: Sequence[CodeUnit],
                   key: str) -> Retriever:
    """Build + index one retriever, turning a missing-artifact failure into a SetupError so
    the run aborts before the first episode instead of logging one error per instance."""
    try:
        return retriever_factory().index(units, key=key)
    except SetupError:
        raise
    except FileNotFoundError as e:
        raise SetupError(str(e)) from e
    except RuntimeError as e:
        if any(m in str(e).lower() for m in _SETUP_MARKERS):
            raise SetupError(str(e)) from e
        raise


def _indexed_retriever(retriever_factory: RetrieverFactory, units: Sequence[CodeUnit],
                       key: str, reuse: bool, retriever_cache: dict,
                       retriever_lock: threading.Lock) -> Retriever:
    if not reuse:
        return _build_indexed(retriever_factory, units, key)
    with retriever_lock:
        retriever = retriever_cache.get(key)
        if retriever is None:
            retriever = _build_indexed(retriever_factory, units, key)
            retriever_cache[key] = retriever
        return retriever


_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)


def _obs_token_count(text: str) -> int:
    """Token count for a tool observation, on ONE fixed ruler across every arm — so structured
    `fetch`-a-section vs flat `visit`-whole-doc vs bm25 doc tokens are compared on the same scale,
    independent of the run's billing tokenizer. See `agent_search.core.tokens.count_tokens`
    (tiktoken o200k_base when installed, whitespace tokens otherwise — never characters)."""
    return count_tokens(text)


def _score_instance(inst: Instance, retriever_factory: RetrieverFactory,
                    ks: Sequence[int], level: str, max_k: int, cache_dir: str,
                    allow_clone: bool = False, units_cache: Optional[dict] = None,
                    units_lock: Optional[threading.Lock] = None,
                    retriever_cache: Optional[dict] = None,
                    retriever_lock: Optional[threading.Lock] = None,
                    reuse_indexed_retriever: bool = False) -> dict:
    """Compute one instance's row. Deterministic skips carry a 'skipped' reason."""
    generic_gold = inst.gold_doc_ids
    units, by_file, files = _units_for_instance(
        inst,
        cache_dir,
        allow_clone,
        units_cache if units_cache is not None else {},
        units_lock if units_lock is not None else threading.Lock(),
    )
    if not units:
        return {"instance_id": inst.instance_id, "skipped": "no_units"}

    corpus_key = _corpus_key(inst)
    retriever = _indexed_retriever(
        retriever_factory,
        units,
        corpus_key,
        reuse_indexed_retriever,
        retriever_cache if retriever_cache is not None else {},
        retriever_lock if retriever_lock is not None else threading.Lock(),
    )
    # the multi-tool localization agent reads the live files (grep/open); plumb the
    # raw {path -> source} in (re-fetch if units came from the disk cache, which
    # keeps only the chunking). Per-instance retriever (reuse off for repo datasets).
    if getattr(retriever, "needs_files", False):
        if files is None:
            files = get_files(inst, cache_dir, allow_clone=allow_clone)
        retriever.set_files(files)
    # Agent retrievers return their FULL accumulated set regardless of k; one-shot
    # floors return top-k, so fetch a large pool (SET_K) for the cutoff-free set
    # metrics. @k still slices [:k] below, so headline @k numbers are unchanged.
    # a retriever that emits its own complete ranking (the agent) declares it; others
    # are padded to _SET_K. Capability flag, not a name check — so a new agentic
    # retriever class just sets `returns_full_set = True`.
    is_agent = getattr(retriever, "returns_full_set", False)
    k_fetch = max_k if is_agent else max(max_k, _SET_K)
    retrieved = retriever.search(inst.problem_statement, k=k_fetch)
    if generic_gold is not None:
        gold = set(generic_gold)
        ranking = retrieved
    elif level == "function":
        gold = gold_units(changed_line_ranges(inst.patch), by_file)
        ranking = retrieved
    else:
        # restrict to .py, non-test: only those files are in the corpus, so other
        # gold files would be unreachable and unfairly deflate file-level Acc@k
        gold = {f for f in gold_files(inst.patch)
                if f.endswith(".py") and not is_test_path(f)}
        ranking = _to_file_ranking(retrieved)

    # An answer-only instance (a QA set with answers but no document labels, e.g. InfoSeek) has
    # nothing to rank against: keep the row, score the answer, leave the rank metrics out.
    answer_only = (generic_gold is None and inst.answer is not None
                   and (inst.docs is not None or inst.docstore is not None))
    if not gold and not answer_only:      # no parseable gold location -> skip (fair)
        return {"instance_id": inst.instance_id, "skipped": "no_gold"}
    units_by_id = getattr(units, "by_id", None)
    if units_by_id is None:
        units_by_id = {u.doc_id: u for u in units}
    row = {
        "instance_id": inst.instance_id,
        "n_gold": len(gold),
        "n_retrieved": len(ranking),
    }
    if answer_only:
        row.update({"answer_only": True, "set_size": len(ranking), "gold_ids": [],
                    "retrieved": _retrieved_detail(ranking, units_by_id, max_k)})
    else:
        row.update({
            **{f"recall@{k}": M.recall_at_k(ranking, gold, k) for k in ks},
            **{f"hit@{k}": M.hit_at_k(ranking, gold, k) for k in ks},
            **{f"acc@{k}": M.acc_at_k(ranking, gold, k) for k in ks},
            **{f"precision@{k}": M.precision_at_k(ranking, gold, k) for k in ks},
            **{f"f1@{k}": M.f1_at_k(ranking, gold, k) for k in ks},
            **{f"map@{k}": M.average_precision_at_k(ranking, gold, k) for k in ks},
            **{f"ndcg@{k}": M.ndcg_at_k(ranking, gold, k) for k in ks},
            "mrr@10": M.mrr_at_k(ranking, gold, 10),
            # cutoff-FREE set metrics over the whole returned set (native to BQL/grep)
            "set_size": len(ranking),
            "set_recall": M.set_recall(ranking, gold),
            "set_precision": M.set_precision(ranking, gold),
            "set_f1": M.set_f1(ranking, gold),
            "gold_ids": sorted(gold),
            "retrieved": _retrieved_detail(ranking, units_by_id, max_k),
        })
    row.update(_retriever_meta(retriever))
    meta = _trajectory_meta(retriever)
    row.update(meta)
    # TOKEN COST, cache-aware — count each prefix-cached span ONCE. The raw per-step prompt_tokens
    # re-sends the whole growing context every turn, so summing it (meta["prompt_tokens"]) triple-
    # counts the cached initial prompt AND every earlier observation — which makes the method look
    # expensive precisely where it's cheap. The meaningful decomposition, each part counted once:
    #   initial_prompt_tokens : the fixed system+task+manual+question — step 0's input, pre-retrieval,
    #                           re-sent (cached) every later turn, so counted ONCE here.
    #   retrieved_doc_tokens  : the accumulated observation/reasoning content that actually ENTERED
    #                           the model's context window, counted once. Derived from the REAL
    #                           per-step prompt_tokens vLLM reports — `max(step.prompt_tokens) -
    #                           initial_prompt_tokens` is the most context the model ever actually
    #                           held at once (windows/caps mean it isn't strictly monotonic, hence
    #                           max, not the last step) — rather than summing raw tool-output text:
    #                           the policy truncates/caps what actually gets inserted into the
    #                           prompt (a single bash "read" can be 85k-370k raw tokens while the
    #                           per-step prompt_tokens tops out at a few thousand), so summing raw
    #                           observation text would overcount badly for whole-doc/bash baselines.
    #                           Renamed internally to context_once_tokens; the field name
    #                           retrieved_doc_tokens stays for downstream consumers
    #                           (runs/_summary/*.py, scripts/summarize_runs.py key off it).
    #   output_tokens         : the model's generated tokens (never cached; naturally once).
    # total_tokens_once sums them — the real marginal work of an episode, using only REAL usage
    # numbers (no raw-observation-text summation). The raw cumulative prompt_tokens/completion_tokens/
    # cached_input_tokens still ride in `meta` for billing reality.
    if "trajectory" in meta or "observations" in meta:
        traj_steps = meta.get("trajectory") or []
        step_prompt_tokens = [s.get("prompt_tokens") for s in traj_steps if s.get("prompt_tokens") is not None]
        row["initial_prompt_tokens"] = int(traj_steps[0].get("prompt_tokens") or 0) if traj_steps else 0
        if step_prompt_tokens:
            context_once_tokens = max(0, int(max(step_prompt_tokens)) - row["initial_prompt_tokens"])
            row["token_source"] = "prompt_tokens"
        else:
            # no per-step prompt_tokens on this trajectory (older run / non-vLLM backend) — fall
            # back to the old raw-observation-text estimate and flag it as such, since it can
            # badly overcount relative to what the model actually saw.
            context_once_tokens = sum(_obs_token_count(o) for o in observations_of(meta))
            row["token_source"] = "fallback"
        row["context_once_tokens"] = context_once_tokens
        row["retrieved_doc_tokens"] = context_once_tokens
        row["output_tokens"] = int(meta.get("completion_tokens") or 0)
        row["total_tokens_once"] = (row["initial_prompt_tokens"] + row["retrieved_doc_tokens"]
                                    + row["output_tokens"])
        row["read_tokens"] = row["retrieved_doc_tokens"]     # back-compat name, now a REAL token count
        # REASONING (generation) tokens — a SUBSET already inside output_tokens, itemized (NOT added
        # on top, so total_tokens_once stays correct). OpenAI reasoning models report it in usage
        # (meta["reasoning_tokens"]); Tongyi/vLLM instead emit it inline as <think>…</think> with no
        # usage field, so fall back to counting those spans on the same ruler. Prefer usage; else spans.
        usage_reason = int(meta.get("reasoning_tokens") or 0)
        think_reason = sum(_obs_token_count(m) for s in traj_steps
                           for m in _THINK_RE.findall(str(s.get("raw_output") or "")))
        row["reasoning_tokens"] = usage_reason or think_reason
    # CODE-FIX arm: the end-to-end metric is fix-file-ok (did the <fix>'s file: hit a gold-patch
    # file), scored on the fix the agent committed to — not the @k of an empty ranking. The
    # retrieval @k columns still compute above (all ~0 for this arm) so the row shape is uniform.
    if meta.get("fix_text"):
        from agent_search.evaluation.fix_scoring import fix_file, score_fix
        ok, _ = score_fix(meta["fix_text"], inst.patch)
        row["fix_file_ok"] = 1.0 if ok else 0.0
        row["predicted_file"] = fix_file(meta["fix_text"])
        # PATCH mode (task taskfix_patch): compile the <fix> SEARCH/REPLACE edits into a
        # git-applyable unified diff against the base_commit `files`, carried in rows.jsonl as
        # `model_patch` for later sb-cli submission (real FAIL_TO_PASS/PASS_TO_PASS grading).
        # A no-op for the prose `codefix` twins (their fix_text has no SEARCH/REPLACE -> "").
        if files:
            from agent_search.evaluation.patch_synthesis import synthesize_patch
            patch, prep = synthesize_patch(meta["fix_text"], files)
            if patch:
                row["model_patch"] = patch
                row["patch_n_edits"] = prep.n_edits
                row["patch_n_applied"] = prep.n_applied
    # DEEP-RESEARCH arm: grounded EM/F1 (the answer must also appear in the tool evidence) plus
    # gold-doc coverage. QA datasets carry a gold answer; the doc arm surfaces evidence.
    if inst.answer:
        from agent_search.evaluation.doc_scoring import gold_doc_coverage, score_answer
        pred = row.get("final_answer", "")
        # Pass surfaced_docs + gold_doc_ids so score_answer emits MuSiQue's SUPPORT F1 (its paired
        # paper metric alongside answer-F1) whenever the dataset carries gold supporting-doc ids.
        row.update(score_answer(pred, inst.answer, observations_of(meta),
                                surfaced_docs=meta.get("surfaced_docs"),
                                gold_doc_ids=inst.gold_doc_ids))
        row["gold_answer"] = inst.answer
        row["question"] = inst.problem_statement   # carried for the optional LLM judge (agent_search/evaluation/llm_judge.py)
        if inst.gold_doc_ids is not None and meta.get("surfaced_docs") is not None:
            row["gold_doc_coverage"] = gold_doc_coverage(meta["surfaced_docs"], inst.gold_doc_ids)
    return row


def _retrieved_detail(ranking: Sequence[str], units_by_id: dict, n: int) -> list[dict]:
    """What was actually retrieved, reviewable without re-running.

    Code units carry their exact source location (path + 1-based line span +
    first-line snippet) — the full code is reproducible from repo@base_commit, so
    rows.jsonl stays small. Non-code/fixed-corpus docs (and file-level rankings)
    carry the doc id, plus title when the unit has one.
    """
    out = []
    for rank, doc_id in enumerate(ranking[:n], start=1):
        u = units_by_id.get(doc_id)
        if u is None:                       # file-level rank entry or unknown id
            out.append({"rank": rank, "doc_id": doc_id})
            continue
        ent = {"rank": rank, "doc_id": doc_id, "path": u.path,
               "lines": [u.start_line, u.end_line]}
        first = next((ln for ln in (u.code or "").strip().splitlines() if ln.strip()), "")
        if first:
            ent["snippet"] = first.strip()
        if u.title:
            ent["title"] = str(u.title)
        out.append(ent)
    return out
