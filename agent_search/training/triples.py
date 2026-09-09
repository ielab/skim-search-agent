"""Turn run records into retriever training data (ITER's trajectory-relative supervision).

Every search the agent issued becomes a candidate training example whose query is rendered from
the agent's history at that moment (`queries.render_query`) and whose labels come from what
happened next:

* **positive**: a document the agent read for the first time after this search, that was
  relevant, and that some search in the episode had surfaced;
* **neg_diversity**: documents the agent had already read before this search and found relevant
  (the retriever should stop returning them, the "diversity" tier);
* **neg_hard**: documents already read and found irrelevant;
* **neg_weak**: documents this search returned that the agent never read in the whole episode.

An example is kept only when it has a positive and at least one negative. "Relevant" is decided by
a *labeller*: ``oracle`` (the document is one of the question's gold documents, which the run
record carries), ``answer`` (the gold answer string appears in the document), or an LLM judge
over the agent's post-read reasoning (ITER's original protocol, `make_llm_judge`).

The output is ITER's record shape, one JSON object per line::

    {"query", "pos", "pos_id", "neg_diversity", "neg_diversity_id", "neg_hard", "neg_hard_id",
     "neg_weak", "neg_weak_id", "reasoning_len", "reweight_rate", "instance_id", "query_style"}

Reading a run record: search steps are the actions whose name contains ``search``; read steps
are ``visit``/``fetch``/``read`` actions. Rows written by this library carry ``hit_ids`` (what a
search listed) and ``read_ids`` (what a read opened) on every trajectory step. For older rows the
listing is parsed from the observation text.
"""
from __future__ import annotations

import json
import math
import re
import statistics
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

from agent_search.core.tokens import count_tokens
from agent_search.evaluation.rows import observations_of
from agent_search.training.queries import DEFAULT_STYLE, reasoning_text, render_query

Labeller = Callable[[str, str, dict], bool]   # (doc_id, reasoning_after_read, row) -> relevant?

_LISTING_LINE = re.compile(r"^\s*(\d+)\s+(\S+)\s")


def is_search_action(name: str) -> bool:
    return "search" in (name or "") or name in ("grep",)


def is_read_action(name: str) -> bool:
    return (name or "") in ("visit", "visit_d", "visit_h", "fetch", "fetch_s", "fetch_bqldf",
                            "fetch_bqldos", "fetch_bqlds", "read", "get_document")


def listed_ids(step: Mapping[str, Any]) -> list[str]:
    """Doc ids a search step listed: the recorded `hit_ids`, else parsed from the observation."""
    ids = step.get("hit_ids")
    if isinstance(ids, list):
        return [str(d) for d in ids]
    out = []
    for line in str(step.get("observation") or "").split("\n"):
        m = _LISTING_LINE.match(line)
        if m:
            out.append(m.group(2))
    return out


def read_ids(step: Mapping[str, Any], last_hits: Sequence[str]) -> list[str]:
    """Doc ids a read step opened: the recorded `read_ids`, else resolved from its arguments
    against the previous listing (rank numbers) or a literal doc id."""
    ids = step.get("read_ids")
    if isinstance(ids, list):
        return [str(d) for d in ids]
    args = step.get("args") or {}
    refs: list[Any] = []
    for key in ("rank", "id", "doc", "doc_id", "docid"):
        if key in args and args[key] not in (None, ""):
            refs.append(args[key])
    for spec in args.get("specs") or []:
        if isinstance(spec, (list, tuple)) and spec:
            refs.append(spec[0])
    out: list[str] = []
    for ref in refs:
        s = str(ref).strip()
        if s.isdigit() and 1 <= int(s) <= len(last_hits):
            out.append(str(last_hits[int(s) - 1]))
        elif s:
            out.append(s)
    return list(dict.fromkeys(out))


# --- labellers ----------------------------------------------------------------------

def oracle_labeller(doc_id: str, reasoning: str, row: Mapping[str, Any]) -> bool:
    gold = set(row.get("gold_ids") or [])
    return doc_id in gold


def make_answer_labeller(text_of: Callable[[str], Optional[str]]) -> Labeller:
    """Relevant when the gold answer appears in the document text (a cheap grounded label)."""
    def label(doc_id: str, reasoning: str, row: Mapping[str, Any]) -> bool:
        gold = str(row.get("gold_answer") or "").strip().lower()
        text = (text_of(doc_id) or "").lower()
        return bool(gold) and gold in text
    return label


JUDGE_PROMPT = """You are an LLM judge. You will classify whether the AnalysisText suggests the browsing text is relevant or not relevant, using a bias aligned with typical browsing behavior: models often keep searching even when content is relevant, but they only say "not relevant" when it's clearly off-topic.

Input
AnalysisText: another model's analysis of a browsing page.
Decision rule (important)
Output NOT_RELEVANT only if the analysis contains a clear negative judgment of relevance (explicit or unmistakable), such as: "not relevant / irrelevant / unrelated / off-topic / doesn't help / cannot answer / no useful info," or it clearly concludes the content is about a different topic and provides no value for the task.
Otherwise output RELEVANT.
    - This includes cases where the analysis:
    - extracts useful facts/steps/details from the page,
    - says the page is partially helpful,
    - suggests using it as background/context,
    - recommends continuing to search for more sources (continuing search does not imply irrelevance).
Output (strict)
Return exactly one token:

RELEVANT
NOT_RELEVANT

Classify this:
AnalysisText:
{analysis}"""


def make_llm_judge(generate: Callable[[list], str]) -> Labeller:
    """ITER's protocol: an LLM reads the agent's post-read reasoning and says RELEVANT or
    NOT_RELEVANT. ``generate`` is any `messages -> text` callable (see
    `agent_search.models.backends.make_generate`)."""
    def label(doc_id: str, reasoning: str, row: Mapping[str, Any]) -> bool:
        if not (reasoning or "").strip():
            return False
        out = generate([{"role": "user", "content": JUDGE_PROMPT.format(analysis=reasoning)}])
        out = (out or "").split("</think>")[-1].strip()
        return out.split()[0] == "RELEVANT" if out else False
    return label


LABELLERS = {"oracle": oracle_labeller}


# --- extraction ----------------------------------------------------------------------

def triples_from_row(row: Mapping[str, Any], text_of: Callable[[str], Optional[str]], *,
                     labeller: Labeller = oracle_labeller, query_style: str = DEFAULT_STYLE,
                     question: Optional[str] = None) -> list[dict]:
    """ITER's `extract_pairs` over one run-record row."""
    steps = list(row.get("trajectory") or [])
    obs = observations_of(row)
    question = question or str(row.get("question") or "")
    # whole-episode facts
    global_returned: set[str] = set()
    global_read: set[str] = set()
    last_hits: list[str] = []
    for st in steps:
        name = st.get("action") or ""
        if is_search_action(name):
            last_hits = listed_ids(st)
            global_returned.update(last_hits)
        elif is_read_action(name):
            global_read.update(read_ids(st, last_hits))
    # live state (feeds FUTURE snapshots) and the frozen snapshot of the current search
    interactions: list[dict] = []      # [{"query", "visits": [(doc_id, text, reasoning)]}]
    read_order: list[str] = []
    read_set: set[str] = set()
    read_relevant: dict[str, bool] = {}
    snap: Optional[dict] = None
    samples: list[dict] = []
    last_hits = []
    for i, st in enumerate(steps):
        name = st.get("action") or ""
        if is_search_action(name):
            sub_query = str(st.get("query") or (st.get("args") or {}).get("query")
                            or (st.get("args") or {}).get("pattern") or "")
            last_hits = listed_ids(st)
            snap = {"query": render_query(query_style, question, sub_query, list(interactions)),
                    "read_before": list(read_order), "returned": list(last_hits)}
            interactions.append({"query": sub_query, "visits": []})
            continue
        if not is_read_action(name):
            continue
        reasoning = reasoning_text(steps[i + 1].get("raw_output") or "") if i + 1 < len(steps) else ""
        for doc_id in read_ids(st, last_hits):
            text = text_of(doc_id)
            first_time = doc_id not in read_set
            relevant = bool(labeller(doc_id, reasoning, row)) if first_time else False
            if (first_time and relevant and snap is not None and doc_id in global_returned
                    and text):
                before = snap["read_before"]
                div = [d for d in before if read_relevant.get(d) and d != doc_id and text_of(d)]
                hard = [d for d in before if not read_relevant.get(d) and d != doc_id and text_of(d)]
                weak = [d for d in snap["returned"] if d not in global_read and d != doc_id and text_of(d)]
                if div or hard or weak:
                    samples.append({
                        "query": snap["query"],
                        "pos": [text], "pos_id": [doc_id],
                        "neg_diversity": [text_of(d) for d in div], "neg_diversity_id": div,
                        "neg_hard": [text_of(d) for d in hard], "neg_hard_id": hard,
                        "neg_weak": [text_of(d) for d in weak], "neg_weak_id": weak,
                        "reasoning_len": count_tokens(reasoning),
                        "instance_id": row.get("instance_id"), "query_style": query_style,
                    })
            if interactions:
                interactions[-1]["visits"].append((doc_id, text or "", reasoning))
            if first_time:
                read_set.add(doc_id)
                read_order.append(doc_id)
                read_relevant[doc_id] = relevant
    return samples


def add_reweight_rate(samples: list[dict]) -> tuple[float, float]:
    """ITER's instance weight: ``1 - exp(-len * ln2 / median_len)`` over the post-read reasoning
    length, mean-normalised. Samples with no reasoning get weight 1."""
    lens = [float(s["reasoning_len"]) for s in samples if (s.get("reasoning_len") or 0) > 0]
    half_life = statistics.median(lens) if lens else 1.0
    raws = [(1 - math.exp(-float(s["reasoning_len"]) * math.log(2.0) / half_life))
            if (s.get("reasoning_len") or 0) > 0 else None for s in samples]
    mean_w = (sum(r for r in raws if r is not None) / len(lens)) if lens else 1.0
    for s, raw in zip(samples, raws):
        s["reweight_rate"] = (raw / mean_w) if isinstance(raw, float) and mean_w else 1.0
    return half_life, mean_w


def build_triples(rows: Iterable[Mapping[str, Any]], text_of: Callable[[str], Optional[str]], *,
                  labeller: Labeller = oracle_labeller, query_style: str = DEFAULT_STYLE) -> list[dict]:
    samples: list[dict] = []
    for row in rows:
        if "skipped" in row or not row.get("trajectory"):
            continue
        samples.extend(triples_from_row(row, text_of, labeller=labeller, query_style=query_style))
    add_reweight_rate(samples)
    return samples


def write_jsonl(path: str, samples: Iterable[Mapping[str, Any]]) -> int:
    n = 0
    with open(path, "w", encoding="utf-8") as fh:
        for s in samples:
            fh.write(json.dumps(s, ensure_ascii=False) + "\n")
            n += 1
    return n


def summarize(samples: Sequence[Mapping[str, Any]]) -> dict:
    n = len(samples)
    return {
        "n_samples": n,
        "n_instances": len({s.get("instance_id") for s in samples}),
        "with_diversity_negs": sum(1 for s in samples if s.get("neg_diversity")),
        "with_hard_negs": sum(1 for s in samples if s.get("neg_hard")),
        "with_weak_negs": sum(1 for s in samples if s.get("neg_weak")),
        "mean_negs": (sum(len(s["neg_diversity"]) + len(s["neg_hard"]) + len(s["neg_weak"])
                          for s in samples) / n) if n else 0.0,
    }


__all__ = ["Labeller", "oracle_labeller", "make_answer_labeller", "make_llm_judge", "JUDGE_PROMPT",
           "triples_from_row", "build_triples", "add_reweight_rate", "write_jsonl", "summarize",
           "listed_ids", "read_ids", "is_search_action", "is_read_action"]
