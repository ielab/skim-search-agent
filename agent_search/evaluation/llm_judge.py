"""LLM-as-judge for deep-research answers: the BrowseComp / BrowseComp-Plus grading protocol.

`agent_search/evaluation/doc_scoring.py` grades a doc answer with exact-match plus token-F1,
gated on grounding. That is strict: a correct answer wrapped in a sentence ("Mike Medavoy, ...
founded the company") scores 0 even though it is right. The BrowseComp and BrowseComp-Plus
benchmarks do not grade that way. They use an LLM judge (the "BrowseComp Appendix F" prompt)
that extracts the final answer from the response and decides whether it matches the gold. This
module reproduces that judge prompt exactly, so our deep-research accuracy is comparable to the
benchmark's own metric, not an over-strict proxy.

This grading step is post-hoc and optional: the core eval stays deterministic and API-free
(`doc_scoring.py`). This module grades a finished run's `rows.jsonl`. It adds `judge_correct`
per row and writes `judge_summary.json` (accuracy). The judge model is a free parameter
(`--judge-model`, default a cheap one), so graders can be swapped for an ablation. A cheap
model is a fine judge here: the task is a yes/no answer-equivalence check, not generation.

    # judge on the OpenAI API (cheap):
    python -m agent_search.evaluation.llm_judge --results-dir runs/... --judge-model gpt-4o-mini --dataset musique_structured
    # judge on a served vLLM (cluster, no API key):  vllm serve <judge_model> --port 8001  then:
    python -m agent_search.evaluation.llm_judge --results-dir runs/... --judge-model <judge_model> \
        --judge-api-base http://localhost:8001/v1 --dataset musique_structured
"""
from __future__ import annotations

import argparse
import json
import os
import time
import sys
import re
from pathlib import Path
from typing import Callable, Optional

# The BrowseComp Appendix F judge prompt, verbatim: matches the BrowseComp-Plus benchmark's own
# judge prompt and the gate this project runs via `scripts/judge_cells.py`. Do not paraphrase:
# fidelity to this exact prompt is what makes the resulting accuracy comparable to the published
# benchmark number instead of a home-grown proxy.
BCP_JUDGE_PROMPT = """Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.

[question]: {question}

[response]: {response}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response]. Put the extracted answer as 'None' if there is no exact, final answer to extract from the response.

[correct_answer]: {correct_answer}

reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], focusing only on if there are meaningful differences between [correct_answer] and the extracted_final_answer. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers match.

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for numerical problems. Answer 'no' otherwise, i.e. if there if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect.

confidence: The extracted confidence score between 0% and 100% from [response]. Put 100 if there is no confidence score available.

Return ONLY a compact JSON object with keys: extracted_final_answer, reasoning, correct, confidence."""


def _parse_judge_json(raw: str) -> dict:
    """The judge is asked for a JSON object; stay forgiving if a model wraps it in prose."""
    try:
        return json.loads(raw)
    except Exception:  # noqa: BLE001
        m = re.search(r"\{.*\}", raw or "", re.DOTALL)
        try:
            return json.loads(m.group(0)) if m else {}
        except Exception:  # noqa: BLE001
            return {}


def _norm_exact(s: str) -> str:
    """Conservative normalization for the exact-match short-circuit: strip whitespace, surrounding
    markdown bold/italic, and trailing punctuation, then casefold. Interior differences escalate to
    the judge, following RISE's short-circuit: a byte-exact answer never needs an LLM call."""
    s = (s or "").strip()
    prev = None
    while prev != s:
        prev = s
        if len(s) >= 4 and s.startswith("**") and s.endswith("**"):
            s = s[2:-2].strip()
        elif len(s) >= 2 and s.startswith("*") and s.endswith("*"):
            s = s[1:-1].strip()
        while s and s[-1] in ".,;:":
            s = s[:-1].rstrip()
    return s.casefold()


# Default judge model: a cheap OpenAI model is sufficient (yes/no equivalence check, not
# generation). Configurable via the `model` param of `judge_answer` or this env var.
# DIVER's judge template (DIVER/scripts_evaluation/evaluate.py JUDGE_TEMPLATE, verbatim). It is the
# Appendix F prompt with two differences: the [correct_answer] line sits after the response, and the
# reasoning clause allows string variations and a more precise or verbose answer. DIVER grades with
# a served Qwen3-30B-A3B-Thinking-2507 at temperature 0 and reads the verdict off a "correct: yes/no"
# line after the think block (`parse_judge_result`), so this prompt is paired with `_parse_verdict_line`.
DIVER_JUDGE_PROMPT = """Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.

[question]: {question}

[response]: {response}

[correct_answer]: {correct_answer}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response]. 

[correct_answer]: Repeat the [correct_answer] given above.

reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], in the context of this [question]. You should judge whether the extracted_final_answer is semantically equivalent to [correct_answer], allowing the extracted_final_answer to be string variations of [correct_answer]. You should also allow the extracted_final_answer to be more precise or verbose than [correct_answer], as long as its additional details are correct. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers are semantically equivalent.

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for numerical problems. Answer 'no' otherwise, i.e. if there if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect.


confidence: The extracted confidence score between 0|\%| and 100|\%| from [response]. Put 100 if there is no confidence score available."""

JUDGE_PROMPTS = {"bcp": None, "diver": DIVER_JUDGE_PROMPT}   # None = BCP_JUDGE_PROMPT below


def _parse_verdict_line(raw: str) -> Optional[dict]:
    """The verdict of a free-text judge reply: the last `correct: yes|no` after any think block."""
    text = re.sub(r"<think>.*?</think>", "", raw or "", flags=re.DOTALL)
    m = list(re.finditer(r"correct:\s*\**\s*(yes|no)", text, re.IGNORECASE))
    if not m:
        return None
    ex = re.search(r"extracted_final_answer:\s*(.*)", text)
    reason = re.search(r"reasoning:\s*(.*)", text)
    return {"correct": m[-1].group(1).lower(), "extracted_final_answer": (ex.group(1).strip() if ex else "None"),
            "reasoning": (reason.group(1).strip() if reason else "")}


DEFAULT_JUDGE_MODEL = os.environ.get("LLM_JUDGE_MODEL", "gpt-4o-mini")


def judge_answer_detail(question: str, gold_answer: str, response: str,
                        generate: Callable[[str], str], *, short_circuit: bool = True,
                        prompt: str = "bcp") -> dict:
    """Grade one (question, gold, response) triple the BrowseComp-Plus way, with full judge detail.

    `generate(prompt) -> str` returns the judge's JSON text; it is injectable for tests and
    backend-agnostic. Returns {judge_correct: bool, judge_extracted: str, judge_reasoning: str}.
    An empty response is 'no' with no call; a normalized exact match is 'yes' with no call
    (cost saver)."""
    resp = (response or "").strip()
    if not resp:
        return {"judge_correct": False, "judge_extracted": "None", "judge_reasoning": "no answer"}
    if short_circuit and gold_answer and _norm_exact(resp) == _norm_exact(gold_answer):
        return {"judge_correct": True, "judge_extracted": resp,
                "judge_reasoning": "exact match after normalization (no judge call)"}
    template = JUDGE_PROMPTS.get(prompt) or BCP_JUDGE_PROMPT
    text = template.format(question=question or "", response=resp, correct_answer=gold_answer or "")
    raw = generate(text)
    data = _parse_judge_json(raw) if prompt == "bcp" else _parse_verdict_line(raw)
    if not data or "correct" not in data:
        # An unparseable judge reply is an error, not a "no": recording it as wrong would
        # silently deflate accuracy. The row keeps the raw reply so it can be re-judged.
        return {"judge_correct": None, "judge_error": "unparseable judge reply",
                "judge_raw": (raw or "")[:], "judge_extracted": "None", "judge_reasoning": ""}
    return {"judge_correct": str(data.get("correct", "no")).strip().lower() == "yes",
            "judge_extracted": data.get("extracted_final_answer", "None"),
            "judge_reasoning": data.get("reasoning", "")}


def judge_answer(question: str, gold_answer: str, response: str, *,
                 model: str = DEFAULT_JUDGE_MODEL,
                 generate: Optional[Callable[[str], str]] = None,
                 api_base: Optional[str] = None, short_circuit: bool = True) -> float:
    """The public entry point: grade one (question, gold_answer, predicted_answer) triple the
    BrowseComp / BrowseComp-Plus way and return 1.0 (correct) or 0.0 (incorrect), the paper's
    LLM-as-judge protocol for this benchmark (see module docstring).

    `model` (or the `LLM_JUDGE_MODEL` env var) selects the judge; default is a cheap OpenAI
    model (gpt-4o-mini). `generate` is injectable (e.g. for tests, to avoid a real API call);
    when omitted, a judge client is built for `model` via `make_judge` (routes to the OpenAI
    API using OPENAI_API_KEY, or another backend by model name; see `make_judge`)."""
    gen = generate if generate is not None else make_judge(model, api_base=api_base)
    detail = judge_answer_detail(question, gold_answer, response, gen, short_circuit=short_circuit)
    return 1.0 if detail["judge_correct"] else 0.0


def make_judge(model: str = "gpt-4o-mini", *, api_base: Optional[str] = None,
               client=None, max_tokens: int = 512, json_mode: bool = True) -> Callable[[str], str]:
    """A judge `generate(prompt) -> str` (JSON object, temperature 0) that auto-routes by model
    name, the same way the agent backend does (`agent_search.agent.backbone.make_generate`). There is no
    separate "gpt-based vs vLLM-based" setting: the model name decides the endpoint.
      - an OpenAI model (gpt-*/o-*/chatgpt-*) -> the OpenAI API (key from OPENAI_API_KEY).
      - a Gemini model (gemini-*) -> Gemini's OpenAI-compatible endpoint (key from GEMINI_API_KEY).
      - anything else -> a served OpenAI-compatible endpoint at `api_base` (a `vllm serve <model>`),
        so the same cluster that serves the agent can serve the judge; pass the run's own api_base.
    `client` is injectable for offline tests. JSON mode uses the server's guided decoding when
    available and falls back to a plain call otherwise (the parser is forgiving; this also covers
    Gemini, whose OpenAI-compat endpoint may not support `response_format` the same way)."""
    if client is None:
        from openai import OpenAI

        from agent_search.agent.backbone import (
            _GEMINI_BASE_URL, _OPENAI_BASE_URL, is_gemini_model, is_openai_model)
        if is_openai_model(model):
            client = OpenAI(base_url=_OPENAI_BASE_URL,
                            api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"))
        elif is_gemini_model(model):
            client = OpenAI(base_url=_GEMINI_BASE_URL,
                            api_key=os.environ.get("GEMINI_API_KEY", "EMPTY"))
        elif api_base:
            client = OpenAI(base_url=api_base, api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"))
        else:
            raise ValueError(
                f"judge model {model!r} is not an OpenAI or Gemini model, so it needs a served "
                f"endpoint: pass api_base (e.g. http://localhost:8001/v1 for `vllm serve {model}`).")

    def generate(prompt: str) -> str:
        kw = dict(model=model, messages=[{"role": "user", "content": prompt}],
                  temperature=0.0, max_tokens=max_tokens)
        if not json_mode:                               # a free-text judge (DIVER's thinking judge)
            resp = client.chat.completions.create(**kw)
            msg = resp.choices[0].message
            text = msg.content or ""
            if not text.strip():                        # a reasoning parser may hold the whole reply
                text = str(getattr(msg, "reasoning_content", None) or getattr(msg, "reasoning", None) or "")
            return text
        try:                                            # JSON mode (OpenAI + vLLM guided decoding)
            resp = client.chat.completions.create(response_format={"type": "json_object"}, **kw)
        except Exception:                               # noqa: BLE001 - server without JSON mode: plain call
            resp = client.chat.completions.create(**kw)
        return resp.choices[0].message.content or "{}"

    return generate


def _questions_from_dataset(dataset: Optional[str]) -> dict:
    """{instance_id: problem_statement}, for rows.jsonl rows that carry no `question` field."""
    if not dataset:
        return {}
    from agent_search.evaluation.datasets import load_dataset_by_name
    return {i.instance_id: (i.problem_statement or "") for i in load_dataset_by_name(dataset)}


def judge_run_dir(results_dir: str, generate: Callable[[str], str], *,
                  judge_model: str = "gpt-4o-mini", dataset: Optional[str] = None,
                  force: bool = False, unfinished_ok: bool = False,
                  prompt: str = "bcp", tag: str = "", workers: int = 1) -> dict:
    """Grade every answered doc row in <results_dir>/rows.jsonl (add `judge_correct` in place, write
    judge_summary.json, return the summary). The question comes from the row (`question`) or, for
    older rows that lack it, from `dataset` by instance_id. Rows with no `gold_answer` (e.g. the code
    arm) are skipped: this metric is doc-QA only.

    Idempotent: a row that already carries a verdict (`judge_correct` is True/False) is not
    re-judged unless ``force=True``; rows whose earlier judge reply was unparseable
    (`judge_correct` is None) are retried. Rows stream through one at a time, so memory stays
    flat however large the trajectories are. rows.jsonl is rewritten atomically (temp file +
    rename) so an interrupt can never truncate the run's only durable artifact.

    `prompt` picks the judge template (`bcp`, the default, or `diver`); `tag` keeps a second judge's
    verdicts next to the first (fields `judge_correct_<tag>` and friends, summary
    `judge_summary_<tag>.json`); `workers` grades that many rows at once, in chunks, so a slow
    served judge is not called one row at a time."""
    rd = Path(results_dir)
    rows_path = rd / "rows.jsonl"
    if not (rd / "results.json").exists() and not unfinished_ok:
        # results.json is written when a run ends; judging while rows.jsonl is still being
        # appended would rewrite it and drop the rows written after the read
        raise SystemExit(f"{results_dir} has no results.json: the run is still writing (or was cut off). "
                         "Wait for it, resume it, or pass unfinished_ok=True (--unfinished-ok).")
    q_by_id = _questions_from_dataset(dataset)
    # One row at a time: a run that reads whole documents (autoread, DCI) keeps every read in its
    # trajectory, and its rows.jsonl runs to tens of gigabytes. Loading it whole would take the
    # judge host down, so rows stream from the file into a temp file that replaces it at the end.
    sfx = f"_{tag}" if tag else ""
    key, err_key = f"judge_correct{sfx}", f"judge_error{sfx}"

    def _grade(r: dict) -> dict:
        question = r.get("question") or q_by_id.get(r.get("instance_id"), "")
        d = judge_answer_detail(question, r.get("gold_answer", ""), r.get("final_answer", ""), generate, prompt=prompt)
        return {k + sfx: v for k, v in d.items()} if sfx else d

    n_new = n_graded = n_correct = n_errored = 0
    tmp = rows_path.with_suffix(f".jsonl.tmp.{os.getpid()}")
    chunk = max(1, int(workers)) * 4
    with open(rows_path, encoding="utf-8") as src, open(tmp, "w", encoding="utf-8") as dst:
        pending: list = []

        def _flush() -> None:
            nonlocal n_new, n_graded, n_correct, n_errored
            t0 = time.monotonic()
            todo = [i for i, r in enumerate(pending) if "gold_answer" in r and (force or r.get(key) is None)]
            if todo:
                if workers > 1:
                    from concurrent.futures import ThreadPoolExecutor
                    with ThreadPoolExecutor(max_workers=int(workers)) as ex:
                        results = list(ex.map(lambda i: _grade(pending[i]), todo))
                else:
                    results = [_grade(pending[i]) for i in todo]
                for i, d in zip(todo, results):
                    pending[i].update(d)
                n_new += len(todo)
            for r in pending:
                if r.get(key) is not None:
                    n_graded += 1
                    n_correct += bool(r.get(key))
                elif err_key in r:
                    n_errored += 1
                dst.write(json.dumps(r, ensure_ascii=False) + "\n")
            dst.flush()
            if todo:
                print(f"[judge] {n_graded} graded so far ({n_correct} correct, {n_errored} errors); "
                      f"chunk of {len(todo)} took {time.monotonic() - t0:.0f}s", file=sys.stderr, flush=True)
            pending.clear()

        for line in src:
            if not line.strip():
                continue
            pending.append(json.loads(line))
            if len(pending) >= chunk:
                _flush()
        _flush()
    if n_new:
        os.replace(tmp, rows_path)
    else:
        os.remove(tmp)
    summary = {"judge_model": judge_model,
               "judge_prompt": ("BrowseComp Appendix F (verbatim)" if prompt == "bcp" else "DIVER evaluate.py JUDGE_TEMPLATE (verbatim)"),
               "n_judged": n_graded, "n_correct": n_correct, "n_judge_errors": n_errored,
               "n_newly_judged": n_new,
               "judge_accuracy": (n_correct / n_graded) if n_graded else 0.0}
    out = rd / f"judge_summary{sfx}.json"
    tmp = out.with_suffix(f".json.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(summary, indent=2))
    os.replace(tmp, out)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="LLM-as-judge (BrowseComp-Plus protocol) over a run dir.")
    ap.add_argument("--results-dir", required=True, help="a run dir containing rows.jsonl")
    ap.add_argument("--judge-model", default="gpt-4o-mini", help="grader model (cheap is fine)")
    ap.add_argument("--judge-api-base", default=None,
                    help="OpenAI-compatible endpoint for the judge, e.g. http://localhost:8000/v1 "
                         "for a served vLLM on the cluster (default: the OpenAI API via OPENAI_API_KEY)")
    ap.add_argument("--dataset", default=None,
                    help="load questions from this dataset for rows that lack a `question` field")
    ap.add_argument("--judge-prompt", default="bcp", choices=sorted(JUDGE_PROMPTS),
                    help="bcp = BrowseComp Appendix F, JSON verdict (default); diver = DIVER's template, verdict line")
    ap.add_argument("--tag", default="", help="keep these verdicts next to the default judge's: judge_correct_<tag>, judge_summary_<tag>.json")
    ap.add_argument("--workers", type=int, default=1, help="rows graded at once")
    ap.add_argument("--max-tokens", type=int, default=None,
                    help="judge generation budget (default 512; a thinking judge needs 8192, DIVER's setting)")
    ap.add_argument("--unfinished-ok", action="store_true",
                    help="judge a run directory that has no results.json yet (the run is still writing or was cut off)")
    a = ap.parse_args()
    gen = make_judge(a.judge_model, api_base=a.judge_api_base,
                     max_tokens=a.max_tokens or (8192 if a.judge_prompt == "diver" else 512),
                     json_mode=(a.judge_prompt == "bcp"))
    s = judge_run_dir(a.results_dir, gen, judge_model=a.judge_model, dataset=a.dataset, unfinished_ok=a.unfinished_ok,
                      prompt=a.judge_prompt, tag=a.tag, workers=a.workers)
    print(f"judge={a.judge_model}  accuracy {s['judge_accuracy'] * 100:.1f}% "
          f"({s['n_correct']}/{s['n_judged']})  ->  {a.results_dir}/judge_summary{('_' + a.tag) if a.tag else ''}.json")


if __name__ == "__main__":
    main()
