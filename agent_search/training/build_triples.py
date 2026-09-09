"""`skimsearchagent-build-triples`: run records -> retriever training triples.

    skimsearchagent-build-triples --runs runs/paper/agent/hotpotqa_structured/gpt-4o-mini/agent_research_snip \\
        --dataset hotpotqa_structured --out train_data/hotpotqa_i2.jsonl --query-style i2 --labeller oracle

`--runs` takes any number of run directories (each holds rows.jsonl) or rows.jsonl files. The
dataset supplies the document texts (positives and negatives are stored as full text, the form
FlagEmbedding reads). Labellers: `oracle` (gold document ids from the run record), `answer`
(the gold answer appears in the document), or `judge:<model>` (ITER's LLM judge over the agent's
post-read reasoning; the model is routed like any other, see agent_search.models.backends).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from agent_search.training import triples as T
from agent_search.training.queries import DEFAULT_STYLE, STYLES


def _rows(paths: list[str]):
    for p in paths:
        path = Path(p)
        f = path / "rows.jsonl" if path.is_dir() else path
        if not f.exists():
            print(f"warning: no rows.jsonl at {p}", file=sys.stderr)
            continue
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def corpus_text_lookup(dataset: str):
    """doc_id -> 'title\\nbody' for the dataset's shared corpus."""
    from agent_search.corpus.units import units_from_documents
    from agent_search.evaluation.datasets import load_dataset_by_name
    instances = load_dataset_by_name(dataset)
    if not instances:
        raise SystemExit(f"dataset {dataset!r} is empty")
    first = instances[0]
    if first.docstore is not None:
        # an on-disk corpus: look documents up by id, never load it
        import functools

        @functools.lru_cache(maxsize=65536)
        def text_of(doc_id):
            u = first.docstore.unit(str(doc_id))
            return "\n".join(p for p in (u.title, u.body or u.code) if p) if u else None
        return text_of
    if first.docs is None:
        raise SystemExit(f"dataset {dataset!r} has no shared document corpus")
    units = units_from_documents(first.docs)
    by_id = {u.doc_id: "\n".join(p for p in (u.title, u.body or u.code) if p) for u in units}
    return by_id.get


def make_labeller(spec: str, text_of) -> T.Labeller:
    if spec == "oracle":
        return T.oracle_labeller
    if spec == "answer":
        return T.make_answer_labeller(text_of)
    if spec.startswith("judge:"):
        from agent_search.models.backends import make_generate
        model = spec.split(":", 1)[1]
        return T.make_llm_judge(make_generate(model=model, backend="api"))
    raise SystemExit(f"unknown labeller {spec!r}: oracle | answer | judge:<model>")


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="skimsearchagent-build-triples", description=__doc__.split("\n\n")[0])
    ap.add_argument("--runs", nargs="+", required=True, help="run directories or rows.jsonl files")
    ap.add_argument("--dataset", required=True, help="registered dataset that supplies the document texts")
    ap.add_argument("--out", required=True, help="output jsonl")
    ap.add_argument("--query-style", default=DEFAULT_STYLE, choices=STYLES)
    ap.add_argument("--labeller", default="oracle", help="oracle | answer | judge:<model>")
    a = ap.parse_args(argv)
    text_of = corpus_text_lookup(a.dataset)
    labeller = make_labeller(a.labeller, text_of)
    samples = T.build_triples(_rows(a.runs), text_of, labeller=labeller, query_style=a.query_style)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    n = T.write_jsonl(a.out, samples)
    summary = T.summarize(samples)
    summary.update({"out": a.out, "query_style": a.query_style, "labeller": a.labeller, "written": n})
    Path(a.out).with_suffix(".summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0 if n else 1


if __name__ == "__main__":
    sys.exit(main())
