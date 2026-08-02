#!/usr/bin/env python3
"""Stage the classic multi-hop QA benchmarks (HotpotQA / 2WikiMultiHopQA / MuSiQue) as
the BEIR-style shared-corpus layout the repo's ``general`` (deep-research) datasets read,
so they drop straight into ``agent_research_tools`` / ``agent_research_tools_bql``.

Default source is HuggingFace (reliable; the cluster already uses HF for everything
else). One invocation stages all three:

    python scripts/stage_multihop.py                     # -> data/hotpotqa, data/2wiki, data/musique
    python scripts/stage_multihop.py --only hotpotqa     # just one

Output under ``data/<name>/``:
    corpus.jsonl     {"_id","title","text"}     # shared paragraph corpus (deduped by title)
    queries.jsonl    {"_id","text","answer"}    # questions (+ gold answer for EM/F1)
    qrels/test.tsv   query-id <TAB> corpus-id <TAB> score   # supporting paragraphs = gold

The corpus is the UNION of the candidate paragraphs across the split (the fixed-corpus
"distractor" setting), qrels are the supporting paragraphs — so evidence Recall/nDCG@k is
well defined and BQL can scope over the ``title`` field.

OFFLINE RULE: run this where there is internet (login node or your laptop) — the GPU node
then reads the staged files with no network. If a HF id/split below is wrong for your
mirror, override it (``--hf-id`` / ``--config`` / ``--split`` with ``--only``), or download
the raw release yourself and pass ``--input`` (a JSON array or JSONL file; the schema is
auto-detected). Then copy ``data/`` to storage the GPU node can read.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

# Best-known HuggingFace sources. Override per dataset with --hf-id/--config/--split when
# using --only, in case a mirror differs. HotpotQA's `hotpot_qa` is canonical; the 2wiki /
# musique mirrors are community uploads, so they're the likeliest to need an override.
HF_SOURCES = {
    "hotpotqa": {"path": "hotpot_qa", "config": "distractor", "split": "validation"},
    # parquet repackaging with HotpotQA-compatible context/supporting_facts (the original
    # xanhho/2WikiMultihopQA is a loading SCRIPT, which recent `datasets` rejects).
    "2wiki":    {"path": "framolfese/2WikiMultihopQA", "config": None, "split": "validation"},
    "musique":  {"path": "dgslibisey/MuSiQue", "config": None, "split": "validation"},
}


# --- read sources -----------------------------------------------------------

def load_examples_file(path: str) -> list[dict]:
    """Read a raw release file: a JSON array (HotpotQA/2Wiki) or JSONL (MuSiQue)."""
    with open(path, encoding="utf-8") as fh:
        raw = fh.read().strip()
    try:
        data = json.loads(raw)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            return data.get("data") or [data]
    except json.JSONDecodeError:
        pass
    return [json.loads(line) for line in raw.splitlines() if line.strip()]


def load_examples_hf(name: str, hf_id=None, config=None, split=None) -> list[dict]:
    """Pull a split from HuggingFace as a list of row dicts."""
    from datasets import load_dataset  # heavy; only needed on the staging node
    src = dict(HF_SOURCES[name])
    if hf_id:
        src["path"] = hf_id
    if config is not None:
        src["config"] = config
    if split is not None:
        src["split"] = split
    # No trust_remote_code / loading scripts: recent `datasets` only accepts standard
    # formats (parquet/json), which all the sources below are.
    ds = load_dataset(src["path"], src["config"], split=src["split"])
    return list(ds)


# --- normalize any schema -> a uniform shape --------------------------------

def _normalize(ex: dict) -> tuple:
    """One release row (raw OR HuggingFace schema) ->
    (qid, question, answer, [(title, text), ...], [gold_title, ...]).

    Handles MuSiQue's `paragraphs`, HotpotQA/2Wiki's `context` in either the raw
    list-of-[title, sentences] form or HF's dict-of-parallel-lists form."""
    qid = str(ex.get("_id") or ex.get("id") or ex.get("question_id") or "")
    question = ex.get("question", "")
    ans = ex.get("answer")
    if ans is None and ex.get("answers"):                 # some mirrors use `answers`
        ans = ex["answers"][0] if isinstance(ex["answers"], list) else ex["answers"]
    answer = str(ans or "")

    if "paragraphs" in ex:                                # MuSiQue
        paras = [(p.get("title", ""), p.get("paragraph_text", "")) for p in ex["paragraphs"]]
        gold = [p.get("title", "") for p in ex["paragraphs"] if p.get("is_supporting")]
        return qid, question, answer, paras, gold

    ctx = ex.get("context")
    sf = ex.get("supporting_facts")
    if isinstance(ctx, dict):                             # HF dict-of-lists
        titles, sents = ctx.get("title", []), ctx.get("sentences", [])
        paras = [(titles[i], " ".join(sents[i])) for i in range(len(titles))]
    else:                                                 # raw [[title, [sent, ...]], ...]
        paras = [(t, " ".join(s)) for t, s in (ctx or [])]
    if isinstance(sf, dict):                              # HF dict-of-lists
        gold = list(dict.fromkeys(sf.get("title", [])))
    else:                                                 # raw [[title, sent_id], ...]
        gold = list(dict.fromkeys(t for t, _ in (sf or [])))
    return qid, question, answer, paras, gold


# --- build the BEIR corpus --------------------------------------------------

class _Corpus:
    """Deduped paragraph corpus, one stable doc_id per title (supporting-fact labels in
    all three datasets reference titles, so title-keying keeps qrels and corpus aligned;
    the first paragraph seen for a title wins — they are the canonical intro paragraph)."""

    def __init__(self) -> None:
        self.docs: dict[str, dict] = {}
        self._by_title: dict[str, str] = {}

    def add(self, title: str, text: str) -> str:
        title = (title or "").strip()
        if title in self._by_title:
            return self._by_title[title]
        base = "d_" + (re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")[:60] or "doc")
        doc_id, i = base, 1
        while doc_id in self.docs:
            i += 1
            doc_id = f"{base}_{i}"
        self.docs[doc_id] = {"title": title, "text": (text or "").strip()}
        self._by_title[title] = doc_id
        return doc_id


def build_beir(examples: list[dict]) -> tuple[dict, dict, set]:
    """release rows -> (corpus{doc_id:{title,text}}, queries{qid:{...}}, qrels{(qid,doc_id)})."""
    corpus = _Corpus()
    queries: dict[str, dict] = {}
    qrels: set = set()
    for ex in examples:
        qid, question, answer, paras, gold = _normalize(ex)
        if not qid:
            continue
        for title, text in paras:
            corpus.add(title, text)
        for title in gold:
            qrels.add((qid, corpus.add(title, "")))
        queries[qid] = {"_id": qid, "text": question, "answer": answer}
    return corpus.docs, queries, qrels


def write_beir(out_dir: str, corpus: dict, queries: dict, qrels: set) -> None:
    os.makedirs(os.path.join(out_dir, "qrels"), exist_ok=True)
    with open(os.path.join(out_dir, "corpus.jsonl"), "w", encoding="utf-8") as fh:
        for doc_id, doc in corpus.items():
            fh.write(json.dumps({"_id": doc_id, **doc}, ensure_ascii=False) + "\n")
    with open(os.path.join(out_dir, "queries.jsonl"), "w", encoding="utf-8") as fh:
        for q in queries.values():
            fh.write(json.dumps(q, ensure_ascii=False) + "\n")
    with open(os.path.join(out_dir, "qrels", "test.tsv"), "w", encoding="utf-8") as fh:
        fh.write("query-id\tcorpus-id\tscore\n")          # BEIR header
        for qid, doc_id in sorted(qrels):
            fh.write(f"{qid}\t{doc_id}\t1\n")


# --- driver -----------------------------------------------------------------

def stage_one(name: str, out_root: str, input_path=None,
              hf_id=None, config=None, split=None) -> bool:
    try:
        examples = (load_examples_file(input_path) if input_path
                    else load_examples_hf(name, hf_id, config, split))
    except Exception as e:  # noqa: BLE001 — surface a clear, actionable message
        src = HF_SOURCES[name]
        print(f"  [error] {name}: could not load ({type(e).__name__}: {e}).\n"
              f"          default HF source: {src['path']} (config={src['config']}, "
              f"split={src['split']}). Override with --only {name} --hf-id <id> "
              f"[--config <c>] [--split <s>], or download the raw file and pass --input.",
              file=sys.stderr)
        return False
    corpus, queries, qrels = build_beir(examples)
    if not (corpus and queries and qrels):
        print(f"  [error] {name}: produced empty output (corpus={len(corpus)} "
              f"queries={len(queries)} qrels={len(qrels)}) — unexpected schema for "
              f"this source.", file=sys.stderr)
        return False
    out_dir = os.path.join(out_root, name)
    write_beir(out_dir, corpus, queries, qrels)
    print(f"  {name}: {len(queries)} queries, {len(corpus)} corpus docs, "
          f"{len(qrels)} qrels -> {out_dir}/")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", choices=list(HF_SOURCES),
                    help="stage just one dataset (default: all three)")
    ap.add_argument("--out-root", default="data", help="parent dir (default: data/)")
    ap.add_argument("--input", help="local raw release file instead of HF (use with --only)")
    ap.add_argument("--hf-id", help="override the HuggingFace dataset id (use with --only)")
    ap.add_argument("--config", help="override the HF config name (use with --only)")
    ap.add_argument("--split", help="override the HF split (use with --only)")
    a = ap.parse_args()

    if a.input and not a.only:
        ap.error("--input requires --only <dataset> (it stages a single file)")
    names = [a.only] if a.only else list(HF_SOURCES)

    print(f"staging {', '.join(names)} -> {a.out_root}/  "
          f"(source: {'local file' if a.input else 'HuggingFace'})")
    ok = 0
    for name in names:
        if stage_one(name, a.out_root, input_path=a.input, hf_id=a.hf_id,
                     config=a.config, split=a.split):
            ok += 1
    print(f"done: {ok}/{len(names)} staged. They are auto-registered as "
          f"--dataset {', '.join(names)} (general domain).")
    return 0 if ok == len(names) else 1


if __name__ == "__main__":
    raise SystemExit(main())
