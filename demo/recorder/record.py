"""Record demo episodes: gpt-4o-mini drives the library's REAL search/fetch workspace.

The tools, Boolean executor, result cards, and snippets are the library's own
(`DocSearchFetch` over a `StructuralExecutor`); only the outer chat loop lives here, kept
deliberately tiny so the demo is readable end to end.

    OPENAI_API_KEY=... python demo/recorder/record.py          # runs all demo questions, saves episodes.jsonl
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# THE PIPELINE SETTINGS — the same knobs (and values) the paper's `research_snip` condition
# runs under; run.py exposes the same names. Change them here and they flow through
# build_web.py into both demo UIs automatically.
SETTINGS = {
    "strategy": "sieve_bm25 (research_snip)",
    "model": "gpt-4o-mini",
    "k": 5,                        # results per search (paper: k=5)
    "max_section_tokens": 12000,   # per-fetch read ceiling (paper: 12,000)
    "max_steps": 20,               # demo budget (paper: 100) — BrowseComp questions need headroom
    "snippets": True,              # result-card snippets on (the _snip arm)
    "engine": "BQL reference executor",
}
import os                                              # noqa: E402
os.environ.setdefault("MAX_VISIT_TOKENS", str(SETTINGS["max_section_tokens"]))
os.environ.setdefault("MAX_SECTION_TOKENS", str(SETTINGS["max_section_tokens"]))

from demo.recorder.browsecomp_corpus import CORPUS, QUESTIONS  # noqa: E402
from agent_search.agent.tools.doc_research import DocSearchFetch  # noqa: E402

SYSTEM = """You are a research agent answering a question over a document collection you can
only access through two tools. Reply with EXACTLY ONE tool call per turn, nothing else:

search: <query>       — Boolean search. Plain terms rank; operators narrow: AND(a, b), OR(a, b),
                        NOT(x), IN(title, term), IN(section, term), IN(body, "a phrase"), PREFIX(st).
fetch: <doc> :: <section>   — read ONE named section of a listed document (use ids/headings from
                              the result cards).
answer: <final answer>      — a short answer span, only once you have read supporting text.

Search first, skim the result cards, fetch only what you need, then answer.
Prefer 2-4 plain keywords per search; add operators only to NARROW once plain terms work.
If a search errors or returns nothing, SIMPLIFY the query — never repeat a failing query."""


def run_episode(client, question: str, max_steps: int = SETTINGS["max_steps"]) -> dict:
    ws = DocSearchFetch(CORPUS, snippets=True)
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"Question: {question}"}]
    traj = []
    for _ in range(max_steps):
        out = client.chat.completions.create(model=SETTINGS["model"], messages=msgs,
                                             temperature=0.2, max_tokens=200)
        text = (out.choices[0].message.content or "").strip()
        m = re.match(r"(search|fetch|answer)\s*:\s*(.+)", text, re.S | re.I)
        if not m:
            obs = "Reply with exactly one tool call: search: / fetch: / answer:"
            traj.append({"action": "malformed", "args": {"raw": text[:120]}, "observation": obs})
        else:
            act, arg = m.group(1).lower(), m.group(2).strip()
            if act == "answer":
                traj.append({"action": "answer", "args": {"answer": arg}, "observation": ""})
                return {"question": question, "final_answer": arg, "trajectory": traj,
                        "llm_calls": len(traj)}
            if act == "search":
                obs = ws.search(arg, k=SETTINGS["k"])
                traj.append({"action": "search", "args": {"query": arg}, "observation": obs})
            else:
                doc, _, section = arg.partition("::")
                obs = ws.fetch([[doc.strip(), section.strip()]])
                traj.append({"action": "fetch",
                             "args": {"doc": doc.strip(), "section": section.strip()},
                             "observation": obs})
        msgs.append({"role": "assistant", "content": text})
        msgs.append({"role": "user", "content": f"Observation:\n{traj[-1]['observation']}"})
    # Budget exhausted without an answer — mirror the library's own forced-answer nudge
    # (agent_search/agent/loop.py's reserved-final-turn elicitation): one last call with the
    # tools disabled, so a hard question ends with a best guess instead of an empty answer.
    msgs.append({"role": "user", "content":
                 "Observation:\nSTEP BUDGET REACHED — do NOT search or fetch again. Give your "
                 "single best-effort answer NOW as: answer: <your answer>. A best guess scores "
                 "better than an empty answer."})
    out = client.chat.completions.create(model=SETTINGS["model"], messages=msgs,
                                         temperature=0.2, max_tokens=300)
    text = (out.choices[0].message.content or "").strip()
    m = re.match(r"answer\s*:\s*(.+)", text, re.S | re.I)
    final = (m.group(1) if m else text).strip()
    traj.append({"action": "answer", "args": {"answer": final}, "observation": ""})
    return {"question": question, "final_answer": final, "trajectory": traj,
            "llm_calls": len(traj)}


def main():
    import openai
    client = openai.OpenAI()
    out = Path(__file__).resolve().parent / "episodes.jsonl"
    with open(out, "w") as fh:
        for q, gold in QUESTIONS:
            print(f">> {q}")
            row = run_episode(client, q)
            row["gold"] = gold
            row["settings"] = SETTINGS
            row["correct"] = gold.lower() in row["final_answer"].lower()
            print(f"   -> {row['final_answer']!r}  correct={row['correct']}  "
                  f"steps={len(row['trajectory'])}")
            fh.write(json.dumps(row) + "\n")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
