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

from demo.recorder.corpus import CORPUS, QUESTIONS  # noqa: E402
from agent_search.agent.tools.doc_research import DocSearchFetch  # noqa: E402

SYSTEM = """You are a research agent answering a question over a document collection you can
only access through two tools. Reply with EXACTLY ONE tool call per turn, nothing else:

search: <query>       — Boolean search. Plain terms rank; operators narrow: AND(a, b), OR(a, b),
                        NOT(x), IN(title, term), IN(section, term), IN(body, "a phrase"), PREFIX(st).
fetch: <doc> :: <section>   — read ONE named section of a listed document (use ids/headings from
                              the result cards).
answer: <final answer>      — a short answer span, only once you have read supporting text.

Search first, skim the result cards, fetch only what you need, then answer."""


def run_episode(client, question: str, max_steps: int = 12) -> dict:
    ws = DocSearchFetch(CORPUS, snippets=True)
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"Question: {question}"}]
    traj = []
    for _ in range(max_steps):
        out = client.chat.completions.create(model="gpt-4o-mini", messages=msgs,
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
                obs = ws.search(arg, k=5)
                traj.append({"action": "search", "args": {"query": arg}, "observation": obs})
            else:
                doc, _, section = arg.partition("::")
                obs = ws.fetch([[doc.strip(), section.strip()]])
                traj.append({"action": "fetch",
                             "args": {"doc": doc.strip(), "section": section.strip()},
                             "observation": obs})
        msgs.append({"role": "assistant", "content": text})
        msgs.append({"role": "user", "content": f"Observation:\n{traj[-1]['observation']}"})
    return {"question": question, "final_answer": "", "trajectory": traj, "llm_calls": len(traj)}


def main():
    import openai
    client = openai.OpenAI()
    out = Path(__file__).resolve().parent / "episodes.jsonl"
    with open(out, "w") as fh:
        for q, gold in QUESTIONS:
            print(f">> {q}")
            row = run_episode(client, q)
            row["gold"] = gold
            row["correct"] = gold.lower() in row["final_answer"].lower()
            print(f"   -> {row['final_answer']!r}  correct={row['correct']}  "
                  f"steps={len(row['trajectory'])}")
            fh.write(json.dumps(row) + "\n")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
