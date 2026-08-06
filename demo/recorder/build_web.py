"""Refresh the web demo's data after re-recording episodes.

    python demo/build_web.py             # writes demo/app/src/data.json
    cd demo/app && npm run build         # rebuilds the single-file dist/index.html

Parses demo/episodes.jsonl (raw workspace observations) into the structured steps the React
player renders: per search — query, compiled form, status, and each result card's title,
sections, matched fields, and snippet; per fetch — the doc, section, text, and error flag.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))

HIT = re.compile(r"^\s*(\d+)\s+(\S+)\s+'([^']*)'\s+§\[([^\]]*)\]\s+ib\[([^\]]*)\]"
                 r"(?:\s+matched:\s+(\S+))?\s+»\s+(.*)$")
FETCH = re.compile(r"^\s*\[(\S+)\s+§(.*?)\]\s+(.*)$", re.S)


def parse_search(obs: str) -> dict:
    lines = obs.split("\n")
    m = re.match(r"search:\s*(.*?)\s+->\s+(.*?)(?:\s{2,}\((.*)\))?\s*$", lines[0])
    out = {"compiled": m.group(2) if m else "", "status": (m.group(3) if m else "") or "",
           "hits": []}
    for ln in lines[1:]:
        h = HIT.match(ln)
        if h:
            out["hits"].append({
                "rank": int(h.group(1)), "id": h.group(2), "title": h.group(3),
                "sections": [s for s in h.group(4).split("·") if s],
                "infobox": [s for s in h.group(5).split("·") if s],
                "matched": h.group(6) or "", "snippet": h.group(7).strip()})
    return out


def parse_fetch(obs: str) -> dict:
    body = obs.split("fetch:", 1)[-1].strip()
    m = FETCH.match(body)
    if not m:
        return {"doc": "?", "section": "?", "text": body, "error": body.startswith("ERROR")}
    text = m.group(3).strip()
    return {"doc": m.group(1), "section": m.group(2),
            "text": text, "error": text.startswith("ERROR")}


def build_episode(row: dict) -> dict:
    steps = []
    for s in row.get("trajectory") or []:
        act = s.get("action")
        if act == "search":
            steps.append({"type": "search", "query": (s.get("args") or {}).get("query", ""),
                          **parse_search(s.get("observation", ""))})
        elif act == "fetch":
            steps.append({"type": "fetch",
                          "ask": (s.get("args") or {}).get("doc", "") + " § " +
                                 (s.get("args") or {}).get("section", ""),
                          **parse_fetch(s.get("observation", ""))})
    return {"question": row.get("question", ""), "answer": row.get("final_answer", ""),
            "gold": row.get("gold", ""), "correct": bool(row.get("correct")),
            "calls": row.get("llm_calls", len(steps)), "steps": steps,
            "settings": row.get("settings") or {}}


def main():
    from demo.recorder.corpus import CORPUS
    episodes = [build_episode(json.loads(l)) for l in open(HERE / "episodes.jsonl")]
    corpus = [{"id": u.doc_id, "title": u.title,
               "sections": re.findall(r"^## (.+)$", u.body, re.M)} for u in CORPUS]
    out = HERE.parent / "app" / "src" / "data.json"
    json.dump({"episodes": episodes, "corpus": corpus}, open(out, "w"), ensure_ascii=False)
    print(f"wrote {out} ({len(episodes)} episodes). Rebuild: cd demo/app && npm run build")


if __name__ == "__main__":
    main()
