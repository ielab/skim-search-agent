"""Observation -> card-dict parsers shared by the offline replay builder (build_web.py) and the
live SSE server (demo/live/server.py).

Each function parses ONE workspace observation string — the exact text renderings of
`DocSearchFetch.search`/`.fetch` and `Bm25Visit.search`/`.visit` (agent_search/agent/tools/
doc_research.py) — into the dict shape the React player's cards render. Parsers are lenient:
an unparseable observation degrades to a raw-text dict, never a raised exception."""
from __future__ import annotations

import re

HIT = re.compile(r"^\s*(\d+)\s+(\S+)\s+'([^']*)'\s+§\[([^\]]*)\]\s+ib\[([^\]]*)\]"
                 r"(?:\s+matched:\s+(\S+))?\s+»\s+(.*)$")
FETCH = re.compile(r"^\s*\[(\S+)\s+§(.*?)\]\s+(.*)$", re.S)
# Bm25Visit.search line: "  {rank}  {id}  {title!r}  {snippet}…" — {title!r} is Python repr,
# so both quote styles must match (repr picks double quotes when the title has an apostrophe).
BM25_HIT = re.compile(r"^\s*(\d+)\s+(\S+)\s+(?:'([^']*)'|\"([^\"]*)\")\s+(.*?)…?\s*$")
# Bm25Visit.visit header: "{doc_id}  {title!r}:" then the full text on following lines.
VISIT = re.compile(r"^(\S+)\s+(?:'([^']*)'|\"([^\"]*)\"):\n(.*)$", re.S)


def parse_search(obs: str) -> dict:
    """DocSearchFetch.search observation -> {compiled, status, hits:[{rank,id,title,sections,
    infobox,matched,snippet}]} (moved verbatim from build_web.py)."""
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
    """DocSearchFetch.fetch observation -> {doc, section, text, error} (moved verbatim from
    build_web.py)."""
    body = obs.split("fetch:", 1)[-1].strip()
    m = FETCH.match(body)
    if not m:
        return {"doc": "?", "section": "?", "text": body, "error": body.startswith("ERROR")}
    text = m.group(3).strip()
    return {"doc": m.group(1), "section": m.group(2),
            "text": text, "error": text.startswith("ERROR")}


def parse_bm25_search(obs: str) -> dict:
    """Bm25Visit.search observation (flat listing, NO structure chips) -> the SAME hit shape
    parse_search emits (sections/infobox empty, matched "") so HitCard renders both."""
    lines = obs.split("\n")
    m = re.match(r"search:\s*(.*?)\s+\((.*?)\)", lines[0])
    out = {"compiled": "", "status": (m.group(2) if m else "") or "", "hits": []}
    for ln in lines[1:]:
        h = BM25_HIT.match(ln)
        if h:
            out["hits"].append({
                "rank": int(h.group(1)), "id": h.group(2),
                "title": h.group(3) if h.group(3) is not None else (h.group(4) or ""),
                "sections": [], "infobox": [], "matched": "",
                "snippet": h.group(5).strip()})
    return out


def parse_visit(obs: str) -> dict:
    """Bm25Visit.visit observation (whole-document read) -> {doc, title, text, error}."""
    if obs.startswith("ERROR"):
        return {"doc": "?", "title": "", "text": obs, "error": True}
    m = VISIT.match(obs)
    if not m:
        return {"doc": "?", "title": "", "text": obs, "error": False}
    return {"doc": m.group(1),
            "title": m.group(2) if m.group(2) is not None else (m.group(3) or ""),
            "text": m.group(4).strip(), "error": False}
