"""`fetch`: one named section (or the facts) of a document a previous search ranked.

The call is flat: `{"rank": 3, "section": "Career"}`. `rank` is a row number from the last
search listing (a document id also works). `section` is a heading from that row's section
list, `"infobox"` for the document's facts (infobox fields, or title, author and date when the
corpus carries them), or `""` for the opening text. One section per call, capped at
`MAX_SECTION_TOKENS` tokens. Never the whole document.

The flat shape replaced the paper's `{"specs": [[rank, section]]}` list of pairs. Tongyi
mis-closed that nested list in a quarter of Sieve's fetch calls and three quarters of
Search-Fetch's, and 99.9% of the calls carried a single pair anyway. The old shape is still
accepted, so recorded trajectories and older configs keep working.

Requests that are not a section name on the referenced document are resolved instead of
refused, because every refusal costs the agent a step and a third of all fetch calls in the
BrowseComp-Plus cells were such refusals:
  - a whole-document word (`body`, `full text`, `*`) returns the whole document, sections in
    order under their headings, capped like a visit;
  - a facts word (`infobox`, `author`, `date`, `title`) returns the document's facts;
  - a name is matched case- and punctuation-insensitively, then by prefix, substring and
    word overlap; a pasted list (`§[A·B·C]`) is read as its first section that exists;
  - a section that is not on this document but is on exactly one other document of the
    current listing (the agent misread a rank) is read from that document, and the reply
    says so; the same for exactly one document of an earlier listing.
The remaining error names every section the document has.

Section text comes from the corpus's explicit sections or the same `##`-marker split the
search listing used, cached per document in `state.listing`, so a fetch right after a search
does not redo the split.
"""
from __future__ import annotations

import re

from agent_search.tools.base import Tool
from agent_search.tools.budgets import MAX_SECTION_TOKENS, MAX_VISIT_TOKENS
from agent_search.tools.common import _INTRO, _cap_tokens, _infobox, sections_from_body

# words that ask for the whole document: every section in order, capped like a visit
_WHOLE_WORDS = {"body", "text", "content", "contents", "full", "full text", "fulltext", "all",
                "(all)", "*", "document", "doc", "page", "article", "whole", "everything",
                "main", "main text", "summary"}
_LINE_RANGE = re.compile(r"^l?\d+\s*-\s*l?\d+$", re.IGNORECASE)   # code-style `L1-40` on a document
# words that ask for the document's facts
_FACTS_WORDS = {"infobox", "info", "facts", "fact box", "metadata", "meta", "author", "authors",
                "byline", "date", "published", "publication date", "title", "header", "details"}
_INTRO_WORDS = {"intro", "introduction", "lead", "opening", "(intro)", "overview", "top", "start"}
_FACT_FIELDS = ("title", "author", "date")


def _norm(s: str) -> str:
    """Lower-cased words of any script (section names are often not Latin), punctuation dropped."""
    return re.sub(r"[\W_]+", " ", (s or "").lower()).strip()


def _facts(u) -> dict:
    """The document's facts: title, author and date when the corpus carries them, then the
    infobox fields. Empty values are left out."""
    meta = u.metadata or {}
    out: dict = {}
    if u.title:
        out["title"] = u.title
    for key in _FACT_FIELDS[1:]:
        val = str(meta.get(key) or "").strip()
        if val:
            out[key] = val
    out.update(_infobox(u))
    return out


class Fetch(Tool):
    name = "fetch"
    aliases = ("fetch", "fetch_v2", "fetch_s", "fetch_bqlds", "fetch_bqldf", "fetch_bqldos")
    description = ("Read a document the last search ranked: a named section from its section list "
                   "(preferred), \"infobox\" for its facts (title, author, date, infobox fields), "
                   "\"\" for the opening text, or \"body\" for the whole document (costs as much as a "
                   "visit). One read per call.")
    parameters = {"type": "object",
                  "properties": {
                      "rank": {"type": "integer",
                               "description": "The row number of the document in the last search "
                                              "listing (1 is the first row)."},
                      "section": {"type": "string",
                                  "description": "A section name from that row's list, \"infobox\" "
                                                 "for the facts, \"\" for the opening text, or \"body\" "
                                                 "for the whole document."}},
                  "required": ["rank", "section"]}

    # -- section/infobox cache, shared with a paired `search` tool via state.listing -----

    def _secs(self, doc_id: str) -> dict:
        entry = self.state.listing.get(doc_id)
        if entry is not None and "sections" in entry:
            return entry["sections"]
        u = self.ubyid.get(doc_id)
        explicit = getattr(u, "sections", None) if u is not None else None
        if explicit:
            out: dict = {}
            for h, t in explicit:
                name = h or _INTRO
                key, n = name, 2
                while key in out:
                    key, n = f"{name} ({n})", n + 1
                out[key] = t
            secs = out or {_INTRO: ""}
        else:
            secs = sections_from_body(u.body if (u and u.body is not None)
                                      else (u.code if u else ""))
        entry = self.state.listing.setdefault(doc_id, {})
        entry["sections"] = secs
        entry["infobox"] = _infobox(u) if u is not None else {}
        return secs

    # -- resolving the document ---------------------------------------------------------

    def _resolve_doc(self, ref):
        last = self.state.last_hits
        if isinstance(ref, (int, float)) or (isinstance(ref, str) and ref.strip().isdigit()):
            rank = int(ref)
            if 1 <= rank <= len(last):
                return self.ubyid[last[rank - 1]], None
            # A digit that is not a valid rank may be a real doc_id (numeric doc_ids do
            # occur), so resolve it as one before erroring.
            if str(ref).strip() in self.ubyid:
                return self.ubyid[str(ref).strip()], None
            if not last:
                return None, f"ERROR: no prior search: rank {rank} has nothing to refer to."
            return None, (f"ERROR: rank {rank} out of range "
                          f"(last search had {len(last)} results).")
        ref = (ref or "").strip()
        if ref in self.ubyid:
            return self.ubyid[ref], None
        low = ref.lower()
        cands = [i for i, u in self.ubyid.items()
                 if low in i.lower() or low in (u.title or u.qualname or "").lower()]
        if len(cands) == 1:
            return self.ubyid[cands[0]], None
        if cands:
            opts = ", ".join(f"{i} ({self.ubyid[i].title or self.ubyid[i].qualname})"
                             for i in cands[:5])
            return None, f"ERROR: ambiguous doc {ref!r}. Did you mean: {opts}"
        return None, f"ERROR: no such doc {ref!r}. Use a rank from the last search or a doc_id."

    # -- resolving the section name -----------------------------------------------------

    @staticmethod
    def _match(names: list, want: str) -> list:
        """The section names `want` resolves to on one document: exact (normalised), then
        unique prefix, then substring either way, then word overlap of at least half. An
        empty list means no match; more than one means ambiguous."""
        w = _norm(want)
        if not w:
            return []
        normed = [(n, _norm(n)) for n in names]
        exact = [n for n, nn in normed if nn == w]
        if exact:
            return exact
        pref = [n for n, nn in normed if nn.startswith(w)]
        if pref:
            return pref
        sub = [n for n, nn in normed if w in nn or (len(nn) >= 4 and nn in w)]
        if sub:
            return sub
        wt = set(w.split())
        scored = []
        for n, nn in normed:
            nt = set(nn.split())
            if not nt:
                continue
            j = len(wt & nt) / len(wt | nt)
            if j >= 0.5:
                scored.append((j, n))
        if not scored:
            return []
        best = max(s for s, _ in scored)
        return [n for s, n in scored if s == best]

    @staticmethod
    def _pieces(section: str) -> list:
        """The candidate names in a request: markup stripped, a pasted `A·B·C` list split."""
        s = (section or "").strip().strip("§").strip("[]{}()").strip("\"'`").strip()
        if "·" in s:
            return [p.strip().strip("§[]") for p in s.split("·") if p.strip()]
        return [s]

    def _render_section(self, u, name: str, note: str = "") -> tuple:
        text = _cap_tokens(self._secs(u.doc_id)[name], MAX_SECTION_TOKENS,
                           " ...(truncated; fetch a narrower section)")
        return (f"{u.doc_id} §{name}{note}", text)

    def _render_facts(self, u) -> tuple:
        facts = _facts(u)
        if not facts:
            names = [n for n in self._secs(u.doc_id)]
            return (f"{u.doc_id} §infobox",
                    f"(no facts on this document: no infobox, author or date. Sections: {'·'.join(names)})")
        return (f"{u.doc_id} §infobox", "; ".join(f"{k}={v}" for k, v in facts.items()))

    def _render_whole(self, u) -> tuple:
        """The whole document: every section in order under its heading, capped like a visit."""
        named = self._secs(u.doc_id)
        parts = []
        for name, text in named.items():
            parts.append(text if name == _INTRO else f"## {name}\n{text}")
        body = _cap_tokens("\n\n".join(parts), MAX_VISIT_TOKENS,
                           " ...(truncated at the whole-document cap; fetch a named section for the rest)")
        return (f"{u.doc_id} §body", body)

    def _elsewhere(self, u, want: str) -> tuple:
        """A section that is not on `u` but is on exactly one other listed document: the
        current listing first (a misread rank), then earlier listings. Returns (doc, name,
        note) or (None, candidates, None) when several documents have it, or (None, [], None)."""
        def find(doc_ids):
            hits = []
            for d in doc_ids:
                if d == u.doc_id or d not in self.ubyid:
                    continue
                m = self._match(list(self._secs(d)), want)
                if len(m) == 1 and _norm(m[0]) == _norm(want):
                    hits.append((d, m[0]))
            return hits
        current = find(self.state.last_hits)
        if len(current) == 1:
            d, name = current[0]
            rank = self.state.last_hits.index(d) + 1
            return self.ubyid[d], name, f" (read from rank {rank}, which has this section)"
        if len(current) > 1:
            return None, [f"rank {self.state.last_hits.index(d) + 1}" for d, _ in current], None
        older = find([d for d in self.state.listing if d not in self.state.last_hits])
        if len(older) == 1:
            d, name = older[0]
            return self.ubyid[d], name, f" (read from an earlier listing, doc {d})"
        return None, [], None

    def _fetch_one(self, doc_ref, section_ref) -> tuple:
        u, err = self._resolve_doc(doc_ref)
        if err:
            return (str(doc_ref), err)
        self.state.seen.add(u.doc_id)
        named = self._secs(u.doc_id)
        names = list(named)
        raw = "" if section_ref is None else str(section_ref)
        pieces = self._pieces(raw)
        s = pieces[0] if pieces else ""
        low = s.lower().strip()
        if not low or low in _INTRO_WORDS:
            if _INTRO in named:
                return self._render_section(u, _INTRO)
            return self._render_section(u, names[0])
        if len(pieces) >= 3 and len(names) > 1:
            return self._render_whole(u)                  # a pasted section list: the whole document
        # a name the document really has wins over every special word ("Details" may be a section)
        exact = [n for n in names if _norm(n) == _norm(s)]
        if len(exact) == 1:
            return self._render_section(u, exact[0])
        if low in _FACTS_WORDS:
            return self._render_facts(u)
        if len(names) == 1:
            # A flat document has exactly one section; any name means its text.
            return self._render_section(u, names[0])
        if low in _WHOLE_WORDS or _LINE_RANGE.match(low):
            return self._render_whole(u)
        for piece in pieces:
            match = self._match(names, piece)
            if len(match) == 1:
                return self._render_section(u, match[0])
            if len(match) > 1:
                return (f"{u.doc_id} §{piece}",
                        f"ERROR: {piece!r} is ambiguous on {u.doc_id}: {'·'.join(match[:8])}. Name one.")
        other, name_or_cands, note = self._elsewhere(u, s)
        if other is not None:
            return self._render_section(other, name_or_cands, note)
        avail = "·".join(names) + (" and infobox" if _infobox(u) else "")
        if name_or_cands:
            return (f"{u.doc_id} §{s}",
                    f"ERROR: no section {s!r} on {u.doc_id}; {' and '.join(name_or_cands)} have it. "
                    f"Fetch one by its rank. Sections here: {avail}")
        return (f"{u.doc_id} §{s}", f"ERROR: no section {s!r} on {u.doc_id}. Sections here: {avail}")

    # -- the call -------------------------------------------------------------------------

    @staticmethod
    def _requests(args: dict) -> list:
        """The (doc, section) requests in a call: the flat shape, or the paper's `specs`
        list of pairs (a single flat pair, or dicts, also accepted)."""
        specs = args.get("specs") or args.get("parts")
        if specs is None:
            if any(k in args for k in ("rank", "doc", "id", "doc_id", "docid", "section", "part", "name", "heading")):
                specs = [args]
            else:
                specs = []
        if isinstance(specs, dict):
            specs = [specs]
        if (isinstance(specs, (list, tuple)) and len(specs) == 2
                and not isinstance(specs[0], (list, tuple, dict))
                and not isinstance(specs[1], (list, tuple, dict))):
            specs = [specs]                                  # one flat pair [rank, "section"]
        out = []
        for s in specs:
            if isinstance(s, dict):
                doc = next((s[k] for k in ("rank", "doc", "id", "doc_id", "docid") if s.get(k) not in (None, "")), None)
                sec = next((s[k] for k in ("section", "part", "name", "heading") if s.get(k) is not None), "")
                out.append((doc, sec) if doc is not None else (None, "missing rank"))
            elif isinstance(s, (list, tuple)) and len(s) == 2:
                out.append((s[0], s[1]))
            elif isinstance(s, (list, tuple)) and len(s) == 1:
                out.append((s[0], ""))
            else:
                out.append((None, f"bad spec {s!r}"))
        return out

    def fetch(self, specs) -> str:
        """The paper's call shape: a list of [doc, section] pairs. Kept for recorded
        trajectories and tests; `run` is the declared entry point."""
        return self.run({"specs": specs})

    def run(self, args: dict) -> str:
        requests = self._requests(args)
        if not requests:
            return 'ERROR: fetch needs a rank and a section name, for example {"rank": 1, "section": "infobox"}.'
        lines = ["fetch:"]
        for doc_ref, section_ref in requests:
            if doc_ref is None:
                lines.append(f"  ERROR: {section_ref}. Expected {{\"rank\": <row number>, \"section\": <name>}}.")
                continue
            label, text = self._fetch_one(doc_ref, section_ref)
            lines.append(f"  [{label}]  {text}")
        return "\n".join(lines)


__all__ = ["Fetch"]
