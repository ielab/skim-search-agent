"""`fetch`: a named section (or the infobox) of a document a previous search ranked.

`specs` is a list of `[doc, section]` pairs. `doc` is a rank from the last search listing
(for example an `agent_search.tools.search_bql.SearchBql` instance sharing the same episode
state), or a document id. `section` is a heading name, `""`/omitted for the lead/opening
content, or `"infobox"` for the infobox facts. A read is capped at `MAX_SECTION_TOKENS`
tokens.

Section text is derived from the same `##`-marker split (or the corpus's explicit matched
sections) the search listing used, cached per document id in `state.listing`. This is the
same cache a paired search tool populates, so a doc fetched right after being surfaced does
not redo the split.
"""
from __future__ import annotations

import re

from agent_search.tools.base import Tool
from agent_search.tools.budgets import MAX_SECTION_TOKENS
from agent_search.tools.common import _INTRO, _cap_tokens, _infobox, sections_from_body


class Fetch(Tool):
    name = "fetch"
    aliases = ("fetch", "fetch_v2", "fetch_s", "fetch_bqlds", "fetch_bqldf", "fetch_bqldos")
    description = ("Pull a specific part of a candidate a previous search ranked — code: a "
                   "function/method name or a line range like L1-40; docs: a named section or the "
                   "infobox. Never the whole file/document.")
    parameters = {"type": "object",
                  "properties": {
                      "specs": {"type": "array",
                                "description": "List of [rank, part] pairs referencing the last "
                                               "search's numbering; part is a name from that "
                                               "candidate's structure list (or an L-range for code).",
                                "items": {"type": "array"}}},
                  "required": ["specs"]}

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

    # -- fetch: rank/doc_id + section name -> aggregated slices -------------------------

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
                return None, f"ERROR: no prior search — rank {rank} has nothing to refer to."
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
            return None, f"ERROR: ambiguous doc {ref!r} — did you mean: {opts}"
        return None, f"ERROR: no such doc {ref!r} — use a rank from the last search or a doc_id."

    def _fetch_one(self, doc_ref, section_ref: str) -> tuple:
        u, err = self._resolve_doc(doc_ref)
        if err:
            return (str(doc_ref), err)
        self.state.seen.add(u.doc_id)
        named = self._secs(u.doc_id)
        infobox = _infobox(u)
        avail = "·".join(list(named)[:12]) + (",infobox" if infobox else "")
        s = (section_ref or "").strip()
        if re.sub(r"[\s_-]", "", s.lower()) in ("infobox", "info", "facts"):
            if not infobox:
                return (f"{u.doc_id} §infobox", f"(no infobox — sections: {avail})")
            body = "; ".join(f"{k}={v}" for k, v in infobox.items())
            return (f"{u.doc_id} §infobox", body)
        low = s.lower()
        names = list(named)
        if not s:                            # no section named -> the lead/opening content
            match = [_INTRO] if _INTRO in named else [names[0]] if names else []
        elif len(names) == 1:
            # A flat doc (no '##' markers) has exactly one section, always named '(intro)'.
            # Any part name on a single-section doc means "the body", so honor it instead of
            # erroring "no section 'body'".
            match = names
        else:
            match = ([n for n in names if n.lower() == low]
                     or [n for n in names if n.lower().startswith(low)]
                     or [n for n in names if low in n.lower()])
        if not match:
            return (f"{u.doc_id} §{s}",
                    f"ERROR: no section {s!r} on {u.doc_id}. Available: {avail}")
        if len(match) > 1:
            return (f"{u.doc_id} §{s}",
                    f"ERROR: {s!r} is ambiguous: {'·'.join(match[:8])}")
        text = _cap_tokens(named[match[0]], MAX_SECTION_TOKENS,
                           " …(truncated — fetch a narrower section)")
        return (f"{u.doc_id} §{match[0]}", text)

    def fetch(self, specs: list) -> str:
        if not specs:
            return "ERROR: fetch needs at least one (doc, section) pair."
        # a single flat pair [rank, "section"] -> one spec (same recovery as the code arm)
        if (isinstance(specs, (list, tuple)) and len(specs) == 2
                and not isinstance(specs[0], (list, tuple))
                and not isinstance(specs[1], (list, tuple))):
            try:
                int(specs[0])
                specs = [specs]
            except (TypeError, ValueError):
                pass
        lines = ["fetch:"]
        for spec in specs:
            if not (isinstance(spec, (list, tuple)) and len(spec) == 2):
                lines.append(f"  ERROR: bad spec {spec!r} — expected [doc, section].")
                continue
            doc_ref, section_ref = spec
            label, text = self._fetch_one(doc_ref, section_ref)
            lines.append(f"  [{label}]  {text}")
        return "\n".join(lines)

    def run(self, args: dict) -> str:
        specs = args.get("specs") or args.get("parts") or []
        if not specs and ("doc" in args or "section" in args or "rank" in args):
            specs = [args]
        if isinstance(specs, dict):
            specs = [specs]
        norm = []
        for s in specs:
            if isinstance(s, dict):
                norm.append((s.get("rank") or s.get("doc") or s.get("id"),
                             s.get("section") or s.get("part") or s.get("name") or ""))
            else:
                norm.append(s)
        return self.fetch(norm)


__all__ = ["Fetch"]
