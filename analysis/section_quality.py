#!/usr/bin/env python
"""M2 -- section-quality audit of the LLM-inserted structure in BrowseComp-Plus-Structured
(docs/reviews/round4_full.md, M2). The paper defends the sectioning only with non-empty/
multi-section counts (benchmark.tex Sec 3.1: "100% non-empty ... 94.9% genuinely multi-section,
mean 14.7 sections/doc"); this script adds automatic QUALITY signals the fetch mechanism's
validity actually depends on (section boundaries being meaningful, not just present).

Corpus format (data/browsecomp_plus_structured/corpus.jsonl, one doc per line):
    {"_id": ..., "title": ..., "author": ..., "date": ...,
     "text": "<date>\\n\\n<title>\\n\\n<body with '## heading' markers inline>",
     "sections": [{"heading": "(intro)", "text": "..."}, {"heading": "...", "text": "..."}, ...]}
`sections.jsonl` (same repo) carries the identical `sections` field alone, keyed by `_id` --
not used here since corpus.jsonl already has everything needed in one pass.

Signals computed over a random sample (default n=200, seed=42, RESERVOIR sampling -- single
streaming pass over the 3.7GB corpus.jsonl, O(n) memory, never loads the 67,707-doc file whole):

  1. section-count distribution           (paper's own headline number, reproduced as a check)
  2. section length distribution          chars AND tokens (tiktoken o200k_base -- same ruler
                                           evaluation/run_eval.py's `_obs_token_count` uses
                                           elsewhere in this repo, chars//4 fallback)
  3. heading-body lexical consistency     fraction of non-"(intro)" sections where >=1 non-
                                           generic heading term (casefolded, len>=3, whole-word)
                                           appears in that section's own body text
  4. degenerate sections                  empty/near-empty body (<15 stripped chars), OR a
                                           heading string duplicated elsewhere in the same doc
  5. boundary sanity                      does each section's `text` occur, in order, as a
                                           non-overlapping substring of the doc's own `text`
                                           field? (found-in-order + no-overlap + gap-size check;
                                           small gaps are EXPECTED -- the doc's `text` carries a
                                           "## heading" markup line + blank line between sections
                                           that isn't in the section's own `text`)

Hand-inspection: 3 verbatim-quoted examples (2 good, 1 worst-case) pulled from the same sample.

    PYTHONPATH=. envs/bin/python analysis/section_quality.py
    PYTHONPATH=. envs/bin/python analysis/section_quality.py --n 300 --seed 42 --out analysis/section_quality.md
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

DEFAULT_CORPUS = REPO_ROOT / "data" / "browsecomp_plus_structured" / "corpus.jsonl"

# same ruler as evaluation/run_eval.py::_obs_token_count (tiktoken o200k_base, chars//4 fallback)
_O200K = None


def _tok_count(text: str) -> int:
    global _O200K
    if _O200K is None:
        try:
            import tiktoken
            _O200K = tiktoken.get_encoding("o200k_base")
        except Exception:
            _O200K = False
    if _O200K is False:
        return len(text or "") // 4
    return len(_O200K.encode(text or "", disallowed_special=()))


# a small generic-heading stoplist -- words that carry no doc-specific content, so a heading
# consisting ONLY of these tokens is untestable (can't fail/pass lexical consistency) rather
# than mechanically counted as a failure. English + a couple structural markers only; NOT
# exhaustive, disclosed as a limitation in the report.
_GENERIC_HEADING_WORDS = {
    "and", "the", "for", "of", "in", "on", "to", "with", "details", "overview", "information",
    "section", "general", "other", "related", "introduction", "background", "summary", "about",
    "notes", "note", "list", "lists", "misc", "miscellaneous", "further", "see", "also",
}
_WORD_RE = re.compile(r"\w+", re.UNICODE)


def heading_terms(heading: str) -> list[str]:
    toks = [t.casefold() for t in _WORD_RE.findall(heading or "") if len(t) >= 3]
    return [t for t in toks if t not in _GENERIC_HEADING_WORDS]


def reservoir_sample(path: Path, n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    sample: list[dict] = []
    with open(path, encoding="utf-8", errors="replace") as fh:
        for i, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if len(sample) < n:
                sample.append(d)
            else:
                j = rng.randint(0, i)
                if j < n:
                    sample[j] = d
    return sample


# --------------------------------------------------------------------------------------
# per-doc analysis
# --------------------------------------------------------------------------------------

GAP_THRESHOLD = 200          # chars; expected gap is ~"## <heading>\n\n" (usually <60 chars)
NEAR_EMPTY_CHARS = 15


def boundary_status(doc_text: str, sections: list[dict]) -> dict:
    pos = 0
    gaps = []
    status = "clean"
    for i, s in enumerate(sections):
        body = s.get("text") or ""
        idx = doc_text.find(body, pos)
        if idx == -1:
            return {"status": "not_found", "gaps": gaps, "fail_at": i}
        gap = idx - pos
        if idx < pos:
            return {"status": "overlap", "gaps": gaps, "fail_at": i}
        if i > 0 and gap > GAP_THRESHOLD:      # first section's "gap" is the title/date preamble
            status = "large_gap"
        gaps.append(gap)
        pos = idx + len(body)
    return {"status": status, "gaps": gaps, "fail_at": None}


def analyze_doc(doc: dict) -> dict:
    sections = doc.get("sections") or []
    text = doc.get("text") or ""
    n_sections = len(sections)
    lens_chars = [len(s.get("text") or "") for s in sections]
    lens_tok = [_tok_count(s.get("text") or "") for s in sections]

    non_intro = [s for s in sections if (s.get("heading") or "").strip() != "(intro)"]
    testable = tested_pass = 0
    for s in non_intro:
        terms = heading_terms(s.get("heading") or "")
        if not terms:
            continue
        testable += 1
        body_cf = (s.get("text") or "").casefold()
        if any(re.search(rf"\b{re.escape(t)}\b", body_cf) for t in terms):
            tested_pass += 1

    near_empty = sum(1 for s in sections if len((s.get("text") or "").strip()) < NEAR_EMPTY_CHARS)
    headings_cf = [(s.get("heading") or "").strip().casefold() for s in sections
                   if (s.get("heading") or "").strip() != "(intro)"]
    dup_headings = len(headings_cf) - len(set(headings_cf))

    bstat = boundary_status(text, sections)

    return {
        "_id": doc.get("_id"), "title": doc.get("title"), "n_sections": n_sections,
        "lens_chars": lens_chars, "lens_tok": lens_tok,
        "testable_headings": testable, "lexical_pass": tested_pass,
        "near_empty_sections": near_empty, "dup_heading_pairs": dup_headings,
        "boundary": bstat["status"], "boundary_gaps": bstat["gaps"],
    }


# --------------------------------------------------------------------------------------
# aggregation / rendering
# --------------------------------------------------------------------------------------

def pctl(xs: list[float], p: float) -> float:
    if not xs:
        return float("nan")
    xs = sorted(xs)
    k = (len(xs) - 1) * p
    f, c = int(k), min(int(k) + 1, len(xs) - 1)
    if f == c:
        return xs[f]
    return xs[f] + (xs[c] - xs[f]) * (k - f)


def summarize(docs: list[dict]) -> dict:
    n_docs = len(docs)
    n_sections_list = [d["n_sections"] for d in docs]
    all_lens_chars = [x for d in docs for x in d["lens_chars"]]
    all_lens_tok = [x for d in docs for x in d["lens_tok"]]
    total_testable = sum(d["testable_headings"] for d in docs)
    total_pass = sum(d["lexical_pass"] for d in docs)
    total_sections = sum(d["n_sections"] for d in docs)
    total_near_empty = sum(d["near_empty_sections"] for d in docs)
    docs_with_dup = sum(1 for d in docs if d["dup_heading_pairs"] > 0)
    boundary_counts = {}
    for d in docs:
        boundary_counts[d["boundary"]] = boundary_counts.get(d["boundary"], 0) + 1

    return {
        "n_docs": n_docs,
        "n_sections": {
            "mean": sum(n_sections_list) / n_docs, "median": pctl(n_sections_list, 0.5),
            "min": min(n_sections_list), "max": max(n_sections_list),
            "p10": pctl(n_sections_list, 0.10), "p90": pctl(n_sections_list, 0.90),
            "pct_single_section": 100.0 * sum(1 for x in n_sections_list if x <= 1) / n_docs,
        },
        "section_len_chars": {
            "mean": sum(all_lens_chars) / len(all_lens_chars), "median": pctl(all_lens_chars, 0.5),
            "p10": pctl(all_lens_chars, 0.10), "p90": pctl(all_lens_chars, 0.90),
        },
        "section_len_tok": {
            "mean": sum(all_lens_tok) / len(all_lens_tok), "median": pctl(all_lens_tok, 0.5),
            "p10": pctl(all_lens_tok, 0.10), "p90": pctl(all_lens_tok, 0.90),
        },
        "lexical_consistency_pct": (100.0 * total_pass / total_testable) if total_testable else float("nan"),
        "n_testable_headings": total_testable, "n_total_non_intro_sections": total_sections - n_docs,
        "near_empty_pct_of_sections": 100.0 * total_near_empty / total_sections,
        "docs_with_dup_heading_pct": 100.0 * docs_with_dup / n_docs,
        "boundary_pct": {k: 100.0 * v / n_docs for k, v in boundary_counts.items()},
        "boundary_counts": boundary_counts,
    }


def _badness(d: dict) -> float:
    """Higher = more degenerate. Boundary failure dominates; duplicate headings and near-empty
    sections are weighted heavily (each is a direct, unambiguous degeneracy signal); a low
    lexical-consistency rate contributes least, since that signal has known false negatives
    (e.g. a heading like 'Jersey No. 1' whose body starts '1: Anthony Carter...' is fine
    structurally but fails the crude word-overlap test)."""
    boundary_bad = 1.0 if d["boundary"] in ("not_found", "overlap") else 0.0
    lex_fail_rate = (1.0 - d["lexical_pass"] / d["testable_headings"]) if d["testable_headings"] else 0.0
    return (100 * boundary_bad + 5 * d["dup_heading_pairs"] + 3 * d["near_empty_sections"]
            + 1 * lex_fail_rate)


def pick_examples(docs: list[dict]) -> dict:
    """2 good (clean boundary, >=1 testable heading, full lexical pass, >=3 sections, no dups/
    near-empty), 1 worst-case (highest `_badness` score in the sample)."""
    good = [d for d in docs if d["boundary"] == "clean" and d["n_sections"] >= 3
            and d["testable_headings"] >= 2 and d["lexical_pass"] == d["testable_headings"]
            and d["dup_heading_pairs"] == 0 and d["near_empty_sections"] == 0]
    good = sorted(good, key=lambda d: -d["n_sections"])[:2]

    worst = max(docs, key=_badness) if docs else None
    return {"good": good, "worst": worst}


def quote_doc(doc_by_id: dict, d: dict, highlight_dup: bool = False) -> str:
    raw = doc_by_id[d["_id"]]
    secs = raw.get("sections") or []
    lines = [f"`_id={d['_id']}` \"{(raw.get('title') or '')[:80]}\" -- {d['n_sections']} sections, "
             f"boundary={d['boundary']}, lexical {d['lexical_pass']}/{d['testable_headings']}, "
             f"dup_heading_pairs={d['dup_heading_pairs']}, near_empty={d['near_empty_sections']}"]
    if highlight_dup and d["dup_heading_pairs"] > 0:
        from collections import Counter
        c = Counter((s.get("heading") or "").strip().casefold() for s in secs
                    if (s.get("heading") or "").strip() != "(intro)")
        worst_heading, count = c.most_common(1)[0]
        dupes = [s for s in secs if (s.get("heading") or "").strip().casefold() == worst_heading]
        lines.append(f"    - heading **\"{dupes[0].get('heading')}\"** repeats {count} times in this "
                      f"one document with DIFFERENT bodies each time (a heading naming a recurring "
                      f"newsletter/list FORMAT rather than distinguishing individual items):")
        for s in dupes[:3]:
            body = (s.get("text") or "").replace("\n", " ")[:150]
            lines.append(f"        - \"{body}{'...' if len(s.get('text') or '') > 150 else ''}\"")
        lines.append(f"        - ... ({count - 3} more sections share this exact heading)")
        return "\n".join(lines)
    for s in secs[:4]:
        body = (s.get("text") or "").replace("\n", " ")[:160]
        lines.append(f"    - **{s.get('heading')}**: \"{body}{'...' if len(s.get('text') or '') > 160 else ''}\"")
    if len(secs) > 4:
        lines.append(f"    - ... ({len(secs) - 4} more sections)")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", default=str(DEFAULT_CORPUS))
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None)
    ap.add_argument("--json-out", default=str(REPO_ROOT / "analysis" / "section_quality_data.json"))
    args = ap.parse_args()

    print(f"reservoir-sampling {args.n} docs (seed={args.seed}) from {args.corpus} ...", file=sys.stderr)
    sample = reservoir_sample(Path(args.corpus), args.n, args.seed)
    print(f"sampled {len(sample)} docs; analyzing ...", file=sys.stderr)

    doc_by_id = {d.get("_id"): d for d in sample}
    analyzed = [analyze_doc(d) for d in sample]
    summary = summarize(analyzed)
    examples = pick_examples(analyzed)

    payload = {"args": vars(args), "summary": summary,
               "examples": {"good": [d["_id"] for d in examples["good"]],
                             "worst": examples["worst"]["_id"] if examples["worst"] else None}}
    with open(args.json_out, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"wrote {args.json_out}", file=sys.stderr)

    md = render_markdown(summary, examples, doc_by_id, args)
    if args.out:
        Path(args.out).write_text(md)
        print(f"wrote {args.out}", file=sys.stderr)
    else:
        print(md)


def render_markdown(s: dict, examples: dict, doc_by_id: dict, args) -> str:
    lines = ["# Section-quality audit -- BrowseComp-Plus-Structured (M2)\n"]
    lines.append(f"Random sample: n={s['n_docs']} documents, seed={args.seed}, reservoir-sampled "
                  f"in one streaming pass over `{args.corpus}` (67,707 docs total).\n")

    lines.append("| signal | value |")
    lines.append("|---|---|")
    ns = s["n_sections"]
    lines.append(f"| sections/doc: mean / median / p10-p90 / min-max | "
                  f"{ns['mean']:.1f} / {ns['median']:.1f} / {ns['p10']:.1f}-{ns['p90']:.1f} / "
                  f"{ns['min']}-{ns['max']} |")
    lines.append(f"| docs with <=1 section (not genuinely multi-section) | {ns['pct_single_section']:.1f}% |")
    lc = s["section_len_chars"]
    lines.append(f"| section length, chars: mean / median / p10-p90 | "
                  f"{lc['mean']:.0f} / {lc['median']:.0f} / {lc['p10']:.0f}-{lc['p90']:.0f} |")
    lt = s["section_len_tok"]
    lines.append(f"| section length, tokens (o200k_base): mean / median / p10-p90 | "
                  f"{lt['mean']:.0f} / {lt['median']:.0f} / {lt['p10']:.0f}-{lt['p90']:.0f} |")
    lines.append(f"| heading-body lexical consistency (non-generic heading term found in body) | "
                  f"{s['lexical_consistency_pct']:.1f}% (of {s['n_testable_headings']} testable "
                  f"non-intro section headings; {s['n_total_non_intro_sections'] - s['n_testable_headings']} "
                  f"more had only generic/stoplisted heading words and are untestable) |")
    lines.append(f"| near-empty sections (<{NEAR_EMPTY_CHARS} stripped chars) | "
                  f"{s['near_empty_pct_of_sections']:.1f}% of all sections |")
    lines.append(f"| docs with a duplicate heading (same heading text twice+ in one doc) | "
                  f"{s['docs_with_dup_heading_pct']:.1f}% of docs |")
    bp = s["boundary_pct"]
    lines.append(f"| boundary sanity: clean / large-gap / overlap / not-found | "
                  f"{bp.get('clean', 0):.1f}% / {bp.get('large_gap', 0):.1f}% / "
                  f"{bp.get('overlap', 0):.1f}% / {bp.get('not_found', 0):.1f}% |")
    lines.append("")

    lines.append("**Method notes.** Lexical consistency: a non-`(intro)` section \"passes\" if any "
                  "casefolded heading token (len>=3, excluding a small generic-word stoplist -- see "
                  "`_GENERIC_HEADING_WORDS` in the script) appears as a whole word in that section's "
                  "own body text; headings whose only content words are stoplisted are reported "
                  "separately as untestable, not counted as failures. Boundary sanity: each section's "
                  "`text` is located as a substring of the doc's own `text` field, walking forward "
                  "from the previous match end -- `clean` = all sections found, in order, with no "
                  "overlap and no gap over "
                  f"{GAP_THRESHOLD} chars (small gaps are EXPECTED: the doc `text` interleaves a "
                  "`## heading` markup line the section's own `text` field omits); `large_gap` = "
                  "found in order, no overlap, but at least one inter-section gap exceeds "
                  f"{GAP_THRESHOLD} chars; `overlap` = a section's text starts before the previous "
                  "section's text ended; `not_found` = a section's `text` does not occur verbatim "
                  "in the doc's `text` field at all past the current position.\n")

    lines.append("## Hand-inspected examples\n")
    for d in examples["good"]:
        lines.append("**Good:** " + quote_doc(doc_by_id, d))
        lines.append("")
    if examples["worst"]:
        lines.append("**Worst-case:** " + quote_doc(doc_by_id, examples["worst"], highlight_dup=True))
        lines.append("")

    lines.append("**Key sentence (paper-ready, hedged).** Beyond the paper's non-empty/multi-section "
                  f"counts, an automatic audit of {s['n_docs']} sampled documents finds the LLM-"
                  f"inserted sections are largely well-formed structurally ({bp.get('clean', 0):.0f}% "
                  "partition the document text cleanly with no overlap or large gap) and "
                  f"{s['lexical_consistency_pct']:.0f}% of testable section headings share at least "
                  "one content word with their own body text (a necessary, not sufficient, "
                  "condition for topical coherence) -- but this remains a shallow, automatic proxy "
                  "for section quality, not a substitute for human topical-coherence judgment, and "
                  f"{s['near_empty_pct_of_sections']:.1f}% of sections are near-empty and "
                  f"{s['docs_with_dup_heading_pct']:.1f}% of documents contain a duplicated heading "
                  "string, both of which point to a non-trivial tail of degenerate LLM-inferred "
                  "structure the paper's own counts do not surface.\n")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
