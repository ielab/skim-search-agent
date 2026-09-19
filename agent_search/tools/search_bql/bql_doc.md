# Structured document search: field-tagged `term[field]` Boolean + fetch

Search the corpus with a Boolean query language: one expression selects the documents whose
named fields contain your words. It matches words, not meaning — no embeddings, so exact
terms matter (`OR` or `*` for variants). The language: `term[field]`, `AND`/`OR`/`NOT`,
parentheses, wildcard `*`, quoted `"phrase"`. A **search never shows document bodies** — only
structure (title, matched fields, section names, infobox keys). **fetch** then pulls the one
slice that should hold the fact. Terms are case-insensitive. Two operations:

1. **search** a query → ranked DOCUMENTS: each hit shows the title, the fields your terms
   matched, the doc's section names, and its infobox keys — numbered for `fetch`.
2. **fetch** a named section (or `infobox`) of a numbered doc — a name from that doc's list —
   to read its text. Never the whole document.

## The fields

| field | matches | reach for it when |
|---|---|---|
| `title` \| `ti` | the document's name | the entity should be the doc's SUBJECT |
| `section` \| `sec` | the heading names | a doc devotes a whole section to it |
| `body` \| `ab` \| `text` | the section texts | it's merely mentioned, not the subject |
| `infobox` \| `ib` | the key: value facts | the reverse link — whose facts name it |
| `tiab` | title OR body (combo) | a first broad pass before narrowing |

Not every corpus has real sections/infobox — a flat (non-Wikipedia-structured) document is just
title + body, so `[section]`/`[infobox]` 0-hit there on EVERY doc; if one 0-hits immediately,
fall back to `[body]`/`[tiab]` rather than re-trying it on a different entity.

Atoms: a bare term matches anywhere in the scoped field. Several bare words parse as one exact
adjacent PHRASE — reliably 0-hits unless truly contiguous (an album title). `word*` widens to
any token starting with it. `term[f1,f2]` matches if EITHER field has it. Combine: `A OR B`
(either), `A AND B` (both — use sparingly; 3+ clauses usually over-specify and 0-hit),
`A NOT B`. Parentheses group as expected.

## Fetch — reading the slice you found

`search` lists each doc's section names as `§[History·Career·Legacy]` and infobox keys as
`ib[Born·Spouse]`. `fetch` takes the hit's rank and one section name from that numbering:

```
{"rank": 1, "section": "infobox"}
```

- Fetch the ONE slice the fact lives in — the section names already told you where: `infobox`
  for relational facts, an early section for what/who it is.
- If that slice truncates or lacks the fact, fetch a DIFFERENT named section that would hold
  it (biography / career / legacy / discography), not the same intro again — an unopened named
  section is a lead; never abandon a retrieved doc or fill the gap from memory.
- `section` is a name from that doc's `§[...]` list (or `infobox`), NEVER a doc id, the
  question's wording, or a name you only hope exists — the `rank` already picks the doc.

## Worked examples

- `harbor[title]` — the doc about the place; fetch its `infobox` for a relational fact.
- `festival[title] AND film[body]` — which same-named hit is the film (a work), not the event.
- `studio[body]` — nothing is titled for it; find the pages that mention it (reverse link).
- `munoz[title] OR "muñoz"[title]` — accent/spelling variants in one search.
- fetch `{"rank": 1, "section": "infobox"}` — read result 1's facts, then chain or answer.

