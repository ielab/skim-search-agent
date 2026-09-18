# Structured document search: field-tagged `term[field]` Boolean + fetch (web documents with sections)

Search the corpus with a Boolean query language: one expression selects the documents whose
named fields contain your words. It matches words, not meaning (no embeddings), so exact terms
matter; use `OR` or `*` for variants. The language: `term[field]`, `AND`/`OR`/`NOT`,
parentheses, wildcard `*`, quoted `"phrase"`. Terms are case-insensitive. A **search never
shows document bodies**: each hit shows the title, which fields your terms matched, and the
document's section names. **fetch** then reads ONE named section. Two moves, in order:

1. **search** a query -> ranked DOCUMENTS, numbered for `fetch`: each row shows the title, a
   `matched:` list of the fields your terms landed in (`title`/`section`/`author`/`date`/`body`),
   and the section names `§[...]`.
2. **fetch** one named section of a numbered hit to read its text. Never the whole document.

## The fields

| field | matches | reach for it when |
|---|---|---|
| `title` \| `ti` | the document's title/headline | the entity should be the document's SUBJECT |
| `section` \| `sec` | the section headings | the document devotes a whole section to it |
| `body` \| `ab` \| `text` | the full document text | it is merely mentioned, not the subject |
| `author` | the byline (person or org) | "written/published/reported BY x" |
| `date` | the publication date | "published/dated/from" a year, month, or full date |
| `tiab` | title OR body (combo) | a first broad pass before narrowing |

Not every document carries a byline (`author` is empty on wire copy and anonymous posts). If
`x[author]` 0-hits on a name you are confident about, look for the same fact in `[body]`
instead of re-guessing the byline.

Atoms: a bare term matches anywhere in the scoped field. Several bare words parse as one exact
adjacent PHRASE, which 0-hits unless truly contiguous (a quoted headline). `word*` widens to any
token starting with it. `term[f1,f2]` matches if EITHER field has it. Combine: `A OR B`
(either), `A AND B` (both; use sparingly, 3+ clauses usually over-specify and 0-hit),
`A NOT B`. Parentheses group as expected.

## Fetch: reading the section you found

Every document is split into named sections. The listing shows up to eight section names per
hit (`,…` means there are more). `fetch` takes the hit's rank and ONE section name:

```
{"rank": 1, "section": "Career"}
```

- `section` is a name from that hit's `§[...]` list. Three special names: `""` reads the
  opening text, `infobox` reads the document's facts (title, author, date), and `body` reads the
  whole page, every section in order. `body` costs as much as visiting the page (up to the
  12,000-token cap), so use it only when the listing gives no section to aim at; a named section
  is the cheap read.
- Fetch the ONE section the fact should live in; the section names already tell you where. A
  results table sits under a "Results" or "Final" heading, a biography fact under "Early life"
  or "Career", a byline or date under `infobox`.
- If that section truncates or lacks the fact, fetch a DIFFERENT named section that would hold
  it, not the same one again. An unopened named section is a lead; never abandon a retrieved
  document or fill the gap from memory.
- The rank refers to the LAST listing. A new search renumbers the rows, so fetch from the
  listing you are looking at, or search again.

## Worked examples

- `harbor[title]`: the document about the place; fetch the section whose name fits the fact.
- `smith[author]`: documents bylined to someone named Smith.
- `2014[date]`: documents published in 2014.
- `final[section] AND championship[title]`: a championship page with a section on the final.
- `festival[title] AND film[body]`: which same-named hit is the film, not the event.
- `studio[body]`: nothing is titled for it; find the pages that mention it.
- `munoz[title] OR "muñoz"[title]`: accent/spelling variants in one search.
- fetch `{"rank": 1, "section": "Career"}`: read result 1's Career section, then chain or answer.
- fetch `{"rank": 1, "section": "infobox"}`: result 1's title, byline and date.

## Common mistakes

- A comma or "and" meant loosely: write real `AND`/`OR`; `AND` requires ALL clauses.
- `A AND B` to CONNECT two entities from different hops (a person `AND` their team's founder):
  they never share ONE document, so it 0-hits. Hop instead: search A, fetch, then search what
  you found. A tight `AND` is for ONE entity's own distinctive words (`telescope AND 1893`).
- Searching the question's framing words (described, mentioned) instead of an entity NAME.
- Fetching `body` when the listing already names the section that holds the fact: the whole page
  costs many times the tokens of one section.
- Fetching a section name from a different row, or from an earlier listing: check the rank.
- Fetching section after section on one document: the listing already named which section
  should hold the fact; if two named sections lack it, move to a different document.
- Re-searching to confirm a fact a section already shows, or hunting a title equal to the
  answer: answer from the section you already have.
- Answering the intermediate entity, or a compound, instead of the TYPE actually asked for.
- Swapping in a more familiar entity that merely sounds like the question's name: search the
  LITERAL name bare first; only widen if that 0-hits.
- Assuming a 0-hit `[author]` means the fact is missing: the document may be unbylined; look
  for the SAME fact in `[body]` before giving up on the document.
