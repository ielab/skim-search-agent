<!-- Advice sections of the Sieve manual, composed onto the reference by scripts/compose_manuals.py; never rendered on their own. -->

## How to search

1. Query entity NAMES, never the question's wording. Relation words (wrote, won, founded) are
   what you look FOR in the fetched section, not what you search for. Unsure between spellings?
   `OR` them: `zurich[title] OR "zürich"[title]`. When the question only DESCRIBES a thing, do
   not AND the whole description: each added clause loses more documents, and a long AND of
   common words 0-hits. Start from the 1-2 tokens most likely to appear VERBATIM in the target
   document (a proper name, a domain term, an exact number like `1897`), not the framing words.
2. Field-scope tightly. `x[title]` = the document is ABOUT x. `x[section]` = the document has a
   whole section on x (a results table, a career, a cast list). `x[body]` = x is mentioned
   somewhere. `x[tiab]` = title or body. `x[author]` = the document is bylined to x. `x[date]`
   = the publication date carries x (a year, or a month name). There is no `infobox` field on
   this corpus; `author` and `date` hold its structured facts.
3. Use the listing to decide which hit to fetch: the `matched:` fields show whether your terms
   landed in the title, a heading, the byline, the date, or only the body, and the section
   names show where the fact would sit. The listing never shows a field's VALUE; fetch to
   read it.

## Hops: chaining across searches

Each hop is its own search + fetch, not a bigger query. The fetched section names the next
entity: search THAT. No document of its own? Flip direction: search it as `[tiab]` instead of
`[title]`; the fact usually sits on a page that mentions it. 0 hits means your SURFACE is
wrong, not that the document is absent. Recover in ONE move: drop a long name to its 1-2 most
distinctive words, try `word*`, or move `[title]` to `[tiab]`. Two loosenings of the SAME
entity both 0-hit? PIVOT to a different entity the question names; never answer from memory.

Before you stop: check the fact is the ASKED-FOR TYPE, not just the next entity in the chain.
Once a section shows a fact of the right type, ANSWER; do not re-search to confirm. The answer
is the shortest span COPIED VERBATIM from the section: exact spelling and accents, and just the
asked-for unit, not a compound (the year alone, not "March 2009").

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
