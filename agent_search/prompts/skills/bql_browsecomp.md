# Structured document search: field-tagged `term[field]` Boolean + fetch (web/news documents)

Search the corpus with a Boolean query language: one expression selects the documents whose
named fields contain your words. It matches words, not meaning — no embeddings, so exact
terms matter (`OR` or `*` for variants). The language: `term[field]`, `AND`/`OR`/`NOT`,
parentheses, wildcard `*`, quoted `"phrase"`. A **search never shows document bodies or field
values** — only structure (the title, and WHICH fields your terms matched). **fetch** then
pulls the one slice that should hold the fact. Terms are case-insensitive. Two moves, in order:

1. **search** a query → ranked DOCUMENTS: each hit shows the title and which fields your
   terms matched (`title`/`author`/`date`/`body`) — numbered for `fetch`.
2. **fetch** the body of a numbered doc (there are no named sections or an infobox on this
   corpus — every doc is one slice) to read its title, byline, date, and text.

## How to search

1. Query entity NAMES, never the question's wording — relation words (wrote, published,
   dated) are what you look FOR in the fetched slice, not what you search for. Unsure between
   spellings? `OR` them: `zurich[title] OR "zürich"[title]`. When the question only DESCRIBES
   a thing (no name given), don't AND the whole description — each added clause loses more
   docs, so a long AND of common words 0-hits. Start from the 1–2 tokens most likely to appear
   VERBATIM in the target doc — a proper name, a domain term, an exact number like `1897` —
   not the framing words (described, mentioned).
2. Field-scope tightly. `x[title]` = the doc is ABOUT x; `x[body]` = x is merely mentioned
   somewhere; `x[tiab]` = title-or-body; `x[author]` = the doc is BYLINED to x (a person or
   org name); `x[date]` = the doc's publication date carries x (a year, or a month name).
   This corpus has NO `section` or `infobox` field — every document is flat (one slice, no
   headings, no fact box) — so scoping to those 0-hits on EVERY document; use `author`/`date`
   for the relational/structured facts a wiki-style corpus would put in an infobox.
3. Use the listing to CONFIRM a hit is worth fetching (its `matched:` line shows whether your
   terms landed in title/author/date/body), then fetch it to read the actual byline/date/text
   — the listing proves the scope worked, it never shows the field's VALUE.

## The fields

| field | matches | reach for it when |
|---|---|---|
| `title` \| `ti` | the document's title/headline | the entity should be the doc's SUBJECT |
| `body` \| `ab` \| `text` | the full document text | it's merely mentioned, not the subject |
| `author` | the byline (person or org) | "written/published/reported BY x" |
| `date` | the publication date | "published/dated/from" a year, month, or full date |
| `tiab` | title OR body (combo) | a first broad pass before narrowing |

`section` and `infobox` are NOT fields on this corpus — every document is one flat slice with
no headings and no fact box; scoping to them always 0-hits. Reach for `author`/`date` instead
when the question needs a structured, relational fact (who wrote it, when it ran) — those are
this corpus's real "infobox-equivalent" fields, populated on ~96% of documents. Not every
document carries a byline (`author` is sometimes empty, e.g. wire copy or anonymous posts) —
if `x[author]` 0-hits on a name you're confident is right, the doc may simply be unbylined;
fall back to finding the SAME fact stated in `[body]` instead of re-guessing the byline.

Atoms: a bare term matches anywhere in the scoped field. Several bare words parse as one exact
adjacent PHRASE — reliably 0-hits unless truly contiguous (a quoted headline). `word*` widens
to any token starting with it. `term[f1,f2]` matches if EITHER field has it. Combine: `A OR B`
(either), `A AND B` (both — use sparingly; 3+ clauses usually over-specify and 0-hit),
`A NOT B`. Parentheses group as expected.

## Fetch — reading the slice you found

`search` marks `matched: author,date,...` on a hit when your query's terms landed in that
doc's byline/date (proof the scope worked) — but it does NOT print the byline/date VALUE;
`fetch` the doc to read it. `fetch` takes `[rank, part]` pairs; since there is only ONE slice
per document on this corpus, any part name (or an empty string) returns the same full body:

```
{"specs": [[1, "body"]]}
```

- There is nothing else to fetch on a given doc — don't hunt for a "section" or "infobox" name
  that the listing never showed; the body is the whole document.
- If the fetched body lacks the fact, that fact is not on THIS doc — pivot to a different
  document (a different search hit, or a new search), never fill the gap from memory.

## Hops — chaining across searches

Each hop is its own search + fetch, not a bigger query. The fetched slice names the next
entity — search THAT. No doc of its own? Flip direction: search it as `[tiab]` instead of
`[title]` — the fact usually sits on the page that mentions it. 0 hits means your SURFACE is
wrong, not that the doc is absent — recover in ONE move: drop a long name to its 1–2 most
distinctive words, try `word*`, or move `[title]` to `[tiab]`. Two loosenings of the SAME
entity both 0-hit? PIVOT to a different entity the question names; never answer from memory.

Before you stop: check the fact is the ASKED-FOR TYPE, not just the next entity in the chain.
Once a slice shows a fact of the right type, ANSWER — don't re-search to confirm. The answer is
the shortest span COPIED VERBATIM from the slice — exact spelling and accents, and just the
asked-for unit, not a compound (the year alone, not "March 2009").

## Worked examples

- `harbor[title]` — the doc about the place; fetch its `body` for the facts (no infobox here).
- `smith[author]` — documents bylined to someone named Smith (a real, working field on this
  corpus — unlike `infobox`, which never matches).
- `2014[date]` — documents published/dated in 2014.
- `AND(election[title], smith[author])` — an "election" doc bylined to Smith.
- `festival[title] AND film[body]` — which same-named hit is the film, not the event.
- `studio[body]` — nothing is titled for it; find the pages that mention it.
- `munoz[title] OR "muñoz"[title]` — accent/spelling variants in one search.
- fetch `{"specs": [[1, "body"]]}` — read result 1's full text, then chain or answer.

## Common mistakes

- Scoping to `[section]` or `[infobox]` — this corpus has neither; you will 0-hit every time.
  Use `[author]`/`[date]` for structured facts, `[body]` for anything stated in the text.
- A comma or "and" meant loosely — write real `AND`/`OR`; `AND` requires ALL clauses.
- Searching the question's framing words (described, mentioned) instead of an entity NAME.
- Re-fetching the same doc hoping for a different "section" — there is only one slice; if it
  lacks the fact, move to a different document.
- Re-searching to confirm a fact a slice already shows, or hunting a title equal to the answer
  — just answer from the slice you already have.
- Answering the intermediate entity, or a compound, instead of the TYPE actually asked for.
- Swapping in a more familiar entity that merely sounds like the question's name — search the
  LITERAL name bare first; only widen if that 0-hits.
- Assuming a 0-hit `[author]` means the fact is missing — it may just mean the doc is
  unbylined; look for the SAME fact in `[body]` before giving up on the document.
