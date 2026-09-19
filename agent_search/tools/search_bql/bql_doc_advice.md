<!-- Advice sections of the Sieve manual, composed onto the reference by scripts/compose_manuals.py; never rendered on their own. -->

## How to search

1. Query entity NAMES, never the question's wording — relation words (spouse, owner,
   founder) are what you look FOR in the fetched slice, not what you search for. Unsure
   between spellings? `OR` them: `zurich[title] OR "zürich"[title]`. When the question only
   DESCRIBES a thing (no name given), don't AND the whole description — each added clause
   loses more docs, so a long AND of common words 0-hits. Start from the 1–2 tokens most
   likely to appear VERBATIM in the target doc — a proper name, a domain term, an exact
   number like `1897` — not the framing words (described, mentioned).
2. Field-scope tightly. `x[title]` = the doc is ABOUT x; `x[body]` = x is merely mentioned
   somewhere; `x[tiab]` = title-or-body; `x[infobox]` = the reverse link — whose FACTS name
   x (a founder, an owner) when x has no page of its own.
3. Pick the hit by the TYPE the question implies, from the listing, not a fetch. Infobox keys
   type a hit at a glance — Released/Label = a work, Born/Spouse = a person, Country/State =
   a place. A title carrying a work marker — `(album)`/`(film)`/`(song)` — is that work, not
   a same-named person or place.

## Hops — chaining across searches

Each hop is its own search + fetch, not a bigger query. The fetched slice names the next
entity — search THAT. No doc of its own? Flip direction: search it as `[tiab]`/`[infobox]`
instead of `[title]` — the fact usually sits on the page that mentions it. 0 hits means your
SURFACE is wrong, not that the doc is absent — recover in ONE move: drop a long name to its
1–2 most distinctive words, try `word*`, or move `[title]` to `[tiab]`. Two loosenings of the
SAME entity both 0-hit? PIVOT to a different entity the question names; never answer from
memory.

Before you stop: check the fact is the ASKED-FOR TYPE, not just the next entity in the chain.
Once a slice shows a fact of the right type, ANSWER — don't re-search to confirm. The answer is
the shortest span COPIED VERBATIM from the slice — exact spelling and accents, and just the
asked-for unit, not a compound (the state alone, not "City, State").

## Common mistakes

- A comma or "and" meant loosely — write real `AND`/`OR`; `AND` requires ALL clauses.
- `A AND B` to CONNECT two entities from different hops (a person `AND` their team's founder)
  — they never share ONE document, so it 0-hits. Hop instead: search A, fetch, then search
  what you found. A tight `AND` is for ONE entity's own distinctive words (`telescope AND 1893`).
- Searching the question's framing words (described, mentioned) instead of an entity NAME.
- Fetching section after section — the listing already named which slice to open.
- Re-searching to confirm a fact a slice already shows, or hunting a title equal to the answer
  — just answer from the slice you already have.
- Answering the intermediate entity, or a compound, instead of the TYPE actually asked for.
- Swapping in a more familiar entity that merely sounds like the question's name — search the
  LITERAL name bare first; only widen if that 0-hits.
