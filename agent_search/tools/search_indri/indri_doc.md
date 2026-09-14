# Indri structured query language — graded belief search + fetch

Adapted from the Indri Query Language reference (lemurproject.org / cs.cmu.edu "IndriQueryLanguage").
`isearch` runs a query through a Dirichlet-smoothed LANGUAGE-MODEL scorer, not a hard Boolean
filter — a query with several terms NEVER hard-zeros; documents missing a term are ranked
lower, not excluded. **fetch** then pulls a named section (or the infobox) of a numbered hit —
the same structured read as the BQL arms. Two moves, in order:

1. **isearch** a query → ranked DOCUMENTS by belief (log-probability the document "generated"
   your query), plus a `weakest constraint for top hit` diagnostic naming which clause of your
   query contributed LEAST to the top hit's score — numbered for `fetch`.
2. **fetch** a named section (or `infobox`) of a numbered doc, to read its text.

## RULE: prefer operators over bare keywords

Prefer operators over bare keywords: exact names/titles go in `#1(...)`, field-scope names with
`.title`/`.author`, temporal clues with `#date:between(...)`; combine constraints under one
`#combine(...)`. A bare-keyword query still ranks (isearch never hard-zeros), but it throws away
the structure that makes the ranking sharp — the belief scorer can't tell "the exact phrase" from
"these words somewhere" or "the byline" from "a body mention" unless you tell it.

Worked example — "a 2002 bank management ceremony" (a name, a topic, and a date, combined):

```
#combine( #1(bank management ceremony) #date:between(2002-01-01 2002-12-31) )
```

## Only this subset is implemented — everything else errors

- **Terms**: `dog` (stemmed), `"NASA"` (unstemmed/exact), `term*` (suffix wildcard).
- **Proximity windows**: `#odN(a b)` / `#N(a b)` — ORDERED, at most N-1 terms between each
  (`#1(white house)` = exact phrase; `#2(white house)` matches "white * house"); `#od(a b)` =
  unlimited ordered window. `#uwN(a b)` — UNORDERED, all terms within a span of N (any order);
  `#uw(a b)` = unlimited.
- **Synonyms**: `#syn(a b)` / `{a b}` / `<a b>` — "a OR b" counted as one occurrence type;
  `#wsyn(1.0 car 0.5 automobile)` — weighted synonym set.
- **Belief (combination) operators**: `#combine(dog train)` (mean of log-beliefs — the safe
  default combiner), `#weight(1.0 dog 0.5 train)` (weighted mean), `#or(dog cat)`, `#not(dog)`,
  `#max(dog train)` (best child), `#band(dog train)` (ALL children must literally match, else
  the whole thing is excluded — the one hard-Boolean operator in this subset).
- **Field restriction**: `term.field` restricts COUNTING to that field (`dog.title` — only
  title occurrences of "dog" count); `term.(field)` restricts SMOOTHING/context only; combine
  as `term.field.(field)`. Multi-field: `dog.title,section`. Available fields: `title`,
  `body`/`text`, `section`, `author`, `date`.
- **Filters** (first argument must be a term/proximity expression): `#filreq(A Q)` — only
  documents matching `A` are candidates, ranked by belief in `Q`; `#filrej(A Q)` — only
  documents NOT matching `A`, ranked by `Q`.
- **Date operators** (over `metadata['date']`, ISO `YYYY-MM-DD`/`YYYY-MM`/`YYYY`):
  `#date:before(D)`, `#date:after(D)`, `#date:between(D_low D_high)` — also spelled
  `#datebefore(D)` / `#dateafter(D)` / `#datebetween(D_low D_high)` (both spellings accepted);
  `#between(date LOW HIGH)` is the same as `#date:between`. A document with a missing or
  malformed date never satisfies ANY date operator.

## NOT implemented — these WILL error, naming the unsupported op

`#prior(...)`, `#any:FIELD`, `#base64hash(...)`/`#base64string(...)`, extent/passage forms
(`#combine[passage200:100](...)`, `#weight[field](...)`), `#wsum(...)`, `#wand(...)`,
`#sum(...)`, and numeric field operators other than `#between(date ...)` (`#less`, `#greater`,
`#equals`). If you get `ERROR: #<op>@<pos>: ...`, drop that operator — reach for `#combine`/
`#weight`/`#filreq`/the date operators above instead; don't retry the same unsupported op.

## Reading the results

`isearch` never returns zero hits for a non-empty query (unless every operator in it errors) —
even a `#combine` of 5 rare terms where NO document has all 5 still ranks the corpus's closest
documents by belief. Use the score gap, not "0 vs. nonzero", to judge relevance: a huge score
gap between rank 1 and rank 2 is a strong signal; a small gap means several candidates are
worth fetching. The `weakest constraint for top hit: <clause> (logb=...)` line names which part
of your query the TOP hit satisfies least well — if that's the clause carrying your key fact,
the top hit is probably the wrong document; pivot rather than fetch it.

## Worked examples (multi-constraint, browsecomp-shaped clues)

- `#combine(bank management ceremony.body #date:between(2002-01-01 2002-12-31))` — "an article
  from 2002 about a bank management ceremony": three soft constraints combined; nothing needs
  to match ALL of them for the doc to rank, but the doc satisfying the most (and the date gate)
  rises to the top.
- `#filreq(sheep #combine(dolly cloning))` — restrict candidates to documents that literally
  mention "sheep", then rank those by belief in "dolly"/"cloning" — use `#filreq` when one
  clue is a hard MUST (a name, a place) and the rest are softer supporting details.
- `#od1(nobel peace prize)` — an exact ordered phrase (adjacent, in order); use `#odN` with a
  small N when the clue names something that should appear as a near-exact phrase.
- `#weight(3.0 "Marie Curie" 1.0 radioactivity.body)` — heavily favor an exact byline/name
  match over a loosely-related body mention, when the name is the more reliable clue.
- `#band(nobel.title physics.title)` — a hard requirement that a title carry BOTH terms (use
  sparingly — `#band` is the one operator here that can 0-hit, since it's real Boolean AND).

## Fetch — reading the slice you found

Same contract as the BQL arms: `fetch` takes a rank and one section name against `isearch`'s
numbering (`{"rank": 1, "section": "infobox"}`). Fetch the ONE slice most likely to hold the fact
(an early section for what/who it is, `infobox` for a relational fact); if it lacks the fact,
fetch a DIFFERENT named section or pivot to the next-ranked hit — never fill the gap from
memory.

## Common mistakes

- Reaching for `#prior`/`#any`/`#wsum`/numeric `#less`/`#greater` — none are implemented; you
  will get a structured `ERROR: #<op>@...` every time. Use `#combine`/`#filreq`/the date ops.
- Treating `#combine`'s low-but-nonzero score as "no match" — it graded-ranks everything; check
  the score GAP and the `weakest constraint` line, don't assume rank 1 is wrong just because
  no single doc had every term.
- Using `#band` for soft/multi-clue questions — it hard-excludes any doc missing ONE child,
  reintroducing the 0-hit brittleness `#combine`/`#weight` exist to avoid. Reserve `#band` for
  a genuine "both must be literally true" requirement.
- Forgetting field restriction — a bare `dog` counts occurrences everywhere; `dog.title` is
  the "this document is ABOUT dog" signal, closer to a title-field search in the BQL skill.
