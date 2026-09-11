"""Field schema for the Lucene structured backend, shared by `index_builder.py`
(writes it) and the compilers and engine (read it).

## Field layout

One Lucene `Document` per `CodeUnit`: one per corpus document for the
`browsecomp_plus_structured`/`hotpotqa_structured`/... datasets this backend targets
(`agent_search.evaluation.datasets`'s `units_from_documents` already maps one input JSONL row to
one `CodeUnit`, with no chunking; see its module docstring). This matches how both
Python reference engines score: `indri.model.IndriExecutor` and
`bql.executor.StructuralExecutor` both treat one `CodeUnit` as one scorable and
matchable entity, with `sections` (the structured corpus's ordered
`[{heading, text}, ...]` list) folded into a single joined `section` string field on
that unit, not split into separate child documents. Modeling `sections` as Lucene
child docs (`IndexWriter` block-join `addDocuments(List<Document>)`) would require
`ToParentBlockJoinQuery` throughout the compiler for every field-scoped op, and
would score at a granularity (section) neither Python engine uses, so it would stop
being comparable for the validation suite's "same match sets, correlated rankings"
requirement. Section-level retrieval would need a granularity decision made for
both engines together, not something this backend can decide alone.

**Storage.** Only `id` is `FieldStore.YES`; every other field below is indexed
(tokenized and positioned, for matching and scoring) but not stored. The tool layer
that calls this engine already holds the live corpus (`doc_id -> CodeUnit` mapping)
and reads title, section, date, author and body from there, not from the search
index, so storing them a second time here would only bloat the on-disk index for
data this engine's callers never read back through it. `LuceneHit` carries
`doc_id`, `score` and `matched_fields` only, no title, date or author.

**Per field**:
  - `id` (StringField, stored): the `CodeUnit.doc_id` / corpus `_id`, how a hit maps
    back to a document.
  - `body` (TextField, `EnglishAnalyzer`: Porter stemming plus English stopwords,
    not stored): the primary LMDirichlet-scored field for `#combine`/`#weight`/`#or`/
    `#max` and BQL's coverage/BM25-style ranking. Stemmed for the same reason
    `bm25_pyserini` stems (see that module's docstring): broader recall, standard IR
    practice, and what Indri itself does by default. The reference engine's
    "Deviations" section documents that the Python engine does not stem, so this
    Lucene backend is a graded-ranking equivalent, not a byte-identical one; see
    `indri_compiler.py`'s Deviations for how that is validated.
  - `body_exact` (TextField, `SimpleAnalyzer`: lowercasing plus letter-run
    tokenization, no stemming, no stopwords, not stored): the literal/positional
    field for span/window ops (`#odN`/`#uwN`/`#syn` as SpanOr) and for exact boolean
    "matches" tests (`#band`/`#filreq`/`#filrej`) that the validation suite requires
    to agree with the Python reference's match set (analyzer-light). This is the
    field that `SpanTermQuery`/`SpanNearQuery` run against.
  - `title` / `title_exact`, `section` / `section_exact`: the same stemmed/exact pair
    repeated for the `title` and `section` fields, so field-scoped ops
    (`dog.title`, `#od2(a b).section`) get the same match-set-vs-graded-ranking
    treatment as the default `body` scope. Without this pair, every field-scoped
    span/boolean op would have to degrade to stemmed-only matching, which would
    diverge from the Python reference's match set since it never stems. Neither the
    stemmed nor `_exact` copies are stored (see "Storage" above).
  - `date` (StringField, ISO `YYYY-MM-DD`, not stored): filter/range field only,
    matching `#date:before/after/between` and BQL's `date[RANGE]`. ISO strings sort
    lexicographically the same as chronological order, so `TermRangeQuery` over this
    field is an exact range filter with no numeric-field machinery needed. The field
    is omitted entirely (not written as an empty value) for a document with no
    date, because an empty string would sort lexicographically before every real
    date and would match every open-lower-bound `#date:before(...)` query; see
    `index_builder.py`'s `_build_document`.
  - `author` (StringField, not stored): an exact, unanalyzed whole-value filter
    (`author == "Jane Smith"`). Approximation, documented in `indri_compiler.py`: the
    Python Indri reference allows `dog.author` to LM-Dirichlet-score the `author`
    field, since its `FIELDS` tuple treats author/date like any other text field;
    this Lucene backend does not replicate that, because author values are short,
    sparse proper nouns and LMD scoring over them is rarely meaningful. A
    `.author`-scoped Indri scoring leaf compiles to an exact `TermQuery` on this
    StringField instead of Dirichlet-smoothed LM scoring.
  - `author_text` / `date_text` (TextField, `SimpleAnalyzer`, same unstemmed
    analyzer as the `_exact` fields, not stored): a tokenized sibling of
    `author`/`date`, needed because BQL's Python reference (`bql/executor.py`'s
    `_field_bag`) tokenizes `metadata['author']`/`metadata['date']` word by word for
    `IN(author, x)`/`IN(date, x)` matching (`code_tokenize(str(meta.get('author')))`),
    a per-token match rather than a whole-string equality test. Without this pair,
    `bql_compiler.py` would have to approximate `IN(author, x)` as "author field
    equals x exactly", which diverges from the reference on any multi-word author
    string. `date_text` is unused by the range operators
    (`#date:before/after/between`, BQL's `date[RANGE]`), which use the ISO
    StringField `date` for `TermRangeQuery`; it exists only for a bare
    `IN(date, "1980")`-style token match, mirroring `_field_bag`'s date branch.
"""
from __future__ import annotations

# Public field names, so callers never hand-type a string that could typo-diverge
# between index_builder.py (writer) and the compilers (reader).
F_ID = "id"
F_BODY = "body"
F_BODY_EXACT = "body_exact"
F_TITLE = "title"
F_TITLE_EXACT = "title_exact"
F_SECTION = "section"
F_SECTION_EXACT = "section_exact"
F_DATE = "date"
F_AUTHOR = "author"
F_AUTHOR_TEXT = "author_text"
F_DATE_TEXT = "date_text"

# Indri/BQL field name -> (scored field, exact field) for field-restriction ops.
# Any field name NOT in this map (e.g. an arbitrary/unknown field, or "author"/
# "date") has no stemmed/exact scoring pair -- callers fall back to the filter-only
# StringField path documented above.
SCORED_FIELD_MAP = {
    "body": (F_BODY, F_BODY_EXACT),
    "title": (F_TITLE, F_TITLE_EXACT),
    "section": (F_SECTION, F_SECTION_EXACT),
}

# Every stemmed-scored field (used to build the default `#combine` disjunction when
# no field restriction is given, and to construct PerFieldAnalyzerWrapper).
STEMMED_FIELDS = (F_BODY, F_TITLE, F_SECTION)
EXACT_FIELDS = (F_BODY_EXACT, F_TITLE_EXACT, F_SECTION_EXACT)
DEFAULT_SCORED_FIELD = F_BODY
DEFAULT_EXACT_FIELD = F_BODY_EXACT
