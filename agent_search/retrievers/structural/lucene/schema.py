"""Field-schema decision for the `lucene` structured backend, shared by
`index_builder.py` (writes it) and the compilers/engine (read it).

## Field-schema decision (documented per task instructions)

**One Lucene `Document` per `CodeUnit`** (i.e. per corpus document for the
`browsecomp_plus_structured`/`hotpotqa_structured`/... datasets this backend targets
-- `evaluation.datasets`'s `units_from_documents` already maps one input JSONL row to
one `CodeUnit`, with no chunking; see its module docstring). This mirrors EXACTLY how
both Python reference engines score: `indri.model.IndriExecutor` and
`bql.executor.StructuralExecutor` both treat one `CodeUnit` as one scorable/matchable
entity, with `sections` (the structured corpus's ordered `[{heading, text}, ...]`
list) folded into a single joined `section` STRING field on that unit, not split into
separate child documents. Modeling `sections` as Lucene child docs (`IndexWriter`
block-join `addDocuments(List<Document>)`) would require `ToParentBlockJoinQuery`
throughout the compiler for every field-scoped op, and would score at a GRANULARITY
(section) neither Python engine uses -- it would stop being a comparable backend for
the validation suite's "same match sets / correlated rankings" requirement. If a
future corpus needs section-level retrieval, that is a new granularity decision for
BOTH engines together, not something this backend should invent unilaterally.

**Storage: lean by design.** Only `id` is `FieldStore.YES`; every other field below
is indexed (tokenized/positioned, for matching + scoring) but NOT stored. The tool
layer that calls this engine already holds the live corpus (`doc_id -> CodeUnit`
mapping) and reads title/section/date/author/body from THAT, not from the search
index -- storing them a second time here would only bloat the on-disk index for
data this engine's callers never read back through it. (`title`/`section` were
originally also stored for standalone listings/diagnostics; changed to unstored
once it was clear the tool layer maps ids to its own corpus, per the coordinator's
efficiency requirement -- `LuceneHit` below carries `doc_id`/`score`/
`matched_fields` only, no title/date/author.)

**Per field**:
  - `id` (StringField, stored): the `CodeUnit.doc_id` / corpus `_id` -- how a hit maps
    back to a document.
  - `body` (TextField, `EnglishAnalyzer` -- Porter stemming + English stopwords, NOT
    stored): the primary LMDirichlet-SCORED field for `#combine`/`#weight`/`#or`/
    `#max` and BQL's coverage/BM25-style ranking. Stemmed for the same reason
    `bm25_pyserini` stems (see that module's docstring): broader recall, standard IR
    practice, and it's what Indri itself does by default (the reference engine's
    "Deviations" section already documents that OUR Python engine does NOT stem --
    this Lucene backend is a graded-ranking-equivalent, not a byte-identical one; see
    `indri_compiler.py`'s Deviations for how this is validated).
  - `body_exact` (TextField, `SimpleAnalyzer` -- lowercasing + letter-run
    tokenization, NO stemming, NO stopwords, NOT stored): the literal/positional
    field for span/window ops (`#odN`/`#uwN`/`#syn`-as-SpanOr) and for exact boolean
    "matches" tests (`#band`/`#filreq`/`#filrej`) that the validation suite requires
    to agree with the Python reference's match SET (analyzer-light). Required by the
    task; this is the one field the prereq POC (`pyserini_probe/lucene_poc.py`)
    specifically exercised SpanTermQuery/SpanNearQuery against.
  - `title` / `title_exact`, `section` / `section_exact`: the SAME stemmed/exact pair
    repeated for the `title` and `section` fields, so field-scoped ops
    (`dog.title`, `#od2(a b).section`) get the same match-set-vs-graded-ranking
    treatment as the default `body` scope. This is a DELIBERATE EXTENSION beyond the
    task's literal minimum schema ("title, section names") -- without it, every
    field-scoped span/boolean op would have to silently degrade to stemmed-only
    matching (a real match-set divergence from the Python reference, which never
    stems). Neither the stemmed nor `_exact` copies are stored (see "Storage" above).
  - `date` (StringField, ISO `YYYY-MM-DD`, NOT stored): filter/range field only,
    matching `#date:before/after/between` and BQL's `date[RANGE]`. ISO strings sort
    lexicographically identically to chronological order, so `TermRangeQuery` over
    this field is an exact range filter with no numeric-field machinery needed.
    OMITTED ENTIRELY (not just an empty value) for a doc with no date -- an empty
    string would sort lexicographically BEFORE every real date, silently matching
    every open-lower-bound "#date:before(...)" query (a real bug caught during
    validation; see `index_builder.py`'s `_build_document`).
  - `author` (StringField, NOT stored): the task's literal schema field -- an exact,
    unanalyzed WHOLE-VALUE filter (`author == "Jane Smith"`). **Approximation**
    (documented, see `indri_compiler.py`): the Python Indri reference technically
    allows `dog.author` to LM-Dirichlet-SCORE the `author` field (its `FIELDS` tuple
    treats author/date like any other text field); this Lucene backend does not
    replicate that (author values are short/sparse proper nouns -- LMD scoring over
    them is rarely meaningful) -- a `.author`-scoped Indri scoring leaf compiles to
    an exact `TermQuery` on this StringField instead of Dirichlet-smoothed LM
    scoring.
  - `author_text` / `date_text` (TextField, `SimpleAnalyzer` -- same unstemmed
    analyzer as the `_exact` fields, NOT stored): a TOKENIZED sibling of
    `author`/`date`, added because BQL's Python reference (`bql/executor.py`'s
    `_field_bag`) tokenizes `metadata['author']`/`metadata['date']` word-by-word for
    `IN(author, x)`/`IN(date, x)` matching (`code_tokenize(str(meta.get('author')))`)
    -- a per-token match, not a whole-string equality test. Without this pair,
    `bql_compiler.py` would have to approximate `IN(author, x)` as "author field
    equals x exactly", which silently diverges from the reference on any
    multi-word author string. `date_text` is unused by the range operators
    (`#date:before/after/between`, BQL's `date[RANGE]`), which use the ISO
    StringField `date` for `TermRangeQuery`; it exists only for a bare
    `IN(date, "1980")`-style TOKEN match, mirroring `_field_bag`'s DATE branch.
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
