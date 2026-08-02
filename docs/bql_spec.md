# BQL — the executor query language

A small, regular Boolean query language, typed by *granularity* so ill-formed queries are
rejected before execution and compiled to multiple backends from one abstract form.

**This is the EXECUTOR language, not the agent surface.** The agent writes a field-tagged
Boolean surface — `term[field]`, `AND`/`OR`/`NOT`, wildcard `*`, quoted `"phrase"` — which
`agent_search/retrievers/structural/bql/surface.py::to_bql` lowers to the BQL below (e.g.
`isnan[call]` → `IN(call, isnan)`; `save[def] NOT test[file]` → `IN(def, save) AND
NOT(IN(file, test))`; `melanoma[title,body]` → `OR(IN(title, melanoma), IN(body, melanoma))`).
The parser / type checker / executor and their tests are reused verbatim; only the surface the
agent types changed. The direct `bql` retriever accepts BQL strings here directly (oracle/debug).

Design rules:
- **Small regular operator set.** Every operator is a precision or recall knob.
- **Granularity typing.** Once proximity and scope exist, the unit of retrieval is
  a *region*, not a document; operators must agree on the region.
- **Field-tagged surface, functional executor.** The agent's `term[field]` surface maps to the
  functional `IN(field, term)` the executor evaluates; an unknown field passes through so the
  type checker rejects it with a readable reason (useful agent feedback).

---

## 0. Provenance: this is NOT a new Boolean logic

BQL deliberately does **not** invent retrieval semantics. Its operator set is the classical
professional-search Boolean kit that systematic-review databases, legal search, and Lucene have
converged on over five decades — that convergence is evidence these are *the* operators expert
searchers need. Correspondence:

| BQL | systematic-review / Ovid | Westlaw / Lexis | Lucene / ProQuest |
|---|---|---|---|
| `AND(a, b)` / `OR` / `AND(a, NOT(b))` | `a AND b`, `OR`, `NOT` | same | same |
| `PHRASE(w1, w2)` | `"w1 w2"` | `"w1 w2"` | `"w1 w2"` |
| `NEAR/w5(a, b)` | Ovid `a adj5 b` | `a /5 b`, `a w/5 b` | `a NEAR/5 b`, `"a b"~5` |
| `NEAR/sent` / `NEAR/para` | — | `/s`, `/p` | — |
| `PREFIX(auth)` | `auth*` (truncation) | `auth!` | `auth*` |
| `IN(title, x)` | `x[ti]`, Ovid `x.ti.` | `TI(x)` | `title:x` |
| `EXPAND(x, synonym)` | thesaurus term mapping / explosion | — | — |
| `IN(def/call/sig/comment/string, x)` | **no equivalent** | **no equivalent** | **no equivalent** |

What BQL adds beyond the classical kit (the contribution):
1. **Code-native fields.** `IN(def/call/sig/comment/string, ·)` is the AST-scope analog of a
   field tag like `[ti]`/`[ab]` — field search where the "fields" are structural roles in code,
   computed live from the AST. The agent reaches these as `x[def]` / `x[call]` / `x[string]`.
2. **A granularity type system.** No classical system type-checks scope nesting; BQL rejects
   `NEAR/w5(NEAR/file(a,b), c)` as ill-formed instead of guessing.
3. **Index-free execution.** Classical Boolean operators are defined over inverted indexes;
   BQL's reference executor evaluates by direct scan of live files, so there is nothing to
   build, ship, or invalidate.
4. **Two surfaces, one executor.** The functional form (`AND(a, b)`, `IN(def, x)`) is the
   canonical executor input; the parser ALSO accepts the classical infix surface (`(a OR b)
   AND c`, `NEAR/5`, case-insensitive operators), and the agent's field-tagged `term[field]`
   surface lowers to the functional form — so all three parse to the same AST.

---

## 1. Grammar (EBNF)

```ebnf
Query    := Expr
Expr     := And | Or | Diff | Near | In | Expand | Phrase | Prefix | Term

And      := "AND" "(" Expr "," Expr { "," Expr } ")"
Or       := "OR"  "(" Expr "," Expr { "," Expr } ")"
Diff     := "AND" "(" Expr "," "NOT" "(" Expr ")" ")"     ; NOT only as set-difference
Near     := "NEAR" "/" Gran "(" Expr "," Expr ")"
In       := "IN" "(" Region "," Expr ")"
Expand   := "EXPAND" "(" Term [ "," Strategy ] ")"
Phrase   := "PHRASE" "(" Term { "," Term } ")"
Prefix   := "PREFIX" "(" stem ")"                          ; truncation / wildcard
Term     := ident | quoted

Gran     := "w" int | int | "line" int | "sent" | "para" | "block" | "func" | "file"
Region   := "title" | "body" | "section" | "comment" | "string"
          | "def" | "call" | "sig" | "file" | "doc"
Strategy := "lexical" | "symbol" | "synonym"              ; how EXPAND resolves
```

Surface tolerance (canonical form is prefix; the parser additionally accepts the
classical Boolean surface so raw model output parses): infix `a AND b` / `a OR b`
(precedence OR < AND), parenthesized grouping `(a OR b) AND c`, top-level comma as
implicit AND, case-insensitive operator/region/strategy names, Lucene-style numeric
proximity `NEAR/5(a, b)` (= `NEAR/w5`), escaped quotes in quoted terms, and a
**bare multi-word run** (`IN(string, must be positive)`) read as an implicit
`PHRASE(...)` — the model's natural form, which used to hard-error. All tolerance is
purely syntactic — it canonicalizes to the same AST; semantics and the type system
are unchanged. A genuine parse failure returns a structured error that suggests the
fix (e.g. quote or `PHRASE(...)` a multi-word operand).

Three deliberate constraints:
- **NOT exists only as the right child of AND** (set-difference). Unbounded
  negation is semantically dangerous (matches almost everything) and a pruning
  worst case. The grammar forbids standalone `NOT`.
- **EXPAND carries a strategy tag**, so one operator covers text synonym expansion
  and code symbol-graph expansion.
- **Gran unifies proximity and scope** — "within 5 tokens", "same function", "same
  file" are one parameter on one operator.

## 2. Operator → retrieval-polarity spine

| Tightens precision | Widens recall | Sets matching region |
|---|---|---|
| `AND`, `NOT`(=Diff), `PHRASE`, `NEAR` | `OR`, `EXPAND`, `PREFIX` | `IN`, the `/Gran` on `NEAR` |

The agent's learned job: pick which column to reach into given trajectory
feedback — tighten when the last query over-returned, widen when it returned
nothing. This table is the paper's one-line statement of what the policy learns.

## 3. Granularity type system

Lattice: `token < line < block < func < file < doc`.

- `Term`, `Phrase`, `Prefix`, `Expand` → **token**-level matches.
- `NEAR/g` and `IN(region, ·)` **lift** their argument to level `g` (resp. the
  region's level).
- `AND`/`OR` require children to **share a level**, else lift to the coarser one.

This rejects nonsense before execution. Example:
`AND(IN(comment, x), IN(def, y))` is satisfiable only if a `comment` region and a
`def` region can co-occur in a shared binding region — the type system resolves
the binding region to the enclosing function/file, not the token. Without this,
the agent emits queries with undefined region semantics and gets silent garbage.

Type checker output is part of the **observation** when a query is ill-typed, so
the agent (and, during RL, the reward) can react to malformedness.

## 4. Worked example

Intent: *find where auth tokens are **defined** (not in tests), with `auth` and
`token` close together.*

```
AND(
  IN(def, NEAR/func( EXPAND(auth, symbol), EXPAND(token, symbol) )),
  NOT( IN(file, PREFIX(test)) )
)
```

- **Text backend:** `SpanNearQuery` over the `def` field, wrapped in a
  `BooleanQuery` with a `MUST_NOT` clause.
- **Code backend:** ripgrep prefilter for the expanded identifiers → ast-grep
  verifies both appear in a definition node within one function, minus test paths.

## 5. Backend targets

One typed AST can be evaluated by the current reference executor, and can later be
lowered to backend-specific programs:

| BQL node | Text (Lucene/Pyserini) | Code (ripgrep + ast-grep) |
|---|---|---|
| `AND/OR` | `BooleanQuery` MUST / SHOULD | piped greps / `(a\|b)` alternation |
| `NOT`(Diff) | `MUST_NOT` clause | `-v` / negative pattern, applied as post-filter |
| `PHRASE` | `PhraseQuery` (in order, adjacent) | exact substring / token sequence |
| `PREFIX` | `PrefixQuery` (stemming = soft form) | identifier-boundary-aware regex (`auth\w*`, camel/snake) |
| `NEAR/g` | `SpanNearQuery` slop=g | within N lines / same AST scope |
| `IN(region,·)` | `field:` (title/body/section) | AST role: def / call / sig / comment / string / test-file |
| `EXPAND` | corpus-validated synonym OR-set | identifier-variant set + symbol-graph (def→calls) |

**Field generalises to structural scope, and in code that is the AST — the single
strongest argument for the project.** `IN(def,·)`/`IN(call,·)`/`IN(comment,·)` is
exactly what dense embeddings blur into a vector. Make it first-class.

**PREFIX in code needs identifier-boundary awareness**: `PREFIX(auth)` should reach
`authToken`, `auth_token`, `Authenticate` across naming conventions — a
code-specific flavour classic truncation lacks.

**EXPAND is where corpus statistics are load-bearing**: the LLM proposes a variant
set; a future live-corpus resolver can prune df=0 / ultra-rare variants under a
recall budget before they enter the OR. `synonym` strategy for text, `symbol`
(naming variants + call graph) for code, `lexical` for raw prefix/wildcard families.

## 6. Cost model & efficient execution

> The **method (code) is index-free**: ripgrep prefilter → ast-grep/Python-AST verify
> over live files, ranked in-memory (no persisted index). The inverted-index / WAND
> notes below apply to the **text/Lucene path** (the BM25 *baseline*). At repo scale
> ripgrep scans in milliseconds, so the method needs no index.

**Implemented today (the reference executor, `structural/bql/executor.py`):** for a
LARGE corpus (≥ `AGENT_SEARCH_BQL_PREFILTER_MIN` units, default 5000 — i.e. the shared
document corpus, not a small per-query repo) selection is **two-phase filter-then-
verify**: a pure-Python **inverted index** (`token → units`, built once per corpus)
computes a recall-safe candidate superset from the query's positive leaves, then the
exact `_eval` verifies only those candidates. It is provably **result-identical** to
the full live scan (the index only narrows; `_eval` is still the arbiter), so it adds
scale with zero semantic change. Below the threshold (per-query code repos) the plain
index-free O(N) scan is used unchanged. The Lucene/WAND notes below are the standard
sublinear machinery the text path can additionally inherit.

Core retrieval is a **solved, sublinear** problem — inherit it where it applies:

- **Text/baseline core (`AND/OR/PHRASE/IN` over a Lucene index):** dynamic top-k
  pruning (MaxScore / WAND / BlockMax-WAND) skips most postings unscored. Cost ≈
  standard BM25 query.
- **EXPAND — the one real risk, and it is bounded.** Each EXPAND is a wide OR;
  wide disjunctions weaken WAND score upper bounds → less skipping. Control:
  resolving an EXPAND is `k` df-lookups (µs); cap width by a **recall budget** —
  sort variants by df, add until marginal recall gain < τ, drop df=0 and
  ultra-rare. Expansion width becomes a tunable knob, not an explosion. *Corpus
  statistics convert an open-ended disjunction into a bounded, selectivity-ordered
  one.*
- **Code structural scope — two-phase filter-then-verify (the systems
  contribution).** Never run AST matching over the whole repo.
  - *Phase A:* cheap lexical prefilter (inverted index / ripgrep) on EXPANDed terms
    → small candidate file set.
  - *Phase B:* ast-grep structural verification (`IN(def,·)`, `NEAR/func`) only on
    candidates.
  The genuine tension: wide EXPAND → big candidate set → expensive Phase B.
  Expansion width and verify cost are **coupled**; the recall-budget cap bounds the
  coupling. State this with the cost model — reviewers will probe it.
- **NOT and proximity resist pruning → schedule late.** NOT as a post-filter on the
  already-small candidate set; NEAR after cheap conjuncts have shrunk the set.
  Evaluation order: selective conjuncts → expansion → proximity → negation.

**Pseudocode — EXPAND under recall budget:**
```
resolve_expand(term, strategy, budget τ):
    cands = propose_variants(term, strategy)          # LLM / symbol graph / wildcard family
    cands = [c for c in cands if df(c) > 0]           # corpus-validate (stats service)
    cands.sort(key=lambda c: -df(c))                  # selectivity order
    kept, covered = [], 0
    for c in cands:
        gain = marginal_recall_estimate(c, kept)      # df-based coverage proxy
        if gain < τ: break
        kept.append(c); covered += gain
    return OR(kept)                                    # bounded disjunction
```

**Pseudocode — two-phase code execution:**
```
execute_code(ast):
    lex = lexical_subquery(ast)                        # EXPANDed terms, no structure
    candidates = ripgrep_or_index(lex)                 # Phase A: cheap, narrows files
    hits = []
    for f in candidates:                               # Phase B: only on candidates
        if ast_grep_verify(ast, f):                    # IN(region,·), NEAR/func
            hits.append(f)
    hits = bm25_rank(hits); hits = apply_not_filter(ast, hits)
    return hits
```

## 7. Observation format (what the agent sees back)

The corpus feedback that drives refinement. Per query, return:
- `n_hits` (total, pre-truncation) — selectivity signal.
- per-clause / per-EXPAND-variant `df` — lets the agent see a dead clause.
- top-k snippets with provenance (path:line, region) — evidence.
- type-check status (ok / which clause is ill-typed).

This makes the **global statistics a prior** and the **observed result the
conditional truth** (df *given the constraints already added*), which only
execution reveals.

## 8. Validity & robustness

- Parser returns structured errors → fed back as observation (not a crash).
- A query that fails type-check is never executed; the agent gets the reason.
- During RL, format/type validity is a reward term (penalise malformed) so the
  policy learns to emit well-formed BQL.
