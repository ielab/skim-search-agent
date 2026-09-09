# BQL: the executor query language

A small, regular Boolean query language. It is typed by *granularity*, so ill-formed queries are
rejected before they execute, and one abstract form compiles to several backends.

**This is the EXECUTOR language, not what the agent types.** The agent writes a field-tagged
Boolean surface (`term[field]`, `AND`/`OR`/`NOT`, wildcard `*`, quoted `"phrase"`), and
`agent_search/retrievers/structural/bql/surface.py::to_bql` lowers that to the BQL below. So
`isnan[call]` becomes `IN(call, isnan)`, `save[def] NOT test[file]` becomes `IN(def, save) AND
NOT(IN(file, test))`, and `melanoma[title,body]` becomes `OR(IN(title, melanoma), IN(body,
melanoma))`. The parser, type checker and executor are shared verbatim between the two; only the
surface differs. The direct `bql` retriever takes BQL strings as-is, for oracle runs and
debugging.

Three design rules hold the language together:
- **A small, regular operator set.** Every operator is either a precision knob or a recall knob.
- **Granularity typing.** With proximity and scope, the unit of retrieval is a *region*, not a
  document, and operators must agree on which region they address.
- **Field-tagged surface, functional executor.** The agent's `term[field]` maps to the functional
  `IN(field, term)` the executor evaluates. An unknown field passes straight through so the type
  checker can reject it with a reason the agent can read and act on.

---

## 0. Provenance: this is NOT a new Boolean logic

BQL does not invent retrieval semantics. Its operator set is the classical professional-search
Boolean kit that systematic-review databases, legal search and Lucene converged on over five
decades. That convergence is the evidence these are the operators expert searchers need. The
correspondence:

| BQL | systematic-review / Ovid | Westlaw / Lexis | Lucene / ProQuest |
|---|---|---|---|
| `AND(a, b)` / `OR` / `AND(a, NOT(b))` | `a AND b`, `OR`, `NOT` | same | same |
| `PHRASE(w1, w2)` | `"w1 w2"` | `"w1 w2"` | `"w1 w2"` |
| `NEAR/w5(a, b)` | Ovid `a adj5 b` | `a /5 b`, `a w/5 b` | `a NEAR/5 b`, `"a b"~5` |
| `NEAR/sent` / `NEAR/para` |, | `/s`, `/p` |, |
| `PREFIX(auth)` | `auth*` (truncation) | `auth!` | `auth*` |
| `IN(title, x)` | `x[ti]`, Ovid `x.ti.` | `TI(x)` | `title:x` |
| `EXPAND(x, synonym)` | thesaurus term mapping / explosion |, |, |
| `IN(def/call/sig/comment/string, x)` | **no equivalent** | **no equivalent** | **no equivalent** |

Four things BQL adds on top of that classical kit:

1. **Code-native fields.** `IN(def/call/sig/comment/string, ·)` is the AST-scope analog of a field
   tag like `[ti]` or `[ab]`. It is field search where the "fields" are structural roles in code,
   computed live from the AST. The agent reaches them as `x[def]`, `x[call]`, `x[string]`.
2. **A granularity type system.** No classical system type-checks scope nesting. BQL rejects
   `NEAR/w5(NEAR/file(a,b), c)` as ill-formed instead of guessing the intent.
3. **Index-free execution.** Classical Boolean operators are defined over inverted indexes. BQL's
   reference executor evaluates by scanning live files, so there is nothing to build, ship or
   invalidate.
4. **Two surfaces, one executor.** The functional form (`AND(a, b)`, `IN(def, x)`) is the canonical
   executor input, the parser also accepts the classical infix surface (`(a OR b) AND c`,
   `NEAR/5`, case-insensitive operators), and the agent's `term[field]` surface lowers to the
   functional form. All three land on the same AST.

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

The canonical form is prefix. The parser is tolerant so raw model output parses. It also accepts
infix `a AND b` and `a OR b` (precedence OR below AND), parenthesized grouping like
`(a OR b) AND c`, a top-level comma as an implicit AND, case-insensitive operator, region and
strategy names, Lucene-style numeric proximity `NEAR/5(a, b)` (the same as `NEAR/w5`), escaped
quotes inside quoted terms, and a **bare multi-word run** such as `IN(string, must be positive)`
read as an implicit `PHRASE(...)`. That last form used to hard-error. All of this tolerance is
syntactic: everything canonicalizes to the same AST, and semantics and the type system do not
change. A real parse failure returns a structured error that suggests the fix, such as quoting a
multi-word operand or wrapping it in `PHRASE(...)`.

Three constraints are deliberate:
- **NOT only exists as the right child of AND**, as set-difference. Unbounded negation matches
  almost everything, which is both semantically dangerous and a pruning worst case. The grammar
  does not parse a standalone `NOT`.
- **EXPAND carries a strategy tag**, so one operator covers text synonym expansion and code
  symbol-graph expansion.
- **Gran unifies proximity and scope.** "Within 5 tokens", "same function" and "same file" are one
  parameter on one operator.

## 2. Operator → retrieval-polarity spine

| Tightens precision | Widens recall | Sets matching region |
|---|---|---|
| `AND`, `NOT`(=Diff), `PHRASE`, `NEAR` | `OR`, `EXPAND`, `PREFIX` | `IN`, the `/Gran` on `NEAR` |

The agent's learned job is picking which column to reach into, given what the trajectory
reported: tighten when the last query over-returned, widen when it came back empty. This table is
the one-line version of what the policy learns.

## 3. Granularity type system

The lattice is `token < line < block < func < file < doc`.

- `Term`, `Phrase`, `Prefix` and `Expand` produce **token**-level matches.
- `NEAR/g` and `IN(region, ·)` **lift** their argument to level `g` (or the region's level).
- `AND` and `OR` need their children to **share a level**, otherwise both lift to the coarser one.

That rejects ill-formed queries before anything executes. Take `AND(IN(comment, x), IN(def, y))`:
it is only satisfiable if a `comment` region and a `def` region can co-occur in a shared binding
region, and the type system resolves that binding region to the enclosing function or file, not the
token. Without it, the agent writes queries with undefined region semantics and gets silent
garbage back.

When a query is ill-typed, the type checker's output goes into the **observation**, so the agent
(and, during RL, the reward) can react to a malformed query instead of guessing.

## 4. Worked example

Find where auth tokens are **defined**, not in tests, with `auth` and `token` close together:

```
AND(
  IN(def, NEAR/func( EXPAND(auth, symbol), EXPAND(token, symbol) )),
  NOT( IN(file, PREFIX(test)) )
)
```

- **Text backend:** a `SpanNearQuery` over the `def` field, wrapped in a `BooleanQuery` with a
  `MUST_NOT` clause.
- **Code backend:** a ripgrep prefilter for the expanded identifiers, then ast-grep verifies both
  appear in a definition node within one function, minus the test paths.

## 5. Backend targets

One typed AST runs on the current reference executor today, and can be lowered to backend-specific
programs later:

| BQL node | Text (Lucene/Pyserini) | Code (ripgrep + ast-grep) |
|---|---|---|
| `AND/OR` | `BooleanQuery` MUST / SHOULD | piped greps / `(a\|b)` alternation |
| `NOT`(Diff) | `MUST_NOT` clause | `-v` / negative pattern, applied as post-filter |
| `PHRASE` | `PhraseQuery` (in order, adjacent) | exact substring / token sequence |
| `PREFIX` | `PrefixQuery` (stemming = soft form) | identifier-boundary-aware regex (`auth\w*`, camel/snake) |
| `NEAR/g` | `SpanNearQuery` slop=g | within N lines / same AST scope |
| `IN(region,·)` | `field:` (title/body/section) | AST role: def / call / sig / comment / string / test-file |
| `EXPAND` | corpus-validated synonym OR-set | identifier-variant set + symbol-graph (def→calls) |

Two points from that table. First, field generalises to structural scope, and in code that scope
is the AST. `IN(def,·)`, `IN(call,·)` and `IN(comment,·)` name the distinctions a dense embedding
blurs into a single vector, which is why they are first-class here. Second, `PREFIX` in code has to
be identifier-boundary aware: `PREFIX(auth)` should reach `authToken`, `auth_token` and
`Authenticate` across naming conventions, and classic truncation cannot do that.

`EXPAND` is where corpus statistics carry weight. The LLM proposes a variant set, and a live corpus
resolver can prune the df=0 and ultra-rare variants under a recall budget before any of them enter
the OR. Use `synonym` for text, `symbol` (naming variants plus the call graph) for code, and
`lexical` for raw prefix and wildcard families.

## 6. Cost model & efficient execution

> The **code method is index-free**: ripgrep prefilter, then ast-grep or Python-AST verification
> over live files, ranked in memory, with nothing persisted. The inverted-index and WAND notes
> below are about the **text/Lucene path**, which is the BM25 *baseline*. At repo scale ripgrep
> scans in milliseconds, so the method needs no index at all.

**What the reference executor does today** (`structural/bql/executor.py`): for a large corpus (at
least `AGENT_SEARCH_BQL_PREFILTER_MIN` units, default 5000, meaning the shared document corpus
rather than a small per-query repo) selection is **two-phase filter-then-verify**. A pure-Python
**inverted index** (`token → units`, built once per corpus) computes a recall-safe candidate
superset from the query's positive leaves, and the exact `_eval` then verifies only those
candidates. The result is identical to the full live scan, since the index only narrows and
`_eval` stays the arbiter, so it scales with no semantic change. Below the threshold (per-query
code repos) the plain index-free O(N) scan runs unchanged.

Core retrieval is a solved, sublinear problem, so it is inherited wherever it applies:

- **Text/baseline core** (`AND`/`OR`/`PHRASE`/`IN` over a Lucene index): dynamic top-k pruning
  (MaxScore, WAND, BlockMax-WAND) skips most postings unscored. Cost lands around a standard BM25
  query.
- **EXPAND is the main cost risk, and it is bounded.** Each EXPAND is a wide OR, and wide
  disjunctions weaken WAND's score upper bounds, which means less skipping. Resolving an EXPAND is
  `k` df-lookups (microseconds), so cap the width with a **recall budget**. Sort variants by df,
  add them until the marginal recall gain drops below τ, and drop df=0 and ultra-rare terms.
  Expansion width becomes a tunable knob, because corpus statistics turn an open-ended disjunction
  into a bounded, selectivity-ordered one.
- **Code structural scope: two-phase filter-then-verify.** Never run AST matching over a whole
  repo. *Phase A* is a cheap lexical prefilter (inverted index or ripgrep) on the EXPANDed terms,
  giving a small candidate file set. *Phase B* runs ast-grep structural verification (`IN(def,·)`,
  `NEAR/func`) only on those candidates. A wide EXPAND gives a large candidate set and an
  expensive Phase B. Expansion width and verify cost are coupled, and the recall-budget cap bounds
  the coupling.
- **NOT and proximity resist pruning, so schedule them late.** Run NOT as a post-filter on the
  already-small candidate set, and NEAR after the cheap conjuncts have shrunk things. The
  evaluation order is: selective conjuncts, expansion, proximity, negation.

**Pseudocode, EXPAND under a recall budget:**
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

**Pseudocode, two-phase code execution:**
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

This is the corpus feedback that drives query refinement. Every query returns:
- `n_hits`, the total before truncation, which is the selectivity signal.
- per-clause and per-EXPAND-variant `df`, so the agent can spot a dead clause.
- top-k snippets with provenance (path:line, region), which is the evidence.
- type-check status: ok, or which clause is ill-typed.

Together these make the global statistics a prior and the observed result the conditional truth,
that is, df *given the constraints already added*, which only execution reports.

## 8. Validity & robustness

- The parser returns structured errors, which are fed back as an observation instead of crashing.
- A query that fails the type check never executes, and the agent is told why.
- During RL, format and type validity are a reward term (malformed queries are penalised), so the
  policy learns to emit well-formed BQL.
