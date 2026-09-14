# Structured code search: field-tagged `term[field]` Boolean + fetch

Search code with a Boolean query language: `term[field]`, `AND`/`OR`/`NOT`, `"phrase"`,
`wildcard*`. No index — every term is matched live, case-insensitively, split on
camelCase/underscores (`worldToPixel` ≡ `world_to_pixel`). Two moves, in order:

1. **search** a query → ranked FILES (not code): each hit shows the path and the matched
   function/method names in it, numbered for `fetch`.
2. **fetch** a named part of a numbered file — a name from that file's list, or a line
   range — to read its source. Never the whole file.

The game: **one good search finds the owning file; fetch the named method and read it.**
Turns are few — most failures waste turns re-searching or mis-typing a fetch, not failing
to find the code. Every turn should move forward.

## How to search

Open with the strongest single anchor in the problem:

- **A traceback/exception is quoted** → search the operation on the raising line, not the
  API in the title. If the title says `build_report(...)` but it raises inside `sanitize`,
  search `sanitize[call]` — the entry point the user called is rarely the buggy frame, and
  a printed path may have been renamed. Search first; never `<fix>` from a mentioned name.
- **An error message is quoted** → its exact text: `"could not convert to float"[string]`.
- **Otherwise** → one distinctive name: `middleware[def]`. Unsure between names? `OR` them:
  `open[def] OR connect[def]`.

Use ONE word or field-scoped name per clause. Several bare words parse as one exact phrase
(see Atoms), not "any of these", and reliably 0-hit.

Then read the ranked list, not just the count — a dozen files is fine; scan the top few for
the one whose matched method OWNS the behavior. An off-target hit still helps: it lists that
file's OTHER names, a better anchor than re-guessing. If the obvious term 0-hits after one
loosening, the fix may ADD code that isn't there yet — anchor on the entry point the problem
names (the outer call in `export(render(x))` is `export`, even if `render` never appears).
Never re-issue a query or `AND` more onto one that already returned the right file: if it
worked, FETCH; if it failed, change it.

## Atoms — the finest-grained match

A bare **term** matches that token anywhere in a file. Several bare words separated by
spaces parse as an exact **phrase** (adjacent, in order) — real code rarely satisfies it, so
quote a phrase only when the words truly are contiguous, e.g. an error string. `stem*`
widens to any token starting with `stem` (`seri*` → serialize, serializer) — use it when
unsure of a word's ending.

## Field — where a match must occur

`term[field]` scopes `term` to one region; `term[f1,f2]` matches either. The fields are
EXACTLY these six:

| field | matches | example |
|---|---|---|
| `def` | definition names (function/method/class) | `Widget[def]` |
| `call` | names at call sites | `warn[call]` |
| `string` | string/f-string text | error messages |
| `comment` | `#` comments | TODOs |
| `sig` | name + parameters + annotations | `buffer[sig]` |
| `file` | every token in the file (path + all code) | co-occurs in-file |

Use the simple name, not a dotted path: `validate[def]`, not `Widget.validate[def]`.
`term[file]` is a token match over the whole file — the widest net, a good first probe when
unsure whether def/call/string is right.

## Combine — building queries from scoped atoms

`a OR b` matches either — for names you're unsure between. `A AND B` needs both in the SAME
file; use it sparingly, since three-plus clauses usually describe a file you're guessing at
and 0-hit — anchor on one sure clause, add a second only if the first floods. `A NOT B`
excludes files where `B` matches (`save[def] NOT test[file]` skips tests). Parentheses
group: `(open[def] OR connect[def]) AND socket[file]`.

## Fetch — reading the part you found

`search` lists each file's names as `defs:[Report.render . Report.export]`. `fetch` takes
`[rank, part]` pairs against that numbering:

```
{"rank": 1, "part": "Report.render"}
```

- `part` is a bare name copied verbatim from that file's `defs:[...]` list (`Report.render`,
  or `render` if unambiguous). NEVER the word `def`/`class`, a file path, the path glued to
  the name (`"file.py:render"`), or a name you only hope exists — the `rank` already picks
  the file.
- If fetch replies `no part named X … Available: A, B, C`, pick one of A/B/C — don't reword
  X, don't re-search.
- Module-level code outside any function: use a line range, `[1, "L1-40"]`. A part over ~40
  lines is truncated with its span shown; follow up with a narrower `L`-range.

**Commit to the file your search surfaced.** Once a search lists the method that owns the
behavior, fetch it, read it, and write the `<fix>` on it — don't re-search to "confirm". Two
traps waste the turns you needed for the fix:

- *Caller, not owner:* if the part you read is one or two lines that just forward arguments
  into another object's call (`return obj.method(*args, **kwargs)`), it's a caller — patching
  it guards one call site, not the behavior. Fetch the callee, usually ranked a step away in
  the same result.
- *Hunting for a name that isn't there:* the fix often ADDS code, so no existing function
  matches the issue's wording. Don't chase a hoped-for `validate_*`/helper across new
  searches; the behavior lives in the method you already surfaced. Drifting into invented
  syntax (a parenthesized `f(x)[def]`, a `path:` pseudo-field) is the tell you're lost —
  stop and fix in what's on screen.

## Worked examples

- `sanitize[call]` — where the failing operation is invoked (not the API in the title).
- `middleware[def]` — a distinctive name; the ranked files list def names to fetch.
- `open[def] OR connect[def]` — either name the fix might use.
- `seri*[def]` — `serialize`/`serializer`, spelling unsure.
- `Logger[call] NOT test[file]` — non-test files that use it.
- `"could not convert to float"[string]` — the exact error text.
- fetch `{"rank": 1, "part": "Report.render"}` — read result 1's method, then fix it.

## Common mistakes

Rejected before running (read the reason, retry):

- `term[module]`/`term[attribute]`/`term[variable]` — no such region; only
  `def`/`call`/`string`/`comment`/`sig`/`file` exist. A class attribute matches in `def`
  (set inside a method) or as `string`/`comment` text.
- `NOT x` alone — `NOT` is binary (`A NOT B`), not a standalone negation.
- a parenthesized call `f(x)[def]`, an invented `filename:`/`path:` field, or a
  leading-wildcard `*name` — none are valid; a name is `name[def]`, a file token `name[file]`.
- `class Foo[def]` — that's the phrase "class Foo", which no name literally is; write
  `Foo[def]`.

Wastes a turn:

- fetch part = `"def"`, `"class"`, a file path, or path-glued name (`"path:Name"`) — use a
  bare name from the file's `defs:[...]`.
- re-searching after a search already showed the owning method — fetch instead.
- re-issuing a query (or a near-duplicate with clauses reordered) — it returns the same.
- regex (`a|b`, `X.*Y`) — unsupported; use `OR` / `stem*`.
- proposing a fix from a part you didn't read, or a name the problem only mentioned.
