# Structured document search: `term[field]` Boolean + fetch (web documents with sections)

Query language: `term[field]`, `AND` / `OR` / `NOT`, parentheses, wildcard `word*`, quoted
`"phrase"`; terms are case-insensitive. Fields: `title` (the document's subject), `section`
(a section heading), `body` (anywhere in the text), `author` (the byline), `date` (the
publication date), `tiab` (title or body). A search lists documents: the title, which fields
matched, and the section names `§[...]`; it never shows text.

`fetch` reads one named section of a listed hit: `{"rank": 1, "section": "Career"}`. Special
names: `""` for the opening text, `infobox` for the title, byline and date, `body` for the
whole page.
