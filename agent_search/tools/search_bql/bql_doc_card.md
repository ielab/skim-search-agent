# Structured document search: `term[field]` Boolean + fetch

Query language: `term[field]`, `AND` / `OR` / `NOT`, parentheses, wildcard `word*`, quoted
`"phrase"`; terms are case-insensitive. Fields: `title` (the document's subject), `section`
(a section heading), `body` (anywhere in the text), `infobox` (the structured facts), `tiab`
(title or body). A search lists documents: the title, which fields matched, and the section
and infobox names; it never shows text.

`fetch` reads one named section of a listed hit: `{"rank": 1, "section": "Career"}`. Special
names: `""` for the opening text, `infobox` for the facts, `body` for the whole page.
