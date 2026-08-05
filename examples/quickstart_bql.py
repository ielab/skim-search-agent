"""Minimal SkimSearchAgent demo: index-free Boolean search over an in-memory structured corpus.

Runs with zero heavy dependencies (no Java, no torch, no index build) — this is the
pure-Python REFERENCE executor. At paper scale the same query language compiles to Lucene
(STRUCTURED_BACKEND=lucene over an index built by evaluation/build_indexes.py), and the
agent-facing `term[field]` suffix sugar is provided by the tool layer on top of it.

    python examples/quickstart_bql.py
"""
from agent_search.corpus.units import CodeUnit
from agent_search.retrievers.structural.bql.executor import StructuralExecutor, execute_bql


def doc(doc_id: str, title: str, body: str, date: str | None = None) -> CodeUnit:
    """A structured document: title + body + optional date field."""
    meta = {"date": date} if date else {}
    return CodeUnit(doc_id=doc_id, path=f"{doc_id}.txt", qualname=doc_id,
                    start_line=1, end_line=1, code=body, body=body, title=title, metadata=meta)


corpus = [
    doc("curie", "Marie Curie",
        "Pioneering work on radioactivity earned two Nobel Prizes in physics and chemistry.",
        date="1867-11-07"),
    doc("meitner", "Lise Meitner",
        "Co-discovered nuclear fission; the element meitnerium honors her work in physics.",
        date="1878-11-07"),
    doc("hopper", "Grace Hopper",
        "Invented the first compiler and popularized machine-independent programming languages."),
]

executor = StructuralExecutor(corpus)
by_id = {u.doc_id: u for u in corpus}

QUERIES = (
    'radioactivity',                                     # bare ranked term
    'IN(title, "Marie Curie")',                          # field-scoped phrase
    'IN(body, "nuclear fission") OR IN(body, compiler)', # disjunction across documents
    'AND(IN(body, physics), NOT(IN(body, fission)))',    # boolean composition
    'PREFIX(physic)',                                    # prefix expansion
)

if __name__ == "__main__":
    for bql in QUERIES:
        obs = execute_bql(bql, executor, by_id, k=5)
        print(f"{bql:52s} -> {obs.n_hits} hit(s): {[h.doc_id for h in obs.hits]}")
