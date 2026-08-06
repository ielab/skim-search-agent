"""The demo's mini collection: 8 structured articles about pioneering scientists.

Small enough to read in one sitting, rich enough that the agent must actually search, skim the
result cards, and fetch the right section — the answers are IN here, not in the model's memory
alone (dates and named sections force grounding)."""
from agent_search.corpus.units import CodeUnit


def _doc(doc_id, title, body, date=None):
    meta = {"date": date} if date else {}
    return CodeUnit(doc_id=doc_id, path=f"{doc_id}.md", qualname=doc_id, start_line=1,
                    end_line=1, code=body, body=body, title=title, metadata=meta)


CORPUS = [
    _doc("curie", "Marie Curie",
         "## Early life\nBorn Maria Sklodowska in Warsaw in 1867.\n"
         "## Research\nIsolated radium and polonium; coined the term radioactivity.\n"
         "## Recognition\nFirst person to win Nobel Prizes in two sciences: physics (1903, shared) "
         "and chemistry (1911, unshared).", date="1867-11-07"),
    _doc("meitner", "Lise Meitner",
         "## Career\nCo-discovered nuclear fission with Otto Hahn in 1938; excluded from the 1944 "
         "Nobel Prize.\n## Legacy\nElement 109, meitnerium, is named after her.", date="1878-11-07"),
    _doc("franklin", "Rosalind Franklin",
         "## Work\nProduced Photo 51, the X-ray diffraction image of DNA taken in May 1952 that "
         "revealed the double helix.\n## Later research\nPioneered X-ray studies of viruses "
         "including tobacco mosaic virus.", date="1920-07-25"),
    _doc("hopper", "Grace Hopper",
         "## Computing\nWrote the A-0 compiler in 1952, the first compiler; her team's FLOW-MATIC "
         "shaped COBOL.\n## Navy\nRetired as a rear admiral of the United States Navy.",
         date="1906-12-09"),
    _doc("noether", "Emmy Noether",
         "## Mathematics\nProved Noether's theorem in 1915, linking symmetries to conservation "
         "laws.\n## Recognition\nEinstein called her the most significant creative mathematical "
         "genius thus far produced.", date="1882-03-23"),
    _doc("lovelace", "Ada Lovelace",
         "## Analytical Engine\nPublished the first algorithm intended for a machine in her 1843 "
         "notes on Babbage's Analytical Engine.\n## Vision\nForesaw computers manipulating symbols "
         "beyond numbers.", date="1815-12-10"),
    _doc("wu", "Chien-Shiung Wu",
         "## Physics\nHer 1956 cobalt-60 experiment demonstrated parity violation in weak "
         "interactions.\n## Recognition\nThe 1957 Nobel Prize went to Lee and Yang for the theory "
         "her experiment confirmed.", date="1912-05-31"),
    _doc("hodgkin", "Dorothy Hodgkin",
         "## Crystallography\nSolved the structures of penicillin (1945) and vitamin B12, and "
         "later insulin in 1969.\n## Recognition\nWon the 1964 Nobel Prize in Chemistry.",
         date="1910-05-12"),
]

QUESTIONS = [
    ("Which element is named after the scientist who co-discovered nuclear fission?",
     "meitnerium"),
    ("What is the name of the X-ray image of DNA taken in May 1952?", "Photo 51"),
    ("Who wrote the first compiler, and what was it called?", "A-0"),
]
