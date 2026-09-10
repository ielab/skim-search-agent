"""Tiny inline fixtures: no staging, so the eval and doc-retrieval paths run anywhere.
`fixture_instances` is the code-localization fixture (SWE-bench-shaped); `_doc_corpus_fixture`
is the shared-corpus deep-research fixture (mirrors BrowseComp-Plus / multi-hop QA).
"""
from __future__ import annotations

from agent_search.evaluation.datasets.base import Instance, register_dataset


def fixture_instances() -> list[Instance]:
    """One tiny self-contained instance, so the eval runs anywhere. The gold patch
    edits `create_session_token`; the issue text mentions session-token expiry, so
    a lexical retriever should localize it."""
    session_py = (
        "import time\n"
        "\n"
        "\n"
        "def create_session_token(user):\n"
        "    token = make_token(user)\n"
        "    return token\n"
        "\n"
        "\n"
        "def make_token(user):\n"
        "    return str(user) + \"-static\"\n"
    )
    html_py = (
        "def render_page(title, body):\n"
        "    return \"<html>\" + title + body + \"</html>\"\n"
    )
    patch = (
        "diff --git a/auth/session.py b/auth/session.py\n"
        "--- a/auth/session.py\n"
        "+++ b/auth/session.py\n"
        "@@ -4,3 +4,4 @@ def create_session_token(user):\n"
        " def create_session_token(user):\n"
        "     token = make_token(user)\n"
        "-    return token\n"
        "+    return token  # TODO: attach expiry\n"
        "+    # expiry handling\n"
    )
    return [
        Instance(
            instance_id="fixture__session-expiry-1",
            repo="fixture/app",
            base_commit="0" * 40,
            problem_statement=(
                "Session tokens never expire. create_session_token should attach an "
                "expiry so user sessions time out."
            ),
            patch=patch,
            files={"auth/session.py": session_py, "render/html.py": html_py},
        )
    ]


def _doc_corpus_fixture(name: str) -> list[Instance]:
    """Tiny inline shared-corpus instance (no staging) so the doc-retrieval path is
    testable anywhere — mirrors the BrowseComp/multi-hop shape: question + evidence + answer."""
    docs = [
        {"_id": "d_guadalupe", "title": "Treaty of Guadalupe Hidalgo",
         "text": "The Treaty of Guadalupe Hidalgo ended the Mexican-American War in 1848."},
        {"_id": "d_paris", "title": "Treaty of Paris (1898)",
         "text": "The 1898 Treaty of Paris ended the Spanish-American War."},
        {"_id": "d_adams", "title": "Adams-Onis Treaty",
         "text": "The Adams-Onis Treaty of 1819 concerned Florida."},
    ]
    return [Instance(
        instance_id=f"{name}__mexican_war",
        repo=f"fixture/{name}", base_commit="0" * 40,
        problem_statement="Which treaty ended the Mexican-American War, and in what year?",
        patch="", docs=docs, gold_doc_ids={"d_guadalupe"},
        answer="Treaty of Guadalupe Hidalgo, 1848", corpus_id=f"{name}_fixture")]


register_dataset("fixture")(lambda limit=None, corpus_limit=None: fixture_instances())
# `doc_fixture` is the library's first-touch corpus: three documents, one question, no staging.
# `code_fixture` is the SWE-bench-shaped code-localization twin; `fixture` remains its alias.
register_dataset("doc_fixture", domain="general")(
    lambda limit=None, corpus_limit=None: _doc_corpus_fixture("doc"))
register_dataset("code_fixture")(lambda limit=None, corpus_limit=None: fixture_instances())
register_dataset("browsecomp_plus_fixture", domain="general")(
    lambda limit=None, corpus_limit=None: _doc_corpus_fixture("browsecomp_plus"))
register_dataset("hotpotqa_fixture", domain="general")(
    lambda limit=None, corpus_limit=None: _doc_corpus_fixture("hotpotqa"))
register_dataset("musique_fixture", domain="general")(
    lambda limit=None, corpus_limit=None: _doc_corpus_fixture("musique"))
