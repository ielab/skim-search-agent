"""Repo-root conftest: puts the repo root on sys.path during collection so test modules
can import the non-packaged `scripts/` helpers with a bare `pytest` (not only `python -m
pytest`, which adds the cwd itself)."""
