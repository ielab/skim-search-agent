"""Offline repo provider: fixtures inline; missing cache fails with a clear error."""
import pytest

from agent_search.evaluation.datasets import fixture_instances
from agent_search.corpus.code_repo import RepoError, ensure_repo, get_files


def test_fixture_files_are_inline_no_git():
    files = get_files(fixture_instances()[0])      # instance.files is set -> no git
    assert "auth/session.py" in files


def test_missing_cache_raises_actionable_error(tmp_path):
    with pytest.raises(RepoError) as ei:
        ensure_repo("astropy/astropy", str(tmp_path / "empty"), allow_clone=False)
    msg = str(ei.value).lower()
    assert "git clone" in msg and "internet" in msg   # tells the user what to do
