"""The run record refuses to start when an import-time knob was set after its module loaded
(`agent_search/evaluation/identity.py`). This is the guard for the bug where a file's
`max_visit_tokens: 512` ran at the default 12,000 because the CLI exported the env after
loading the file had already imported the tool modules."""
import pytest

from agent_search.errors import SetupError
from agent_search.evaluation.identity import IMPORT_TIME_KNOBS, _check_import_time_knobs


def test_matching_environment_passes(monkeypatch):
    from agent_search.tools import budgets
    monkeypatch.setenv("MAX_VISIT_TOKENS", str(budgets.MAX_VISIT_TOKENS))
    _check_import_time_knobs()


def test_stale_constant_is_refused(monkeypatch):
    from agent_search.tools import budgets
    monkeypatch.setenv("MAX_VISIT_TOKENS", str(budgets.MAX_VISIT_TOKENS + 1))
    with pytest.raises(SetupError, match="MAX_VISIT_TOKENS"):
        _check_import_time_knobs()


def test_every_listed_knob_names_a_real_constant():
    import importlib
    for var, module, attr in IMPORT_TIME_KNOBS:
        assert isinstance(getattr(importlib.import_module(module), attr), int), (var, module, attr)
