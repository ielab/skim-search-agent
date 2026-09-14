"""Result cards cut their section-name and infobox-key lists at LISTING_SECTIONS /
LISTING_INFOBOX_KEYS (agent_search/tools/budgets.py); 0 shows every name."""
import importlib

from agent_search.tools import budgets, common


def _reload(sections, keys, monkeypatch):
    monkeypatch.setenv("LISTING_SECTIONS", str(sections))
    monkeypatch.setenv("LISTING_INFOBOX_KEYS", str(keys))
    importlib.reload(budgets)


def test_default_cuts_at_eight_and_six(monkeypatch):
    _reload(8, 6, monkeypatch)
    names = [f"S{i}" for i in range(10)]
    sec, ib = common.structure_str(names, [f"K{i}" for i in range(7)])
    assert sec == "·".join(names[:8]) + ",…" and ib.endswith("K5,…")


def test_zero_shows_every_name(monkeypatch):
    _reload(0, 0, monkeypatch)
    names = [f"S{i}" for i in range(30)]
    sec, ib = common.structure_str(names, ["A", "B"])
    assert sec == "·".join(names) and "…" not in sec and ib == "A·B"
    _reload(8, 6, monkeypatch)


def test_short_lists_are_not_marked(monkeypatch):
    _reload(8, 6, monkeypatch)
    assert common.structure_str(["History", "Career"], []) == ("History·Career", "")
