"""``OrderedSeen``: the set of documents an episode has surfaced, in first-seen order.

Every workspace keeps one of these as ``seen``. It behaves like a ``set`` for membership,
``add``/``update``/``discard``, iteration, ``len`` and truthiness, but it also remembers the
order in which ids arrived. That order is the agent's *retrieval ranking* for a document
episode (the loop reads ``workspace.surfaced``): the first document a search surfaced ranks
first, and so on. Without an ordered structure the harness could only report which documents
were seen, not rank-based metrics such as hit@k or nDCG.
"""
from __future__ import annotations

from typing import Iterable, Iterator


class OrderedSeen:
    __slots__ = ("_order",)

    def __init__(self, items: Iterable[str] = ()):
        self._order: dict[str, None] = {}
        self.update(items)

    # --- set-like API ---------------------------------------------------------------
    def add(self, item: str) -> None:
        self._order.setdefault(item, None)

    def update(self, items: Iterable[str]) -> None:
        for it in items:
            self._order.setdefault(it, None)

    def discard(self, item: str) -> None:
        self._order.pop(item, None)

    def clear(self) -> None:
        self._order.clear()

    def __contains__(self, item: object) -> bool:
        return item in self._order

    def __iter__(self) -> Iterator[str]:
        return iter(self._order)

    def __len__(self) -> int:
        return len(self._order)

    def __bool__(self) -> bool:
        return bool(self._order)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, OrderedSeen):
            return list(self._order) == list(other._order)
        if isinstance(other, (set, frozenset)):
            return set(self._order) == other
        if isinstance(other, (list, tuple)):
            return list(self._order) == list(other)
        return NotImplemented

    def __repr__(self) -> str:
        return f"OrderedSeen({list(self._order)!r})"

    # --- ranking view ---------------------------------------------------------------
    @property
    def order(self) -> list[str]:
        """Ids in first-seen order: the episode's retrieval ranking."""
        return list(self._order)

    def as_set(self) -> set:
        return set(self._order)


__all__ = ["OrderedSeen"]
