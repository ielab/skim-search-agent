"""Compatibility module: `DenseBelief` lives in `agent_search.retrievers.dense.belief` now (it is
the dense engine every arm shares, not an Indri detail). `DEFAULT_MODEL` is the general-domain
dense default resolved when this module is imported (`DENSE_MODEL`)."""
from agent_search.evaluation.datasets import default_dense_model as _default_dense_model
from agent_search.retrievers.dense.belief import DEFAULT_TOP_K, DenseBelief, _plain_terms, _plain_text, default_model

DEFAULT_MODEL = _default_dense_model("general")

__all__ = ["DenseBelief", "DEFAULT_MODEL", "DEFAULT_TOP_K", "default_model", "_plain_terms", "_plain_text"]
