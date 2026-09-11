"""Fusion methods, one file each: `rrf` (ranks only) and `interpolation` (normalised scores).
`build_fusion("rrf", k=60)` or `build_fusion("interpolation", weights=[0.5, 0.5])`."""
from agent_search.retrievers.fusion.base import FUSIONS, Fusion, build_fusion, register_fusion
from agent_search.retrievers.fusion.interpolation import Interpolation
from agent_search.retrievers.fusion.rrf import RRF, rrf_k_from_env

__all__ = ["Fusion", "FUSIONS", "build_fusion", "register_fusion", "RRF", "Interpolation", "rrf_k_from_env"]
