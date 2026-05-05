from collections import namedtuple
from dataclasses import dataclass
from typing import Any, Dict, Tuple

Trajectories_invdyn = namedtuple("Trajectories", "actions observations")


@dataclass
class StitchSearchNode:
    score: float
    chunk_indices: Tuple[int, ...]
    operators: Tuple[str, ...]
    seam_scores: Tuple[float, ...]


@dataclass
class StitchSearchResult:
    blended_traj: Any
    score: float
    chunk_indices: Tuple[int, ...]
    operators: Tuple[str, ...]
    diagnostics: Dict[str, Any]
