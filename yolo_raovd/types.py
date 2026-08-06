from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class ReferenceRecord:
    ref_id: str
    label: str
    modality: str
    embedding: List[float]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Detection:
    label: str
    box: List[float]
    score: float
    confidence: float
    objectness: float
    top_k_scores: List[Dict[str, Any]]
    margin: float
    aggregation: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)

