"""RAOD package.

Dense feature matching for zero-shot open-vocabulary object detection.
Public API entry points are in `raod.cli`.
"""

from .dense import DenseMatcher, SAMPointRefiner, generate_tiles
from .encoding import DINOV3Encoder
from .retrieval import (
    ChromaReferenceStore,
    build_chroma_from_references,
    build_reference_indexes,
    load_indices,
)

__all__ = [
    "DINOV3Encoder",
    "DenseMatcher",
    "SAMPointRefiner",
    "generate_tiles",
    "ChromaReferenceStore",
    "build_chroma_from_references",
    "build_reference_indexes",
    "load_indices",
]
__version__ = "0.1.0"
