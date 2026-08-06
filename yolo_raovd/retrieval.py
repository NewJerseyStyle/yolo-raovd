from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import faiss  # type: ignore

    _HAS_FAISS = True
except Exception:  # pragma: no cover - optional dependency
    faiss = None  # type: ignore
    _HAS_FAISS = False


@dataclass
class ReferenceIndex:
    vectors: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=np.float32))
    metadata: List[Dict[str, Any]] = field(default_factory=list)
    dim: int = 0

    def add(self, vectors: np.ndarray, metadata: List[Dict[str, Any]]) -> None:
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim == 1:
            vectors = vectors[None, :]
        if vectors.size == 0:
            return
        if self.dim == 0:
            self.dim = int(vectors.shape[1])
        if vectors.shape[1] != self.dim:
            raise ValueError("embedding dim mismatch")
        self.vectors = vectors if self.vectors.size == 0 else np.vstack([self.vectors, vectors])
        self.metadata.extend(metadata)

    def normalize(self, x: np.ndarray) -> np.ndarray:
        x = x.astype(np.float32)
        norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-8
        return x / norms

    def search(self, query: np.ndarray, top_k: int = 10, use_faiss: bool = True) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        if self.vectors.size == 0:
            return np.empty((0,), dtype=np.float32), []
        if query.ndim == 1:
            query = query[None, :]
        if query.shape[1] != self.dim:
            raise ValueError("query dim mismatch")
        top_k = int(min(top_k, len(self.vectors)))
        q = self.normalize(query.copy())
        db = self.normalize(self.vectors.copy())
        if use_faiss and _HAS_FAISS and len(self.vectors) >= 10:
            index = faiss.IndexFlatIP(self.dim)
            index.add(db)
            scores, ids = index.search(q, top_k)
            scores = scores[0]
            ids = ids[0]
        else:
            sims = q @ db.T
            ids = np.argsort(-sims[0])[:top_k]
            scores = sims[0][ids]
        selected = [self.metadata[int(i)] for i in ids]
        return scores.astype(np.float32), selected

    def save(self, path: Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        np.save(path / "vectors.npy", self.vectors)
        with (path / "metadata.jsonl").open("w", encoding="utf-8") as f:
            for item in self.metadata:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        manifest = {
            "count": len(self.metadata),
            "dim": self.dim,
        }
        (path / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "ReferenceIndex":
        path = Path(path)
        vec_file = path / "vectors.npy"
        md_file = path / "metadata.jsonl"
        manifest_file = path / "manifest.json"
        if not vec_file.exists() or not md_file.exists():
            raise FileNotFoundError(f"index files not found in {path}")
        vectors = np.load(vec_file).astype(np.float32)
        metadata: List[Dict[str, Any]] = []
        with md_file.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    metadata.append(json.loads(line))
        dim = vectors.shape[1] if vectors.size else 0
        if manifest_file.exists():
            manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
            dim = int(manifest.get("dim", dim))
        return cls(vectors=vectors, metadata=metadata, dim=dim)


