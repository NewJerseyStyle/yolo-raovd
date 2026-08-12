from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .encoding import SimpleImageEncoder, SimpleTextEncoder

try:
    import faiss  # type: ignore

    _HAS_FAISS = True
except Exception:  # pragma: no cover - optional dependency
    faiss = None  # type: ignore
    _HAS_FAISS = False

try:
    import chromadb  # type: ignore

    _HAS_CHROMA = True
except Exception:  # pragma: no cover - optional dependency
    chromadb = None  # type: ignore
    _HAS_CHROMA = False


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

    def _select_candidates(
        self,
        *,
        modality: Optional[str] = None,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[int]:
        idxs: List[int] = []
        for i, item in enumerate(self.metadata):
            if modality is not None:
                item_modality = str(item.get("modality", "")).strip().lower()
                if item_modality and item_modality != str(modality).strip().lower():
                    continue
            if metadata_filter:
                ok = True
                for key, value in metadata_filter.items():
                    if item.get(key) != value:
                        ok = False
                        break
                if not ok:
                    continue
            idxs.append(i)
        return idxs

    def search(
        self,
        query: np.ndarray,
        top_k: int = 10,
        use_faiss: bool = True,
        modality: Optional[str] = None,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        if self.vectors.size == 0:
            return np.empty((0,), dtype=np.float32), []
        if query.ndim == 1:
            query = query[None, :]
        if query.shape[1] != self.dim:
            raise ValueError("query dim mismatch")

        candidates = self._select_candidates(modality=modality, metadata_filter=metadata_filter)
        if not candidates:
            return np.empty((0,), dtype=np.float32), []

        candidate_vecs = self.vectors[candidates]
        top_k = int(min(top_k, len(candidate_vecs)))
        q = self.normalize(query.copy())
        db = self.normalize(candidate_vecs.copy())
        if use_faiss and _HAS_FAISS and len(candidate_vecs) >= 10:
            index = faiss.IndexFlatIP(self.dim)
            index.add(db)
            scores, ids = index.search(q, top_k)
            scores = scores[0]
            ids = ids[0]
        else:
            sims = q @ db.T
            ids = np.argsort(-sims[0])[:top_k]
            scores = sims[0][ids]

        selected = [self.metadata[int(candidates[int(i)])] for i in ids]
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


class ChromaReferenceStore:
    """Optional Chroma-backed multimodal store for metadata-aware retrieval."""

    def __init__(self, collection_name: str = "references", persist_directory: Optional[str] = None):
        if not _HAS_CHROMA:
            raise RuntimeError("chromadb is not installed; install it to use ChromaReferenceStore")
        if persist_directory:
            self.client = chromadb.PersistentClient(path=persist_directory)
        else:
            self.client = chromadb.Client()
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    @property
    def count(self) -> int:
        return self.collection.count()

    def add(self, vector: np.ndarray, metadata: Dict[str, Any], record_id: Optional[str] = None) -> str:
        vector = np.asarray(vector, dtype=np.float32).reshape(-1)
        doc_id = str(record_id or metadata.get("id") or f"ref_{self.count}")
        self.collection.upsert(
            ids=[doc_id],
            embeddings=[vector.tolist()],
            metadatas=[metadata],
        )
        return doc_id

    def add_batch(self, vectors: np.ndarray, metadatas: List[Dict[str, Any]], ids: Optional[List[str]] = None) -> List[str]:
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim == 1:
            vectors = vectors[None, :]
        n = int(vectors.shape[0])
        if ids is None:
            base = self.count
            ids = [f"ref_{base + i}" for i in range(n)]
        elif len(ids) != n:
            raise ValueError("ids length must match vectors length")
        self.collection.upsert(
            ids=ids,
            embeddings=vectors.tolist(),
            metadatas=metadatas,
        )
        return ids

    def search(
        self,
        query: np.ndarray,
        top_k: int = 10,
        metadata_filter: Optional[Dict[str, Any]] = None,
        modality: Optional[str] = None,
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        where = dict(metadata_filter or {})
        if modality is not None:
            where["modality"] = str(modality).strip().lower()
        result = self.collection.query(
            query_embeddings=[np.asarray(query, dtype=np.float32).reshape(-1).tolist()],
            n_results=top_k,
            where=where or None,
        )
        ids = result.get("ids", [[]])[0]
        distances = result.get("distances", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        scores = np.asarray([1.0 - float(d) for d in distances], dtype=np.float32) if distances else np.empty((0,), dtype=np.float32)
        return scores, list(metadatas)

    def search_batch(
        self,
        queries: np.ndarray,
        top_k: int = 10,
        metadata_filter: Optional[Dict[str, Any]] = None,
        modality: Optional[str] = None,
    ) -> Tuple[np.ndarray, List[List[Dict[str, Any]]]]:
        queries = np.asarray(queries, dtype=np.float32)
        if queries.ndim == 1:
            queries = queries[None, :]
        n = int(queries.shape[0])
        where = dict(metadata_filter or {})
        if modality is not None:
            where["modality"] = str(modality).strip().lower()
        result = self.collection.query(
            query_embeddings=queries.tolist(),
            n_results=top_k,
            where=where or None,
        )
        ids = result.get("ids", [[]] * n)
        distances = result.get("distances", [[]] * n)
        metadatas = result.get("metadatas", [[]] * n)
        all_scores = []
        all_payloads = []
        for i in range(n):
            dists = distances[i] if i < len(distances) else []
            metas = metadatas[i] if i < len(metadatas) else []
            scores = np.asarray([1.0 - float(d) for d in dists], dtype=np.float32) if dists else np.empty((0,), dtype=np.float32)
            all_scores.append(scores)
            all_payloads.append(list(metas))
        return np.asarray(all_scores, dtype=np.float32), all_payloads


def _resolve_ref_label(item: Dict[str, Any]) -> str:
    return str(item.get("label", "")).strip()


def _resolve_ref_modality(item: Dict[str, Any]) -> str:
    modality = str(item.get("modality", "")).strip().lower()
    if modality in ("text", "image"):
        return modality
    if any(key in item for key in ("text", "prompt", "query")):
        return "text"
    if any(key in item for key in ("path", "image", "image_path")):
        return "image"
    return "text"


def build_chroma_from_references(
    references_path: str,
    persist_directory: str,
    collection_name: str = "references",
    text_encoder: Optional[Any] = None,
    image_encoder: Optional[Any] = None,
) -> ChromaReferenceStore:
    refs = json.loads(Path(references_path).read_text(encoding="utf-8"))
    if not isinstance(refs, list):
        raise ValueError("references file should be a list")

    store = ChromaReferenceStore(collection_name=collection_name, persist_directory=persist_directory)

    text_encoder = text_encoder or SimpleTextEncoder()
    image_encoder = image_encoder or SimpleImageEncoder()

    batch_vectors: List[np.ndarray] = []
    batch_metadatas: List[Dict[str, Any]] = []
    batch_ids: List[str] = []

    for item in refs:
        if not isinstance(item, dict):
            continue
        label = _resolve_ref_label(item)
        if not label:
            continue
        modality = _resolve_ref_modality(item)
        ref_id = str(item.get("id", f"{modality}_{label}"))
        payload: Dict[str, Any] = {"id": ref_id, "label": label, "modality": modality}

        if modality == "text":
            text = str(item.get("text", item.get("prompt", item.get("query", "")))).strip()
            if not text:
                continue
            vector = text_encoder.encode(text)
            meta = dict(payload)
            meta["text"] = text
            batch_vectors.append(vector)
            batch_metadatas.append(meta)
            batch_ids.append(ref_id)
        else:
            image_path = str(item.get("path", item.get("image", item.get("image_path", "")))).strip()
            if not image_path:
                continue
            vector = image_encoder.encode_image(image_path)
            meta = dict(payload)
            meta["path"] = image_path
            batch_vectors.append(vector)
            batch_metadatas.append(meta)
            batch_ids.append(ref_id)

    if batch_vectors:
        store.add_batch(np.stack(batch_vectors, axis=0), batch_metadatas, batch_ids)

    return store

