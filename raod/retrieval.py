from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .encoding import DINOV3Encoder

try:
    import chromadb  # type: ignore

    _HAS_CHROMA = True
except Exception:  # pragma: no cover - optional dependency
    chromadb = None  # type: ignore
    _HAS_CHROMA = False


class _ForbidBuiltinEmbedding:
    """ChromaDB embedding function that refuses to embed anything.

    All vectors in this pipeline are produced by ``DINOV3Encoder`` (DINOv3 for
    images, dino.txt for text) and passed explicitly to ``add``/``search``. If
    ChromaDB ever tries to call a built-in embedding function (e.g. because a
    caller omitted embeddings), this raises instead of silently falling back to
    ChromaDB's built-in text-only model, which would corrupt the shared
    2048-dim multimodal space.
    """

    def __call__(self, input):
        raise RuntimeError(
            "ChromaDB attempted to embed input with a built-in model. "
            "All embeddings must be produced by DINOV3Encoder and passed "
            "explicitly to add/search."
        )

    def embed_query(self, input):
        return self.__call__(input)

    def embed_documents(self, input):
        return self.__call__(input)

    def __name__(self):
        return "forbid_builtin_embedding"

    def name(self):
        return "forbid_builtin_embedding"


_FORBID_BUILTIN = _ForbidBuiltinEmbedding()


@dataclass
class ChromaReferenceStore:
    """ChromaDB-backed multimodal store for metadata-aware retrieval.

    This is the single reference store used across the pipeline (indexing,
    dense matching, and benchmarking). Rows may be either text or image
    references, all embedded into the same 2048-dim space by ``DINOV3Encoder``.
    """

    collection_name: str = "references"
    persist_directory: Optional[str] = None
    _client: Any = field(default=None, init=False, repr=False)
    _collection: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if not _HAS_CHROMA:
            raise RuntimeError("chromadb is not installed; install it to use ChromaReferenceStore")
        if self.persist_directory:
            self._client = chromadb.PersistentClient(path=self.persist_directory)
        else:
            self._client = chromadb.Client()
        # New collections get an embedding function that refuses to embed, so
        # ChromaDB can never fall back to its built-in text-only model. Existing
        # collections (created before this guard existed) are loaded as-is; they
        # were always written with explicit DINOV3Encoder embeddings, so their
        # persisted "default" config is never actually invoked.
        try:
            self._client.get_collection(name=self.collection_name)
            self._collection = self._client.get_collection(name=self.collection_name)
        except Exception:
            self._collection = self._client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
                embedding_function=_FORBID_BUILTIN,
            )

    @property
    def count(self) -> int:
        return self._collection.count()

    def add(self, vector: np.ndarray, metadata: Dict[str, Any], record_id: Optional[str] = None) -> str:
        vector = np.asarray(vector, dtype=np.float32).reshape(-1)
        doc_id = str(record_id or metadata.get("id") or f"ref_{self.count}")
        self._collection.upsert(
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
        self._collection.upsert(
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
        result = self._collection.query(
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
        result = self._collection.query(
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

    def get_all(self) -> List[Dict[str, Any]]:
        """Return all stored metadata rows (used for benchmarking class lookup)."""
        result = self._collection.get(include=["metadatas"])
        return list(result.get("metadatas", []) or [])


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
    encoder: Optional[DINOV3Encoder] = None,
) -> ChromaReferenceStore:
    """Build a ChromaDB-only store from a references JSON file.

    Each reference row is embedded with the unified DINOv3/dino.txt encoder
    (text rows via ``encode_text``, image rows via ``encode_image``) and stored
    in a single ChromaDB collection.
    """
    refs = json.loads(Path(references_path).read_text(encoding="utf-8"))
    if not isinstance(refs, list):
        raise ValueError("references file should be a list")

    store = ChromaReferenceStore(collection_name=collection_name, persist_directory=persist_directory)
    encoder = encoder or DINOV3Encoder()

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
            vector = encoder.encode_text(text)
            meta = dict(payload)
            meta["text"] = text
            batch_vectors.append(vector)
            batch_metadatas.append(meta)
            batch_ids.append(ref_id)
        else:
            image_path = str(item.get("path", item.get("image", item.get("image_path", "")))).strip()
            if not image_path:
                continue
            vector = encoder.encode_image_path(image_path)
            meta = dict(payload)
            meta["path"] = image_path
            batch_vectors.append(vector)
            batch_metadatas.append(meta)
            batch_ids.append(ref_id)

    if batch_vectors:
        store.add_batch(np.stack(batch_vectors, axis=0), batch_metadatas, batch_ids)

    return store


def build_reference_indexes(
    references_path: str,
    out_dir: str,
    encoder: Optional[DINOV3Encoder] = None,
    chroma_dir: Optional[str] = None,
    chroma_collection_name: str = "references",
) -> Dict[str, Any]:
    """Build the ChromaDB store from a references file.

    ``out_dir`` is retained for CLI compatibility and records a small manifest;
    the actual vectors live in the ChromaDB store at ``chroma_dir``.
    """
    encoder = encoder or DINOV3Encoder()
    chroma_store = build_chroma_from_references(
        references_path=references_path,
        persist_directory=chroma_dir,
        collection_name=chroma_collection_name,
        encoder=encoder,
    )
    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)
    manifest = {
        "store": "chromadb",
        "chroma_dir": chroma_dir,
        "collection_name": chroma_collection_name,
        "count": chroma_store.count,
    }
    (out_dir_path / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def load_indices(
    index_dir: str,
    chroma_dir: Optional[str] = None,
    chroma_collection_name: str = "references",
) -> Optional[ChromaReferenceStore]:
    """Load the ChromaDB store from disk.

    ``index_dir`` is retained for CLI compatibility (it may hold a manifest);
    the store itself is loaded from ``chroma_dir``.
    """
    if not chroma_dir:
        raise ValueError("chroma_dir is required to load the reference store")
    return ChromaReferenceStore(collection_name=chroma_collection_name, persist_directory=chroma_dir)
