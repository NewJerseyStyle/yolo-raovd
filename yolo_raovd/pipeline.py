from __future__ import annotations

import json
from dataclasses import dataclass
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .confidence import aggregate_scores, compute_confidence, nms
from .encoding import SimpleCLIPImageEncoder, SimpleImageEncoder, SimpleTextEncoder
from .proposal import YOLOProposalGenerator
from .retrieval import ChromaReferenceStore, ReferenceIndex, build_chroma_from_references
from .types import Detection


@dataclass
class YoloRaovdConfig:
    top_k: int = 20
    retrieval_agg: str = "weighted_mean"
    agg_top_m: int = 5
    agg_temperature: float = 0.06
    score_threshold: float = 0.15
    margin_eps: float = 1e-6
    nms_iou: float = 0.5
    support_threshold: float = 0.0
    max_proposals: int = 180
    yolo_model_path: str = "yolo11n.pt"
    yolo_conf: float = 0.25
    yolo_iou: float = 0.45
    yolo_max_det: int = 300
    chroma_persist_directory: Optional[str] = None
    chroma_collection_name: str = "references"


class YoloRaovdDetector:
    def __init__(
        self,
        text_encoder: Optional[SimpleTextEncoder] = None,
        image_encoder: Optional[SimpleImageEncoder] = None,
        text_image_encoder: Optional[SimpleCLIPImageEncoder] = None,
        text_index: Optional[ReferenceIndex] = None,
        image_index: Optional[ReferenceIndex] = None,
        chroma_store: Optional[ChromaReferenceStore] = None,
        config: Optional[YoloRaovdConfig] = None,
    ) -> None:
        self.text_encoder = text_encoder or SimpleTextEncoder()
        self.image_encoder = image_encoder or SimpleImageEncoder()
        self.text_image_encoder = text_image_encoder or SimpleCLIPImageEncoder()
        self.text_index = text_index
        self.image_index = image_index
        self.chroma_store = chroma_store
        self.config = config or YoloRaovdConfig()
        self.proposer = YOLOProposalGenerator(
            model_path=self.config.yolo_model_path,
            conf=self.config.yolo_conf,
            iou=self.config.yolo_iou,
            max_det=self.config.yolo_max_det,
        )

    def _load_image_array(self, image_path: str) -> np.ndarray:
        from PIL import Image

        with Image.open(image_path) as im:
            arr = np.array(im.convert("RGB"), dtype=np.float32)
        return arr

    def _crop_region(self, image: np.ndarray, box: Sequence[float]) -> np.ndarray:
        x1, y1, x2, y2 = [int(v) for v in box]
        h, w = image.shape[:2]
        x1 = max(0, min(w - 1, x1))
        x2 = max(x1 + 1, min(w, x2))
        y1 = max(0, min(h - 1, y1))
        y2 = max(y1 + 1, min(h, y2))
        return image[y1:y2, x1:x2]

    def _resolve_query_labels(self, query: str, top_k: int, min_ratio: float = 0.6) -> set[str]:
        """Open-vocabulary matching keeps only the query and semantically close labels.

        In OVD systems, text queries are treated as class prototypes and region features are
        matched against that prompt-conditioned set rather than against the full reference bank.
        """
        q = str(query).strip()
        if not q:
            return set()
        if self.chroma_store is None and (self.text_index is None or self.text_index.vectors.size == 0):
            return {q}

        query_vector = self.text_encoder.encode(q)
        if self.chroma_store is not None:
            q_scores, q_payloads = self.chroma_store.search(query_vector, top_k=top_k, modality="text")
        else:
            q_scores, q_payloads = self.text_index.search(query_vector, top_k=top_k, use_faiss=False)
        candidates: Dict[str, float] = {}
        for s, payload in zip(q_scores, q_payloads):
            label = str(payload.get("label", q) or "").strip()
            if not label:
                continue
            candidates[label] = max(candidates.get(label, -1e9), float(s))

        if not candidates:
            return {q}

        best_score = max(candidates.values())
        threshold = max(0.35, min_ratio * best_score)
        kept = {label for label, score in candidates.items() if label.lower() == q.lower() or score >= threshold}
        if not kept:
            kept = {q}
        return kept

    def _search_text(self, query: np.ndarray, top_k: int) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        if self.chroma_store is not None:
            return self.chroma_store.search(query, top_k=top_k, modality="text")
        if self.text_index is None or self.text_index.vectors.size == 0:
            return np.empty((0,), dtype=np.float32), []
        return self.text_index.search(query, top_k=top_k, use_faiss=False, modality="text")

    def _search_image(self, query: np.ndarray, top_k: int) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        if self.chroma_store is not None:
            return self.chroma_store.search(query, top_k=top_k, modality="image")
        if self.image_index is None or self.image_index.vectors.size == 0:
            return np.empty((0,), dtype=np.float32), []
        return self.image_index.search(query, top_k=top_k, use_faiss=False, modality="image")

    def _search_batch_text(self, queries: np.ndarray, top_k: int) -> Tuple[np.ndarray, List[List[Dict[str, Any]]]]:
        if self.chroma_store is not None:
            return self.chroma_store.search_batch(queries, top_k=top_k, modality="text")
        return self._search_batch_fallback(queries, top_k, self.text_index)

    def _search_batch_image(self, queries: np.ndarray, top_k: int) -> Tuple[np.ndarray, List[List[Dict[str, Any]]]]:
        if self.chroma_store is not None:
            return self.chroma_store.search_batch(queries, top_k=top_k, modality="image")
        return self._search_batch_fallback(queries, top_k, self.image_index)

    def _search_batch_fallback(
        self,
        queries: np.ndarray,
        top_k: int,
        index: Optional[ReferenceIndex],
    ) -> Tuple[np.ndarray, List[List[Dict[str, Any]]]]:
        if index is None or index.vectors.size == 0:
            n = int(queries.shape[0]) if queries.ndim > 1 else 1
            return np.empty((0, 0), dtype=np.float32), [[] for _ in range(n)]
        all_scores: List[np.ndarray] = []
        all_payloads: List[List[Dict[str, Any]]] = []
        qs = queries if queries.ndim > 1 else queries[None, :]
        for i in range(int(qs.shape[0])):
            scores, payloads = index.search(qs[i], top_k=top_k, use_faiss=False)
            all_scores.append(scores)
            all_payloads.append(payloads)
        return np.asarray(all_scores, dtype=np.float32), all_payloads

    def retrieve_candidates(
        self,
        query: str,
        *,
        mode: str = "text",
        query_image: Optional[str] = None,
        top_k: Optional[int] = None,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        top_k = top_k if top_k is not None else self.config.top_k

        if mode == "text":
            vector = self.text_encoder.encode(query)
            return self._search_text(vector, top_k)

        if mode == "image":
            if query_image is None:
                raise ValueError("query_image is required for image mode")
            vector = self.image_encoder.encode_image(query_image)
            return self._search_image(vector, top_k)

        if mode == "hybrid":
            results: List[Tuple[float, Dict[str, Any]]] = []
            text_vector = self.text_encoder.encode(query)
            text_scores, text_payloads = self._search_text(text_vector, top_k)
            for s, p in zip(text_scores, text_payloads):
                results.append((float(s), p))
            if query_image is not None:
                image_vector = self.image_encoder.encode_image(query_image)
                image_scores, image_payloads = self._search_image(image_vector, top_k)
                for s, p in zip(image_scores, image_payloads):
                    results.append((float(s), p))
            if not results:
                return np.empty((0,), dtype=np.float32), []
            merged = {}
            for score, payload in results:
                key = str(payload.get("id") or payload.get("label") or repr(payload))
                merged[key] = {**payload, "_score": max(float(merged.get(key, {}).get("_score", -1e9)), score)}
            ordered = sorted(merged.values(), key=lambda x: float(x.get("_score", 0.0)), reverse=True)[:top_k]
            scores = np.asarray([float(item.get("_score", 0.0)) for item in ordered], dtype=np.float32)
            return scores, ordered

        raise ValueError(f"unsupported retrieval mode: {mode}")

    def detect_with_text_queries(
        self,
        image_path: str,
        queries: Sequence[str],
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None,
        nms_iou: Optional[float] = None,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[Detection]:
        if self.text_index is None or self.text_index.vectors.size == 0:
            raise ValueError("text index is empty, run index first")
        image = self._load_image_array(image_path)
        proposals = self.proposer.propose(image, self.config.max_proposals)
        top_k = top_k if top_k is not None else self.config.top_k
        score_threshold = score_threshold if score_threshold is not None else self.config.score_threshold
        nms_iou = nms_iou if nms_iou is not None else self.config.nms_iou

        results: List[Detection] = []
        for q in queries:
            q = q.strip()
            if not q:
                continue
            query_labels = self._resolve_query_labels(q, top_k=top_k)
            query_label_boost: Dict[str, float] = {}
            query_vector = self.text_encoder.encode(q)
            q_scores, q_payloads = self._search_text(query_vector, top_k)
            for s, p in zip(q_scores, q_payloads):
                label = p.get("label", q)
                if label in query_labels:
                    query_label_boost[label] = max(query_label_boost.get(label, -1e9), float(s))

            candidate_ids = {str(p.get("id")) for p in q_payloads if p.get("id")}

            if self.chroma_store is not None and proposals:
                region_vectors = np.stack([self.text_image_encoder.encode(self._crop_region(image, box)) for box, _ in proposals])
                all_scores, all_payloads = self._search_batch_text(region_vectors, top_k)
            else:
                all_scores, all_payloads = None, None

            for idx, (box, obj_score) in enumerate(proposals):
                if all_scores is not None:
                    scores = all_scores[idx] if idx < len(all_scores) else np.empty((0,), dtype=np.float32)
                    payloads = all_payloads[idx] if idx < len(all_payloads) else []
                else:
                    region = self._crop_region(image, box)
                    region_vector = self.text_image_encoder.encode(region)
                    scores, payloads = self.text_index.search(
                        region_vector,
                        top_k=top_k,
                        candidate_ids=candidate_ids or None,
                    )
                by_label: Dict[str, List[float]] = {}
                topk_meta: Dict[str, List[Dict[str, Any]]] = {}
                for s, p in zip(scores, payloads):
                    label = p.get("label", q)
                    if query_labels and label not in query_labels:
                        continue
                    by_label.setdefault(label, []).append(float(s))
                    topk_meta.setdefault(label, []).append({
                        "ref_id": p.get("id"),
                        "score": float(s),
                        "modality": p.get("modality"),
                    })
                if not by_label:
                    continue

                ranked = []
                for label, svals in by_label.items():
                    agg = aggregate_scores(
                        svals,
                        method=self.config.retrieval_agg,
                        temperature=self.config.agg_temperature,
                        top_m=self.config.agg_top_m,
                    )
                    agg += 0.2 * query_label_boost.get(label, 0.0)
                    ranked.append((label, agg, svals, topk_meta[label]))

                ranked.sort(key=lambda x: x[1], reverse=True)
                best_label, best_agg, best_scores, best_topk = ranked[0]
                second = ranked[1][1] if len(ranked) > 1 else self.config.margin_eps
                margin = float(best_agg - second)
                consistency = float(np.std(best_scores))
                support = float(np.mean(np.array(best_scores) > self.config.support_threshold))
                confidence = compute_confidence(best_agg, margin, consistency, float(obj_score), support)
                score = float(best_agg)

                if confidence >= score_threshold or score >= score_threshold:
                    det = Detection(
                        label=best_label,
                        box=[float(v) for v in box],
                        score=score,
                        confidence=confidence,
                        objectness=float(obj_score),
                        top_k_scores=best_topk[:top_k],
                        margin=margin,
                        aggregation={
                            "method": self.config.retrieval_agg,
                            "temperature": self.config.agg_temperature,
                            "top_m": self.config.agg_top_m,
                            "query": q,
                        },
                        metadata={"source": "text_query"},
                    )
                    results.append(det)

        if not results:
            return []
        boxes = [d.box for d in results]
        scores = [d.confidence for d in results]
        labels = [d.label for d in results]
        keep_idxs = nms(boxes=boxes, scores=scores, labels=labels, iou_threshold=nms_iou)
        filtered = [results[i] for i in keep_idxs if scores[i] >= score_threshold]
        return filtered

    def detect_with_image_queries(
        self,
        image_path: str,
        query_images: Sequence[str],
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None,
        nms_iou: Optional[float] = None,
        query_image_agg: str = "mean",
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[Detection]:
        if self.image_index is None or self.image_index.vectors.size == 0:
            raise ValueError("image index is empty, run index with image references first")
        image = self._load_image_array(image_path)
        proposals = self.proposer.propose(image, self.config.max_proposals)
        top_k = top_k if top_k is not None else self.config.top_k
        score_threshold = score_threshold if score_threshold is not None else self.config.score_threshold
        nms_iou = nms_iou if nms_iou is not None else self.config.nms_iou
        if not query_images:
            return []

        query_image_agg = str(query_image_agg).strip().lower() if query_image_agg else "mean"
        query_label_scores: Dict[str, List[float]] = {}
        query_label_counts: Dict[str, int] = {}
        query_labels = set()
        candidate_ids = set()
        for qimg in query_images:
            q = str(qimg).strip()
            if not q:
                continue
            query_vector = self.image_encoder.encode_image(q)
            q_scores, q_payloads = self._search_image(query_vector, top_k)
            for s, p in zip(q_scores, q_payloads):
                label = p.get("label")
                if not label:
                    continue
                query_labels.add(label)
                query_label_scores.setdefault(label, []).append(float(s))
                query_label_counts[label] = query_label_counts.get(label, 0) + 1
                ref_id = p.get("id")
                if ref_id:
                    candidate_ids.add(str(ref_id))

        if query_image_agg == "mean":
            query_label_boost: Dict[str, float] = {
                label: float(np.mean(scores)) for label, scores in query_label_scores.items() if scores
            }
        elif query_image_agg == "max":
            query_label_boost = {
                label: float(np.max(scores)) for label, scores in query_label_scores.items() if scores
            }
        elif query_image_agg == "mode":
            if not query_label_counts:
                query_label_boost = {}
            else:
                max_count = max(query_label_counts.values())
                top_count_labels = [label for label, count in query_label_counts.items() if count == max_count]
                if len(top_count_labels) > 1:
                    mean_scores = {
                        label: float(np.mean(query_label_scores.get(label, [0.0])))
                        for label in top_count_labels
                    }
                    max_mean = max(mean_scores.values())
                    top_count_labels = [label for label, value in mean_scores.items() if value == max_mean]
                top_label = next(iter(sorted(top_count_labels)))
                query_label_boost = {top_label: float(query_label_counts[top_label])}
                query_labels = {top_label}
        else:
            raise ValueError("query_image_agg must be one of: mean, max, mode")

        if not query_label_boost:
            return []

        all_scores = None
        all_payloads = None
        if proposals:
            crops = [self._crop_region(image, box) for box, _ in proposals]
            region_vectors = self.image_encoder.encode_batch(crops)
            all_scores, all_payloads = self._search_batch_image(region_vectors, top_k)

        results: List[Detection] = []
        for idx, (box, obj_score) in enumerate(proposals):
            if all_scores is not None:
                scores = all_scores[idx] if idx < len(all_scores) else np.empty((0,), dtype=np.float32)
                payloads = all_payloads[idx] if idx < len(all_payloads) else []
            else:
                region = self._crop_region(image, box)
                region_vector = self.image_encoder.encode(region)
                scores, payloads = self.image_index.search(
                    region_vector,
                    top_k=top_k,
                    candidate_ids=candidate_ids or None,
                )
            by_label: Dict[str, List[float]] = {}
            topk_meta: Dict[str, List[Dict[str, Any]]] = {}
            for s, p in zip(scores, payloads):
                label = p.get("label")
                if query_labels:
                    if label not in query_labels:
                        continue
                by_label.setdefault(label, []).append(float(s))
                topk_meta.setdefault(label, []).append({
                    "ref_id": p.get("id"),
                    "score": float(s),
                    "modality": p.get("modality"),
                })
            if not by_label:
                continue

            ranked = []
            for label, svals in by_label.items():
                agg = aggregate_scores(
                    svals,
                    method=self.config.retrieval_agg,
                    temperature=self.config.agg_temperature,
                    top_m=self.config.agg_top_m,
                )
                agg += 0.2 * query_label_boost.get(label, 0.0)
                ranked.append((label, agg, svals, topk_meta[label]))

            ranked.sort(key=lambda x: x[1], reverse=True)
            best_label, best_agg, best_scores, best_topk = ranked[0]
            second = ranked[1][1] if len(ranked) > 1 else self.config.margin_eps
            margin = float(best_agg - second)
            consistency = float(np.std(best_scores))
            support = float(np.mean(np.array(best_scores) > self.config.support_threshold))
            confidence = compute_confidence(best_agg, margin, consistency, float(obj_score), support)
            score = float(best_agg)

            if confidence >= score_threshold or score >= score_threshold:
                det = Detection(
                    label=best_label,
                    box=[float(v) for v in box],
                    score=score,
                    confidence=confidence,
                    objectness=float(obj_score),
                    top_k_scores=best_topk[:top_k],
                    margin=margin,
                    aggregation={
                        "method": self.config.retrieval_agg,
                        "temperature": self.config.agg_temperature,
                        "top_m": self.config.agg_top_m,
                        "query_count": len(query_images),
                        "query_image_agg": query_image_agg,
                    },
                    metadata={"source": "image_query"},
                )
                results.append(det)

        if not results:
            return []
        boxes = [d.box for d in results]
        scores = [d.confidence for d in results]
        labels = [d.label for d in results]
        keep_idxs = nms(boxes=boxes, scores=scores, labels=labels, iou_threshold=nms_iou)
        filtered = [results[i] for i in keep_idxs if scores[i] >= score_threshold]
        return filtered

    def detect_with_image_query(
        self,
        image_path: str,
        query_image: str,
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None,
        nms_iou: Optional[float] = None,
        query_image_agg: str = "mean",
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[Detection]:
        return self.detect_with_image_queries(
            image_path=image_path,
            query_images=[query_image],
            top_k=top_k,
            score_threshold=score_threshold,
            nms_iou=nms_iou,
            query_image_agg=query_image_agg,
            metadata_filter=metadata_filter,
        )

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


def build_reference_indexes(
    references_path: str,
    out_dir: str,
    text_encoder: Optional[SimpleTextEncoder] = None,
    image_encoder: Optional[SimpleImageEncoder] = None,
    chroma_dir: Optional[str] = None,
    chroma_collection_name: str = "references",
) -> Dict[str, Any]:
    text_encoder = text_encoder or SimpleTextEncoder()
    image_encoder = image_encoder or SimpleImageEncoder()
    refs = json.loads(Path(references_path).read_text(encoding="utf-8"))
    if not isinstance(refs, list):
        raise ValueError("references file should be a list")

    text_index = ReferenceIndex()
    image_index = ReferenceIndex()

    for item in refs:
        if not isinstance(item, dict):
            continue
        label = _resolve_ref_label(item)
        if not label:
            continue

        modality = _resolve_ref_modality(item)
        ref_id = str(item.get("id", f"{modality}_{label}"))
        payload = {"id": ref_id, "label": label, "modality": modality}

        if modality == "text":
            text = str(item.get("text", item.get("prompt", item.get("query", "")))).strip()
            if not text:
                continue
            vector = text_encoder.encode(text)
            metadata = dict(payload)
            metadata["text"] = text
            text_index.add(vector, [metadata])
        else:
            image_path = str(item.get("path", item.get("image", item.get("image_path", "")))).strip()
            if not image_path:
                continue
            vector = image_encoder.encode_image(image_path)
            metadata = dict(payload)
            metadata["path"] = image_path
            image_index.add(vector, [metadata])

    out_dir_path = Path(out_dir)
    if text_index.vectors.size > 0:
        text_index.save(out_dir_path / "text")
    if image_index.vectors.size > 0:
        image_index.save(out_dir_path / "image")

    result: Dict[str, Any] = {
        "text": "ok" if text_index.vectors.size > 0 else "empty",
        "image": "ok" if image_index.vectors.size > 0 else "empty",
        "out": str(out_dir_path),
    }

    if chroma_dir:
        chroma_store = build_chroma_from_references(
            references_path=references_path,
            persist_directory=chroma_dir,
            collection_name=chroma_collection_name,
            text_encoder=text_encoder,
            image_encoder=image_encoder,
        )
        result["chroma"] = {
            "persist_directory": chroma_dir,
            "collection_name": chroma_collection_name,
            "count": chroma_store.count,
        }

    return result


def load_indices(
    index_dir: str,
    chroma_dir: Optional[str] = None,
    chroma_collection_name: str = "references",
) -> Tuple[Optional[ReferenceIndex], Optional[ReferenceIndex], Optional[ChromaReferenceStore]]:
    root = Path(index_dir)
    if not root.exists():
        raise FileNotFoundError(f"index directory not found: {index_dir}")

    text_index: Optional[ReferenceIndex] = None
    image_index: Optional[ReferenceIndex] = None

    text_path = root / "text"
    if text_path.exists():
        text_index = ReferenceIndex.load(text_path)
    elif (root / "text_vectors.npy").exists() and (root / "text_metadata.jsonl").exists():
        text_index = ReferenceIndex.load(root)

    image_path = root / "image"
    if image_path.exists():
        image_index = ReferenceIndex.load(image_path)
    elif (root / "image_vectors.npy").exists() and (root / "image_metadata.jsonl").exists():
        image_index = ReferenceIndex.load(root)

    chroma_store: Optional[ChromaReferenceStore] = None
    if chroma_dir:
        chroma_store = ChromaReferenceStore(collection_name=chroma_collection_name, persist_directory=chroma_dir)

    return text_index, image_index, chroma_store


def detections_to_json(detections: Sequence[Detection]) -> Dict[str, Any]:
    return {
        "num_detections": len(detections),
        "detections": [asdict(d) for d in detections],
    }





