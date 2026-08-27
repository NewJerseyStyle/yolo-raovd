"""Dense feature matching for large images.

This module implements a tiling-based dense matching pipeline:

1. The input image is split into overlapping tiles. If the image (or a tile) is
   not a multiple of 16 pixels, it is auto-padded so the backbone's patch grid
   aligns cleanly.
2. Each tile is passed through the unified DINOv3/dino.txt encoder to produce
   per-patch 2048-dim features.
3. The features are compared against query embeddings (a single query vector or
   a ChromaDB reference store) to build a similarity heatmap.
4. High-confidence locations are clustered with HDBSCAN into regions, from which
   bounding boxes are derived.
5. Optionally, each region's center point can be fed to SAM to obtain an exact
   segmentation mask and a refined bounding box.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .encoding import DINOV3Encoder
from .retrieval import ChromaReferenceStore


# --------------------------------------------------------------------------- #
# Tiling
# --------------------------------------------------------------------------- #
@dataclass
class Tile:
    x0: int
    y0: int
    x1: int
    y1: int
    array: np.ndarray


def _snap_to_multiple(size: int, multiple: int = 16) -> int:
    return max(multiple, int(math.floor(size / multiple)) * multiple)


def _pad_to_multiple(image: np.ndarray, multiple: int = 16) -> Tuple[np.ndarray, int, int]:
    """Pad the image (right/bottom) so both dimensions are multiples of ``multiple``.

    Returns (padded_image, pad_right, pad_bottom). Padding uses edge replication
    so the padded border does not introduce artificial high-frequency content.
    """
    h, w = image.shape[:2]
    pad_right = (multiple - w % multiple) % multiple
    pad_bottom = (multiple - h % multiple) % multiple
    if pad_right == 0 and pad_bottom == 0:
        return image, 0, 0
    import numpy as np

    padded = np.pad(
        image,
        ((0, pad_bottom), (0, pad_right), (0, 0)) if image.ndim == 3 else ((0, pad_bottom), (0, pad_right)),
        mode="edge",
    )
    return padded, pad_right, pad_bottom


def generate_tiles(
    image: np.ndarray,
    tile_size: Optional[int] = None,
    tiles_per_axis: Optional[int] = None,
    overlap_ratio: float = 0.15,
    multiple: int = 16,
) -> List[Tile]:
    """Split an image into overlapping tiles.

    Either ``tile_size`` (pixels per side) or ``tiles_per_axis`` (number of tiles
    along each axis) must be given. The image is first auto-padded to a multiple
    of ``multiple`` (default 16) so the backbone patch grid aligns cleanly, and
    each tile's dimensions are snapped to a multiple of ``multiple``.
    """
    image, pad_right, pad_bottom = _pad_to_multiple(image, multiple)
    h, w = image.shape[:2]
    if tile_size is not None:
        tile_size = int(tile_size)
        if tile_size <= 0:
            raise ValueError("tile_size must be positive")
        tile_size = _snap_to_multiple(tile_size, multiple)
        step = max(multiple, int(tile_size * (1.0 - overlap_ratio)))
        tiles: List[Tile] = []
        y = 0
        while y < h:
            x = 0
            while x < w:
                x1 = min(x + tile_size, w)
                y1 = min(y + tile_size, h)
                tx0, ty0, tx1, ty1 = x, y, x1, y1
                tiles.append(Tile(tx0, ty0, tx1, ty1, image[ty0:ty1, tx0:tx1]))
                if x1 >= w:
                    break
                x = max(x + step, x1 - tile_size)
            if y1 >= h:
                break
            y = max(y + step, y1 - tile_size)
        return tiles

    if tiles_per_axis is not None:
        n = int(tiles_per_axis)
        if n <= 0:
            raise ValueError("tiles_per_axis must be positive")
        tw = _snap_to_multiple(int(w / n), multiple)
        th = _snap_to_multiple(int(h / n), multiple)
        tw = max(multiple, tw)
        th = max(multiple, th)
        step_x = max(multiple, int(tw * (1.0 - overlap_ratio)))
        step_y = max(multiple, int(th * (1.0 - overlap_ratio)))
        tiles = []
        y = 0
        while y < h:
            x = 0
            while x < w:
                x1 = min(x + tw, w)
                y1 = min(y + th, h)
                tx0, ty0, tx1, ty1 = x, y, x1, y1
                tiles.append(Tile(tx0, ty0, tx1, ty1, image[ty0:ty1, tx0:tx1]))
                if x1 >= w:
                    break
                x = max(x + step_x, x1 - tw)
            if y1 >= h:
                break
            y = max(y + step_y, y1 - th)
        return tiles

    raise ValueError("provide either tile_size or tiles_per_axis")


# --------------------------------------------------------------------------- #
# HDBSCAN clustering
# --------------------------------------------------------------------------- #
def _dbscan(points: np.ndarray, eps: float, min_samples: int) -> np.ndarray:
    """Cluster 2D points with HDBSCAN. Returns cluster labels (-1 = noise).

    ``eps`` is ignored (HDBSCAN is density-based and does not need a fixed
    neighborhood radius); ``min_samples`` controls the minimum cluster size.
    """
    n = int(points.shape[0])
    if n == 0:
        return np.zeros((0,), dtype=np.int64)
    try:
        import hdbscan
    except Exception as exc:  # pragma: no cover - optional dependency guard
        raise RuntimeError(
            "hdbscan is required for dense clustering; install with `pip install hdbscan`."
        ) from exc
    clusterer = hdbscan.HDBSCAN(min_cluster_size=max(2, int(min_samples)), metric="euclidean")
    labels = clusterer.fit_predict(points)
    return np.asarray(labels, dtype=np.int64)


# --------------------------------------------------------------------------- #
# SAM point-prompt refiner (optional)
# --------------------------------------------------------------------------- #
@dataclass
class SAMPointRefiner:
    """Build a bounding box from a region's center point using SAM (point prompt).

    Used to optionally refine dense-match regions: if SAM can produce a mask/box
    at the region's center point, the region is treated as a real object and the
    SAM-derived box is used. This is optional and not required for validation.
    """

    checkpoint: str = "sam_vit_b_01ec64.pth"
    model_type: str = "vit_b"
    device: str = "cpu"
    _resources: Dict[str, Any] = field(default_factory=dict, init=False, repr=False)

    def _load(self) -> None:
        if self._resources:
            return
        from segment_anything import SamPredictor, sam_model_registry

        sam = sam_model_registry[self.model_type](checkpoint=self.checkpoint)
        sam.to(self.device)
        self._resources["predictor"] = SamPredictor(sam)

    def refine_boxes(self, image: np.ndarray, boxes: Sequence[Sequence[float]]) -> List[Optional[List[float]]]:
        """Return a SAM-derived bbox for each input box's center, or None if SAM fails."""
        try:
            self._load()
        except Exception:  # pragma: no cover - optional dependency guard
            return [None] * len(boxes)

        predictor = self._resources["predictor"]
        predictor.set_image(np.clip(np.asarray(image), 0, 255).astype(np.uint8))
        out: List[Optional[List[float]]] = []
        for box in boxes:
            cx = (float(box[0]) + float(box[2])) / 2.0
            cy = (float(box[1]) + float(box[3])) / 2.0
            masks, scores, _ = predictor.predict(
                point_coords=np.array([[cx, cy]], dtype=np.float32),
                point_labels=np.array([1]),
                multimask_output=True,
            )
            best: Optional[List[float]] = None
            best_score = -1.0
            for m, s in zip(masks, scores):
                if m.sum() == 0:
                    continue
                ys, xs = np.where(m)
                bb = [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]
                if float(s) > best_score:
                    best_score = float(s)
                    best = bb
            out.append(best)
        return out


# --------------------------------------------------------------------------- #
# Dense matcher
# --------------------------------------------------------------------------- #
@dataclass
class DenseMatcher:
    """Dense feature matching over tiles, producing heatmaps and bounding boxes."""

    encoder: Optional[DINOV3Encoder] = None
    tile_size: Optional[int] = 1024
    tiles_per_axis: Optional[int] = None
    overlap_ratio: float = 0.15
    similarity_threshold: float = 0.30
    dbscan_eps: float = 1.5
    dbscan_min_samples: int = 3
    max_regions: int = 200
    device: str = "cpu"
    _resources: Dict[str, Any] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.encoder is None:
            self.encoder = DINOV3Encoder(device=self.device)

    def _tile_heatmap(self, tile: Tile, query: np.ndarray) -> np.ndarray:
        feat, (fh, fw) = self.encoder.encode_image_spatial(tile.array)  # (H*W, 2048)
        norms = np.linalg.norm(feat, axis=1, keepdims=True) + 1e-8
        feat = feat / norms
        q = query / (np.linalg.norm(query) + 1e-8)
        sims = feat @ q  # (H*W,)
        return sims.reshape(fh, fw)

    def _heatmap_to_boxes(self, heatmap: np.ndarray, tile: Tile) -> List[Tuple[List[float], float]]:
        fh, fw = heatmap.shape
        ys, xs = np.where(heatmap >= self.similarity_threshold)
        if len(xs) == 0:
            return []
        points = np.stack([xs.astype(np.float32), ys.astype(np.float32)], axis=1)
        labels = _dbscan(points, eps=self.dbscan_eps, min_samples=self.dbscan_min_samples)
        boxes: List[Tuple[List[float], float]] = []
        for cid in np.unique(labels):
            if cid < 0:
                continue
            mask = labels == cid
            cx, cy = points[mask].mean(axis=0)
            scores = heatmap[ys[mask], xs[mask]]
            score = float(np.max(scores))
            # map feature coords back to pixel coords within the tile
            px = xs[mask] / fw * (tile.x1 - tile.x0) + tile.x0
            py = ys[mask] / fh * (tile.y1 - tile.y0) + tile.y0
            x0, x1 = float(px.min()), float(px.max())
            y0, y1 = float(py.min()), float(py.max())
            boxes.append(([x0, y0, x1, y1], score))
        return boxes

    def _crop_region(self, image: np.ndarray, box: Sequence[float]) -> np.ndarray:
        x1, y1, x2, y2 = [int(v) for v in box]
        h, w = image.shape[:2]
        x1 = max(0, min(w - 1, x1))
        x2 = max(x1 + 1, min(w, x2))
        y1 = max(0, min(h - 1, y1))
        y2 = max(y1 + 1, min(h, y2))
        return image[y1:y2, x1:x2]

    def refine_regions(
        self,
        image: np.ndarray,
        boxes: Sequence[Sequence[float]],
        store: Optional[ChromaReferenceStore] = None,
        modality: str = "image",
        top_k: int = 5,
        sam_refiner: Optional[SAMPointRefiner] = None,
        chroma_threshold: Optional[float] = None,
    ) -> List[Tuple[List[float], float, Optional[str]]]:
        """Optionally refine candidate regions with SAM and/or ChromaDB.

        If ``sam_refiner`` is provided, each region's center is fed to SAM and
        the SAM-derived box is used (regions where SAM fails are dropped). If a
        ``store`` is provided, each region is re-embedded and matched against the
        store to assign a label and score; regions below ``chroma_threshold`` are
        dropped.

        Returns (box, score, label) per accepted region. ``label`` is None when
        no store is given.
        """
        threshold = chroma_threshold if chroma_threshold is not None else self.similarity_threshold

        refined_boxes: List[Optional[List[float]]]
        if sam_refiner is not None:
            refined_boxes = sam_refiner.refine_boxes(image, boxes)
        else:
            refined_boxes = [list(map(float, b)) for b in boxes]

        results: List[Tuple[List[float], float, Optional[str]]] = []
        for box, refined in zip(boxes, refined_boxes):
            if refined is None:
                # SAM could not build a bbox from the center point -> not an object
                continue
            if store is None:
                results.append((refined, 0.0, None))
                continue
            crop = self._crop_region(image, refined)
            vec = self.encoder.encode_image(crop)
            scores, payloads = store.search(vec, top_k=top_k, modality=modality)
            if not len(scores):
                continue
            best = int(np.argmax(scores))
            s = float(scores[best])
            if s < threshold:
                # ChromaDB found no similar object for the region -> drop
                continue
            lab = str(payloads[best].get("label", "unknown"))
            results.append((refined, s, lab))
        return results

    def match_query(
        self,
        image: np.ndarray,
        query: np.ndarray,
    ) -> Tuple[List[Tuple[List[float], float]], np.ndarray]:
        """Dense-match a single query vector against the image.

        Returns (boxes, global_heatmap) where global_heatmap is a downsampled
        similarity map over the whole image (for visualization).
        """
        tiles = generate_tiles(
            image,
            tile_size=self.tile_size,
            tiles_per_axis=self.tiles_per_axis,
            overlap_ratio=self.overlap_ratio,
        )
        all_boxes: List[Tuple[List[float], float]] = []
        gh, gw = image.shape[:2]
        # global heatmap at ~1/16 resolution (patch size 16)
        heat = np.zeros((max(1, gh // 16), max(1, gw // 16)), dtype=np.float32)
        count = np.zeros_like(heat)
        for tile in tiles:
            hm = self._tile_heatmap(tile, query)
            fh, fw = hm.shape
            gy0 = tile.y0 // 16
            gx0 = tile.x0 // 16
            gy1 = min(heat.shape[0], gy0 + fh)
            gx1 = min(heat.shape[1], gx0 + fw)
            if gy1 <= gy0 or gx1 <= gx0:
                continue
            heat[gy0:gy1, gx0:gx1] += hm[: gy1 - gy0, : gx1 - gx0]
            count[gy0:gy1, gx0:gx1] += 1.0
        count[count == 0] = 1.0
        heat /= count
        all_boxes.extend(self._heatmap_to_boxes(heat, Tile(0, 0, gw, gh, image)))
        return all_boxes, heat

    def _cluster_label_points(
        self,
        points: np.ndarray,
        scores: np.ndarray,
        labels: np.ndarray,
    ) -> List[Tuple[List[float], float, str]]:
        """Cluster global-pixel points for one label into bounding boxes."""
        if len(points) == 0:
            return []
        cluster_ids = _dbscan(points, eps=self.dbscan_eps, min_samples=self.dbscan_min_samples)
        boxes: List[Tuple[List[float], float, str]] = []
        for cid in np.unique(cluster_ids):
            if cid < 0:
                continue
            mask = cluster_ids == cid
            score = float(np.max(scores[mask]))
            lab = labels[mask][0]
            px = points[mask, 0]
            py = points[mask, 1]
            x0, x1 = float(px.min()), float(px.max())
            y0, y1 = float(py.min()), float(py.max())
            boxes.append(([x0, y0, x1, y1], score, lab))
        return boxes

    def match_reference_store(
        self,
        image: np.ndarray,
        store: ChromaReferenceStore,
        modality: str = "image",
        top_k: int = 5,
        label: Optional[str] = None,
    ) -> List[Tuple[List[float], float, str]]:
        """Match all tile features against a reference store.

        For each spatial location in each tile, retrieve the nearest reference and
        accumulate a per-label set of high-confidence global pixel locations, then
        cluster each label's locations with HDBSCAN into bounding boxes.
        """
        tiles = generate_tiles(
            image,
            tile_size=self.tile_size,
            tiles_per_axis=self.tiles_per_axis,
            overlap_ratio=self.overlap_ratio,
        )
        # per-label: list of (points, scores) accumulated across tiles
        label_points: Dict[str, List[np.ndarray]] = {}
        label_scores: Dict[str, List[np.ndarray]] = {}
        for tile in tiles:
            feat, (fh, fw) = self.encoder.encode_image_spatial(tile.array)  # (H*W, 2048)
            norms = np.linalg.norm(feat, axis=1, keepdims=True) + 1e-8
            feat = feat / norms
            # batch search against store
            scores, payloads = store.search_batch(feat, top_k=top_k, modality=modality)
            for i in range(feat.shape[0]):
                srow = scores[i] if i < len(scores) else np.empty((0,), dtype=np.float32)
                prow = payloads[i] if i < len(payloads) else []
                if not len(srow):
                    continue
                best = int(np.argmax(srow))
                s = float(srow[best])
                if s < self.similarity_threshold:
                    continue
                lab = str(prow[best].get("label", label or "unknown"))
                if label and lab != label:
                    continue
                fy = i // fw
                fx = i % fw
                px = fx / fw * (tile.x1 - tile.x0) + tile.x0
                py = fy / fh * (tile.y1 - tile.y0) + tile.y0
                label_points.setdefault(lab, []).append(np.array([[px, py]], dtype=np.float32))
                label_scores.setdefault(lab, []).append(np.array([s], dtype=np.float32))

        results: List[Tuple[List[float], float, str]] = []
        for lab, pts in label_points.items():
            points = np.concatenate(pts, axis=0)
            sc = np.concatenate(label_scores[lab], axis=0)
            results.extend(self._cluster_label_points(points, sc, np.full(len(points), lab)))
        return results
