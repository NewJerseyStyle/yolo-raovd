from __future__ import annotations

import math
from typing import Iterable, List

import numpy as np


def softmax(values: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    if temperature <= 0:
        temperature = 1e-6
    v = values / temperature
    v = v - np.max(v)
    exp = np.exp(v)
    return exp / (np.sum(exp) + 1e-8)


def aggregate_scores(scores: Iterable[float], method: str = "weighted_mean", temperature: float = 0.1, top_m: int | None = None) -> float:
    scores = np.array(list(scores), dtype=np.float32)
    if scores.size == 0:
        return 0.0
    scores = np.sort(scores)[::-1]
    if top_m is not None:
        scores = scores[: int(top_m)]
    if method == "max":
        return float(np.max(scores))
    if method == "mean":
        return float(np.mean(scores))
    if method == "lse":
        return float(math.log(np.sum(np.exp(scores)) + 1e-8))
    weights = softmax(scores, temperature=temperature)
    return float(np.sum(weights * scores))


def compute_confidence(agg: float, margin: float, consistency: float, objectness: float, support: float) -> float:
    x = 4.5 * agg + 2.0 * margin - 3.0 * consistency + 0.5 * np.log1p(support * 10.0) + 2.0 * objectness
    return float(1.0 / (1.0 + math.exp(-x)))


def iou(a: Iterable[float], b: Iterable[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    x_left = max(ax1, bx1)
    y_top = max(ay1, by1)
    x_right = min(ax2, bx2)
    y_bottom = min(ay2, by2)
    if x_right <= x_left or y_bottom <= y_top:
        return 0.0
    inter = (x_right - x_left) * (y_bottom - y_top)
    a_area = (ax2 - ax1) * (ay2 - ay1)
    b_area = (bx2 - bx1) * (by2 - by1)
    union = max(a_area + b_area - inter, 1e-8)
    return float(inter / union)


def nms(boxes: List[Iterable[float]], scores: List[float], labels: List[str], iou_threshold: float = 0.5):
    idxs = np.argsort(scores)[::-1]
    keep = []
    while len(idxs) > 0:
        i = int(idxs[0])
        keep.append(i)
        rem = []
        for j in idxs[1:]:
            j = int(j)
            if labels[j] != labels[i]:
                rem.append(j)
                continue
            if iou(boxes[i], boxes[j]) <= iou_threshold:
                rem.append(j)
        idxs = np.array(rem, dtype=np.int64)
    return keep

