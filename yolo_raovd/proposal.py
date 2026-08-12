from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass
class YOLOProposalGenerator:
    """YOLO-backed region proposal generator using ultralytics."""

    model_path: str = "yolo11n.pt"
    conf: float = 0.25
    iou: float = 0.45
    max_det: int = 300

    def __post_init__(self) -> None:
        from ultralytics import YOLO

        self._model = YOLO(self.model_path)

    def propose(self, image: np.ndarray, max_boxes: int = 200) -> List[Tuple[List[float], float]]:
        results = self._model(image, conf=self.conf, iou=self.iou, max_det=self.max_det, verbose=False)
        proposals: List[Tuple[List[float], float]] = []
        for r in results:
            if r.boxes is None or r.boxes.xyxy is None:
                continue
            boxes = r.boxes.xyxy.cpu().numpy()
            scores = r.boxes.conf.cpu().numpy()
            for box, score in zip(boxes, scores):
                x1, y1, x2, y2 = map(float, box)
                proposals.append(([x1, y1, x2, y2], float(score)))
                if len(proposals) >= max_boxes:
                    return proposals
        return proposals
