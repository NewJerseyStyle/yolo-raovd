from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class RegionProposalGenerator:
    """Grid-style proposal fallback before YOLO backbone is plugged in."""

    def propose(self, width: int, height: int, max_boxes: int = 200) -> List[Tuple[List[float], float]]:
        proposals = []
        scales = [0.28, 0.2, 0.12]
        strides = []
        for scale in scales:
            box_w = max(int(width * scale), 24)
            box_h = max(int(height * scale), 24)
            stride_w = max(int(box_w * 0.7), 1)
            stride_h = max(int(box_h * 0.7), 1)
            strides.append((box_w, box_h, stride_w, stride_h))

        for idx, (box_w, box_h, stride_w, stride_h) in enumerate(strides):
            for y in range(0, max(height - box_h + 1, 1), stride_h):
                for x in range(0, max(width - box_w + 1, 1), stride_w):
                    x1, y1 = float(x), float(y)
                    x2, y2 = float(min(x + box_w, width - 1)), float(min(y + box_h, height - 1))
                    if x2 <= x1 or y2 <= y1:
                        continue
                    score = 0.35 + 0.15 * (1.0 - idx * 0.2)
                    proposals.append(([x1, y1, x2, y2], float(min(max(score, 0.05), 0.95))))
                    if len(proposals) >= max_boxes:
                        return proposals
        return proposals

