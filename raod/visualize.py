from __future__ import annotations

from typing import Sequence

from PIL import Image, ImageDraw


def draw_predictions(image_path: str, preds: Sequence[dict], output_path: str) -> None:
    with Image.open(image_path) as im:
        im = im.convert("RGB")
        draw = ImageDraw.Draw(im)
        for d in preds:
            box = d.get("box", [0, 0, 0, 0])
            x1, y1, x2, y2 = map(float, box)
            label = d.get("label", "object")
            conf = d.get("confidence", d.get("score", 0.0))
            color = "lime"
            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
            text = f"{label}: {conf:.2f}"
            draw.text((x1 + 3, y1 + 3), text, fill=color)
        im.save(output_path)

