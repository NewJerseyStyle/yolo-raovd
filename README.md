# YOLO-RAOVD: Zero-Shot Detection by Region Proposals + Vector Retrieval

YOLO-RAOVD is an experimental framework for open-vocabulary object detection with region proposals and retrieval-based matching.

## Features

- Text or image query input (image query supports multiple reference images per label)
- Top-k retrieval-based evidence for each region
- Confidence score generation from top-k evidence
- Optional box visualization
- COCO benchmark script skeleton

## What is implemented now

- `yolo_raovd` package with CLI entrypoint
- Text/image reference indexing
- Detection pipeline (MVP: region proposals + encoder models)
- Drawing and evaluation entrypoints

## Install

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
```

Before running DINOv3-based features, sign in to Hugging Face and accept the model usage agreement:

```bash
huggingface-cli login
```

Then open the DINO model pages and agree to the license terms:

- https://huggingface.co/facebook/dinov3-vits16plus-pretrain-lvd1689m
- https://huggingface.co/facebook/dinov3-vit7b16-pretrain-lvd1689m
- https://huggingface.co/facebook/dinov2-base (fallback)

After installation:

```bash
yolo-raovd --help
```

## File layout

```
.
+-- README.md
+-- design.md
+-- requirements.txt
+-- pyproject.toml
+-- yolo_raovd/
¦   +-- __init__.py
¦   +-- cli.py
¦   +-- confidence.py
¦   +-- encoding.py
¦   +-- pipeline.py
¦   +-- proposal.py
¦   +-- retrieval.py
¦   +-- types.py
¦   +-- visualize.py
+-- examples/
    +-- references_text.json
```

## Quick run

1) Build text references index

```bash
python -m yolo_raovd.cli index \
  --references examples/references_text.json \
  --out ./.yolo_raovd_index
```

2) Detect with text query

```bash
python -m yolo_raovd.cli detect \
  --image /path/to/image.jpg \
  --index ./.yolo_raovd_index \
  --query "red cup" \
  --query "blue bottle" \
  --top-k 20 \
  --score-threshold 0.20 \
  --output outputs/predictions.json
```

3) Detect with image query

```bash
python -m yolo_raovd.cli detect \
  --image /path/to/image.jpg \
  --index ./.yolo_raovd_index \
  --query-image /path/to/query_object.jpg \
  --top-k 20 \
  --score-threshold 0.20 \
  --query-image-agg max \
  --output outputs/predictions_by_image.json
```

4) Draw boxes

```bash
python -m yolo_raovd.cli draw \
  --image /path/to/image.jpg \
  --predictions outputs/predictions.json \
  --output outputs/sample_vis.jpg
```

5) Benchmark (text query)

```bash
python -m yolo_raovd.cli benchmark \
  --index ./.yolo_raovd_index \
  --annotations /path/to/instances_val2017.json \
  --images /path/to/val2017 \
  --prompts examples/references_text.json \
  --benchmark-mode text \
  --output reports/coco_text.json
```

6) Benchmark (image query)

```bash
python -m yolo_raovd.cli benchmark \
  --index ./.yolo_raovd_index \
  --annotations /path/to/instances_val2017.json \
  --images /path/to/val2017 \
  --prompts examples/references_image.json \
  --benchmark-mode image \
  --output reports/coco_image.json
```

For image-query benchmark, each label can provide one or more example images. The `--query-image-agg` option controls how multiple query images are merged.

- `mean` (default): average retrieval scores per label.
- `max`: use maximum retrieval score per label.
- `mode`: pick the single most frequent label across all query top-k retrieval results (majority vote).

```bash
python -m yolo_raovd.cli benchmark \
  --index ./.yolo_raovd_index \
  --annotations /path/to/instances_val2017.json \
  --images /path/to/val2017 \
  --prompts examples/references_image.json \
  --benchmark-mode image \
  --query-image-agg mode \
  --output reports/coco_image.json
```

The query format can be:

```json
{
  "person": [
    "examples/query_images/person_front.jpg",
    "examples/query_images/person_side.jpg"
  ],
  "bicycle": ["examples/query_images/bicycle_1.jpg"]
}
```

`prompts` can also be a JSON array. For each label, each entry may include `query`, `text`, `prompt`, `path`, or `image`:

```json
[
  { "label": "person", "image": "examples/query_images/person_front.jpg" },
  { "label": "person", "image": "examples/query_images/person_side.jpg" },
  { "label": "bicycle", "path": "examples/query_images/bicycle_1.jpg" }
]
```

`--benchmark-mode text` still accepts the existing text format.

## Design notes

- `top-k` results are treated as retrieval evidence, not direct probabilities.
- `detection score` = retrieval aggregation result
- `confidence` = calibrated confidence-like score in `[0,1]` (sigmoid from weighted retrieval term + margin + consistency + objectness)
- default aggregation methods in `confidence.py`:
  - `weighted_mean`
  - `max`
  - `mean`
  - `lse`

## Runtime dependencies for model loading

```bash
pip install torch torchvision transformers
```

- If your workflow uses `--query-image`, DINOv3 is used for image embedding.
- If your workflow uses `--query`, CLIP image embeddings are used for region matching.
