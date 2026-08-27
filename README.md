# RAOD: Object Detection by Dense Feature Matching + Vector Retrieval

Retrieval Augmented Open Object Detection (RAOD) is an experimental framework for open-vocabulary object detection. Instead of a closed-set region-proposal detector (e.g. YOLO), it tiles the input image, embeds each tile with a unified DINOv3 / dino.txt encoder, and matches the resulting spatial features against query embeddings or a ChromaDB reference store to produce bounding boxes.

## Features

- Unified multimodal encoder: images via DINOv3 ViT-L + vision head, text via dino.txt text encoder, both projected into a shared 2048-dim space
- Text-query or image-query matching
- Reference store built from text and/or image samples (ChromaDB only)
- Dense tiling (auto-pads to a multiple of 16) + HDBSCAN clustering into bounding boxes
- Optional SAM point-prompt refinement of cluster centers into precise masks/boxes
- COCO-style benchmark for evaluating detection performance (AP / AR)

## Install

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
```

### Model weights

The DINOv3 / dino.txt checkpoints are large (several GB) and are not shipped with the package. Download them and point the encoder at them via environment variables or CLI flags:

| Variable | Default | Contents |
| --- | --- | --- |
| `RAOD_BACKBONE_WEIGHTS` | `C:/tmp/dinov3_vitl16_backbone.pth` | DINOv3 ViT-L backbone (1.13 GB) |
| `RAOD_DINOTXT_WEIGHTS` | `C:/tmp/dinov3_vitl16_dinotxt.pth` | dino.txt text encoder + vision head (2.25 GB) |
| `RAOD_BPE_PATH` | `https://dl.fbaipublicfiles.com/dinov3/thirdparty/bpe_simple_vocab_16e6.txt.gz` | BPE vocab for the text tokenizer |

The official `dl.fbaipublicfiles.com/dinov3/*` weight URLs are gated (403). Use the Hugging Face mirrors:

```bash
# backbone
https://huggingface.co/PIA-SPACE-LAB/dinov3_vitl16_dinotxt_vision_head_and_text_encoder/resolve/main/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth
# text encoder + vision head
https://huggingface.co/PIA-SPACE-LAB/dinov3_vitl16_dinotxt_vision_head_and_text_encoder/resolve/main/dinov3_vitl16_dinotxt_vision_head_and_text_encoder-a442d8f5.pth
```

After installation:

```bash
raod --help
```

## File layout

```
.
+-- README.md
+-- requirements.txt
+-- pyproject.toml
+-- raod/
¦   +-- __init__.py
¦   +-- cli.py
¦   +-- config.py
¦   +-- dense.py
¦   +-- encoding.py
¦   +-- retrieval.py
¦   +-- visualize.py
¦   +-- _vendor/
¦       +-- dinov3/          # vendored DINOv3 + dino.txt package
+-- scripts/
    +-- run_coco_benchmark.py
+-- examples/
    +-- references_text.json
```

## Quick run

1) Build the ChromaDB reference store. Text rows are embedded with dino.txt, image rows with DINOv3, into the same 2048-dim space:

```bash
python -m raod index \
  --references examples/references_text.json \
  --out ./.raod_index \
  --chroma-dir ./.raod_chroma
```

2) Detect with a text query:

```bash
python -m raod dense \
  --image /path/to/image.jpg \
  --query "a red cup" \
  --tile-size 1024 \
  --similarity-threshold 0.30 \
  --output outputs/dense.json
```

3) Detect with an image query:

```bash
python -m raod dense \
  --image /path/to/image.jpg \
  --query-image /path/to/query_object.jpg \
  --output outputs/dense.json
```

4) Detect against the reference store (per-label boxes):

```bash
python -m raod dense \
  --image /path/to/image.jpg \
  --store ./.raod_chroma \
  --index ./.raod_index \
  --output outputs/dense.json
```

5) Draw boxes:

```bash
python -m raod draw \
  --image /path/to/image.jpg \
  --predictions outputs/dense.json \
  --output outputs/sample_vis.jpg
```

## Benchmark

Evaluate dense matching against COCO-style annotations (AP / AR at IoU 0.5). The reference store supplies both the class (via top-k retrieval) and the bounding box:

```bash
python -m raod benchmark \
  --index ./.raod_index \
  --store ./.raod_chroma \
  --annotations /path/to/instances_val2017.json \
  --images /path/to/val2017 \
  --output reports/dense_coco.json
```

There is also a standalone script that builds a reference index from COCO samples, runs dense matching over a sampled validation set, and reports AP / AR / FPS:

```bash
python scripts/run_coco_benchmark.py \
  --data-dir /path/to/coco \
  --sample-size 1000 \
  --output reports/coco_benchmark.json
```

## Dense feature matching

The image is split into overlapping tiles. If the image (or a tile) is not a multiple of 16 pixels, it is auto-padded (edge replication) so the backbone's patch grid aligns cleanly. Each tile is embedded with the unified DINOv3 encoder to produce per-patch 2048-dim features, which are matched against a query to build a similarity heatmap. High-confidence locations are clustered with HDBSCAN into regions and bounding boxes.

Options:

- `--tile-size` or `--tiles-per-axis`: control tile size (pixels) or the number of tiles per axis. Either can be used to target large or small objects.
- `--backbone-weights`, `--dinotxt-weights`, `--bpe-path`: override the encoder weight paths (defaults come from `config.py` / environment variables).
- `--overlap-ratio`: overlap between adjacent tiles (default 0.15).
- `--similarity-threshold`, `--dbscan-eps`, `--dbscan-min-samples`: heatmap threshold and HDBSCAN clustering parameters.
- `--validate`: optionally refine self-matched regions. Each region's center point is fed to SAM to build a bounding box; if SAM cannot produce a box, the region is dropped. When a `--store` is available, the SAM bbox region is also embedded and queried against ChromaDB, and dropped if no similar object is found (below `--chroma-threshold`).

## Design notes

- The reference store must be built with the same encoder used for matching so the embedding spaces align (the `index` command uses the same DINOv3/dino.txt encoder by default).
- `top-k` results are treated as retrieval evidence, not direct probabilities.
- `score` is the cosine similarity of the best matching reference for a cluster.
- SAM is optional and only used to improve segmentation / bounding-box quality; it is not required for validation.
