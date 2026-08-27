from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .dense import DenseMatcher, SAMPointRefiner
from .encoding import DINOV3Encoder
from .retrieval import build_reference_indexes, load_indices
from .visualize import draw_predictions


def _ensure_parent(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def _load_image(path: str) -> np.ndarray:
    from PIL import Image

    return np.array(Image.open(path).convert("RGB"))


def _build_matcher(args: argparse.Namespace) -> DenseMatcher:
    encoder = DINOV3Encoder(
        backbone_weights=args.backbone_weights,
        dinotxt_weights=args.dinotxt_weights,
        bpe_path_or_url=args.bpe_path,
        device=args.device,
    )
    return DenseMatcher(
        encoder=encoder,
        tile_size=args.tile_size,
        tiles_per_axis=args.tiles_per_axis,
        overlap_ratio=args.overlap_ratio,
        similarity_threshold=args.similarity_threshold,
        dbscan_eps=args.dbscan_eps,
        dbscan_min_samples=args.dbscan_min_samples,
        max_regions=args.max_regions,
        device=args.device,
    )


def _build_encoder(args: argparse.Namespace) -> DINOV3Encoder:
    return DINOV3Encoder(
        backbone_weights=args.backbone_weights,
        dinotxt_weights=args.dinotxt_weights,
        bpe_path_or_url=args.bpe_path,
        device=args.device,
    )


def cmd_index(args: argparse.Namespace) -> int:
    encoder = _build_encoder(args)
    result = build_reference_indexes(
        references_path=args.references,
        out_dir=args.out,
        encoder=encoder,
        chroma_dir=args.chroma_dir,
        chroma_collection_name="references",
    )
    print(f"index built at: {args.out}")
    if result.get("chroma_dir"):
        print(f"chroma collection ready: {result['chroma_dir']}")
    return 0


def cmd_dense(args: argparse.Namespace) -> int:
    """Dense feature matching: tile the image, embed, match, cluster into boxes."""
    image = _load_image(args.image)
    matcher = _build_matcher(args)

    results: List[Dict[str, Any]] = []

    chroma_store = None
    if args.store:
        chroma_store = load_indices(
            args.index,
            chroma_dir=args.store,
            chroma_collection_name="references",
        )
        if chroma_store is None:
            raise ValueError("--store requires a Chroma directory built by `index`")

    if args.store:
        boxes = matcher.match_reference_store(
            image,
            chroma_store,
            modality=args.modality,
            top_k=args.top_k,
            label=args.label,
        )
        for box, score, label in boxes:
            results.append({"box": box, "score": score, "label": label})
    elif args.query_image:
        query = matcher.encoder.encode_image(_load_image(args.query_image))
        boxes, heatmap = matcher.match_query(image, query)
        for box, score in boxes:
            results.append({"box": box, "score": score, "label": "query"})
    elif args.query:
        query = matcher.encoder.encode_text(args.query)
        boxes, heatmap = matcher.match_query(image, query)
        for box, score in boxes:
            results.append({"box": box, "score": score, "label": args.query})
    else:
        raise ValueError("provide --query, --query-image, or --store")

    # Optionally refine self-matched regions: SAM to build a bbox from the region
    # center, and (when a store is available) ChromaDB to assign a label/score.
    if args.validate and results:
        sam_refiner = SAMPointRefiner(
            checkpoint=args.sam_checkpoint,
            model_type=args.sam_model_type,
            device=args.device,
        )
        validated = matcher.refine_regions(
            image,
            [r["box"] for r in results],
            store=chroma_store,
            modality=args.modality,
            top_k=args.top_k,
            sam_refiner=sam_refiner,
            chroma_threshold=args.chroma_threshold,
        )
        results = [
            {"box": box, "score": score, "label": label if label is not None else r["label"]}
            for r, (box, score, label) in zip(results, validated)
        ]

    payload = {"num_regions": len(results), "regions": results, "image": args.image}
    out = args.output or "outputs/dense.json"
    _ensure_parent(out)
    Path(out).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved dense regions: {out}")
    return 0


def cmd_draw(args: argparse.Namespace) -> int:
    pred = json.loads(Path(args.predictions).read_text(encoding="utf-8"))
    draws = pred.get("regions", pred.get("detections", []))
    draw_predictions(args.image, draws, args.output)
    print(f"saved overlay: {args.output}")
    return 0


def _xywh_to_xyxy(box: List[float]) -> List[float]:
    x, y, w, h = box
    return [x, y, x + w, y + h]


def _iou(a: List[float], b: List[float]) -> float:
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


def _coco_read_annotations(ann_path: str) -> Tuple[Dict[int, List[List]], Dict[int, str], Dict[str, int], Dict[int, str]]:
    ann = json.loads(Path(ann_path).read_text(encoding="utf-8"))
    images = {img["id"]: img["file_name"] for img in ann.get("images", [])}
    cat_id_to_name = {cat["id"]: cat.get("name", str(cat["id"])) for cat in ann.get("categories", [])}
    cat_name_to_id = {v: k for k, v in cat_id_to_name.items()}
    gt: Dict[int, List[List]] = {}
    for a in ann.get("annotations", []):
        if "bbox" not in a or "image_id" not in a:
            continue
        gt.setdefault(a["image_id"], []).append([a["category_id"], _xywh_to_xyxy(a["bbox"])])
    return gt, images, cat_name_to_id, cat_id_to_name


def _ap_from_pr(tp: List[int], fp: List[int], total_gts: int) -> Dict[str, float]:
    if total_gts <= 0:
        return {"ap": 0.0, "ar": 0.0, "num_gt": 0}
    tp = np.array(tp, dtype=np.float32)
    fp = np.array(fp, dtype=np.float32)
    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    rec = tp_cum / (total_gts + 1e-8)
    prec = tp_cum / (tp_cum + fp_cum + 1e-8)

    if len(rec) == 0:
        return {"ap": 0.0, "ar": 0.0, "num_gt": total_gts}

    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))
    for i in range(len(mpre) - 2, -1, -1):
        if mpre[i + 1] > mpre[i]:
            mpre[i] = mpre[i + 1]
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))
    ar = float(rec[-1])
    return {"ap": ap, "ar": ar, "num_gt": total_gts}


def _eval_sorted_predictions(
    sorted_records: List[Tuple[float, int, List[float]]],
    gt_by_image: Dict[int, List[List]],
    target_cat_id: int,
    iou_threshold: float = 0.5,
) -> Tuple[List[int], List[int], int]:
    used_map: Dict[int, List[bool]] = {}
    gt_cache: Dict[int, List[List[float]]] = {}
    for image_id, ann in gt_by_image.items():
        class_gts = [b[1] for b in ann if b[0] == target_cat_id]
        if class_gts:
            gt_cache[image_id] = class_gts
            used_map[image_id] = [False] * len(class_gts)

    tp: List[int] = []
    fp: List[int] = []
    total_gt = sum(len(v) for v in gt_cache.values())

    for _, image_id, box in sorted_records:
        class_gts = gt_cache.get(image_id, [])
        if not class_gts:
            fp.append(1)
            tp.append(0)
            continue

        used = used_map.setdefault(image_id, [False] * len(class_gts))
        max_iou = 0.0
        max_idx = -1
        for i, g in enumerate(class_gts):
            if used[i]:
                continue
            value = _iou(box, g)
            if value > max_iou:
                max_iou = float(value)
                max_idx = i
        if max_iou >= iou_threshold and max_idx >= 0:
            used[max_idx] = True
            tp.append(1)
            fp.append(0)
        else:
            tp.append(0)
            fp.append(1)

    return tp, fp, total_gt


def cmd_benchmark(args: argparse.Namespace) -> int:
    """Evaluate dense matching against COCO-style annotations."""
    chroma_store = load_indices(
        args.index,
        chroma_dir=args.store,
        chroma_collection_name="references",
    )
    if chroma_store is None:
        raise ValueError("--store requires a Chroma directory built by `index`")

    matcher = _build_matcher(args)

    gt_by_image, image_files, cat_name_to_id, _ = _coco_read_annotations(args.annotations)

    metrics: Dict[str, Dict[str, float]] = {}
    for label, cat_id in cat_name_to_id.items():
        records: List[Tuple[float, int, List[float]]] = []
        for image_id, fname in image_files.items():
            image_path = str(Path(args.images) / fname)
            if not Path(image_path).exists():
                continue
            image = _load_image(image_path)
            boxes = matcher.match_reference_store(
                image,
                chroma_store,
                modality=args.modality,
                top_k=args.top_k,
                label=label,
            )
            for box, score, lab in boxes:
                if lab == label:
                    records.append((float(score), int(image_id), box))

        records.sort(reverse=True, key=lambda x: x[0])
        tp, fp, total_gt = _eval_sorted_predictions(records, gt_by_image, int(cat_id), iou_threshold=0.5)
        res = _ap_from_pr(tp, fp, total_gt)
        res["num_predictions"] = len(records)
        metrics[label] = res

    per_label_aps = [v.get("ap", 0.0) for v in metrics.values() if "ap" in v]
    mean_ap = float(np.mean(per_label_aps)) if per_label_aps else 0.0
    per_label_ars = [v.get("ar", 0.0) for v in metrics.values() if "ar" in v]
    mean_ar = float(np.mean(per_label_ars)) if per_label_ars else 0.0

    summary = {
        "num_labels": len(metrics),
        "labels": list(metrics.keys()),
        "mean_ap": mean_ap,
        "mean_ar": mean_ar,
        "metrics": metrics,
        "source": {
            "index": args.index,
            "store": args.store,
            "modality": args.modality,
        },
    }
    _ensure_parent(args.output)
    Path(args.output).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved benchmark: {args.output}")
    return 0


def _add_encoder_args(p) -> None:
    from .config import DEFAULT_BACKBONE_WEIGHTS, DEFAULT_BPE_PATH, DEFAULT_DINOTXT_WEIGHTS

    p.add_argument("--backbone-weights", default=DEFAULT_BACKBONE_WEIGHTS,
                   help="Path/URL to DINOv3 backbone weights")
    p.add_argument("--dinotxt-weights", default=DEFAULT_DINOTXT_WEIGHTS,
                   help="Path/URL to dino.txt text encoder + vision head weights")
    p.add_argument("--bpe-path", default=DEFAULT_BPE_PATH,
                   help="Path/URL to the BPE vocabulary for the text tokenizer")


def _add_dense_args(p) -> None:
    _add_encoder_args(p)
    p.add_argument("--device", default="cpu")
    p.add_argument("--tile-size", type=int, default=1024)
    p.add_argument("--tiles-per-axis", type=int, default=None)
    p.add_argument("--overlap-ratio", type=float, default=0.15)
    p.add_argument("--similarity-threshold", type=float, default=0.30)
    p.add_argument("--dbscan-eps", type=float, default=1.5)
    p.add_argument("--dbscan-min-samples", type=int, default=3)
    p.add_argument("--max-regions", type=int, default=200)


def main() -> int:
    parser = argparse.ArgumentParser(prog="raod", description="RAOD dense matching CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_idx = sub.add_parser("index")
    p_idx.add_argument("--references", required=True)
    p_idx.add_argument("--out", required=True)
    p_idx.add_argument("--chroma-dir", default=None)
    _add_encoder_args(p_idx)
    p_idx.add_argument("--device", default="cpu")
    p_idx.set_defaults(func=cmd_index)

    p_dense = sub.add_parser("dense")
    p_dense.add_argument("--image", required=True)
    p_dense.add_argument("--query", default=None)
    p_dense.add_argument("--query-image", default=None)
    p_dense.add_argument("--store", default=None)
    p_dense.add_argument("--index", default=None)
    p_dense.add_argument("--label", default=None)
    p_dense.add_argument("--modality", choices=["text", "image"], default="image")
    p_dense.add_argument("--top-k", type=int, default=5)
    p_dense.add_argument("--validate", action="store_true",
                         help="Validate self-matched regions with SAM bbox + ChromaDB similarity")
    p_dense.add_argument("--sam-checkpoint", default="sam_vit_b_01ec64.pth")
    p_dense.add_argument("--sam-model-type", default="vit_b")
    p_dense.add_argument("--chroma-threshold", type=float, default=None,
                         help="ChromaDB similarity threshold for region validation (default: --similarity-threshold)")
    p_dense.add_argument("--output", default="outputs/dense.json")
    _add_dense_args(p_dense)
    p_dense.set_defaults(func=cmd_dense)

    p_draw = sub.add_parser("draw")
    p_draw.add_argument("--image", required=True)
    p_draw.add_argument("--predictions", required=True)
    p_draw.add_argument("--output", default="outputs/vis.jpg")
    p_draw.set_defaults(func=cmd_draw)

    p_bm = sub.add_parser("benchmark")
    p_bm.add_argument("--index", required=True)
    p_bm.add_argument("--store", required=True)
    p_bm.add_argument("--annotations", required=True)
    p_bm.add_argument("--images", required=True)
    p_bm.add_argument("--modality", choices=["text", "image"], default="image")
    p_bm.add_argument("--top-k", type=int, default=5)
    p_bm.add_argument("--output", default="reports/dense_coco.json")
    _add_dense_args(p_bm)
    p_bm.set_defaults(func=cmd_benchmark)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
