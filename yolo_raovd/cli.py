from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from .confidence import iou
from .pipeline import YoloRaovdConfig, YoloRaovdDetector, build_reference_indexes, detections_to_json, load_indices
from .pdf_stats import analyze_pdf, format_summary, summarize_results
from .visualize import draw_predictions


def _ensure_parent(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def cmd_init(args: argparse.Namespace) -> int:
    for d in args.directories:
        Path(d).mkdir(parents=True, exist_ok=True)
    return 0


def cmd_index(args: argparse.Namespace) -> int:
    build_reference_indexes(
        references_path=args.references,
        out_dir=args.out,
    )
    print(f"index built at: {args.out}")
    return 0


def cmd_detect(args: argparse.Namespace) -> int:
    text_index, image_index = load_indices(args.index)
    cfg = YoloRaovdConfig(
        top_k=args.top_k,
        score_threshold=args.score_threshold,
        nms_iou=args.nms_iou,
        yolo_model_path=args.yolo_model,
        yolo_conf=args.yolo_conf,
        yolo_iou=args.yolo_iou,
        yolo_max_det=args.yolo_max_det,
    )
    detector = YoloRaovdDetector(text_index=text_index, image_index=image_index, config=cfg)

    queries: List[str] = []
    if args.queries:
        queries.extend([q.strip() for q in args.queries.split(",") if q.strip()])
    if args.query:
        queries.extend(args.query)

    query_images: List[str] = []
    if args.query_image:
        query_images.extend(args.query_image)

    if not queries and not query_images:
        raise ValueError("no query is provided")

    detections = []
    mode = str(args.query_mode).strip().lower()
    if mode == "text":
        for q in queries:
            detections.extend(
                detector.detect_with_text_queries(
                    image_path=args.image,
                    queries=[q],
                    top_k=args.top_k,
                    score_threshold=args.score_threshold,
                    nms_iou=args.nms_iou,
                )
            )
    elif mode == "image":
        if not query_images:
            raise ValueError("image mode requires --query-image")
        detections.extend(
            detector.detect_with_image_queries(
                image_path=args.image,
                query_images=query_images,
                top_k=args.top_k,
                score_threshold=args.score_threshold,
                nms_iou=args.nms_iou,
                query_image_agg=args.query_image_agg,
            )
        )
    elif mode == "hybrid":
        if not queries and not query_images:
            raise ValueError("hybrid mode requires text or image queries")
        if queries:
            detections.extend(
                detector.detect_with_text_queries(
                    image_path=args.image,
                    queries=queries,
                    top_k=args.top_k,
                    score_threshold=args.score_threshold,
                    nms_iou=args.nms_iou,
                )
            )
        if query_images:
            detections.extend(
                detector.detect_with_image_queries(
                    image_path=args.image,
                    query_images=query_images,
                    top_k=args.top_k,
                    score_threshold=args.score_threshold,
                    nms_iou=args.nms_iou,
                    query_image_agg=args.query_image_agg,
                )
            )
    else:
        raise ValueError("query mode must be one of: text, image, hybrid")

    payload = detections_to_json(detections)
    payload["image"] = args.image
    payload["queries"] = queries
    payload["query_images"] = query_images
    payload["query_image_agg"] = args.query_image_agg

    out = args.output or "outputs/predictions.json"
    _ensure_parent(out)
    Path(out).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved predictions: {out}")
    return 0


def cmd_draw(args: argparse.Namespace) -> int:
    pred = json.loads(Path(args.predictions).read_text(encoding="utf-8"))
    draws = pred.get("detections", [])
    draw_predictions(args.image, draws, args.output)
    print(f"saved overlay: {args.output}")
    return 0


def _xywh_to_xyxy(box: List[float]) -> List[float]:
    x, y, w, h = box
    return [x, y, x + w, y + h]


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
            value = iou(box, g)
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


def _to_query_list(value: object) -> List[str]:
    values: List[str] = []
    if isinstance(value, str):
        value = value.strip()
        if value:
            return [value]
        return []
    if isinstance(value, (list, tuple)):
        for item in value:
            if isinstance(item, str):
                v = item.strip()
                if v and v not in values:
                    values.append(v)
    return values


def _normalize_prompt_query_file(path: Path, query: str) -> str:
    p = Path(query)
    if not p.is_absolute():
        p = path.parent / p
    return str(p)


def _load_benchmark_queries(prompts: object) -> Dict[str, List[str]]:
    label_to_query: Dict[str, List[str]] = {}

    if isinstance(prompts, dict):
        for label, query in prompts.items():
            if not isinstance(label, str) or not label.strip():
                continue
            values = _to_query_list(query)
            if values:
                label_to_query[label.strip()] = values
        return label_to_query

    if isinstance(prompts, list):
        for item in prompts:
            if not isinstance(item, dict):
                continue
            label = str(item.get("label", "")).strip()
            if not label:
                continue
            for key in ("query", "text", "prompt", "path", "image"):
                values = _to_query_list(item.get(key))
                if values:
                    merged = label_to_query.setdefault(label, [])
                    for value in values:
                        if value not in merged:
                            merged.append(value)
                    break
    return label_to_query


def cmd_benchmark(args: argparse.Namespace) -> int:
    text_index, image_index = load_indices(args.index)
    cfg = YoloRaovdConfig(
        top_k=args.top_k,
        score_threshold=args.score_threshold,
        nms_iou=0.5,
        yolo_model_path=args.yolo_model,
        yolo_conf=args.yolo_conf,
        yolo_iou=args.yolo_iou,
        yolo_max_det=args.yolo_max_det,
    )
    detector = YoloRaovdDetector(text_index=text_index, image_index=image_index, config=cfg)

    if args.benchmark_mode == "image" and image_index is None:
        raise ValueError("image index is empty, run index with image modality references first")
    if args.benchmark_mode == "text" and text_index is None:
        raise ValueError("text index is empty, run index with text modality references first")

    gt_by_image, image_files, cat_name_to_id, _ = _coco_read_annotations(args.annotations)
    prompts_raw = json.loads(Path(args.prompts).read_text(encoding="utf-8"))
    label_to_queries = _load_benchmark_queries(prompts_raw)
    prompts_path = Path(args.prompts)

    metrics: Dict[str, Dict[str, float]] = {}
    for label, queries in {k: v for k, v in label_to_queries.items() if k.strip() and v}.items():
        cat_id = cat_name_to_id.get(label)
        if cat_id is None:
            metrics[label] = {
                "ap": 0.0,
                "ar": 0.0,
                "num_gt": 0,
                "num_predictions": 0,
                "reason": "category_not_in_coco",
            }
            continue

        records: List[Tuple[float, int, List[float]]] = []
        for image_id in image_files.keys():
            image_path = str(Path(args.images) / image_files[int(image_id)])
            if not Path(image_path).exists():
                continue

            if args.benchmark_mode == "text":
                dets = detector.detect_with_text_queries(
                    image_path=image_path,
                    queries=queries,
                    top_k=cfg.top_k,
                    score_threshold=cfg.score_threshold,
                    nms_iou=cfg.nms_iou,
                )
            else:
                query_images = [_normalize_prompt_query_file(prompts_path, q) for q in queries]
                if not query_images:
                    raise ValueError(f"empty query image list for label: {label}")
                for q in query_images:
                    if not Path(q).exists():
                        raise ValueError(f"query image not found: {q}")
                dets = detector.detect_with_image_queries(
                    image_path=image_path,
                    query_images=query_images,
                    top_k=cfg.top_k,
                    query_image_agg=args.query_image_agg,
                )

            for d in dets:
                if d.confidence >= cfg.score_threshold:
                    records.append((float(d.confidence), int(image_id), d.box))

        records.sort(reverse=True, key=lambda x: x[0])
        tp, fp, total_gt = _eval_sorted_predictions(records, gt_by_image, int(cat_id), iou_threshold=0.5)
        res = _ap_from_pr(tp, fp, total_gt)
        res["num_predictions"] = len(records)
        res["num_query_images"] = len(queries)
        metrics[label] = res

    per_label_aps = [v.get("ap", 0.0) for v in metrics.values() if "ap" in v]
    mean_ap = float(np.mean(per_label_aps)) if per_label_aps else 0.0
    per_label_ars = [v.get("ar", 0.0) for v in metrics.values() if "ar" in v]
    mean_ar = float(np.mean(per_label_ars)) if per_label_ars else 0.0

    summary = {
        "num_labels": len(metrics),
        "labels": list(metrics.keys()),
        "mode": args.benchmark_mode,
        "mean_ap": mean_ap,
        "mean_ar": mean_ar,
        "metrics": metrics,
        "source": {
            "index": args.index,
            "prompts": args.prompts,
            "query_image_agg": args.query_image_agg,
        },
    }
    _ensure_parent(args.output)
    Path(args.output).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved benchmark: {args.output}")
    return 0


def cmd_pdf_stats(args: argparse.Namespace) -> int:
    pdf_paths = [Path(p) for p in args.pdfs]
    if args.batch:
        pdf_paths = sorted(Path(args.batch).glob("*.pdf"))

    if not pdf_paths:
        raise ValueError("no PDF files were provided")

    results = [analyze_pdf(path) for path in pdf_paths]
    summary = summarize_results(results)

    out = args.output or "outputs/pdf_stats.json"
    _ensure_parent(out)
    Path(out).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(format_summary(summary))
    print(f"saved pdf stats: {out}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(prog="yolo_raovd", description="YOLO-RAOVD MVP CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_init = sub.add_parser("init")
    p_init.add_argument("--directories", nargs="+", required=True)
    p_init.set_defaults(func=cmd_init)

    p_idx = sub.add_parser("index")
    p_idx.add_argument("--references", required=True)
    p_idx.add_argument("--out", required=True)
    p_idx.set_defaults(func=cmd_index)

    p_det = sub.add_parser("detect")
    p_det.add_argument("--image", required=True)
    p_det.add_argument("--index", required=True)
    p_det.add_argument("--query", action="append")
    p_det.add_argument("--query-image", action="append")
    p_det.add_argument("--query-mode", choices=["text", "image", "hybrid"], default="text")
    p_det.add_argument("--queries")
    p_det.add_argument("--top-k", type=int, default=20)
    p_det.add_argument("--score-threshold", type=float, default=0.15)
    p_det.add_argument("--nms-iou", type=float, default=0.5)
    p_det.add_argument("--query-image-agg", choices=["mean", "max", "mode"], default="mean")
    p_det.add_argument("--yolo-model", default=None)
    p_det.add_argument("--yolo-conf", type=float, default=0.25)
    p_det.add_argument("--yolo-iou", type=float, default=0.45)
    p_det.add_argument("--yolo-max-det", type=int, default=300)
    p_det.add_argument("--output", default="outputs/predictions.json")
    p_det.set_defaults(func=cmd_detect)

    p_draw = sub.add_parser("draw")
    p_draw.add_argument("--image", required=True)
    p_draw.add_argument("--predictions", required=True)
    p_draw.add_argument("--output", default="outputs/vis.jpg")
    p_draw.set_defaults(func=cmd_draw)

    p_bm = sub.add_parser("benchmark")
    p_bm.add_argument("--index", required=True)
    p_bm.add_argument("--annotations", required=True)
    p_bm.add_argument("--images", required=True)
    p_bm.add_argument("--prompts", required=True)
    p_bm.add_argument("--benchmark-mode", choices=["text", "image"], default="text")
    p_bm.add_argument("--top-k", type=int, default=20)
    p_bm.add_argument("--score-threshold", type=float, default=0.15)
    p_bm.add_argument("--query-image-agg", choices=["mean", "max", "mode"], default="mean")
    p_bm.add_argument("--yolo-model", default=None)
    p_bm.add_argument("--yolo-conf", type=float, default=0.25)
    p_bm.add_argument("--yolo-iou", type=float, default=0.45)
    p_bm.add_argument("--yolo-max-det", type=int, default=300)
    p_bm.add_argument("--output", default="reports/coco.json")
    p_bm.set_defaults(func=cmd_benchmark)

    p_pdf = sub.add_parser("pdf-stats")
    p_pdf.add_argument("--pdfs", nargs="+", required=True)
    p_pdf.add_argument("--batch", default=None)
    p_pdf.add_argument("--output", default="outputs/pdf_stats.json")
    p_pdf.set_defaults(func=cmd_pdf_stats)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

