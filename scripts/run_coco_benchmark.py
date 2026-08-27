from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

IOU_THRESHOLD = 0.5


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="COCO benchmark for RAOD dense matching")
    p.add_argument("--data-dir", type=str, default=None, help="Path to COCO dataset root (default: datasets/coco)")
    p.add_argument("--sample-size", type=int, default=1000, help="Number of images to sample (default: 1000)")
    p.add_argument("--full", action="store_true", help="Use full validation set instead of sampling")
    p.add_argument("--seed", type=int, default=42, help="Random seed for sampling (default: 42)")
    p.add_argument("--output", type=str, default=str(REPORTS_DIR / "coco_benchmark.json"), help="Output JSON path")
    p.add_argument("--index-dir", type=str, default=str(REPORTS_DIR / "coco_index"), help="Reference index directory")
    p.add_argument("--store-dir", type=str, default=str(REPORTS_DIR / "coco_chroma"), help="Chroma store directory")
    p.add_argument("--refs-path", type=str, default=str(REPORTS_DIR / "coco_references.json"), help="References JSON path")
    p.add_argument("--backbone-weights", type=str, default=None, help="Path/URL to DINOv3 backbone weights")
    p.add_argument("--dinotxt-weights", type=str, default=None, help="Path/URL to dino.txt text encoder + vision head weights")
    p.add_argument("--bpe-path", type=str, default=None, help="Path/URL to the BPE vocabulary for the text tokenizer")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--tile-size", type=int, default=1024)
    p.add_argument("--tiles-per-axis", type=int, default=None)
    p.add_argument("--overlap-ratio", type=float, default=0.15)
    p.add_argument("--similarity-threshold", type=float, default=0.30)
    p.add_argument("--dbscan-eps", type=float, default=1.5)
    p.add_argument("--dbscan-min-samples", type=int, default=3)
    p.add_argument("--top-k", type=int, default=5)
    return p.parse_args()


def resolve_coco_dir(args: argparse.Namespace) -> Path:
    if args.data_dir:
        return Path(args.data_dir)
    candidates = [
        ROOT / "datasets" / "coco",
        ROOT / "data" / "coco-mini",
        ROOT / "data" / "coco-mini-synth",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError("No COCO dataset found. Use --data-dir to specify the dataset root.")


def load_coco_annotations(ann_path: Path):
    if not ann_path.exists():
        raise FileNotFoundError(f"COCO annotations not found: {ann_path}")
    with open(ann_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_image_list(ann: dict, images_dir: Path):
    image_files = []
    for img in ann.get("images", []):
        fname = img["file_name"]
        candidates = [
            images_dir / fname,
            images_dir / "val2017" / fname,
            images_dir / "train2017" / fname,
            images_dir / "test2017" / fname,
        ]
        for img_path in candidates:
            if img_path.exists():
                image_files.append((img["id"], fname, img.get("width", 640), img.get("height", 640)))
                break
    return image_files


def sample_images(image_files, n, seed=42):
    random.seed(seed)
    return random.sample(image_files, min(n, len(image_files)))


def build_gt_by_image(ann: dict):
    gt = {}
    for a in ann.get("annotations", []):
        img_id = a["image_id"]
        gt.setdefault(img_id, []).append({
            "category_id": a["category_id"],
            "bbox": a["bbox"],
            "area": a.get("area", 0),
            "iscrowd": a.get("iscrowd", 0),
        })
    return gt


def build_category_map(ann: dict):
    cat_id_to_name = {}
    for c in ann.get("categories", []):
        cat_id_to_name[c["id"]] = c["name"]
    return cat_id_to_name


def collect_reference_samples(ann, gt_by_image, cat_id_to_name, image_files_map, images_dir, samples_per_class=1):
    refs = []
    seen = set()
    for img_id, anns in gt_by_image.items():
        for a in anns:
            cat_id = a["category_id"]
            if cat_id not in seen and len([r for r in refs if r["label"] == cat_id_to_name.get(cat_id, str(cat_id))]) < samples_per_class:
                fname = image_files_map.get(img_id)
                if fname:
                    candidates = [
                        images_dir / fname,
                        images_dir / "val2017" / fname,
                        images_dir / "train2017" / fname,
                        images_dir / "test2017" / fname,
                    ]
                    for img_path in candidates:
                        if img_path.exists():
                            refs.append({
                                "label": cat_id_to_name.get(cat_id, str(cat_id)),
                                "image": str(img_path),
                                "modality": "image",
                            })
                            seen.add(cat_id)
                            break
    return refs


def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def evaluate_detections(preds, gt_by_image, cat_name_to_id, iou_threshold=0.5):
    tp = []
    fp = []
    scores = []
    num_gt = 0

    gt_cache = {}
    for img_id, anns in gt_by_image.items():
        class_gts = {}
        for a in anns:
            cid = a["category_id"]
            cname = cat_name_to_id.get(cid, str(cid))
            class_gts.setdefault(cname, []).append(a["bbox"])
        gt_cache[img_id] = class_gts
        num_gt += sum(len(v) for v in class_gts.values())

    used = {}
    for pred in preds:
        img_id = pred["image_id"]
        label = pred["label"]
        box = pred["box"]
        score = pred["score"]
        scores.append(score)
        gts = gt_cache.get(img_id, {}).get(label, [])
        if not gts:
            fp.append(1)
            tp.append(0)
            continue
        used.setdefault(img_id, {label: [False] * len(gts)})
        max_iou = 0.0
        max_idx = -1
        for i, gt_box in enumerate(gts):
            if used[img_id][label][i]:
                continue
            iou_val = compute_iou(box, gt_box)
            if iou_val > max_iou:
                max_iou = iou_val
                max_idx = i
        if max_iou >= iou_threshold and max_idx >= 0:
            used[img_id][label][max_idx] = True
            tp.append(1)
            fp.append(0)
        else:
            tp.append(0)
            fp.append(1)

    tp = np.array(tp, dtype=np.float32)
    fp = np.array(fp, dtype=np.float32)
    scores = np.array(scores, dtype=np.float32)

    if len(tp) == 0:
        return 0.0, 0.0, num_gt

    idx = np.argsort(-scores)
    tp = tp[idx]
    fp = fp[idx]
    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    rec = tp_cum / (num_gt + 1e-8)
    prec = tp_cum / (tp_cum + fp_cum + 1e-8)

    if len(rec) == 0:
        return 0.0, 0.0, num_gt

    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))
    for i in range(len(mpre) - 2, -1, -1):
        if mpre[i + 1] > mpre[i]:
            mpre[i] = mpre[i + 1]
    idx_pr = np.where(mrec[1:] != mrec[:-1])[0]
    ap = float(np.sum((mrec[idx_pr + 1] - mrec[idx_pr]) * mpre[idx_pr + 1]))
    ar = float(rec[-1]) if len(rec) > 0 else 0.0
    return ap, ar, num_gt


def main() -> None:
    args = parse_args()
    coco_dir = resolve_coco_dir(args)
    print(f"Using COCO dataset at: {coco_dir}")
    images_dir = coco_dir / "images"
    ann_file = coco_dir / "annotations" / "instances_val2017_filtered.json"
    if not ann_file.exists():
        ann_file = coco_dir / "annotations" / "instances_val2017.json"
    ann = load_coco_annotations(ann_file)
    image_files = build_image_list(ann, images_dir)
    print(f"Found {len(image_files)} images with files")

    if args.full:
        sampled = image_files
        print(f"Using full dataset: {len(sampled)} images")
    else:
        sampled = sample_images(image_files, args.sample_size, seed=args.seed)
        print(f"Sampled {len(sampled)} images")

    gt_by_image = build_gt_by_image(ann)
    cat_id_to_name = build_category_map(ann)
    cat_name_to_id = {v: k for k, v in cat_id_to_name.items()}
    image_files_map = {img_id: fname for img_id, fname, _, _ in image_files}

    refs = collect_reference_samples(ann, gt_by_image, cat_id_to_name, image_files_map, images_dir, samples_per_class=1)
    print(f"Collected {len(refs)} reference samples")

    refs_path = Path(args.refs_path)
    with open(refs_path, "w", encoding="utf-8") as f:
        json.dump(refs, f, indent=2)

    from raod.dense import DenseMatcher
    from raod.encoding import DINOV3Encoder
    from raod.retrieval import build_reference_indexes

    index_dir = Path(args.index_dir)
    store_dir = Path(args.store_dir)
    encoder_kwargs = {"device": args.device}
    if args.backbone_weights:
        encoder_kwargs["backbone_weights"] = args.backbone_weights
    if args.dinotxt_weights:
        encoder_kwargs["dinotxt_weights"] = args.dinotxt_weights
    if args.bpe_path:
        encoder_kwargs["bpe_path_or_url"] = args.bpe_path
    encoder = DINOV3Encoder(**encoder_kwargs)

    index_exists = (store_dir / "chroma.sqlite3").exists()
    if not index_exists:
        print("Building reference index...")
        build_reference_indexes(str(refs_path), str(index_dir), encoder=encoder, chroma_dir=str(store_dir))
    else:
        print(f"Reusing existing reference index at {index_dir}")

    from raod.retrieval import ChromaReferenceStore

    store = ChromaReferenceStore(collection_name="references", persist_directory=str(store_dir))
    matcher = DenseMatcher(
        encoder=encoder,
        tile_size=args.tile_size,
        tiles_per_axis=args.tiles_per_axis,
        overlap_ratio=args.overlap_ratio,
        similarity_threshold=args.similarity_threshold,
        dbscan_eps=args.dbscan_eps,
        dbscan_min_samples=args.dbscan_min_samples,
        device=args.device,
    )

    print(f"Running dense matching on {len(sampled)} images...")
    all_preds = []
    times = []
    checkpoint_path = Path(args.output).with_suffix('.checkpoint.json')
    processed_ids = set()
    if checkpoint_path.exists():
        try:
            ckpt = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            all_preds = ckpt.get("preds", [])
            times = ckpt.get("times", [])
            processed_ids = {p["image_id"] for p in all_preds}
            print(f"Resumed from checkpoint: {len(processed_ids)} images already processed")
        except Exception:
            pass

    for img_id, fname, width, height in sampled:
        if img_id in processed_ids:
            continue
        candidates = [
            images_dir / fname,
            images_dir / "val2017" / fname,
            images_dir / "train2017" / fname,
            images_dir / "test2017" / fname,
        ]
        img_path = next((p for p in candidates if p.exists()), images_dir / fname)
        from PIL import Image

        image = np.array(Image.open(img_path).convert("RGB"))
        t0 = time.perf_counter()
        try:
            boxes = matcher.match_reference_store(image, store, modality="image", top_k=args.top_k)
        except Exception as e:
            print(f"Error on {fname}: {e}")
            continue
        t1 = time.perf_counter()
        times.append(t1 - t0)
        for box, score, label in boxes:
            all_preds.append({
                "image_id": img_id,
                "label": label,
                "box": box,
                "score": score,
            })
        checkpoint_path.write_text(
            json.dumps({"preds": all_preds, "times": times}, ensure_ascii=False),
            encoding="utf-8",
        )

    print(f"Total predictions: {len(all_preds)}")
    if times:
        print(f"Avg inference time: {np.mean(times):.4f}s ({1.0 / np.mean(times):.2f} FPS)")

    ap, ar, num_gt = evaluate_detections(all_preds, gt_by_image, cat_name_to_id, iou_threshold=IOU_THRESHOLD)
    print(f"AP@{IOU_THRESHOLD}: {ap:.4f}")
    print(f"AR@{IOU_THRESHOLD}: {ar:.4f}")
    print(f"Num GT: {num_gt}")

    summary = {
        "data_dir": str(coco_dir),
        "sample_size": len(sampled),
        "num_predictions": len(all_preds),
        "num_gt": num_gt,
        "ap": ap,
        "ar": ar,
        "avg_time_ms": float(np.mean(times) * 1000) if times else 0.0,
        "fps": float(1.0 / np.mean(times)) if times else 0.0,
        "iou_threshold": IOU_THRESHOLD,
        "seed": args.seed,
        "backbone": "dinov3_vitl16",
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved benchmark to {out_path}")


if __name__ == "__main__":
    main()
