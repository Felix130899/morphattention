"""Batch fish segmentation over a directory of images using SAM ViT-L.

Walks ``data/raw/`` (any common image format), derives one or more point
prompts per image, runs SAM, and writes three parallel output trees under
``data/processed/segmented/`` so you can pick whichever fits later work:

    segmented/
      masks/    <stem>.png            binary mask, 0=background 255=fish
      overlays/ <stem>.png            original image with mask drawn on top
      coco/     annotations.json      one COCO file for the whole run

Prompting strategy (``--prompt``):
  * ``center`` (default) - a single point at the image centre. Best for
    specimen photos that contain exactly one, roughly centred fish.
  * ``yolo``  - run a YOLOv8 detector and use each box centre as a point,
    giving one mask per detected fish. NOTE: stock yolov8*.pt is trained on
    COCO and has NO fish class; pass a fish-trained checkpoint via
    ``--yolo-weights`` for meaningful results.
  * ``dino``  - run Grounding DINO zero-shot with a text prompt (default
    "fish.") and feed each detected box straight into SAM as a box prompt.
    No fish-trained checkpoint needed, but it's a slow, heavy model - boxes
    are a much stronger SAM prompt than a single centre point, so quality
    should beat ``center`` immediately.

Run inside the container, e.g.:

    docker compose run --rm vit-project python scripts/segment_fish.py
    docker compose run --rm vit-project python scripts/segment_fish.py \
        --prompt yolo --yolo-weights /workspace/data/model_cache/fish_yolo.pt
    docker compose run --rm vit-project python scripts/segment_fish.py \
        --prompt dino
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from data.models.dino import load_dino, detect
from data.models.sam import load_sam, segment

# Pillow can decode all of these; we just filter the directory walk by them.
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}


def iter_images(root: Path):
    """Yield every image file under ``root`` (recursively), sorted for determinism."""
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            yield path


def center_points(image):
    """One prompt point at the image centre -> shape expected by segment()."""
    w, h = image.size
    return [[[w / 2, h / 2]]]


def yolo_points(detector, image):
    """Run the detector and turn each box centre into a SAM point prompt.

    Returns a list like [[[x, y]], [[x, y]], ...] - one object per detection,
    or an empty list if nothing was found.
    """
    results = detector(image, verbose=False)
    points = []
    for box in results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        points.append([[(x1 + x2) / 2, (y1 + y2) / 2]])
    return points


def dino_boxes(model, processor, image, text_prompt, box_threshold):
    """Run Grounding DINO and format the detected boxes for SAM's input_boxes.

    Returns a list like [[[x0, y0, x1, y1], ...]] - one image containing all
    detected boxes - or an empty list if nothing cleared the threshold.
    """
    boxes, scores = detect(model, processor, image, text_prompt=text_prompt, box_threshold=box_threshold)
    if len(boxes) == 0:
        return []
    return [boxes.tolist()]


def to_binary(mask_tensor):
    """Collapse SAM's per-point mask tensor into a single bool HxW array.

    SAM returns 3 candidate masks per point; we keep the union of the best
    one per object. ``mask_tensor`` is the masks[i] entry from segment().
    """
    # masks[i] shape: (num_points, 3, H, W). Take the first (highest-IoU is
    # index 0 after SAM's internal sort) candidate of each point, then OR them.
    arr = mask_tensor.numpy().astype(bool)
    if arr.ndim == 4:
        arr = arr[:, 0]  # (num_points, H, W)
    return np.any(arr, axis=0)  # (H, W)


def save_overlay(image, binary, out_path):
    """Write the original image with a translucent red fish mask on top."""
    base = np.array(image.convert("RGB"), dtype=np.float32)
    red = np.zeros_like(base)
    red[..., 0] = 255.0
    alpha = 0.5
    m = binary[..., None]
    blended = np.where(m, (1 - alpha) * base + alpha * red, base)
    Image.fromarray(blended.astype(np.uint8)).save(out_path)


def binary_to_coco_segmentation(binary):
    """Convert a bool mask to COCO polygon segmentation + bbox + area.

    Uses OpenCV contours; falls back to an empty polygon list if the mask is
    empty so the annotation is still well-formed.
    """
    import cv2

    mask_u8 = binary.astype(np.uint8)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    segmentation = []
    for c in contours:
        if cv2.contourArea(c) < 1:
            continue
        segmentation.append(c.flatten().astype(float).tolist())
    ys, xs = np.where(binary)
    if len(xs) == 0:
        bbox = [0.0, 0.0, 0.0, 0.0]
    else:
        x0, y0 = float(xs.min()), float(ys.min())
        bbox = [x0, y0, float(xs.max()) - x0, float(ys.max()) - y0]
    return segmentation, bbox, float(binary.sum())


def choose_directory(parent_dir: Path):
    """List subdirectories and let the user choose one interactively."""
    subdirs = sorted([d for d in parent_dir.iterdir() if d.is_dir()])
    
    if not subdirs:
        raise ValueError(f"No directories found in {parent_dir}")
    
    print(f"\nAvailable datasets in {parent_dir}:")
    for i, d in enumerate(subdirs, start=1):
        print(f"  {i}. {d.name}")
    
    while True:
        try:
            choice = input("\nSelect a dataset number: ").strip()
            idx = int(choice) - 1
            if 0 <= idx < len(subdirs):
                return subdirs[idx]
            else:
                print(f"Please enter a number between 1 and {len(subdirs)}")
        except ValueError:
            print("Invalid input. Please enter a valid number.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-dir", type=Path, default=None,
                        help="Directory of input images to process (searched recursively). "
                             "If not provided, you'll be prompted to choose from available datasets.")
    parser.add_argument("--out-dir", default="/workspace/data/processed/segmented", type=Path,
                        help="Root of the outputs; results nest under <out-dir>/<raw-dir name>/.")
    parser.add_argument("--prompt", choices=["center", "yolo", "dino"], default="center",
                        help="How to derive prompts for SAM.")
    parser.add_argument("--yolo-weights", default="/workspace/data/model_cache/yolov8n.pt",
                        help="Local YOLO checkpoint, pre-cached by scripts/download_yolo.py "
                             "(use a FISH-trained one for real results).")
    parser.add_argument("--dino-text", default="fish.",
                        help="Text prompt for Grounding DINO (--prompt dino). Lowercase, "
                             "each concept ending in a period, e.g. 'fish.'.")
    parser.add_argument("--dino-box-threshold", type=float, default=0.35,
                        help="Minimum confidence for a Grounding DINO box to be kept.")
    args = parser.parse_args()
    
    # If raw_dir not provided, prompt user to choose
    if args.raw_dir is None:
        default_raw = Path("/workspace/data/raw")
        args.raw_dir = choose_directory(default_raw)

    # Nest outputs under the selected folder's name so processing several
    # folders (trout/, salmon/, ...) never overwrites earlier runs.
    run_dir = args.out_dir / args.raw_dir.name
    masks_dir = run_dir / "masks"
    overlays_dir = run_dir / "overlays"
    coco_dir = run_dir / "coco"
    for d in (masks_dir, overlays_dir, coco_dir):
        d.mkdir(parents=True, exist_ok=True)

    model, processor = load_sam()

    detector = None
    dino_model = dino_processor = None
    if args.prompt == "yolo":
        from ultralytics import YOLO  # imported lazily so 'center' needs no install
        detector = YOLO(args.yolo_weights)
    elif args.prompt == "dino":
        dino_model, dino_processor = load_dino()

    # Minimal COCO scaffolding; one category since we only segment "fish".
    coco = {
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": "fish"}],
    }
    ann_id = 1

    images = list(iter_images(args.raw_dir))
    print(f"Found {len(images)} image(s) under {args.raw_dir}")

    for img_id, path in enumerate(images, start=1):
        image = Image.open(path).convert("RGB")
        w, h = image.size

        if args.prompt == "yolo":
            points = yolo_points(detector, image)
            if not points:
                print(f"  [skip] no detections: {path.name}")
                continue
            masks, iou_scores = segment(model, processor, image, input_points=points)
        elif args.prompt == "dino":
            boxes = dino_boxes(dino_model, dino_processor, image, args.dino_text, args.dino_box_threshold)
            if not boxes:
                print(f"  [skip] no detections: {path.name}")
                continue
            masks, iou_scores = segment(model, processor, image, input_boxes=boxes)
        else:
            points = center_points(image)
            masks, iou_scores = segment(model, processor, image, input_points=points)

        # masks is a list with one entry per image in the batch; we pass one
        # image at a time, so take masks[0].
        per_object = masks[0]
        combined = to_binary(per_object)

        stem = path.stem
        Image.fromarray((combined * 255).astype(np.uint8)).save(masks_dir / f"{stem}.png")
        save_overlay(image, combined, overlays_dir / f"{stem}.png")

        coco["images"].append({
            "id": img_id, "file_name": str(path.relative_to(args.raw_dir)),
            "width": w, "height": h,
        })
        segmentation, bbox, area = binary_to_coco_segmentation(combined)
        coco["annotations"].append({
            "id": ann_id, "image_id": img_id, "category_id": 1,
            "segmentation": segmentation, "bbox": bbox, "area": area,
            "iscrowd": 0,
        })
        ann_id += 1
        print(f"  [ok] {path.name} -> {int(combined.sum())} px")

    with open(coco_dir / "annotations.json", "w") as f:
        json.dump(coco, f)
    print(f"Done. Wrote masks/, overlays/ and coco/annotations.json under {run_dir}")


if __name__ == "__main__":
    main()
