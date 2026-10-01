"""Batch fish segmentation over a directory of images: one SAM ViT-L mask per fish.

Walks ``--raw-dir`` (searched recursively, any common image format), derives
prompts per image, runs SAM, and writes one mask per detected fish instance
under ``<out-dir>/<raw-dir name>/``:

    run_config.json          every setting + the git commit of each session
    annotations.jsonl        one line per image, appended as soon as it's done
    coco/annotations.json    merged COCO file, one annotation per instance -
                             the single source of truth (CVAT/X-AnyLabeling
                             import it; it converts straight to YOLO-seg)
    skipped.txt              every image without a usable instance + reason
    masks/    <stem>.png     union of all instances at original resolution
                             (0=background 255=fish), for the attention metric
    overlays/ <stem>.jpg     QA view: each instance in its own colour with its
                             index and flags; removed duplicates as grey boxes

Design and the reasoning behind every rule below:
vault/thesis-log/decisions/2026-09-24-per-instance-mask-output.md

Prompting strategy (``--prompt``):
  * ``center`` (default) - a single point at the image centre. Only fits
    photos with exactly one, roughly centred fish.
  * ``yolo``  - YOLO detector boxes as SAM box prompts. NOTE: stock
    yolov8*.pt is trained on COCO and has NO fish class; pass a fish-trained
    checkpoint via ``--yolo-weights`` for meaningful results.
  * ``dino``  - Grounding DINO zero-shot boxes for a text prompt (default
    "fish.") as SAM box prompts. Slow and heavy, but no training needed.

Per image: detect -> SAM (3 candidates per prompt; threshold the mask logits
at ``--mask-threshold``; keep one candidate per ``--candidate``) -> drop empty
masks -> drop duplicate detections -> give every contested pixel to exactly
one instance -> optionally grow each instance by a size-relative margin
(``--margin-frac``) -> flag tiny/giant instances. Every threshold is a CLI
flag and a starting guess, to be tuned per domain (photo vs. X-ray) on real
overlays - none of the defaults comes from the data.

Fin recovery (SAM tends to cut off fins; see
vault/thesis-log/decisions/2026-10-01-mask-policy.md). All three default to
the original behaviour, so a run with defaults reproduces the old masks:
  * ``--candidate`` ``score`` (default: highest predicted IoU) or
    ``largest`` (largest candidate area after thresholding).
  * ``--mask-threshold`` (default 0.0 = SAM's own cut) applied to the mask
    logits at working resolution; lower (e.g. -2) keeps translucent fins.
  * ``--margin-frac`` (default 0 = off) grows every final instance by
    r = max(1, round(margin_frac * sqrt(area))) px, never into another fish.

Per-instance QA features (COCO annotation + annotations.jsonl), meant as
inputs for the mask-QA model: ``sam_candidate_areas``,
``area_ratio_largest_to_chosen``, ``edge_uncertain_frac`` (fin-loss signal,
see ``edge_uncertain_frac()``), ``touches_border``, ``n_components``,
``margin_px``. Areas and ``margin_px`` are in working-resolution pixels.

``--image-list`` restricts a run to the listed files (e.g. a dev set), one
path per line relative to ``--raw-dir`` as in COCO ``file_name``.

Crash-safe: re-run with ``--resume`` to skip every image already in
annotations.jsonl. A resume with different settings (incl. a different
image list) is refused, so one run never mixes masks made with different
thresholds.

Run inside the container, e.g.:

    docker compose run --rm vit-project python scripts/segment_fish.py \
        --raw-dir /workspace/data/raw/NHM_datensatz/full_body --prompt dino
    docker compose run --rm vit-project python scripts/segment_fish.py \
        --raw-dir /workspace/data/raw/NHM_datensatz/full_body --prompt dino --resume
    docker compose run --rm vit-project python scripts/segment_fish.py \
        --raw-dir /workspace/data/raw/NHM_datensatz/full_body --prompt dino \
        --image-list dev_set.txt --candidate largest --mask-threshold -2 --margin-frac 0.03 \
        --out-dir /workspace/data/processed/segmented/fin_largest_t-2_m0.03
"""

import argparse
import datetime
import hashlib
import json
import math
import os
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from data.models.dino import CHECKPOINT as DINO_CHECKPOINT
from data.models.dino import detect, load_dino
from data.models.sam import CHECKPOINT as SAM_CHECKPOINT
from data.models.sam import load_sam

# NHM scans reach ~150 MP, above Pillow's decompression-bomb guard (89 MP).
# These are trusted local museum files, so lift the limit.
Image.MAX_IMAGE_PIXELS = None

# Pillow can decode all of these; we just filter the directory walk by them.
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}

REPO_ROOT = Path(__file__).resolve().parents[1]

# Distinct overlay colours, cycled per instance.
PALETTE = [
    (230, 25, 75), (60, 180, 75), (0, 130, 200), (245, 130, 48), (145, 30, 180),
    (70, 240, 240), (240, 50, 230), (210, 245, 60), (250, 190, 212), (0, 128, 128),
]

# edge_uncertain_frac(): ring width as a fraction of sqrt(mask area), and how
# far below the mask threshold a logit still counts as "almost fish".
EDGE_RING_FRAC = 0.02
EDGE_BAND_LOGITS = 3.0

# Settings added after the first full run, with the value that reproduces it.
# A run_config.json written before they existed lacks them; resuming such a
# run is allowed only with exactly these values.
LEGACY_DEFAULTS = {"candidate": "score", "mask_threshold": 0.0, "margin_frac": 0.0,
                   "image_list": None, "image_list_sha256": None}


def iter_images(root: Path):
    """Yield every image file under ``root`` (recursively), sorted for determinism."""
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            yield path


def parse_image_list(text):
    """File names from an ``--image-list`` file, in file order, without repeats.

    One name per line, relative to ``--raw-dir`` with forward slashes (the
    COCO ``file_name`` form, e.g. ``Abramis_brama_NMW12345_left_WEB.jpg``).
    Surrounding whitespace is stripped; blank lines and lines starting with
    ``#`` are ignored (no trailing comments - '#' may occur in a name).
    """
    names = []
    for line in text.splitlines():
        name = line.strip()
        if name and not name.startswith("#") and name not in names:
            names.append(name)
    return names


def select_images(rel_names, wanted):
    """Restrict ``{path: rel_name}`` to the names in ``wanted``.

    Fails loudly (SystemExit listing every offender) if a wanted name is not
    an image under ``--raw-dir``, so a typo can't silently shrink a dev set.
    """
    available = set(rel_names.values())
    missing = [n for n in wanted if n not in available]
    if missing:
        raise SystemExit(f"{len(missing)} name(s) in --image-list are not images under --raw-dir:\n  "
                         + "\n  ".join(missing))
    wanted = set(wanted)
    return {p: r for p, r in rel_names.items() if r in wanted}


def load_working_image(path, max_side):
    """Open an image as RGB, downscaled so its longest side is <= ``max_side``.

    SAM sees every image at 1024 px on the long side and upsamples a 256x256
    mask, so working above ~2048 px adds no mask detail - only memory (a
    150 MP scan would need GBs per instance). Results are mapped back to the
    original size afterwards. Returns (image, (orig_width, orig_height)).
    """
    image = Image.open(path)
    orig_size = image.size
    if max_side:
        # JPEG-only shortcut: decode straight at 1/2, 1/4 or 1/8 scale
        # (never below the requested size), instead of the full 150 MP.
        image.draft("RGB", (max_side, max_side))
    image = image.convert("RGB")
    w, h = image.size
    if max_side and max(w, h) > max_side:
        scale = max_side / max(w, h)
        image = image.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS)
    return image, orig_size


def best_candidates(candidate_masks, iou_scores):
    """Keep SAM's highest-predicted-IoU candidate mask per prompt.

    candidate_masks: (N, 3, H, W) bool; iou_scores: (N, 3). SAM returns its
    3 candidates in fixed token order, NOT sorted by quality (transformers'
    mask decoder just slices tokens 1-3), so index 0 is not the best one.
    Returns (masks (N, H, W) bool, chosen candidate index (N,)).
    """
    chosen = iou_scores.argmax(axis=1)
    return candidate_masks[np.arange(len(chosen)), chosen], chosen


def sam_logits(model, processor, image, input_points=None, input_boxes=None):
    """Like ``data.models.sam.segment`` but returns SAM's raw mask logits.

    Same processor/model call; the only difference is ``binarize=False`` in
    ``post_process_masks``, which then skips its final ``masks > 0.0`` and
    returns the logits upscaled (bilinear) to the working image size. So
    ``logits > 0.0`` is bit-identical to ``segment()``'s masks.
    Returns (logits (N, 3, H, W) float32, iou_scores (N, 3)) as numpy.
    """
    import torch

    device = next(model.parameters()).device
    inputs = processor(
        image, input_points=input_points, input_boxes=input_boxes, return_tensors="pt"
    ).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = processor.image_processor.post_process_masks(
        outputs.pred_masks.cpu(),
        inputs["original_sizes"].cpu(),
        inputs["reshaped_input_sizes"].cpu(),
        binarize=False,
    )
    return logits[0].numpy(), outputs.iou_scores.cpu()[0].numpy()


def choose_candidates(logits, iou_scores, threshold=0.0, rule="score"):
    """Threshold SAM's 3 candidates per prompt and keep one of them.

    logits: (N, 3, H, W) float; iou_scores: (N, 3). A pixel is in a
    candidate when its logit is > ``threshold`` (strict, like HF's own
    binarisation). ``rule`` "score" keeps the highest predicted IoU (=
    ``best_candidates``); "largest" keeps the candidate with the most pixels
    (ties -> lowest candidate index).
    Returns (masks (N, H, W) bool, chosen index (N,), candidate areas (N, 3)).
    """
    candidates = logits > threshold
    areas = candidates.reshape(*candidates.shape[:2], -1).sum(axis=2)
    if rule == "score":
        chosen = iou_scores.argmax(axis=1)
    elif rule == "largest":
        chosen = areas.argmax(axis=1)
    else:
        raise ValueError(f"unknown candidate rule {rule!r}")
    return candidates[np.arange(len(chosen)), chosen], chosen, areas


def mask_containment(inner, outer):
    """Fraction of ``inner``'s pixels that also lie inside ``outer``."""
    area = inner.sum()
    return float(np.logical_and(inner, outer).sum() / area) if area else 0.0


def box_containment(inner, outer):
    """Fraction of box ``inner``'s area that lies inside box ``outer`` (xyxy)."""
    w = min(inner[2], outer[2]) - max(inner[0], outer[0])
    h = min(inner[3], outer[3]) - max(inner[1], outer[1])
    area = (inner[2] - inner[0]) * (inner[3] - inner[1])
    return float(max(0, w) * max(0, h) / area) if area > 0 else 0.0


def remove_duplicates(masks, boxes, scores, giant, threshold):
    """Drop detections that are a part or a copy of another detection of the same fish.

    Grounding DINO often boxes one fish several times (whole body + head).
    Two instances count as duplicates when the smaller mask lies >=
    ``threshold`` inside the larger one AND the smaller one's box lies >=
    ``threshold`` inside the other's box. The box check spares a touching
    neighbour that a bleeding SAM mask happens to cover. Of a duplicate pair
    the higher detector score wins, so a DINO box around a whole group of
    fish (which usually scores lower than each single fish) can't swallow
    them. Giant instances are flag-only: never dropped, never cause a drop.

    masks: (N, H, W) bool; boxes: (N, 4) xyxy or None; scores: (N,) or None;
    giant: (N,) bool. Returns (kept indices ascending, [(dropped, kept_by)]).
    """
    n = len(masks)
    areas = masks.reshape(n, -1).sum(axis=1)
    score = scores if scores is not None else np.zeros(n)
    order = sorted(range(n), key=lambda i: (-score[i], -areas[i], i))
    kept, dropped = [], []
    for i in order:
        duplicate_of = None
        if not giant[i]:
            for j in kept:
                if giant[j]:
                    continue
                small, large = (i, j) if areas[i] <= areas[j] else (j, i)
                if mask_containment(masks[small], masks[large]) < threshold:
                    continue
                if boxes is not None and box_containment(boxes[small], boxes[large]) < threshold:
                    continue
                duplicate_of = j
                break
        if duplicate_of is None:
            kept.append(i)
        else:
            dropped.append((i, duplicate_of))
    return sorted(kept), dropped


def resolve_overlaps(masks, boxes, scores):
    """Give every pixel claimed by several instances to exactly one of them.

    Fish in these photos don't physically overlap, so a shared pixel is a SAM
    mask bleeding into a touching neighbour. The pixel goes to the instance
    whose detector box contains it; if several (or none) do, the higher
    detector score wins. Masks are deliberately NOT clipped to their box:
    boxes are often tight and would cut off fins.

    Returns (masks, number of contested pixels).
    """
    if len(masks) < 2:
        return masks, 0
    ys, xs = np.nonzero(masks.sum(axis=0) > 1)
    if len(ys) == 0:
        return masks, 0
    claims = masks[:, ys, xs]  # (N, K)
    priority = np.zeros(claims.shape)
    if scores is not None:
        priority += np.asarray(scores)[:, None]  # detector scores are in [0, 1]
    if boxes is not None:
        cx, cy = xs + 0.5, ys + 0.5  # pixel centres
        in_box = ((cx >= boxes[:, 0:1]) & (cx <= boxes[:, 2:3])
                  & (cy >= boxes[:, 1:2]) & (cy <= boxes[:, 3:4]))
        priority += 2 * in_box  # box membership outranks any score difference
    winner = np.where(claims, priority, -1).argmax(axis=0)
    out = masks.copy()
    out[:, ys, xs] = False
    out[winner, ys, xs] = True
    return out, len(ys)


def _crop(mask, pad):
    """Slices of ``mask``'s bounding box grown by ``pad`` px, clipped to the image."""
    ys, xs = np.nonzero(mask)
    h, w = mask.shape
    return (slice(max(0, ys.min() - pad), min(h, ys.max() + pad + 1)),
            slice(max(0, xs.min() - pad), min(w, xs.max() + pad + 1)))


def grow_margins(masks, margin_frac):
    """Grow every instance by a size-relative margin without touching other fish.

    Instance k gets r_k = max(1, round(margin_frac * sqrt(area_k))) px: every
    background pixel whose Euclidean distance (pixel centre to nearest mask
    pixel centre) is <= r_k. Pixels of any instance's input mask are never
    given away. A background pixel reached by several margins goes to the
    nearest instance; at an exact distance tie the lower instance index
    wins. So every pixel still belongs to at most one instance.
    margin_frac <= 0 returns the masks unchanged (copy) with r = 0.

    masks: (N, H, W) bool, non-empty, disjoint. Returns (masks, r (N,) int).
    """
    n = len(masks)
    if margin_frac <= 0 or n == 0:
        return masks.copy(), np.zeros(n, dtype=int)
    from scipy.ndimage import distance_transform_edt

    occupied = masks.any(axis=0)
    best = np.full(occupied.shape, np.inf)
    owner = np.full(occupied.shape, -1)
    radii = np.zeros(n, dtype=int)
    for k in range(n):
        radii[k] = max(1, round(margin_frac * math.sqrt(masks[k].sum())))
        # The crop holds the whole mask, so in-crop distances are exact.
        sl = _crop(masks[k], radii[k])
        dist = distance_transform_edt(~masks[k][sl])
        take = (dist <= radii[k]) & ~occupied[sl] & (dist < best[sl])  # strict: ties keep lower k
        best[sl][take] = dist[take]
        owner[sl][take] = k
    out = masks.copy()
    for k in range(n):
        out[k] |= owner == k
    return out, radii


def edge_uncertain_frac(mask, logits, threshold, blocked=None):
    """Fin-loss signal: share of a thin ring just outside ``mask`` that SAM almost kept.

    Exact definition (all at working resolution):
      w    = max(1, round(EDGE_RING_FRAC * sqrt(mask area)))   (2 % -> w px)
      ring = pixels NOT in ``mask`` (and not in ``blocked``, i.e. other
             instances) whose Euclidean distance to the nearest mask pixel
             is <= w
      band = ring pixels with threshold - EDGE_BAND_LOGITS <= logit <= threshold
      edge_uncertain_frac = |band| / |ring|        (0.0 if the ring is empty)
    ``logits`` are the chosen SAM candidate's logits; a pixel is in the
    candidate iff logit > threshold, so ``band`` is everything just outside
    the cut that was within 3 logits of being kept. Translucent fins sit
    there at slightly negative logits, so a high value suggests a cut fin;
    a clean edge against background has logits far below the threshold.
    ``mask`` is the final instance before any margin.
    """
    from scipy.ndimage import distance_transform_edt

    w = max(1, round(EDGE_RING_FRAC * math.sqrt(mask.sum())))
    sl = _crop(mask, w)
    ring = distance_transform_edt(~mask[sl]) <= w
    ring &= ~mask[sl]
    if blocked is not None:
        ring &= ~blocked[sl]
    n_ring = int(ring.sum())
    if n_ring == 0:
        return 0.0
    lg = logits[sl]
    band = ring & (lg >= threshold - EDGE_BAND_LOGITS) & (lg <= threshold)
    return float(band.sum() / n_ring)


def touches_border(mask):
    """True if any mask pixel lies in the first/last row or column."""
    return bool(mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any())


def n_components(mask):
    """Number of 8-connected components (8 like cv2.findContours' outlines)."""
    import cv2

    return int(cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)[0] - 1)


def size_flags(area_frac, tiny_frac, giant_frac):
    """Flag (never drop) instances whose area is suspiciously small or large."""
    flags = []
    if area_frac < tiny_frac:
        flags.append("tiny")
    if area_frac > giant_frac:
        flags.append("giant")
    return flags


def mask_to_coco(binary, sx, sy):
    """Convert a working-resolution bool mask to COCO polygons + bbox + area.

    (sx, sy) scale working pixels back to original-image pixels, so the COCO
    geometry lines up with the original file that CVAT/YOLO will open.
    """
    import cv2

    contours, _ = cv2.findContours(binary.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    segmentation = []
    for c in contours:
        if cv2.contourArea(c) < 1:
            continue
        pts = c.reshape(-1, 2) * (sx, sy)
        segmentation.append([round(float(v), 1) for v in pts.flatten()])
    ys, xs = np.nonzero(binary)
    x0, y0 = xs.min(), ys.min()
    bbox = [round(float(v), 1) for v in (x0 * sx, y0 * sy, (xs.max() + 1 - x0) * sx, (ys.max() + 1 - y0) * sy)]
    return segmentation, bbox, round(float(binary.sum() * sx * sy), 1)


def scale_box(box, sx, sy):
    return [round(float(v), 1) for v in (box[0] * sx, box[1] * sy, box[2] * sx, box[3] * sy)]


def _font(size):
    try:
        return ImageFont.load_default(size=size)
    except (TypeError, OSError):  # Pillow < 10.1 or no FreeType
        return ImageFont.load_default()


def save_overlay(image, masks, labels, dropped_boxes, out_path):
    """Write the QA overlay: each instance in its own colour, labelled
    "<index> <flags>" at its top-left corner; removed duplicates as grey boxes."""
    base = np.asarray(image, dtype=np.float32)
    blended = base.copy()
    for k, m in enumerate(masks):
        blended[m] = 0.5 * base[m] + 0.5 * np.array(PALETTE[k % len(PALETTE)], np.float32)
    out = Image.fromarray(blended.astype(np.uint8))
    draw = ImageDraw.Draw(out)
    size = max(14, max(out.size) // 50)
    font = _font(size)
    line = max(1, size // 8)
    for box in dropped_boxes:
        draw.rectangle([float(v) for v in box], outline=(160, 160, 160), width=line)
        draw.text((float(box[0]), float(box[1])), "dup", fill=(160, 160, 160), font=font,
                  stroke_width=line, stroke_fill=(0, 0, 0))
    for k, (m, label) in enumerate(zip(masks, labels)):
        ys, xs = np.nonzero(m)
        draw.text((int(xs.min()), max(0, int(ys.min()) - size)), label, fill=PALETTE[k % len(PALETTE)],
                  font=font, stroke_width=line, stroke_fill=(0, 0, 0))
    out.save(out_path, quality=90)


def segment_image(image, prompt, sam, detector, args):
    """Run detection + SAM + the per-instance rules on one working-size image.

    Returns (record fields, final masks, labels, dropped boxes) or a skip
    reason string when no usable instance remains.
    """
    model, processor = sam
    w, h = image.size
    if prompt == "center":
        boxes = scores = None
        logits, iou = sam_logits(model, processor, image, input_points=[[[w / 2, h / 2]]])
    else:
        boxes, scores = detector(image)
        if len(boxes) == 0:
            return "no_detection"
        logits, iou = sam_logits(model, processor, image, input_boxes=[boxes.tolist()])
    masks, chosen, cand_areas = choose_candidates(logits, iou, args.mask_threshold, args.candidate)
    # Only the chosen candidate's logits are needed later (edge feature).
    chosen_logits = logits[np.arange(len(chosen)), chosen]
    del logits
    n_detections = len(masks)

    def keep(idx):
        nonlocal masks, boxes, scores, iou, chosen, cand_areas, chosen_logits
        masks, iou, chosen = masks[idx], iou[idx], chosen[idx]
        cand_areas, chosen_logits = cand_areas[idx], chosen_logits[idx]
        boxes = boxes[idx] if boxes is not None else None
        scores = scores[idx] if scores is not None else None

    img_area = float(w * h)
    keep(np.nonzero(masks.reshape(len(masks), -1).sum(axis=1) > 0)[0])
    empty_dropped = n_detections - len(masks)

    giant = masks.reshape(len(masks), -1).sum(axis=1) / img_area > args.giant_area_frac
    kept, dropped = remove_duplicates(masks, boxes, scores, giant, args.dedup_containment)
    dropped_info = [(boxes[i] if boxes is not None else None,
                     float(scores[i]) if scores is not None else None, j) for i, j in dropped]
    keep(np.array(kept, dtype=int))

    masks, contested = resolve_overlaps(masks, boxes, scores)
    # An instance can lose every pixel to its neighbours; treat it as empty.
    nonempty = np.nonzero(masks.reshape(len(masks), -1).sum(axis=1) > 0)[0]
    empty_dropped += len(masks) - len(nonempty)
    new_index = {old: new for new, old in enumerate(np.array(kept)[nonempty])}
    keep(nonempty)
    if len(masks) == 0:
        return "all_masks_empty"

    # Edge feature on the pre-margin masks; other fish are not "missing fin".
    occupied = masks.any(axis=0)
    edge_uncertain = [edge_uncertain_frac(m, lg, args.mask_threshold, blocked=occupied & ~m)
                      for m, lg in zip(masks, chosen_logits)]
    masks, margin_px = grow_margins(masks, args.margin_frac)
    return {
        "masks": masks, "boxes": boxes, "scores": scores, "iou": iou, "chosen": chosen,
        "cand_areas": cand_areas, "edge_uncertain": edge_uncertain, "margin_px": margin_px,
        "n_detections": n_detections, "empty_dropped": empty_dropped,
        "dropped": [(b, s, new_index.get(j)) for b, s, j in dropped_info],
        "contested_frac": contested / img_area,
    }


def build_record(rel_name, orig_size, work_size, result, args):
    """Turn one image's segmentation result into its annotations.jsonl line."""
    W, H = orig_size
    sx, sy = W / work_size[0], H / work_size[1]
    img_area = float(work_size[0] * work_size[1])
    instances, labels = [], []
    for k, m in enumerate(result["masks"]):
        segmentation, bbox, area = mask_to_coco(m, sx, sy)
        flags = size_flags(m.sum() / img_area, args.tiny_area_frac, args.giant_area_frac)
        instances.append({
            "segmentation": segmentation, "bbox": bbox, "area": area,
            "det_box": scale_box(result["boxes"][k], sx, sy) if result["boxes"] is not None else None,
            "det_score": round(float(result["scores"][k]), 4) if result["scores"] is not None else None,
            "sam_iou_scores": [round(float(v), 4) for v in result["iou"][k]],
            "sam_mask_index": int(result["chosen"][k]),
            "flags": flags,
            # QA features (areas/margin in working-resolution px, see module docstring).
            # Candidate areas: raw SAM candidates; edge: pre-margin mask;
            # border/components: the final mask as written (incl. margin).
            "sam_candidate_areas": [int(a) for a in result["cand_areas"][k]],
            "area_ratio_largest_to_chosen": round(float(result["cand_areas"][k].max()
                                                        / result["cand_areas"][k][result["chosen"][k]]), 4),
            "edge_uncertain_frac": round(result["edge_uncertain"][k], 4),
            "touches_border": touches_border(m),
            "n_components": n_components(m),
            "margin_px": int(result["margin_px"][k]),
        })
        labels.append(" ".join([str(k)] + flags))
    image_flags = sorted({f for inst in instances for f in inst["flags"]})
    if len(instances) > 1:
        image_flags.append("multi_instance")
    if result["dropped"]:
        image_flags.append("duplicates_removed")
    if result["contested_frac"] > 0:
        image_flags.append("overlap_resolved")
    if result["empty_dropped"]:
        image_flags.append("empty_masks_dropped")
    record = {
        "file_name": rel_name, "status": "ok", "width": W, "height": H,
        "qa": {
            "flags": image_flags,
            "n_detections": result["n_detections"],
            "empty_dropped": result["empty_dropped"],
            "duplicates_removed": [
                {"det_box": scale_box(b, sx, sy) if b is not None else None,
                 "det_score": round(s, 4) if s is not None else None,
                 "duplicate_of": j}
                for b, s, j in result["dropped"]
            ],
            "contested_frac": round(result["contested_frac"], 6),
        },
        "instances": instances,
    }
    return record, labels


def git_state():
    """Commit + dirty flag of the code producing this run (for traceability)."""
    try:
        run = lambda *a: subprocess.run(["git", "-C", str(REPO_ROOT), *a], capture_output=True,
                                        text=True, check=True).stdout.strip()
        return {"git_commit": run("rev-parse", "HEAD"),
                "git_dirty": bool(run("status", "--porcelain", "--untracked-files=no"))}
    except (OSError, subprocess.CalledProcessError):
        return {"git_commit": "unknown", "git_dirty": None}


def run_settings(args, image_names=None):
    """Everything that changes the masks - must be identical to resume a run.

    ``image_names``: the parsed --image-list (None = all images). Its hash
    is stored, so resuming with an edited list is refused too (comments,
    blank lines and line order don't count).
    """
    settings = {
        "raw_dir": str(args.raw_dir), "prompt": args.prompt, "sam_checkpoint": SAM_CHECKPOINT,
        "max_side": args.max_side, "dedup_containment": args.dedup_containment,
        "tiny_area_frac": args.tiny_area_frac, "giant_area_frac": args.giant_area_frac,
        "candidate": args.candidate, "mask_threshold": args.mask_threshold,
        "margin_frac": args.margin_frac,
        "image_list": str(Path(args.image_list).resolve()) if args.image_list is not None else None,
        "image_list_sha256": (hashlib.sha256("\n".join(sorted(image_names)).encode()).hexdigest()
                              if image_names is not None else None),
    }
    if args.prompt == "dino":
        settings.update(dino_checkpoint=DINO_CHECKPOINT, dino_text=args.dino_text,
                        dino_box_threshold=args.dino_box_threshold)
    elif args.prompt == "yolo":
        settings.update(yolo_weights=str(args.yolo_weights))
    return json.loads(json.dumps(settings))  # normalise types for comparison


def prepare_run_config(run_dir, settings, resume):
    """Write run_config.json, or on --resume check the settings still match.

    A setting missing from an older run_config.json counts as its
    LEGACY_DEFAULTS value (what that run implicitly used) and is written
    back explicitly.
    """
    cfg_path = run_dir / "run_config.json"
    session = {"started": datetime.datetime.now().isoformat(timespec="seconds"), **git_state()}
    if resume and cfg_path.exists():
        cfg = json.loads(cfg_path.read_text())
        cfg["settings"] = {**{k: v for k, v in LEGACY_DEFAULTS.items() if k in settings}, **cfg["settings"]}
        changed = {k: (cfg["settings"].get(k), v) for k, v in settings.items() if cfg["settings"].get(k) != v}
        changed.update({k: (v, None) for k, v in cfg["settings"].items() if k not in settings})
        if changed:
            raise SystemExit(f"Refusing to resume {run_dir}: settings differ (old, new): {changed}. "
                             "Use a new --out-dir or the old settings.")
        if cfg["sessions"][-1].get("git_commit") != session["git_commit"]:
            print(f"  [warn] code changed since the last session ({cfg['sessions'][-1].get('git_commit')} "
                  f"-> {session['git_commit']}); recorded in run_config.json")
        cfg["sessions"].append(session)
    else:
        cfg = {"settings": settings, "sessions": [session]}
    cfg_path.write_text(json.dumps(cfg, indent=2))


def load_done(jsonl_path):
    """Read finished records keyed by file_name; drop a truncated last line
    left by a crash and rewrite the file so appending continues cleanly."""
    if not jsonl_path.exists():
        return {}
    lines = [ln for ln in jsonl_path.read_text().splitlines() if ln.strip()]
    records = {}
    for i, line in enumerate(lines):
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            if i == len(lines) - 1:
                print("  [warn] dropped a truncated last line in annotations.jsonl (crash mid-write)")
                break
            raise SystemExit(f"Corrupt line {i + 1} in {jsonl_path} - not a crash artefact, check by hand.")
        records[rec["file_name"]] = rec
    jsonl_path.write_text("".join(json.dumps(r) + "\n" for r in records.values()))
    return records


def write_outputs(records, run_dir):
    """Merge annotations.jsonl records into coco/annotations.json + skipped.txt."""
    coco = {"images": [], "annotations": [], "categories": [{"id": 1, "name": "fish"}]}
    ok = sorted((r for r in records if r["status"] == "ok"), key=lambda r: r["file_name"])
    ann_id = 1
    for img_id, rec in enumerate(ok, start=1):
        coco["images"].append({"id": img_id, "file_name": rec["file_name"], "width": rec["width"],
                               "height": rec["height"], "qa": rec["qa"]})
        for k, inst in enumerate(rec["instances"]):
            coco["annotations"].append({"id": ann_id, "image_id": img_id, "category_id": 1, "iscrowd": 0,
                                        "instance_index": k, **inst})
            ann_id += 1
    (run_dir / "coco" / "annotations.json").write_text(json.dumps(coco))
    skipped = sorted((r for r in records if r["status"] == "skipped"), key=lambda r: r["file_name"])
    (run_dir / "skipped.txt").write_text("".join(f"{r['file_name']}\t{r['reason']}\n" for r in skipped))
    return coco, skipped


def print_summary(coco, skipped):
    from collections import Counter

    image_flags = Counter(f for img in coco["images"] for f in img["qa"]["flags"])
    reasons = Counter(r["reason"].split(":")[0] for r in skipped)
    print(f"Segmented {len(coco['images'])} image(s) -> {len(coco['annotations'])} instance(s).")
    print(f"Skipped {len(skipped)} image(s): {dict(reasons) or 'none'} (see skipped.txt)")
    print(f"Images per QA flag: {dict(image_flags) or 'none'}")


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
                        help="Minimum confidence for a Grounding DINO box to be kept. Tunable guess.")
    parser.add_argument("--dedup-containment", type=float, default=0.8,
                        help="Duplicate if the smaller mask AND its box lie at least this fraction "
                             "inside the other. Tunable guess.")
    parser.add_argument("--tiny-area-frac", type=float, default=0.001,
                        help="Flag instances smaller than this fraction of the image. Flag only. Tunable guess.")
    parser.add_argument("--giant-area-frac", type=float, default=0.9,
                        help="Flag instances larger than this fraction of the image. Flag only. Tunable guess.")
    parser.add_argument("--max-side", type=int, default=2048,
                        help="Process at most this many px on the long side (SAM works at 1024 "
                             "anyway); outputs map back to original resolution. 0 = no limit.")
    parser.add_argument("--candidate", choices=["score", "largest"], default="score",
                        help="Which of SAM's 3 candidate masks to keep per prompt: highest predicted "
                             "IoU (score, original behaviour) or most pixels after --mask-threshold "
                             "(largest; tends to keep fins).")
    parser.add_argument("--mask-threshold", type=float, default=0.0,
                        help="Logit above which a pixel is in a SAM mask (0.0 = SAM's own cut). "
                             "Lower, e.g. -2, recovers translucent fins. Tunable guess.")
    parser.add_argument("--margin-frac", type=float, default=0.0,
                        help="Grow each final instance by max(1, round(f*sqrt(area))) px, never into "
                             "another instance. 0 = off. Tunable guess.")
    parser.add_argument("--image-list", type=Path, default=None,
                        help="Text file with one file name per line, relative to --raw-dir as in COCO "
                             "file_name ('#' comment lines and blank lines ignored). Only these images "
                             "are processed; names not found under --raw-dir are an error.")
    parser.add_argument("--resume", action="store_true",
                        help="Continue an interrupted run, skipping images already in annotations.jsonl.")
    args = parser.parse_args()

    # If raw_dir not provided, prompt user to choose
    if args.raw_dir is None:
        default_raw = Path("/workspace/data/raw")
        args.raw_dir = choose_directory(default_raw)

    images = list(iter_images(args.raw_dir))
    # Per-image outputs are named by stem, so two files sharing a stem in
    # different subfolders would overwrite each other's mask/overlay.
    stems = {}
    for p in images:
        stems.setdefault(p.stem, []).append(p)
    clashes = {s: ps for s, ps in stems.items() if len(ps) > 1}
    if clashes:
        raise SystemExit(f"{len(clashes)} filename stem(s) occur more than once, e.g. {next(iter(clashes.values()))}")

    rel_names = {p: p.relative_to(args.raw_dir).as_posix() for p in images}
    image_names = None
    if args.image_list is not None:
        image_names = parse_image_list(args.image_list.read_text())
        rel_names = select_images(rel_names, image_names)
        images = list(rel_names)
        print(f"--image-list: {len(images)} image(s) selected from {args.image_list}")

    # Nest outputs under the selected folder's name so processing several
    # folders (full_body/, Röntgen/, ...) never overwrites earlier runs.
    run_dir = args.out_dir / args.raw_dir.name
    masks_dir = run_dir / "masks"
    overlays_dir = run_dir / "overlays"
    for d in (masks_dir, overlays_dir, run_dir / "coco"):
        d.mkdir(parents=True, exist_ok=True)

    jsonl_path = run_dir / "annotations.jsonl"
    if jsonl_path.exists() and not args.resume:
        raise SystemExit(f"{jsonl_path} already exists - pass --resume to continue that run, "
                         f"or delete {run_dir} to start over.")
    prepare_run_config(run_dir, run_settings(args, image_names), args.resume)
    done = load_done(jsonl_path) if args.resume else {}

    todo = [p for p in images if rel_names[p] not in done]
    print(f"Found {len(images)} image(s) under {args.raw_dir}; {len(images) - len(todo)} already done, "
          f"{len(todo)} to process.")

    sam = load_sam() if todo else None
    detector = None
    if todo and args.prompt == "yolo":
        from ultralytics import YOLO  # imported lazily so 'center' needs no install

        yolo = YOLO(args.yolo_weights)

        def detector(image):
            r = yolo(image, verbose=False)[0]
            return r.boxes.xyxy.cpu().numpy().reshape(-1, 4), r.boxes.conf.cpu().numpy()
    elif todo and args.prompt == "dino":
        dino_model, dino_processor = load_dino()

        def detector(image):
            boxes, scores = detect(dino_model, dino_processor, image, text_prompt=args.dino_text,
                                   box_threshold=args.dino_box_threshold)
            return boxes.numpy().reshape(-1, 4), scores.numpy()

    with open(jsonl_path, "a") as jsonl:
        for n, path in enumerate(todo, start=1):
            rel = rel_names[path]
            try:
                image, orig_size = load_working_image(path, args.max_side)
            except Exception as e:  # corrupt/truncated files: log and move on
                result = f"unreadable: {type(e).__name__}: {e}"
            else:
                result = segment_image(image, args.prompt, sam, detector, args)

            if isinstance(result, str):
                record = {"file_name": rel, "status": "skipped", "reason": result}
                print(f"  [{n}/{len(todo)}] [skip] {result.split(':')[0]}: {rel}")
            else:
                record, labels = build_record(rel, orig_size, image.size, result, args)
                union = np.any(result["masks"], axis=0)
                Image.fromarray(union.astype(np.uint8) * 255).resize(orig_size, Image.NEAREST).save(
                    masks_dir / f"{path.stem}.png")
                dropped_boxes = [b for b, _, _ in result["dropped"] if b is not None]
                save_overlay(image, result["masks"], labels, dropped_boxes, overlays_dir / f"{path.stem}.jpg")
                flags = record["qa"]["flags"]
                print(f"  [{n}/{len(todo)}] [ok] {rel} -> {len(record['instances'])} instance(s)"
                      + (f" {flags}" if flags else ""))

            # Outputs first, jsonl line last: a crash in between just means the
            # image is redone on --resume.
            jsonl.write(json.dumps(record) + "\n")
            jsonl.flush()
            os.fsync(jsonl.fileno())
            done[rel] = record

    current = set(rel_names.values())
    stale = [r for r in done if r not in current]
    if stale:
        print(f"  [warn] {len(stale)} record(s) in annotations.jsonl are for files no longer under "
              f"{args.raw_dir} (or not in --image-list); left out of the COCO file.")
    coco, skipped = write_outputs([r for k, r in done.items() if k in current], run_dir)
    print_summary(coco, skipped)
    print(f"Done. Wrote coco/annotations.json, skipped.txt, masks/ and overlays/ under {run_dir}")


if __name__ == "__main__":
    main()
