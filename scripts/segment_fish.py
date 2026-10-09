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

Prompts: Grounding DINO zero-shot boxes for a text prompt (default "fish.")
as SAM box prompts. (A centre-point mode and a YOLO mode existed until
2026-10-09; neither was used after stage 0, see
vault/thesis-log/experiments/mask-quality-history.md.)

**Defaults = the thesis setting D** (vault/thesis-log/experiments/
2026-10-04-random-review-fixes.md): mask threshold -1, specks < 1 % dropped,
bleed fallback at 2 % outside the box, wrong-region fix on, far pieces and
small holes at 0.02. One setting differs per domain: ``--bleed-max-box-edge``
0.1 for photos (default), 0.3 for X-rays. ``scripts/run_part1.py`` passes it
from ``pipeline/config.yaml``.

Per image: detect -> SAM (3 candidates per prompt; threshold the mask logits
at ``--mask-threshold``; keep one candidate per ``--candidate``) -> drop empty
masks -> drop duplicate detections -> give every contested pixel to exactly
one instance -> optionally drop specks (``--min-component-frac``) ->
optionally grow each instance by a size-relative margin
(``--margin-frac``) -> flag tiny/giant instances. Every threshold is a CLI
flag and a starting guess, to be tuned per domain (photo vs. X-ray) on real
overlays - none of the defaults comes from the data.

Fin recovery (SAM tends to cut off fins; see
vault/thesis-log/decisions/2026-10-01-mask-policy.md):
  * ``--candidate`` ``score`` (default: highest predicted IoU) or
    ``largest`` (largest candidate area after thresholding).
  * ``--mask-threshold`` (default -1; 0.0 = SAM's own cut) applied to the
    mask logits at working resolution; lower keeps more translucent fin.
  * ``--margin-frac`` (default 0 = off) grows every final instance by
    r = max(1, round(margin_frac * sqrt(area))) px, never into another fish.
  * ``--min-component-frac`` (default 0.01) drops connected pieces of an
    instance smaller than this fraction of it (low thresholds leave specks;
    the largest piece always stays).

Per-instance QA features (COCO annotation + annotations.jsonl), meant as
inputs for the mask-QA model: ``sam_candidate_areas``,
``area_ratio_largest_to_chosen``, ``edge_uncertain_frac`` (fin-loss signal,
see ``edge_uncertain_frac()``), ``touches_border``, ``n_components``,
``margin_px``. Areas and ``margin_px`` are in working-resolution pixels.

Bleed fallback (``--bleed-outside-frac``, default 0.02; 0 = off): SAM sometimes
picks a candidate that fills the whole background, printed names included.
Such a mask spills out of its detector box; when it does, a SAM candidate
that stays inside the box is used instead, or the inverse of a background
candidate. Instances record ``bleed_fallback`` (none / candidate / inverse /
unresolved), ``sam_outside_box_frac`` and ``outside_box_frac``; flags
``bleed_fixed`` / ``bleed_unresolved``. See ``bleed_fallback()``.

Wrong-region fix (``--wrong-region-fix``, default on): inside a box that
spans the whole image a mask of the background around the fish, or of the
corners of an X-ray canvas, never spills out of the box. It leaves the middle
of the box empty and runs along the box outline instead (``is_wrong_region()``);
SAM is then asked again with a point on the fish and points in the wrong
region. Instances record ``wrong_region`` (none / reprompt / unresolved),
``box_center_cover`` and ``box_edge_frac``; flags ``wrong_region_fixed`` /
``wrong_region_unresolved``; a spurious detection without a fish is dropped
(image flag ``wrong_region_dropped``). See ``wrong_region_fix()``.

Clean-up (both default 0.02): ``--keep-near-frac`` drops small pieces far from the
fish (printed names, rulers, background blobs, the line SAM leaves along an
image edge; ``far_pieces_dropped``), ``--fill-holes-frac`` fills small holes
(speckled fins, X-ray grids; ``holes_filled_px``). Numbers behind all three:
vault/thesis-log/experiments/2026-10-04-random-review-fixes.md

Fin extension (``--fin-threshold``, default off): after all rules above, each
mask thresholded from its own SAM candidate also gets the connected parts of
that candidate above the lower fin threshold (faint X-ray fin rays), except
added pieces touching the image border; if that grows it by more than
``--fin-max-growth`` it stays as is (``fin_extension``, ``fin_growth``,
``fin_border_px``; flag ``fin_guarded``). See ``extend_fins()``.

``--image-list`` restricts a run to the listed files (e.g. a dev set), one
path per line relative to ``--raw-dir`` as in COCO ``file_name``.

Crash-safe: re-run with ``--resume`` to skip every image already in
annotations.jsonl. A resume with different settings (incl. a different
image list) is refused, so one run never mixes masks made with different
thresholds.

Run inside the container, e.g.:

    docker compose run --rm vit-project python scripts/segment_fish.py \
        --raw-dir /workspace/data/raw/NHM_datensatz/full_body
    docker compose run --rm vit-project python scripts/segment_fish.py \
        --raw-dir /workspace/data/raw/NHM_datensatz/Röntgen --bleed-max-box-edge 0.3 --resume
    docker compose run --rm vit-project python scripts/segment_fish.py \
        --raw-dir /workspace/data/raw/NHM_datensatz/full_body \
        --image-list dev_set.txt --mask-threshold -2 --margin-frac 0.03 \
        --out-dir /workspace/data/processed/segmented/dev_t-2_m0.03
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

# bleed_fallback(): an inverted background candidate is cut to the detector
# box grown by this fraction of its width/height (DINO boxes can be tight on
# fins); besides its largest piece, inverse_pieces() keeps only pieces of at
# least this fraction of it (drops printed letters, specks).
BLEED_BOX_PAD = 0.02
BLEED_INVERSE_SPECK_FRAC = 0.05

# Wrong-region fix (see wrong_region_fix()): a mask is the wrong region when it
# covers less than WRONG_CENTER_MAX of its box's central rectangle
# (CENTER_FRAC x the box width/height) and more than WRONG_EDGE_MIN of the box
# outline. A re-prompted replacement may cover at most REPROMPT_EDGE_MAX of
# the outline and needs REPROMPT_MIN_REL_AREA of the largest other instance's
# area; otherwise a spurious detection (several instances) is dropped.
CENTER_FRAC = 0.3
WRONG_CENTER_MAX = 0.25
WRONG_EDGE_MIN = 0.3
REPROMPT_EDGE_MAX = 0.15
REPROMPT_MAX_NEG = 4
REPROMPT_MIN_REL_AREA = 0.05

# keep_near_pieces(): a far piece is dropped only when it is also smaller than
# this fraction of the instance's largest piece (a ruler lying across a fish
# splits it into big pieces that must stay).
FAR_PIECE_MAX_FRAC = 0.1

# Settings added after the first full run, with the value that reproduces it.
# A run_config.json written before they existed lacks them; resuming such a
# run is allowed only with exactly these values.
LEGACY_DEFAULTS = {"candidate": "score", "mask_threshold": 0.0, "margin_frac": 0.0,
                   "min_component_frac": 0.0, "image_list": None, "image_list_sha256": None,
                   "bleed_outside_frac": 0.0, "wrong_region_fix": False, "keep_near_frac": 0.0,
                   "fill_holes_frac": 0.0, "fin_threshold": None}


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


def sam_logits(model, processor, image, input_points=None, input_boxes=None, input_labels=None):
    """Like ``data.models.sam.segment`` but returns SAM's raw mask logits.

    Same processor/model call; the only difference is ``binarize=False`` in
    ``post_process_masks``, which then skips its final ``masks > 0.0`` and
    returns the logits upscaled (bilinear) to the working image size. So
    ``logits > 0.0`` is bit-identical to ``segment()``'s masks.
    ``input_labels`` (1 = foreground, 0 = background point) pairs with
    ``input_points``; None makes every point foreground.
    Returns (logits (N, 3, H, W) float32, iou_scores (N, 3)) as numpy.
    """
    import torch

    device = next(model.parameters()).device
    inputs = processor(
        image, input_points=input_points, input_labels=input_labels, input_boxes=input_boxes,
        return_tensors="pt",
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


def box_slices(box, shape, pad=0.0):
    """Row/column slices of an xyxy box grown by ``pad`` x its width/height, clipped to ``shape``."""
    x0, y0, x1, y1 = box
    px, py = pad * (x1 - x0), pad * (y1 - y0)
    h, w = shape
    return (slice(max(0, int(math.floor(y0 - py))), min(h, int(math.ceil(y1 + py)))),
            slice(max(0, int(math.floor(x0 - px))), min(w, int(math.ceil(x1 + px)))))


def box_fit(mask, box):
    """(outside_box_frac, box_fill) of a mask against its detector box (xyxy).

    outside_box_frac = share of the mask's pixels outside the box; box_fill =
    share of the box's pixels inside the mask. A fish mask sits inside its
    box (outside ~0, fins aside) and fills part of it; a mask that bled into
    the background spills out of the box and fills it almost completely.
    """
    sl = box_slices(box, mask.shape)
    inside = int(mask[sl].sum())
    area = int(mask.sum())
    box_px = (sl[0].stop - sl[0].start) * (sl[1].stop - sl[1].start)
    return ((area - inside) / area if area else 0.0), (inside / box_px if box_px else 0.0)


def box_edge_frac(mask, box):
    """Share of the detector box's outline (1 px, clipped to the image) covered by ``mask``.

    A fish touches its box at its tips; background left inside the box runs
    along the box edges.
    """
    sl = box_slices(box, mask.shape)
    inner = mask[sl]
    if inner.size == 0:
        return 0.0
    edge = np.concatenate([inner[0], inner[-1], inner[1:-1, 0], inner[1:-1, -1]])
    return float(edge.mean())


def inverse_pieces(inv, min_frac):
    """Clean an inverted background candidate: keep its largest piece, plus other
    pieces >= ``min_frac`` of it that do not touch the image border (corners of
    the X-ray canvas, labels and rulers at the image edge do)."""
    import cv2

    n, labels, stats, _ = cv2.connectedComponentsWithStats(inv.astype(np.uint8), connectivity=8)
    if n <= 2:
        return inv.copy()
    areas = stats[1:, cv2.CC_STAT_AREA]
    h, w = inv.shape
    x, y = stats[1:, cv2.CC_STAT_LEFT], stats[1:, cv2.CC_STAT_TOP]
    on_border = (x == 0) | (y == 0) | (x + stats[1:, cv2.CC_STAT_WIDTH] == w) | (y + stats[1:, cv2.CC_STAT_HEIGHT] == h)
    keep = (areas == areas.max()) | ((areas >= min_frac * areas.sum()) & ~on_border)
    return np.isin(labels, 1 + np.nonzero(keep)[0])


def bleed_fallback(candidates, chosen, box, args):
    """Replace a mask that bled into the background by a better one from the same SAM call.

    Trigger: the chosen candidate has more than ``--bleed-outside-frac`` of
    its pixels outside the detector box. Then, in this order:
      1. the largest of SAM's 3 candidates that stays inside the box
         (outside <= the same limit) and fills ``--bleed-fill`` MIN..MAX of
         it. MAX rejects the box-shaped blob a bleed leaves inside the box,
         MIN rejects an eye or a vertebra;
      2. the inverse of a background-like candidate (it covers >=
         ``--bleed-bg-border`` of the image frame): its complement inside
         the box grown by BLEED_BOX_PAD, cleaned by inverse_pieces(). SAM
         sometimes returns only the background, never the fish. Same limits,
         and it may cover at most ``--bleed-max-box-edge`` of the box outline
         (else it is background/labels left inside the box);
      3. otherwise the chosen candidate stays, marked unresolved.
    Reasoning and the numbers behind the defaults:
    vault/thesis-log/experiments/2026-10-03-bleed-fallback.md

    candidates: (3, H, W) bool; box: xyxy at working resolution.
    Returns (mask, source, candidate index, inverted) with source one of
    "none" (not triggered), "candidate", "inverse", "unresolved".
    """
    outside, _ = box_fit(candidates[chosen], box)
    if args.bleed_outside_frac <= 0 or outside <= args.bleed_outside_frac:
        return candidates[chosen], "none", chosen, False
    lo, hi = args.bleed_fill

    def valid(mask):
        if not mask.any():
            return False
        out, fill = box_fit(mask, box)
        return out <= args.bleed_outside_frac and lo <= fill <= hi

    fits = [k for k in range(len(candidates)) if valid(candidates[k])]
    if fits:
        k = max(fits, key=lambda k: candidates[k].sum())
        return candidates[k], "candidate", k, False

    sl = box_slices(box, candidates.shape[1:], BLEED_BOX_PAD)
    inverses = []
    for k, cand in enumerate(candidates):
        frame = np.concatenate([cand[0], cand[-1], cand[:, 0], cand[:, -1]])
        if frame.mean() < args.bleed_bg_border:
            continue
        inv = np.zeros_like(cand)
        inv[sl] = ~cand[sl]
        inv = inverse_pieces(inv, BLEED_INVERSE_SPECK_FRAC)
        if valid(inv) and box_edge_frac(inv, box) <= args.bleed_max_box_edge:
            inverses.append((int(inv.sum()), k, inv))
    if inverses:
        _, k, inv = max(inverses, key=lambda t: t[0])
        return inv, "inverse", k, True
    return candidates[chosen], "unresolved", chosen, False


def box_center_cover(mask, box):
    """Share of the box's central rectangle (CENTER_FRAC x its width and height) covered by ``mask``."""
    sl = box_slices(box, mask.shape)
    inner = mask[sl]
    h, w = inner.shape
    if h == 0 or w == 0:
        return 0.0
    lo, hi = (1 - CENTER_FRAC) / 2, (1 + CENTER_FRAC) / 2
    y0, x0 = int(h * lo), int(w * lo)
    centre = inner[y0:max(y0 + 1, int(h * hi)), x0:max(x0 + 1, int(w * hi))]
    return float(centre.mean())


def is_wrong_region(mask, box):
    """True if ``mask`` is the background around the fish (or the canvas corners) instead of the fish.

    A fish covers the middle of its detector box and touches the box outline
    only at its tips; an inverted mask or the corners of an X-ray canvas
    leave the middle empty and run along the outline. Inside a box that spans
    the whole image this never shows as "outside the box", so the bleed
    fallback can't see it.
    """
    return box_center_cover(mask, box) < WRONG_CENTER_MAX and box_edge_frac(mask, box) > WRONG_EDGE_MIN


def deepest_point(region):
    """(x, y) of the ``region`` pixel farthest from the region's border."""
    from scipy.ndimage import distance_transform_edt

    dist = distance_transform_edt(np.pad(region, 1))[1:-1, 1:-1]
    y, x = np.unravel_index(int(dist.argmax()), dist.shape)
    return float(x), float(y)


def reprompt_points(wrong, box):
    """Point prompts that steer SAM away from a wrong-region mask: one positive
    point on the fish (the box centre, or the in-box pixel farthest from the
    wrong region if the centre lies in it) and a negative point deep inside
    each of the REPROMPT_MAX_NEG largest pieces of the wrong region.
    Returns (points [[x, y], ...], labels [1, 0, ...])."""
    import cv2

    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    h, w = wrong.shape
    if wrong[min(h - 1, int(cy)), min(w - 1, int(cx))]:
        free = np.zeros_like(wrong)
        sl = box_slices(box, wrong.shape)
        free[sl] = ~wrong[sl]
        cx, cy = deepest_point(free)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(wrong.astype(np.uint8), connectivity=8)
    biggest = 1 + np.argsort(-stats[1:, cv2.CC_STAT_AREA], kind="stable")[:REPROMPT_MAX_NEG]
    negatives = [list(deepest_point(labels == k)) for k in biggest]
    return [[cx, cy]] + negatives, [1] + [0] * len(negatives)


def wrong_region_fix(wrong, box, image, sam, args, other_max_area=0):
    """Replace a wrong-region mask (see is_wrong_region()) by re-prompting SAM.

    SAM gets the detector box again plus reprompt_points(). A new candidate is
    valid if it is not a wrong region itself, covers at most REPROMPT_EDGE_MAX
    of the box outline, lies at most 2 % outside the box, fills
    ``--bleed-fill`` MIN..MAX of it and, when other instances exist, has at
    least REPROMPT_MIN_REL_AREA of the largest one's area. The valid candidate
    with the highest predicted IoU wins. Without one, an instance among
    others is a spurious detection and is dropped; a lone instance stays,
    marked unresolved.

    Returns (mask, its logits, iou (3,), candidate areas (3,), candidate index,
    status) with status "reprompt"; for "dropped" / "unresolved" all but the
    status are None.
    """
    model, processor = sam
    points, labels = reprompt_points(wrong, box)
    logits, iou = sam_logits(model, processor, image, input_points=[[points]], input_labels=[[labels]],
                             input_boxes=[[[float(v) for v in box]]])
    logits, iou = logits[0], iou[0]
    cands = logits > args.mask_threshold
    lo, hi = args.bleed_fill
    valid = []
    for k, cand in enumerate(cands):
        if not cand.any() or is_wrong_region(cand, box) or box_edge_frac(cand, box) > REPROMPT_EDGE_MAX:
            continue
        outside, fill = box_fit(cand, box)
        if outside <= 0.02 and lo <= fill <= hi and cand.sum() >= REPROMPT_MIN_REL_AREA * other_max_area:
            valid.append(k)
    if valid:
        k = max(valid, key=lambda j: iou[j])
        return cands[k], logits[k], iou, cands.reshape(len(cands), -1).sum(axis=1), k, "reprompt"
    return None, None, None, None, None, ("dropped" if other_max_area > 0 else "unresolved")


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


def drop_small_components(mask, min_frac):
    """Remove 8-connected pieces smaller than ``min_frac`` of the mask's area.

    The largest piece always stays, so the mask never becomes empty.
    min_frac <= 0 returns the mask unchanged (copy).
    """
    if min_frac <= 0:
        return mask.copy()
    import cv2

    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if n <= 2:  # background + at most one piece
        return mask.copy()
    areas = stats[1:, cv2.CC_STAT_AREA]
    keep = (areas >= min_frac * areas.sum()) | (areas == areas.max())
    return np.isin(labels, 1 + np.nonzero(keep)[0])


def keep_near_pieces(mask, near_frac):
    """Drop pieces of an instance that lie far from the fish: printed names,
    rulers, background blobs, the line SAM leaves along an image edge.

    The largest 8-connected piece stays. Every piece within r = max(1,
    round(near_frac * sqrt(largest piece area))) px of a kept piece stays too
    (repeated, so chains of fin pieces survive), and so does every piece of at
    least FAR_PIECE_MAX_FRAC of the largest one (a fish cut in two by a ruler).
    near_frac <= 0 returns the mask unchanged (copy). Returns (mask, number of
    pieces dropped).
    """
    if near_frac <= 0:
        return mask.copy(), 0
    import cv2

    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if n <= 2:
        return mask.copy(), 0
    areas = stats[1:, cv2.CC_STAT_AREA]
    keep = set((1 + np.nonzero(areas >= FAR_PIECE_MAX_FRAC * areas.max())[0]).tolist())
    r = max(1, round(near_frac * math.sqrt(areas.max())))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    while True:
        kept = np.isin(labels, sorted(keep))
        near = cv2.dilate(kept.astype(np.uint8), kernel) > 0
        reached = set(np.unique(labels[near & (labels > 0)]).tolist()) - keep
        if not reached:
            return kept, n - 1 - len(keep)
        keep |= reached


def fill_small_holes(mask, max_frac, blocked=None):
    """Fill holes of an instance (background enclosed by it) of at most
    ``max_frac`` of its area: speckled translucent fins, an open mouth, the
    grid SAM sometimes leaves in X-ray bodies. Bigger holes (e.g. background
    inside a curled body) and ``blocked`` pixels (other fish) stay out.
    max_frac <= 0 returns the mask unchanged (copy). Returns (mask, pixels added).
    """
    if max_frac <= 0:
        return mask.copy(), 0
    import cv2
    from scipy.ndimage import binary_fill_holes

    holes = binary_fill_holes(mask) & ~mask
    if not holes.any():
        return mask.copy(), 0
    n, labels, stats, _ = cv2.connectedComponentsWithStats(holes.astype(np.uint8), connectivity=4)
    small = 1 + np.nonzero(stats[1:, cv2.CC_STAT_AREA] <= max_frac * mask.sum())[0]
    add = np.isin(labels, small)
    if blocked is not None:
        add &= ~blocked
    return mask | add, int(add.sum())


def extend_fins(mask, logits, fin_threshold, max_growth):
    """Add the faint parts of the same SAM candidate (logits > ``fin_threshold``)
    that are connected to the mask: X-ray fin rays and spines SAM's cut leaves out.

    ``logits`` are the candidate's own logits, so a lower ``fin_threshold`` only
    widens the mask (8-connected pieces not touching the mask are ignored).
    Added pieces (8-connected, outside the mask) that touch the image border
    are dropped: at low thresholds the dark plate and scanner edges come in
    from the border, fins grow off the body. Guard: if the mask would still
    grow by more than ``max_growth`` (fraction of its area), it is kept as is.
    Numbers behind both: vault/thesis-log/decisions/2026-10-08-xray-fin-extension.md
    Returns (mask, "extended" / "guarded", growth = added px / mask px,
    px dropped as border pieces).
    """
    import cv2

    ext = mask | (logits > fin_threshold)
    n, labels = cv2.connectedComponents(ext.astype(np.uint8), connectivity=8)
    if n > 2:
        ext = np.isin(labels, np.unique(labels[mask]))
    add = ext & ~mask
    n, labels = cv2.connectedComponents(add.astype(np.uint8), connectivity=8)
    border = np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]]))
    on_border = np.isin(labels, border[border > 0])
    border_px = int(on_border.sum())
    ext = mask | (add & ~on_border)
    growth = float(ext.sum() / mask.sum() - 1)
    if growth > max_growth:
        return mask.copy(), "guarded", growth, border_px
    return ext, "extended", growth, border_px


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


def segment_image(image, sam, detector, args):
    """Run detection + SAM + the per-instance rules on one working-size image.

    Returns (record fields, final masks, labels, dropped boxes) or a skip
    reason string when no usable instance remains.
    """
    model, processor = sam
    w, h = image.size
    boxes, scores = detector(image)
    if len(boxes) == 0:
        return "no_detection"
    logits, iou = sam_logits(model, processor, image, input_boxes=[boxes.tolist()])
    masks, chosen, cand_areas = choose_candidates(logits, iou, args.mask_threshold, args.candidate)
    n_detections = len(masks)
    # Edge feature threshold per instance: an inverted candidate is "in" where
    # its negated logits are >= -threshold (see bleed_fallback()).
    edge_thr = np.full(n_detections, args.mask_threshold)
    bleed_source = np.array(["none"] * n_detections, dtype=object)
    chosen_outside = np.array([box_fit(m, b)[0] for m, b in zip(masks, boxes)])
    inverted = np.zeros(n_detections, dtype=bool)
    if args.bleed_outside_frac > 0:
        for k in range(n_detections):
            masks[k], bleed_source[k], chosen[k], inverted[k] = bleed_fallback(
                logits[k] > args.mask_threshold, chosen[k], boxes[k], args)
    # Only the chosen candidate's logits are needed later (edge feature).
    chosen_logits = logits[np.arange(len(chosen)), chosen]
    chosen_logits[inverted] *= -1
    edge_thr[inverted] *= -1
    del logits

    wrong_source = np.array(["none"] * n_detections, dtype=object)
    if args.wrong_region_fix:
        areas = masks.reshape(n_detections, -1).sum(axis=1)
        for k in range(n_detections):
            if not is_wrong_region(masks[k], boxes[k]):
                continue
            others = np.delete(areas, k)
            new, new_logits, new_iou, new_areas, new_k, wrong_source[k] = wrong_region_fix(
                masks[k], boxes[k], image, sam, args, int(others.max()) if len(others) else 0)
            if wrong_source[k] == "reprompt":
                masks[k], chosen_logits[k], edge_thr[k] = new, new_logits, args.mask_threshold
                iou[k], cand_areas[k], chosen[k] = new_iou, new_areas, new_k
            elif wrong_source[k] == "dropped":
                masks[k] = False
            areas[k] = masks[k].sum()
    wrong_dropped = int((wrong_source == "dropped").sum())

    # Fin extension: only for masks thresholded from their own logits. An
    # inverted background, an unresolved bleed or wrong region would only
    # take in more background at a lower threshold.
    fin_source = np.array(["none"] * n_detections, dtype=object)
    fin_growth = np.zeros(n_detections)
    fin_border = np.zeros(n_detections, dtype=int)
    if args.fin_threshold is not None:
        for k in range(n_detections):
            if not masks[k].any():
                continue
            own = wrong_source[k] == "reprompt" or (
                wrong_source[k] == "none" and not inverted[k] and bleed_source[k] != "unresolved")
            if not own:
                fin_source[k] = "skipped"
                continue
            masks[k], fin_source[k], fin_growth[k], fin_border[k] = extend_fins(
                masks[k], chosen_logits[k], args.fin_threshold, args.fin_max_growth)
            if fin_source[k] == "extended":
                edge_thr[k] = args.fin_threshold

    def keep(idx):
        nonlocal masks, boxes, scores, iou, chosen, cand_areas, chosen_logits
        nonlocal edge_thr, bleed_source, chosen_outside, wrong_source, fin_source, fin_growth, fin_border
        masks, iou, chosen = masks[idx], iou[idx], chosen[idx]
        cand_areas, chosen_logits = cand_areas[idx], chosen_logits[idx]
        edge_thr, bleed_source, chosen_outside = edge_thr[idx], bleed_source[idx], chosen_outside[idx]
        wrong_source, fin_source, fin_growth = wrong_source[idx], fin_source[idx], fin_growth[idx]
        fin_border = fin_border[idx]
        boxes, scores = boxes[idx], scores[idx]

    img_area = float(w * h)
    keep(np.nonzero(masks.reshape(len(masks), -1).sum(axis=1) > 0)[0])
    empty_dropped = n_detections - len(masks) - wrong_dropped

    giant = masks.reshape(len(masks), -1).sum(axis=1) / img_area > args.giant_area_frac
    kept, dropped = remove_duplicates(masks, boxes, scores, giant, args.dedup_containment)
    dropped_info = [(boxes[i], float(scores[i]), j) for i, j in dropped]
    keep(np.array(kept, dtype=int))

    masks, contested = resolve_overlaps(masks, boxes, scores)
    # An instance can lose every pixel to its neighbours; treat it as empty.
    nonempty = np.nonzero(masks.reshape(len(masks), -1).sum(axis=1) > 0)[0]
    empty_dropped += len(masks) - len(nonempty)
    new_index = {old: new for new, old in enumerate(np.array(kept)[nonempty])}
    keep(nonempty)
    if len(masks) == 0:
        return "all_masks_empty"

    masks = np.stack([drop_small_components(m, args.min_component_frac) for m in masks])
    far_dropped = np.zeros(len(masks), dtype=int)
    for k in range(len(masks)):
        masks[k], far_dropped[k] = keep_near_pieces(masks[k], args.keep_near_frac)
    holes_filled = np.zeros(len(masks), dtype=int)
    for k in range(len(masks)):  # one by one, so a filled hole is never given to two fish
        masks[k], holes_filled[k] = fill_small_holes(masks[k], args.fill_holes_frac,
                                                     blocked=masks.any(axis=0) & ~masks[k])
    # Edge feature on the pre-margin masks; other fish are not "missing fin".
    occupied = masks.any(axis=0)
    edge_uncertain = [edge_uncertain_frac(m, lg, thr, blocked=occupied & ~m)
                      for m, lg, thr in zip(masks, chosen_logits, edge_thr)]
    masks, margin_px = grow_margins(masks, args.margin_frac)
    return {
        "masks": masks, "boxes": boxes, "scores": scores, "iou": iou, "chosen": chosen,
        "cand_areas": cand_areas, "edge_uncertain": edge_uncertain, "margin_px": margin_px,
        "bleed_source": list(bleed_source), "chosen_outside": chosen_outside,
        "wrong_source": list(wrong_source), "far_dropped": far_dropped, "holes_filled": holes_filled,
        "fin_source": list(fin_source), "fin_growth": fin_growth, "fin_border": fin_border,
        "n_detections": n_detections, "empty_dropped": empty_dropped, "wrong_dropped": wrong_dropped,
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
            # Bleed: share outside the detector box of SAM's own choice
            # (before the fallback) and of the final mask; which fallback ran.
            "sam_outside_box_frac": round(float(result["chosen_outside"][k]), 4),
            "outside_box_frac": (round(box_fit(m, result["boxes"][k])[0], 4)
                                 if result["boxes"] is not None else None),
            "bleed_fallback": result["bleed_source"][k],
            # Wrong-region fix (none / reprompt / unresolved) and the box
            # features it uses, on the final mask; far pieces dropped and hole
            # pixels filled (working resolution).
            "wrong_region": result["wrong_source"][k],
            "box_center_cover": (round(box_center_cover(m, result["boxes"][k]), 4)
                                 if result["boxes"] is not None else None),
            "box_edge_frac": (round(box_edge_frac(m, result["boxes"][k]), 4)
                              if result["boxes"] is not None else None),
            "far_pieces_dropped": int(result["far_dropped"][k]),
            "holes_filled_px": int(result["holes_filled"][k]),
            # Fin extension (none = off / skipped / extended / guarded), the
            # growth it found (also when the guard rejected it) and the pixels
            # it dropped as border pieces (working resolution).
            "fin_extension": result["fin_source"][k],
            "fin_growth": round(float(result["fin_growth"][k]), 4),
            "fin_border_px": int(result["fin_border"][k]),
        })
        if result["bleed_source"][k] in ("candidate", "inverse"):
            flags.append("bleed_fixed")
        elif result["bleed_source"][k] == "unresolved":
            flags.append("bleed_unresolved")
        if result["wrong_source"][k] == "reprompt":
            flags.append("wrong_region_fixed")
        elif result["wrong_source"][k] == "unresolved":
            flags.append("wrong_region_unresolved")
        if result["fin_source"][k] == "guarded":
            flags.append("fin_guarded")
        # Fix flags stay off the overlay: they would tell a blind
        # review_masks.py --compare which run an overlay comes from.
        labels.append(" ".join([str(k)] + [f for f in flags
                                           if not f.startswith(("bleed_", "wrong_region_", "fin_"))]))
    image_flags = sorted({f for inst in instances for f in inst["flags"]})
    if len(instances) > 1:
        image_flags.append("multi_instance")
    if result["dropped"]:
        image_flags.append("duplicates_removed")
    if result["contested_frac"] > 0:
        image_flags.append("overlap_resolved")
    if result["empty_dropped"]:
        image_flags.append("empty_masks_dropped")
    if result["wrong_dropped"]:
        image_flags.append("wrong_region_dropped")
    record = {
        "file_name": rel_name, "status": "ok", "width": W, "height": H,
        "qa": {
            "flags": image_flags,
            "n_detections": result["n_detections"],
            "empty_dropped": result["empty_dropped"],
            "wrong_region_dropped": result["wrong_dropped"],
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
        "raw_dir": str(args.raw_dir), "prompt": "dino", "sam_checkpoint": SAM_CHECKPOINT,
        "max_side": args.max_side, "dedup_containment": args.dedup_containment,
        "tiny_area_frac": args.tiny_area_frac, "giant_area_frac": args.giant_area_frac,
        "candidate": args.candidate, "mask_threshold": args.mask_threshold,
        "margin_frac": args.margin_frac, "min_component_frac": args.min_component_frac,
        "bleed_outside_frac": args.bleed_outside_frac, "wrong_region_fix": args.wrong_region_fix,
        "keep_near_frac": args.keep_near_frac, "fill_holes_frac": args.fill_holes_frac,
        "fin_threshold": args.fin_threshold,
        "image_list": str(Path(args.image_list).resolve()) if args.image_list is not None else None,
        "image_list_sha256": (hashlib.sha256("\n".join(sorted(image_names)).encode()).hexdigest()
                              if image_names is not None else None),
    }
    if args.bleed_outside_frac > 0 or args.wrong_region_fix:
        settings.update(bleed_fill=list(args.bleed_fill))
    if args.bleed_outside_frac > 0:
        settings.update(bleed_bg_border=args.bleed_bg_border, bleed_max_box_edge=args.bleed_max_box_edge,
                        bleed_box_pad=BLEED_BOX_PAD, bleed_inverse_speck_frac=BLEED_INVERSE_SPECK_FRAC)
    if args.wrong_region_fix:
        settings.update(center_frac=CENTER_FRAC, wrong_center_max=WRONG_CENTER_MAX, wrong_edge_min=WRONG_EDGE_MIN,
                        reprompt_edge_max=REPROMPT_EDGE_MAX, reprompt_max_neg=REPROMPT_MAX_NEG,
                        reprompt_min_rel_area=REPROMPT_MIN_REL_AREA)
    if args.keep_near_frac > 0:
        settings.update(far_piece_max_frac=FAR_PIECE_MAX_FRAC)
    if args.fin_threshold is not None:
        settings.update(fin_max_growth=args.fin_max_growth)
    settings.update(dino_checkpoint=DINO_CHECKPOINT, dino_text=args.dino_text,
                    dino_box_threshold=args.dino_box_threshold)
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


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-dir", type=Path, default=None,
                        help="Directory of input images to process (searched recursively). "
                             "If not provided, you'll be prompted to choose from available datasets.")
    parser.add_argument("--out-dir", default="/workspace/data/processed/segmented", type=Path,
                        help="Root of the outputs; results nest under <out-dir>/<raw-dir name>/.")
    parser.add_argument("--dino-text", default="fish.",
                        help="Text prompt for Grounding DINO. Lowercase, "
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
    parser.add_argument("--mask-threshold", type=float, default=-1.0,
                        help="Logit above which a pixel is in a SAM mask (0.0 = SAM's own cut; default "
                             "%(default)s keeps more of the translucent fins).")
    parser.add_argument("--margin-frac", type=float, default=0.0,
                        help="Grow each final instance by max(1, round(f*sqrt(area))) px, never into "
                             "another instance. 0 = off. Tunable guess.")
    parser.add_argument("--min-component-frac", type=float, default=0.01,
                        help="Drop connected pieces of an instance smaller than this fraction of its "
                             "area (the largest piece always stays). 0 = off (default %(default)s).")
    parser.add_argument("--bleed-outside-frac", type=float, default=0.02,
                        help="Bleed fallback: if more than this fraction of a mask lies outside its "
                             "detector box, replace it (see bleed_fallback()). 0 = off; the default "
                             "%(default)s separated the reviewed bleed from OK masks.")
    parser.add_argument("--bleed-fill", type=float, nargs=2, default=(0.2, 0.9), metavar=("MIN", "MAX"),
                        help="Bleed fallback: a replacement must fill MIN..MAX of its detector box "
                             "(default: %(default)s).")
    parser.add_argument("--bleed-bg-border", type=float, default=0.3,
                        help="Bleed fallback: a candidate covering at least this fraction of the image "
                             "frame counts as background and may be inverted (default: %(default)s).")
    parser.add_argument("--bleed-max-box-edge", type=float, default=0.1,
                        help="Bleed fallback: an inverted background candidate may cover at most this "
                             "fraction of its box outline (default: %(default)s, photos; tightly cropped "
                             "X-rays touch their box legitimately, 0.3 there).")
    parser.add_argument("--wrong-region-fix", action=argparse.BooleanOptionalAction, default=True,
                        help="Re-prompt SAM (box + a point on the fish + points in the wrong region) when "
                             "a mask leaves the middle of its box empty and runs along the box outline - "
                             "the background around the fish or the corners of an X-ray canvas (see "
                             "wrong_region_fix()). On by default.")
    parser.add_argument("--keep-near-frac", type=float, default=0.02,
                        help="Drop pieces of an instance farther than f*sqrt(largest piece area) px from "
                             "the fish and smaller than %(far)s of its largest piece (labels, rulers, "
                             "blobs, edge lines). 0 = off; the default 0.02 fit the reviewed sample."
                             % {"far": FAR_PIECE_MAX_FRAC})
    parser.add_argument("--fill-holes-frac", type=float, default=0.02,
                        help="Fill holes of an instance of at most this fraction of its area (speckled "
                             "fins, X-ray grids), never into another fish. 0 = off; the default 0.02 fit "
                             "the reviewed sample.")
    parser.add_argument("--fin-threshold", type=float, default=None,
                        help="After all other rules, add the parts of each mask's own SAM candidate with "
                             "logits above this value that touch the mask but not the image border "
                             "(faint X-ray fins; see "
                             "extend_fins()). Must be below --mask-threshold. Default off; -6 for X-rays.")
    parser.add_argument("--fin-max-growth", type=float, default=0.3,
                        help="With --fin-threshold: keep the mask unchanged if the extension would grow it "
                             "by more than this fraction of its area (default 0.3).")
    parser.add_argument("--image-list", type=Path, default=None,
                        help="Text file with one file name per line, relative to --raw-dir as in COCO "
                             "file_name ('#' comment lines and blank lines ignored). Only these images "
                             "are processed; names not found under --raw-dir are an error.")
    parser.add_argument("--resume", action="store_true",
                        help="Continue an interrupted run, skipping images already in annotations.jsonl.")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.fin_threshold is not None and args.fin_threshold >= args.mask_threshold:
        parser.error("--fin-threshold must be below --mask-threshold")

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
    if todo:
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
                result = segment_image(image, sam, detector, args)

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
