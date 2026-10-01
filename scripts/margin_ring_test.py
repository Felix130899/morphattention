"""Margin ring test: does the background margin around a fish mask leak the label?

Setting C (vault/thesis-log/decisions/2026-10-01-fin-fix-setting.md) grows
every fish mask by a margin of 3 % of fish size. That margin is background
(plus fin bits SAM missed). If the background near a fish carries information
about the label (tray, photo session, printed text), a model trained on
masked images could still learn a Clever Hans shortcut through it.

Test: a frozen DINOv2 linear probe tries to predict the GENUS (genera with
>= ``--per-genus`` specimens, balanced, one image per specimen) from image
variants that show only certain pixels; everything else is flat grey:

  full         the raw image (reference: how easy is the task at all)
  background   everything except fish + margin (positive control: the probe
               must be able to read background, or the test says nothing)
  fish         the fish mask without margin
  fish_ring    fish + margin (what training would see with setting C)
  ring         only the margin pixels
  ring_shape   the margin pixels painted flat white (outline only, no content)
  outer        a band just outside the margin: distance (r, 2r] from the
               grown mask, r = the margin radius. Pure background near the
               fish; the r-wide gap keeps fin bits the margin caught out of it
  outer_shape  that band painted flat white

The ring traces the fish's outline, so a ring-only probe can recognise a
genus by its silhouette without any background information. The ``_shape``
twins measure that; the leak is the difference between a variant and its
twin, not its distance to chance. ``fish``/``fish_ring``/``ring*``/``outer*``
share one crop (bounding box of grown mask + outer band), so position and
size can't differ between them; ``full``/``background`` use the whole image.

Verdict (per domain, from ``evaluate``):
  * control fails (background not above chance)  -> test uninformative
  * outer beats outer_shape significantly        -> background near the fish
    leaks the label: shrink / blur / noise the margin and re-test
  * otherwise                                    -> margin is safe
``ring`` vs ``ring_shape`` and ``fish_ring`` vs ``fish`` are reported too
(the ring may legitimately add fin information).

Steps (``data/processed/ring_test/run_ring_test.sh`` runs them all):

    python scripts/margin_ring_test.py select
    python scripts/segment_fish.py ... --image-list data/processed/ring_test/photos.txt \\
        --mask-threshold -1 --min-component-frac 0.01        # setting B, no margin
    python scripts/margin_ring_test.py features --domain photos --run-dir <B run dir>
    python scripts/margin_ring_test.py evaluate --domain photos

The probe is linear on frozen features, so it is a lower bound on what a
fine-tuned ViT could pick up; the raw-image Clever Hans baseline and LRP on
the real model remain the second line of defence.
"""

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
from segment_fish import grow_margins  # noqa: E402

Image.MAX_IMAGE_PIXELS = None

WORKSPACE = Path("/workspace")
OUT_DIR = WORKSPACE / "data/processed/ring_test"
DOMAINS = {"photos": "full_body", "xrays": "Röntgen"}
VARIANTS = ["full", "background", "fish", "fish_ring", "ring", "ring_shape", "outer", "outer_shape"]
GREY, INK = 128, 255
WORK_SIDE = 1024      # long side the masks/variants are built at
INPUT_SIDE = 448      # DINOv2 input (multiple of the 14 px patch)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)


# --------------------------------------------------------------------- select

def select(args):
    """Balanced genus sample: genera with >= per_genus specimens, one image per specimen."""
    rng = random.Random(args.seed)
    with args.labels.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for domain, folder in DOMAINS.items():
        by_genus = defaultdict(lambda: defaultdict(list))  # genus -> specimen -> [file]
        for r in rows:
            if r["photo_type"] != folder or not r["species"] or r["needs_review"] == "True":
                continue
            by_genus[r["species"].split()[0]][r["catalog_number"]].append(r["file_name"].split("/", 1)[1])
        genera = sorted(g for g, specs in by_genus.items() if len(specs) >= args.per_genus)
        picked = []
        for genus in genera:
            specimens = sorted(by_genus[genus])
            rng.shuffle(specimens)
            for spec in specimens[:args.per_genus]:
                picked.append((rng.choice(sorted(by_genus[genus][spec])), genus, spec))
        (args.out_dir / f"{domain}.txt").write_text("\n".join(p[0] for p in picked) + "\n", encoding="utf-8")
        with (args.out_dir / f"{domain}_labels.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["file_name", "genus", "catalog_number"])
            w.writerows(picked)
        print(f"{domain}: {len(genera)} genera x {args.per_genus} specimens = {len(picked)} images")


# ------------------------------------------------------------------- variants

def load_record_masks(record, size):
    """Rasterise one annotations.jsonl record's instance polygons at ``size`` (w, h).

    Returns (N, h, w) bool, disjoint (a pixel two polygons share goes to the
    earlier instance), or None when no instance has pixels.
    """
    import cv2

    sx, sy = size[0] / record["width"], size[1] / record["height"]
    taken = np.zeros((size[1], size[0]), bool)
    masks = []
    for inst in record["instances"]:
        m = np.zeros((size[1], size[0]), np.uint8)
        polys = [np.round(np.asarray(p, np.float64).reshape(-1, 2) * (sx, sy)).astype(np.int32)
                 for p in inst["segmentation"] if len(p) >= 6]
        if polys:
            cv2.fillPoly(m, polys, 1)
        m = m.astype(bool) & ~taken
        if m.any():
            masks.append(m)
            taken |= m
    return np.stack(masks) if masks else None


def make_variants(img, masks, margin_frac):
    """Build every test variant for one image.

    img: (H, W, 3) uint8; masks: (N, H, W) bool, disjoint instances without
    margin. Returns (dict variant -> uint8 image, stats dict).
    """
    from scipy.ndimage import distance_transform_edt

    fish = masks.any(axis=0)
    grown, radii = grow_margins(masks, margin_frac)
    fish_ring = grown.any(axis=0)
    ring = fish_ring & ~fish
    r = int(radii.max())
    dist = distance_transform_edt(~fish_ring)
    outer = (dist > r) & (dist <= 2 * r)

    ys, xs = np.nonzero(fish_ring | outer)
    crop = (slice(ys.min(), ys.max() + 1), slice(xs.min(), xs.max() + 1))

    def show(keep):
        out = np.full_like(img, GREY)
        out[keep] = img[keep]
        return out

    def paint(keep):
        out = np.full_like(img, GREY)
        out[keep] = INK
        return out

    variants = {
        "full": img,
        "background": show(~fish_ring),
        "fish": show(fish)[crop],
        "fish_ring": show(fish_ring)[crop],
        "ring": show(ring)[crop],
        "ring_shape": paint(ring)[crop],
        "outer": show(outer)[crop],
        "outer_shape": paint(outer)[crop],
    }
    stats = {"margin_px": r, "n_instances": len(masks), "ring_px": int(ring.sum()),
             "outer_px": int(outer.sum()), "fish_px": int(fish.sum())}
    return variants, stats


def load_working_image(path):
    """RGB uint8 array with the long side at most WORK_SIDE (JPEG draft decode for speed)."""
    with Image.open(path) as im:
        im.draft("RGB", (WORK_SIDE, WORK_SIDE))
        im = im.convert("RGB")
        im.thumbnail((WORK_SIDE, WORK_SIDE), Image.BILINEAR)
        return np.asarray(im)


def to_input(arr):
    """Pad to a grey square, resize to INPUT_SIDE, ImageNet-normalise -> (3, S, S) float32."""
    h, w = arr.shape[:2]
    side = max(h, w)
    sq = np.full((side, side, 3), GREY, np.uint8)
    sq[(side - h) // 2:(side - h) // 2 + h, (side - w) // 2:(side - w) // 2 + w] = arr
    im = Image.fromarray(sq).resize((INPUT_SIDE, INPUT_SIDE), Image.BILINEAR)
    x = (np.asarray(im, np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    return x.transpose(2, 0, 1)


def save_example(variants, out_path):
    """One sheet with every variant side by side (for checking by eye)."""
    tiles = []
    for name in VARIANTS:
        im = Image.fromarray(variants[name])
        im.thumbnail((360, 360), Image.BILINEAR)
        tile = Image.new("RGB", (360, 384), (255, 255, 255))
        tile.paste(im, ((360 - im.width) // 2, 24 + (360 - im.height) // 2))
        ImageDraw.Draw(tile).text((6, 4), name, fill=(0, 0, 0))
        tiles.append(tile)
    sheet = Image.new("RGB", (360 * 4, 384 * 2), (255, 255, 255))
    for i, t in enumerate(tiles):
        sheet.paste(t, ((i % 4) * 360, (i // 4) * 384))
    sheet.save(out_path, quality=90)


# ------------------------------------------------------------------- features

def features(args):
    import torch

    from data.models.dinov2 import load_dinov2

    folder = DOMAINS[args.domain]
    labels = {}
    with (args.out_dir / f"{args.domain}_labels.csv").open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            labels[row["file_name"]] = row
    records = {}
    with (args.run_dir / "annotations.jsonl").open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                if rec.get("status") == "ok" and rec["file_name"] in labels:
                    records[rec["file_name"]] = rec

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_dinov2(device)
    raw_dir = args.raw_root / folder
    example_dir = args.out_dir / f"examples_{args.domain}"
    example_dir.mkdir(parents=True, exist_ok=True)

    feats = {v: [] for v in VARIANTS}
    kept, stats_rows, missing = [], [], []
    batch, batch_names = {v: [] for v in VARIANTS}, []

    def flush():
        with torch.no_grad(), torch.autocast(device_type=device, enabled=device == "cuda"):
            for v in VARIANTS:
                x = torch.from_numpy(np.stack(batch[v])).to(device)
                h = model(pixel_values=x).last_hidden_state.float()
                feats[v].append(torch.cat([h[:, 0], h[:, 1:].mean(dim=1)], dim=1).cpu().numpy())
                batch[v].clear()
        kept.extend(batch_names)
        batch_names.clear()

    names = sorted(labels)
    for i, name in enumerate(names):
        rec = records.get(name)
        if rec is None:
            missing.append(name)  # skipped by segment_fish (no fish found)
            continue
        img = load_working_image(raw_dir / name)
        masks = load_record_masks(rec, (img.shape[1], img.shape[0]))
        if masks is None:
            missing.append(name)
            continue
        variants, stats = make_variants(img, masks, args.margin_frac)
        stats_rows.append({"file_name": name, **stats})
        if len(stats_rows) <= args.examples:
            save_example(variants, example_dir / (Path(name).stem + ".jpg"))
        for v in VARIANTS:
            batch[v].append(to_input(variants[v]))
        batch_names.append(name)
        if len(batch_names) == args.batch_size:
            flush()
        if (i + 1) % 50 == 0:
            print(f"  [{i + 1}/{len(names)}]", flush=True)
    if batch_names:
        flush()

    out = args.out_dir / f"features_{args.domain}.npz"
    np.savez_compressed(
        out, file_names=np.array(kept), genus=np.array([labels[n]["genus"] for n in kept]),
        margin_frac=np.array(args.margin_frac),
        **{f"X_{v}": np.concatenate(feats[v]) for v in VARIANTS})
    with (args.out_dir / f"variant_stats_{args.domain}.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(stats_rows[0]))
        w.writeheader()
        w.writerows(stats_rows)
    print(f"{args.domain}: {len(kept)} images -> {out}; {len(missing)} without masks"
          + (f" (e.g. {missing[:3]})" if missing else ""))
    print(f"Example sheets for the first {min(args.examples, len(kept))} images in {example_dir}")


# ------------------------------------------------------------------- evaluate

def exact_mcnemar(a_correct, b_correct):
    """Two-sided exact McNemar p for paired correctness vectors. Returns (a_only, b_only, p)."""
    from scipy.stats import binomtest

    a_only = int(np.sum(a_correct & ~b_correct))
    b_only = int(np.sum(~a_correct & b_correct))
    n = a_only + b_only
    p = binomtest(a_only, n, 0.5).pvalue if n else 1.0
    return a_only, b_only, float(p)


def evaluate(args):
    from scipy.stats import binomtest
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    data = np.load(args.out_dir / f"features_{args.domain}.npz")
    y = data["genus"]
    classes = np.unique(y)
    chance = 1 / len(classes)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)

    results, correct = {}, {}
    for v in VARIANTS:
        X = data[f"X_{v}"]
        clf = make_pipeline(StandardScaler(), PCA(n_components=min(args.pca, X.shape[1]), random_state=args.seed),
                            LogisticRegression(C=args.C, max_iter=5000))
        pred = cross_val_predict(clf, X, y, cv=cv)
        correct[v] = pred == y
        k = int(correct[v].sum())
        test = binomtest(k, len(y), chance, alternative="greater")
        ci = test.proportion_ci(confidence_level=0.95, method="wilson")
        results[v] = {"accuracy": k / len(y), "ci_low": ci.low, "ci_high": ci.high,
                      "balanced_accuracy": balanced_accuracy_score(y, pred), "p_vs_chance": test.pvalue}

    pairs = {}
    for a, b in (("outer", "outer_shape"), ("ring", "ring_shape"), ("fish_ring", "fish")):
        a_only, b_only, p = exact_mcnemar(correct[a], correct[b])
        pairs[f"{a}_vs_{b}"] = {"diff": results[a]["accuracy"] - results[b]["accuracy"],
                                f"only_{a}_right": a_only, f"only_{b}_right": b_only, "p": p}

    control_ok = results["background"]["p_vs_chance"] < 0.05
    leak = pairs["outer_vs_outer_shape"]
    if not control_ok:
        verdict = "UNINFORMATIVE: the probe can't read the background at all, so a clean ring proves nothing"
    elif leak["diff"] > 0 and leak["p"] < 0.05:
        verdict = "LEAK: background near the fish predicts the genus beyond its outline - shrink/blur/noise the margin and re-test"
    else:
        verdict = "SAFE: background near the fish adds no label information beyond the outline"

    print(f"\n{args.domain}: {len(y)} images, {len(classes)} genera, chance = {chance:.1%}\n")
    print(f"{'variant':<12} {'acc':>6} {'95% CI':>15} {'bal.acc':>8} {'p vs chance':>12}")
    for v in VARIANTS:
        r = results[v]
        print(f"{v:<12} {r['accuracy']:>6.1%} {r['ci_low']:>7.1%}-{r['ci_high']:<7.1%} "
              f"{r['balanced_accuracy']:>8.1%} {r['p_vs_chance']:>12.2g}")
    print()
    for name, pr in pairs.items():
        a, b = name.split("_vs_")
        print(f"{name:<22} diff {pr['diff']:+.1%}  ({pr[f'only_{a}_right']} vs {pr[f'only_{b}_right']} "
              f"images right only by one)  McNemar p = {pr['p']:.3g}")
    print(f"\nVerdict: {verdict}")

    out = {"domain": args.domain, "n_images": int(len(y)), "n_genera": int(len(classes)), "chance": chance,
           "margin_frac": float(data["margin_frac"]), "pca": args.pca, "C": args.C, "seed": args.seed,
           "variants": results, "pairs": pairs, "control_ok": bool(control_ok), "verdict": verdict}
    path = args.out_dir / f"results_{args.domain}.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"Wrote {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--seed", type=int, default=0)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("select", help="pick the balanced genus sample")
    p.add_argument("--labels", type=Path, default=WORKSPACE / "data/processed/labels.csv")
    p.add_argument("--per-genus", type=int, default=40,
                   help="specimens per genus; genera with fewer are left out (default: %(default)s)")

    p = sub.add_parser("features", help="build the variants and extract DINOv2 features")
    p.add_argument("--domain", choices=list(DOMAINS), required=True)
    p.add_argument("--run-dir", type=Path, required=True,
                   help="segment_fish.py run dir with masks WITHOUT margin (setting B)")
    p.add_argument("--raw-root", type=Path, default=WORKSPACE / "data/raw/NHM_datensatz")
    p.add_argument("--margin-frac", type=float, default=0.03, help="margin to test (default: %(default)s)")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--examples", type=int, default=12, help="example sheets to save (default: %(default)s)")

    p = sub.add_parser("evaluate", help="cross-validated linear probe per variant + verdict")
    p.add_argument("--domain", choices=list(DOMAINS), required=True)
    p.add_argument("--pca", type=int, default=256)
    p.add_argument("--C", type=float, default=1.0)

    args = parser.parse_args()
    {"select": select, "features": features, "evaluate": evaluate}[args.cmd](args)


if __name__ == "__main__":
    main()
