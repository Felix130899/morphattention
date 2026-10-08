"""Which masks changed between two segment_fish.py runs of the same images, and why.

Per image, the IoU of the union masks ``<run>/masks/<stem>.png`` (0/255 PNGs at
original size) of the ``--before`` and ``--after`` run; an image is "changed"
when IoU < ``--threshold``. Only images that are status ok in both runs are
compared; the ones ok in only one run are counted separately. Also counts how
often each setting-D rule fired in the after run (instance fields
``bleed_fallback``, ``wrong_region``, ``far_pieces_dropped``,
``holes_filled_px``, ``fin_extension``; image field ``qa.wrong_region_dropped``).

Writes into ``--after`` (nothing else is changed):
    changed_vs_<name>.tsv     every compared image: file_name, iou, pixels before/after, rules fired
    changed_vs_<name>.txt     the changed file names (a review_masks.py --image-list)
    changed_vs_<name>.json    the counts printed at the end
    changed_vs_<name>_pick<N>_seed<S>.txt   with --pick: N seeded random changed images,
                                            for a blind review_masks.py --compare

    python scripts/mask_changes.py --name B \\
        --before data/processed/segmented/B_merged_2026-10-03/full_body \\
        --after data/processed/segmented/D_merged_2026-10-05/full_body --pick 100 --seed 0
"""

import argparse
import json
import random
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None  # the NHM scans are large but trusted

FALLBACK_SOURCES = ("candidate", "inverse", "unresolved")
WRONG_REGION_SOURCES = ("reprompt", "unresolved")


def read_ok_records(run_dir):
    """{file_name: record} of the status-ok images of a run."""
    records = {}
    for line in (Path(run_dir) / "annotations.jsonl").read_text().splitlines():
        if line.strip():
            rec = json.loads(line)
            if rec["status"] == "ok":
                records[rec["file_name"]] = rec
    return records


def mask_iou(a, b):
    """IoU of two boolean masks; two empty masks count as identical (1.0)."""
    if a.shape != b.shape:
        raise ValueError(f"mask sizes differ: {a.shape} vs {b.shape}")
    union = np.logical_or(a, b).sum()
    return 1.0 if union == 0 else float(np.logical_and(a, b).sum() / union)


def load_mask(path):
    return np.asarray(Image.open(path).convert("L")) > 127


def compare_one(job):
    """(file_name, iou, pixels before, pixels after) of one image."""
    name, before_png, after_png = job
    a, b = load_mask(before_png), load_mask(after_png)
    return name, mask_iou(a, b), int(a.sum()), int(b.sum())


def rules_fired(rec):
    """Rules (setting D + fin extension) that fired in this record, e.g. ['bleed_inverse', 'holes_filled'].

    ``fin_guarded`` changed nothing: the guard kept the mask as it was.
    """
    fired = set()
    for inst in rec.get("instances", []):
        if inst.get("bleed_fallback") in FALLBACK_SOURCES:
            fired.add(f"bleed_{inst['bleed_fallback']}")
        if inst.get("wrong_region") in WRONG_REGION_SOURCES:
            fired.add(f"wrong_region_{inst['wrong_region']}")
        if inst.get("far_pieces_dropped", 0) > 0:
            fired.add("far_pieces_dropped")
        if inst.get("holes_filled_px", 0) > 0:
            fired.add("holes_filled")
        if inst.get("fin_extension") == "extended" and inst.get("fin_growth", 0) > 0:
            fired.add("fin_extended")
        elif inst.get("fin_extension") == "guarded":
            fired.add("fin_guarded")
    if rec.get("qa", {}).get("wrong_region_dropped", 0) > 0:
        fired.add("wrong_region_dropped")
    return sorted(fired)


def summarize(rows, before, after, threshold):
    """Counts for the json/printout. rows: [(name, iou, px_before, px_after, rules)]."""
    changed = [r for r in rows if r[1] < threshold]
    rule_images = Counter(rule for r in rows for rule in r[4])
    rule_changed = Counter(rule for r in changed for rule in r[4])
    return {
        "threshold": threshold,
        "compared": len(rows),
        "changed": len(changed),
        "changed_frac": round(len(changed) / len(rows), 4) if rows else None,
        "changed_without_rule": sum(not r[4] for r in changed),
        "ok_only_before": len(set(before) - set(after)),
        "ok_only_after": len(set(after) - set(before)),
        "images_with_rule": sum(bool(r[4]) for r in rows),
        "rule_images": dict(sorted(rule_images.items())),
        "rule_images_changed": dict(sorted(rule_changed.items())),
        "iou_changed_quartiles": ([round(float(q), 4) for q in np.quantile([r[1] for r in changed], [.25, .5, .75])]
                                  if changed else None),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--before", type=Path, required=True, help="earlier run dir (e.g. B_merged/<domain>)")
    parser.add_argument("--after", type=Path, required=True, help="new run dir; the outputs go here")
    parser.add_argument("--name", default="before", help="short name of --before for the output files (e.g. B)")
    parser.add_argument("--threshold", type=float, default=0.99, help="changed = IoU below this (default 0.99)")
    parser.add_argument("--pick", type=int, metavar="N", help="also write N seeded random changed images")
    parser.add_argument("--seed", type=int, default=0, help="seed for --pick (default 0)")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    before, after = read_ok_records(args.before), read_ok_records(args.after)
    common = sorted(set(before) & set(after))
    jobs = [(n, args.before / "masks" / f"{Path(n).stem}.png", args.after / "masks" / f"{Path(n).stem}.png")
            for n in common]
    with Pool(args.workers) as pool:
        results = pool.map(compare_one, jobs, chunksize=16)
    rows = [(name, iou, pa, pb, rules_fired(after[name])) for name, iou, pa, pb in results]

    stem = args.after / f"changed_vs_{args.name}"
    with open(f"{stem}.tsv", "w") as f:
        f.write("file_name\tiou\tpx_before\tpx_after\trules\n")
        for name, iou, pa, pb, rules in rows:
            f.write(f"{name}\t{iou:.4f}\t{pa}\t{pb}\t{','.join(rules)}\n")
    changed = [r[0] for r in rows if r[1] < args.threshold]
    Path(f"{stem}.txt").write_text(f"# IoU < {args.threshold} vs {args.before}\n" + "".join(n + "\n" for n in changed))
    summary = {"before": str(args.before), "after": str(args.after),
               **summarize(rows, before, after, args.threshold)}
    if args.pick is not None:
        if args.pick > len(changed):
            raise SystemExit(f"--pick {args.pick} but only {len(changed)} changed images")
        picked = sorted(changed)
        random.Random(args.seed).shuffle(picked)
        pick_path = Path(f"{stem}_pick{args.pick}_seed{args.seed}.txt")
        pick_path.write_text(f"# {args.pick} random changed images (seed {args.seed}) of {stem}.txt\n"
                             + "".join(n + "\n" for n in sorted(picked[:args.pick])))
        summary["pick"] = {"n": args.pick, "seed": args.seed, "file": str(pick_path)}
    Path(f"{stem}.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
