"""Pick the stratified dev set for comparing segmentation settings (fin fix).

The dev set is judged blind across several segment_fish.py runs (old baseline
+ new settings) with review_masks.py, see
vault/thesis-log/decisions/2026-10-01-mask-policy.md. It is NOT a random
sample (fix cases are oversampled, strata get fixed quotas), so it must stay
out of the later random review / QA-model acceptance sample.

Rules:
  * at most one image per specimen (catalog number), across both domains,
    so no specimen is judged twice
  * at most ``--max-per-genus`` images per genus, to spread over taxa
  * photos: ``--n-known-fix`` images from the old review's fix.txt (known
    hard cases, mostly cut fins), then quotas per QA flag and per
    view x instance-count stratum
  * X-rays: quotas per QA flag and per view x instance-count stratum
  * only images the baseline run has an overlay for (skipped images can't be
    compared)

Writes to ``--out-dir``: ``photos.txt`` and ``xrays.txt`` (one file name per
line, relative to the domain's raw dir, as used by ``--image-list``) and
``devset.csv`` (file name, domain, stratum, species, catalog number,
instances, flags).

    python scripts/make_devset.py
"""

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path

WORKSPACE = Path("/workspace")

# (flag, quota) are drawn first, then (view, instance bucket, quota).
PHOTO_FLAG_QUOTAS = [("giant", 2), ("tiny", 2), ("overlap_resolved", 3), ("duplicates_removed", 3)]
PHOTO_STRATA = [
    ("lateral", "1", 14), ("lateral", "2-3", 8), ("lateral", "4+", 8),
    ("dorsal", "1", 5), ("dorsal", "2+", 5),
    ("ventral", "1", 5), ("ventral", "2+", 5),
    ("head", "1", 6), ("head", "2+", 4),
]
XRAY_FLAG_QUOTAS = [("giant", 3), ("tiny", 1), ("overlap_resolved", 2), ("duplicates_removed", 2)]
XRAY_STRATA = [
    ("lateral", "1", 14), ("lateral", "2-3", 12), ("lateral", "4+", 11),
    ("dorsal", "any", 5),
]


def view_of(extra):
    """Coarse view from the filename tail: head > dorsal > ventral > lateral."""
    e = extra.lower()
    for view in ("head", "dorsal", "ventral"):
        if view in e:
            return view
    return "lateral"


def bucket_matches(bucket, n):
    return {"1": n == 1, "2-3": 2 <= n <= 3, "4+": n >= 4, "2+": n >= 2, "any": True}[bucket]


def load_domain(run_dir, labels):
    """Candidate records for one domain: images with an overlay in ``run_dir``."""
    coco = json.loads((run_dir / "coco" / "annotations.json").read_text())
    n_inst = Counter(a["image_id"] for a in coco["annotations"])
    records = []
    for img in coco["images"]:
        name = img["file_name"]
        label = labels.get((run_dir.name, name))
        if label is None or not (run_dir / "overlays" / name).exists():
            continue
        records.append({
            "file_name": name,
            "domain": run_dir.name,
            "species": label["species"],
            "genus": label["species"].split()[0] if label["species"] else "?",
            "catalog_number": label["catalog_number"],
            "view": view_of(label["extra"]),
            "instances": n_inst[img["id"]],
            "flags": img.get("qa", {}).get("flags", []),
        })
    return records


class Picker:
    """Draws images while enforcing the specimen and genus limits globally."""

    def __init__(self, rng, max_per_genus):
        self.rng = rng
        self.max_per_genus = max_per_genus
        self.specimens = set()
        self.genera = Counter()
        self.picked = []

    def allowed(self, rec):
        return (rec["catalog_number"] not in self.specimens
                and self.genera[rec["genus"]] < self.max_per_genus)

    def draw(self, pool, n, stratum):
        pool = list(pool)
        self.rng.shuffle(pool)
        got = 0
        for rec in pool:
            if got == n:
                break
            if self.allowed(rec):
                self.specimens.add(rec["catalog_number"])
                self.genera[rec["genus"]] += 1
                self.picked.append({**rec, "stratum": stratum})
                got += 1
        if got < n:
            print(f"  warning: stratum {stratum!r} only got {got}/{n}")


def pick_domain(picker, records, flag_quotas, strata):
    for flag, n in flag_quotas:
        picker.draw([r for r in records if flag in r["flags"]], n, f"flag:{flag}")
    for view, bucket, n in strata:
        picker.draw([r for r in records if r["view"] == view and bucket_matches(bucket, r["instances"])],
                    n, f"{view}:{bucket}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--labels", type=Path, default=WORKSPACE / "data/processed/labels.csv")
    parser.add_argument("--photo-run", type=Path, default=WORKSPACE / "data/processed/segmented/full_body")
    parser.add_argument("--xray-run", type=Path, default=WORKSPACE / "data/processed/segmented/Röntgen")
    parser.add_argument("--out-dir", type=Path, default=WORKSPACE / "data/processed/devset_fin_fix")
    parser.add_argument("--n-known-fix", type=int, default=30,
                        help="photos drawn from the old review's fix.txt (default: %(default)s)")
    parser.add_argument("--max-per-genus", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    labels = {}
    with args.labels.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            folder, name = row["file_name"].split("/", 1)
            labels[(folder, name)] = row

    rng = random.Random(args.seed)
    picker = Picker(rng, args.max_per_genus)

    photos = load_domain(args.photo_run, labels)
    known_fix = set((args.photo_run / "review" / "fix.txt").read_text().split("\n"))
    picker.draw([r for r in photos if r["file_name"] in known_fix], args.n_known_fix, "known_fix")
    pick_domain(picker, photos, PHOTO_FLAG_QUOTAS, PHOTO_STRATA)

    xrays = load_domain(args.xray_run, labels)
    pick_domain(picker, xrays, XRAY_FLAG_QUOTAS, XRAY_STRATA)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for domain, out_name in ((args.photo_run.name, "photos.txt"), (args.xray_run.name, "xrays.txt")):
        names = sorted(r["file_name"] for r in picker.picked if r["domain"] == domain)
        (args.out_dir / out_name).write_text("\n".join(names) + "\n", encoding="utf-8")
    fields = ["file_name", "domain", "stratum", "species", "catalog_number", "view", "instances", "flags"]
    with (args.out_dir / "devset.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for rec in picker.picked:
            writer.writerow({**rec, "flags": " ".join(rec["flags"])})

    by_domain = Counter(r["domain"] for r in picker.picked)
    print(f"Picked {len(picker.picked)} images {dict(by_domain)}, "
          f"{len(picker.genera)} genera, {len(picker.specimens)} specimens -> {args.out_dir}")


if __name__ == "__main__":
    main()
