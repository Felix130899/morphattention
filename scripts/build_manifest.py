"""Write the dataset manifest: one CSV row per image, the input for training.

Every row of labels.csv is in it, also dropped and skipped images, so it is
visible why an image is not used. Columns:

    file_name        <domain>/<name>, as in labels.csv
    sha256           of the raw image bytes; the key git-tracked files may use
                     (file names hold catalog numbers, which stay out of git)
    domain           full_body / Röntgen
    species, genus   species as in labels.csv; genus = the word before any "(Subgenus)"
    catalog_number   as parsed by extract_labels.py
    specimen         specimen group from split_specimens.py (usable images only)
    mask_path        the image's mask PNG in the curated run (usable images only)
    n_instances      number of fish masks
    status, reason   usable / dropped / skipped; reason for dropped and skipped
                     (mask review, no detection, or no catalog number)
    needs_review     from labels.csv
    split            train / val / test / train_only, "none" when not usable
    genus_evaluated  the genus is an evaluated class in this domain

The 3 % mask margin is not applied here; training applies it when building
the images. SHA-256 sums are cached in ``--hash-cache`` by path, size and
modification time, so a rerun only hashes new or changed files.

Output (``--out``, in gitignored data/): manifest.csv, run_config.json

    python scripts/build_manifest.py \\
        --segmented data/processed/segmented/B_merged_2026-10-03 \\
        --split-dir data/processed/split/B_merged_2026-10-03_seed0 \\
        --out data/processed/manifest/B_merged_2026-10-03_seed0
"""

import argparse
import collections
import csv
import datetime
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from segment_fish import git_state  # noqa: E402
from split_specimens import file_sha256, genus_of, load_status, raw_dirs_of, resolve  # noqa: E402

COLUMNS = ["file_name", "sha256", "domain", "species", "genus", "catalog_number", "specimen", "mask_path",
           "n_instances", "status", "reason", "needs_review", "split", "genus_evaluated"]


def build_rows(labels, segmented, split_rows, raw_dirs, hash_cache):
    status = load_status(segmented, raw_dirs)
    rows = []
    for r in labels:
        name, domain = r["file_name"], r["photo_type"]
        state, reason, n_inst = status[name]
        s = split_rows.get(name)
        stem = Path(name).stem
        if state == "ok" and s is None:
            state, reason = "dropped", "no catalog number"
        elif state == "ok":
            state = "usable"
        mask = Path(segmented) / domain / "masks" / f"{stem}.png" if state == "usable" else None
        if mask is not None and not resolve(mask).exists():
            raise SystemExit(f"mask missing for a usable image: {mask}")
        rows.append({
            "file_name": name, "sha256": file_sha256(raw_dirs[domain] / Path(name).relative_to(domain), hash_cache),
            "domain": domain, "species": r["species"], "genus": genus_of(r["species"]) if r["species"] else "",
            "catalog_number": r["catalog_number"], "specimen": s["specimen"] if s else "",
            "mask_path": str(mask) if mask else "", "n_instances": n_inst if state == "usable" else "",
            "status": state, "reason": reason, "needs_review": r["needs_review"],
            "split": s["split"] if s else "none", "genus_evaluated": s["genus_evaluated"] if s else "False"})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--labels", type=Path, default=Path("data/processed/labels.csv"))
    parser.add_argument("--segmented", type=Path, required=True,
                        help="Curated run with one subfolder per domain (full_body, Röntgen).")
    parser.add_argument("--split-dir", type=Path, required=True, help="Output folder of split_specimens.py.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--hash-cache", type=Path, default=Path("data/processed/sha256_cache.json"))
    args = parser.parse_args()

    with open(args.labels, newline="") as f:
        labels = list(csv.DictReader(f))
    with open(args.split_dir / "split.csv", newline="") as f:
        split_rows = {r["file_name"]: r for r in csv.DictReader(f)}
    split_config = json.loads((args.split_dir / "run_config.json").read_text())
    labels_sha = hashlib.sha256(args.labels.read_bytes()).hexdigest()
    if split_config["labels_sha256"] != labels_sha:
        raise SystemExit(f"{args.labels} changed since the split was made; rerun split_specimens.py first")
    if resolve(split_config["segmented"]).resolve() != resolve(args.segmented).resolve():
        raise SystemExit(f"the split was made from {split_config['segmented']}, not {args.segmented}")

    domains = sorted({r["photo_type"] for r in labels})
    raw_dirs = raw_dirs_of(args.segmented, domains)
    cache = json.loads(args.hash_cache.read_text()) if args.hash_cache.exists() else {}
    n_cached = len(cache)
    try:
        rows = build_rows(labels, args.segmented, split_rows, raw_dirs, cache)
    finally:
        if len(cache) != n_cached:
            args.hash_cache.parent.mkdir(parents=True, exist_ok=True)
            args.hash_cache.write_text(json.dumps(cache))

    dup = [h for h, n in collections.Counter(r["sha256"] for r in rows).items() if n > 1]
    args.out.mkdir(parents=True, exist_ok=True)
    with open(args.out / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    counts = collections.Counter((r["domain"], r["status"], r["split"]) for r in rows)
    (args.out / "run_config.json").write_text(json.dumps({
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "labels": str(args.labels), "labels_sha256": labels_sha, "segmented": str(args.segmented),
        "split_dir": str(args.split_dir), "split_settings": split_config["settings"],
        "manifest_sha256": hashlib.sha256((args.out / "manifest.csv").read_bytes()).hexdigest(),
        "rows": len(rows), "duplicate_sha256": len(dup),
        "counts": [{"domain": d, "status": s, "split": sp, "images": n} for (d, s, sp), n in sorted(counts.items())],
        **git_state()}, indent=1, ensure_ascii=False))

    print(f"{len(rows)} rows ({len(cache) - n_cached} images newly hashed)")
    for (d, s, sp), n in sorted(counts.items()):
        print(f"  {d:10s} {s:8s} {sp:10s} {n:6d}")
    if dup:
        print(f"WARNING: {len(dup)} SHA-256 values occur more than once (byte-identical images)")
    print(f"Wrote {args.out}/manifest.csv, run_config.json")


if __name__ == "__main__":
    main()
