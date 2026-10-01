"""Flag suspected whole-background bleed in a segment_fish.py run.

Some SAM masks swallow almost the entire background, including the printed
species names and catalog numbers, which is the Clever Hans leak masking is
meant to remove (vault/thesis-log/decisions/2026-10-01-mask-policy.md: bleed
is never OK). The per-instance ``giant`` flag (> 90 % of the image) only
catches the worst of them. A fish almost never covers most of the image, so:

    suspect = total mask coverage (all instances) > --max-cover
              AND at least one instance touches the image border

Tightly cropped X-rays can legitimately exceed the coverage threshold, so the
list is for review, not for automatic exclusion.

Writes ``<run-dir>/suspect_bleed.txt`` (file names, highest coverage first;
usable as ``review_masks.py --image-list``) and ``suspect_bleed.csv``
(file name, coverage, instances, flags).

    python scripts/flag_bleed.py --run-dir data/processed/segmented/B_full_2026-10-01/full_body
    python scripts/review_masks.py --run-dir data/processed/segmented/B_full_2026-10-01/full_body \\
        --image-list data/processed/segmented/B_full_2026-10-01/full_body/suspect_bleed.txt
"""

import argparse
import csv
import json
from pathlib import Path


def coverage(record):
    """Fraction of the image covered by all instances together (instances are disjoint)."""
    return sum(i["area"] for i in record["instances"]) / (record["width"] * record["height"])


def is_suspect(record, max_cover):
    return (coverage(record) > max_cover
            and any(i.get("touches_border") for i in record["instances"]))


def flag_run(run_dir, max_cover):
    """Return [(file_name, coverage, n_instances, flags)] of suspects, highest coverage first."""
    suspects = []
    with (run_dir / "annotations.jsonl").open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("status") == "ok" and is_suspect(rec, max_cover):
                suspects.append((rec["file_name"], coverage(rec), len(rec["instances"]), rec["qa"]["flags"]))
    return sorted(suspects, key=lambda s: -s[1])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, required=True, action="append",
                        help="segment_fish.py run dir (repeatable)")
    parser.add_argument("--max-cover", type=float, default=0.7,
                        help="coverage above which a border-touching mask is suspect (default: %(default)s)")
    args = parser.parse_args()

    for run_dir in args.run_dir:
        suspects = flag_run(run_dir, args.max_cover)
        (run_dir / "suspect_bleed.txt").write_text("".join(s[0] + "\n" for s in suspects), encoding="utf-8")
        with (run_dir / "suspect_bleed.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["file_name", "coverage", "n_instances", "flags"])
            for name, cov, n, flags in suspects:
                w.writerow([name, round(cov, 4), n, " ".join(flags)])
        print(f"{run_dir}: {len(suspects)} suspect image(s) (coverage > {args.max_cover:.0%} + border) "
              f"-> suspect_bleed.txt / .csv")


if __name__ == "__main__":
    main()
