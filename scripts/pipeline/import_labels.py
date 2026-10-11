"""Labels from your own CSV (the clean route): one row per image -> labels CSV.

The other route, ``extract_labels.py``, parses NHM Wien file names; this one
reads a table you made. Both write the labels CSV the later steps read
(split_specimens.py, build_manifest.py).

Your CSV (``--csv``), comma-separated, UTF-8, with a header:

    file_name     path of the image relative to --raw-dir, e.g. photos/a.jpg;
                  the first folder is the domain
    <specimen>    the specimen (one physical object, e.g. one fish) the image
                  shows: images of one specimen always land in the same split.
                  The column name is ``--specimen`` (default "specimen")
    any others    your labels, e.g. species, origin; any of them can be the
                  split's class (split_specimens.py --class)

Every image under --raw-dir must have exactly one row and every row an image;
otherwise nothing is written and the mismatches are listed. ``--specimen none``
means you have no specimen information: every image is then its own specimen
(id "image:<sha256>"), so photos of one fish can end up in train and test;
split_specimens.py warns about it.

``--decisions DIR`` applies your fixes on top, both keyed so no file names
are needed (copies of the empty files in pipeline/decisions_template/labels/):

  1. ``value_renames.tsv`` (column, from, to, decision): every row with
     decision "accept" replaces the value ``from`` by ``to`` in that column
     (spellings, synonyms); an empty decision = kept apart. ``from`` must
     occur in the column
  2. ``image_fixes.tsv`` (sha256, note, decision): per image (SHA-256 of its
     bytes; byte-identical copies all get it) "drop", or "set <column>
     <value>", several joined with "; "
  3. byte-identical images (same SHA-256): the alphabetically first file name
     is kept, the others are dropped

Output: the labels CSV with file_name, photo_type (the domain), specimen_id,
your other columns in their order, label_change (what was changed) and
label_drop (why the image is not used; empty = used). Dropped rows stay in the
CSV, so the row count still matches the image count.

    python scripts/pipeline/import_labels.py --raw-dir data/raw/my_dataset --csv data/raw/my_dataset/labels.csv \\
        --decisions pipeline/decisions_my_dataset/labels --out-csv data/processed/labels_my_dataset.csv
"""

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_labels import iter_images, read_sheet  # noqa: E402
from split_specimens import file_sha256  # noqa: E402

RESERVED = ("photo_type", "specimen_id", "label_change", "label_drop")
MAX_LISTED = 10


def read_table(path, specimen):
    """Rows of the user's CSV (values stripped), with checks on the columns."""
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        columns = reader.fieldnames or []
        rows = [{k: (v or "").strip() for k, v in r.items()} for r in reader]
    problems = []
    if "file_name" not in columns:
        problems.append("no file_name column")
    if specimen != "none" and specimen not in columns:
        problems.append(f"no {specimen!r} column (the specimen; --specimen none if you have no specimen information)")
    problems += [f"column {c!r} is reserved, rename it" for c in columns if c in RESERVED]
    problems += [f"column {c!r} twice" for c, n in Counter(columns).items() if n > 1]
    if None in columns or "" in columns:
        problems.append("a column without a name")
    if problems:
        raise SystemExit(f"{path}: " + "; ".join(problems))
    return columns, rows


def listed(names):
    names = sorted(names)
    return ", ".join(names[:MAX_LISTED]) + (f" (+{len(names) - MAX_LISTED} more)" if len(names) > MAX_LISTED else "")


def match_images(rows, images, raw_dir):
    """{file_name: row}; stops if a row has no image, an image no row, or a name is listed twice."""
    names = [r["file_name"] for r in rows]
    on_disk = {p.relative_to(raw_dir).as_posix() for p in images}
    twice = {n for n, k in Counter(names).items() if k > 1}
    no_image = set(names) - on_disk
    no_row = on_disk - set(names)
    flat = {n for n in on_disk if "/" not in n}
    problems = [f"{len(twice)} file name(s) listed twice: {listed(twice)}" if twice else "",
                f"{len(no_image)} row(s) without an image under {raw_dir}: {listed(no_image)}" if no_image else "",
                f"{len(no_row)} image(s) without a row: {listed(no_row)}" if no_row else "",
                f"{len(flat)} image(s) directly under {raw_dir}, not in a domain folder: {listed(flat)}" if flat else ""]
    problems = [p for p in problems if p]
    if problems:
        raise SystemExit("labels CSV and images don't match, nothing written:\n  " + "\n  ".join(problems))
    return {r["file_name"]: r for r in rows}


def apply_fixes(rows, columns, decisions_dir, sha):
    """Apply value_renames.tsv and image_fixes.tsv, then keep one of each set of byte-identical
    images. ``rows``: output rows (dicts) in file order; ``columns``: the columns fixes may change
    (the CSV's, without file_name); ``sha``: {file_name: sha256}. Returns a Counter."""
    done = Counter()
    if decisions_dir is not None:
        for d in read_sheet(Path(decisions_dir) / "value_renames.tsv"):
            if d["decision"] not in ("accept", ""):
                raise SystemExit(f"value_renames.tsv: unknown decision {d['decision']!r} for {d['column']} {d['from']!r}")
            if d["column"] not in columns:
                raise SystemExit(f"value_renames.tsv: {d['column']!r} is not a label column of the CSV")
            hits = [r for r in rows if r[d["column"]] == d["from"]]
            if not hits:
                raise SystemExit(f"value_renames.tsv: no image has {d['column']} = {d['from']!r}")
            if d["decision"] == "accept":
                for r in hits:
                    r[d["column"]] = d["to"]
                    r["label_change"].append(f"{d['column']} {d['from']} -> {d['to']}")
                    done["renamed"] += 1
        by_sha = defaultdict(list)
        for r in rows:
            by_sha[sha[r["file_name"]]].append(r)
        for d in read_sheet(Path(decisions_dir) / "image_fixes.tsv"):
            if d["sha256"] not in by_sha:
                raise SystemExit(f"image_fixes.tsv: no image with sha256 {d['sha256']}")
            parts = [p.strip() for p in d["decision"].split(";")]
            for part in parts:
                m = re.fullmatch(r"set (\S+) (\S.*)", part)
                if part == "drop" and len(parts) == 1:
                    for r in by_sha[d["sha256"]]:
                        r["label_drop"] = f"image fix: {d['note'] or 'drop'}"
                        done["dropped_image_fix"] += 1
                elif m and m.group(1) in columns:
                    for r in by_sha[d["sha256"]]:
                        r[m.group(1)] = m.group(2).strip()
                        r["label_change"].append(part)
                        done["values_set"] += 1
                else:
                    raise SystemExit(f"image_fixes.tsv: unknown decision {part!r} in {d['decision']!r}")
    by_sha = defaultdict(list)
    for r in rows:
        if not r["label_drop"]:
            by_sha[sha[r["file_name"]]].append(r)
    for copies in by_sha.values():
        if len(copies) < 2:
            continue
        done["duplicate_sets"] += 1
        keep = min(copies, key=lambda r: r["file_name"])
        for r in copies:
            if r is not keep:
                r["label_drop"] = "byte-identical copy of an image kept under another file name"
                done["dropped_duplicate"] += 1
    return done


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-dir", type=Path, required=True, help="Images, one subfolder per domain.")
    parser.add_argument("--csv", type=Path, required=True, help="Your labels CSV (one row per image).")
    parser.add_argument("--specimen", default="specimen",
                        help='Column with the specimen id (default "specimen"), or "none": every image its own specimen.')
    parser.add_argument("--decisions", type=Path, help="Folder with value_renames.tsv and image_fixes.tsv.")
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--hash-cache", type=Path, default=Path("data/processed/sha256_cache.json"),
                        help="SHA-256 cache shared with split_specimens.py and build_manifest.py.")
    args = parser.parse_args()
    raw_dir = args.raw_dir.resolve()  # absolute, so the hash cache keys match split_specimens.py's

    columns, table = read_table(args.csv, args.specimen)
    images = list(iter_images(raw_dir))
    by_name = match_images(table, images, raw_dir)
    cache = json.loads(args.hash_cache.read_text()) if args.hash_cache.exists() else {}
    n_cached = len(cache)
    try:
        sha = {p.relative_to(raw_dir).as_posix(): file_sha256(p, cache) for p in images}
    finally:
        if len(cache) != n_cached:
            args.hash_cache.parent.mkdir(parents=True, exist_ok=True)
            args.hash_cache.write_text(json.dumps(cache))

    spec_col = None if args.specimen == "none" else args.specimen
    editable = [c for c in columns if c != "file_name"]  # fixes name the columns as in the CSV
    rows = []
    for p in images:
        name = p.relative_to(raw_dir).as_posix()
        rows.append({"file_name": name, "photo_type": name.split("/", 1)[0],
                     **{c: by_name[name][c] for c in editable}, "label_change": [], "label_drop": ""})
    done = apply_fixes(rows, editable, args.decisions, sha)
    for r in rows:
        r["specimen_id"] = r.pop(spec_col) if spec_col else f"image:{sha[r['file_name']]}"
        r["label_change"] = "; ".join(r["label_change"])
    labels = [c for c in editable if c != spec_col]

    out_cols = ["file_name", "photo_type", "specimen_id", *labels, "label_change", "label_drop"]
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=out_cols)
        w.writeheader()
        w.writerows(rows)

    print(f"{len(rows)} images under {raw_dir} ({len(cache) - n_cached} newly hashed), one row each in {args.csv}")
    if args.decisions is not None:
        print(f"Applied the fixes in {args.decisions}:")
    for key in ("renamed", "values_set", "dropped_image_fix", "duplicate_sets", "dropped_duplicate"):
        print(f"  {key}: {done[key]}")
    if args.specimen == "none":
        print("WARNING: no specimen column (--specimen none): every image is its own specimen, so images of one "
              "specimen can land in different splits")
    no_spec = sum(not r["specimen_id"] for r in rows if not r["label_drop"])
    if no_spec:
        print(f"Note: {no_spec} used image(s) have an empty {args.specimen}; split_specimens.py leaves them out")
    print("Rows by domain: " + ", ".join(f"{d} {n}" for d, n in sorted(Counter(r['photo_type'] for r in rows).items())))
    print(f"Wrote {args.out_csv}")


if __name__ == "__main__":
    main()
