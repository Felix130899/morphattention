"""Parse species labels out of NHM Wien filenames into a structured CSV.

Expected filename convention (per user, origin encoding not yet confirmed
with Neo/supervisor - see the clean-segmented-dataset task in
vault/thesis-log/tasks/):

    <species_name>_NMW<catalog_number>_<extra stuff>.<ext>

``species_name`` may itself contain underscores between words (e.g.
``Salmo_trutta``). ``NMW<catalog_number>`` is the museum's specimen catalog
number - but against the real NHM_datensatz export, "catalog_number" is not
just a plain integer: syntype/paratype series use ranges ("NMW1234-6"),
comma lists ("NMW12340-45,48-53"), and parenthetical sub-indices glued on
with no separator ("NMW98765(54321_2)"). A few filenames also use "MNW"
instead of "NMW" (a typo in the source data, not fixed silently here).

So the parser only anchors on the literal "_NMW"/"_MNW" that splits species
from everything after it, then peels off a leading run of digits as
``catalog_number`` for convenience. Anything beyond that - the range/list
tail, parenthetical qualifiers, "syntypes"/"left"/"WEB" suffixes - is NOT
assumed to be origin (that mapping isn't confirmed yet) and is kept
verbatim in ``extra`` for manual review instead of being guessed at.

Writes one CSV row per image, even when parsing fails (the row is flagged
via ``needs_review``/``review_reason`` instead of being silently dropped),
so the row count always matches the image count - this lets a later step
cross-check labels.csv against the image set.

Run inside the container, e.g.:

    docker compose run --rm vit-project python scripts/extract_labels.py
    docker compose run --rm vit-project python scripts/extract_labels.py \
        --raw-dir /workspace/data/raw/nmw_specimens \
        --out-csv /workspace/data/processed/labels.csv

``--decisions DIR`` applies Neo's label decision sheets (tab-separated,
tracked in ``pipeline/decisions/labels/``; no file names or catalog numbers in
them) on top of the parse, in this order:

  1. ``genus_spelling.tsv``: every row with decision "accept" respells that
     genus in every species name (an empty decision = rejected, name kept)
  2. ``catalog_conflicts.tsv``: per set of names sharing a catalog number,
     "rename A -> B" renames species A to B everywhere (a synonym), "rename
     in catalog A -> B" only in those catalogs (a re-identification), "drop
     A" drops the images of species A in those catalogs only, "keep" changes
     nothing.
     The catalog sets are recomputed (after step 1, on the parsed catalog
     number) and must match the sheet's catalog and image counts
  3. ``needs_review.tsv``: per image (key: SHA-256 of its bytes), "set
     catalog NMW<number>", "keep, own specimen group XYZ<number>" (another
     museum's number, its own specimen group), or "drop"; ``<number>`` is
     read from the file name
  4. byte-identical images (same SHA-256): one copy is kept, the one whose
     file name already carries its final species name, else the
     alphabetically first; the others are dropped

Two columns are added: ``label_change`` (what was changed) and
``label_drop`` (why the image is not used; empty = used). Dropped rows stay
in the CSV, so the row count still matches the image count.

    python scripts/extract_labels.py --raw-dir data/raw/NHM_datensatz \\
        --decisions pipeline/decisions/labels --out-csv data/processed/labels_final.csv
"""

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from split_specimens import file_sha256, genus_of  # noqa: E402

# Pillow can decode all of these; segment_fish.py filters the same way.
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}

# <species>_NMW<anything> - species is greedy since it may contain
# underscores itself; the literal "_NMW"/"_MNW" anchors the split point.
# Catalog-number *structure* (digits vs ranges vs parens) is handled
# separately below, since it varies too much to fold into one regex.
FILENAME_RE = re.compile(r"^(?P<species>.+)_(?P<prefix>NMW|MNW)(?P<catalog_raw>.*)$")

# Leading digit run of catalog_raw, e.g. "98765" out of "98765(54321_2)".
LEADING_DIGITS_RE = re.compile(r"^(\d+)(.*)$")


def iter_images(root: Path):
    """Yield every image file under ``root`` (recursively), sorted for determinism."""
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            yield path


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
            print(f"Please enter a number between 1 and {len(subdirs)}")
        except ValueError:
            print("Invalid input. Please enter a valid number.")


def parse_filename(stem: str):
    """Split a filename stem into species/catalog_number/extra.

    Returns a dict. ``needs_review`` is True (with a ``review_reason``) when
    something about the stem doesn't fit the expected shape - no "_NMW"/
    "_MNW" anchor at all, or a prefix with no digits after it - so the row
    still gets written (never silently dropped) but is easy to filter to for
    manual inspection.
    """
    match = FILENAME_RE.match(stem)
    if not match:
        return {
            "species": None,
            "species_raw": None,
            "catalog_number": None,
            "extra": stem,
            "needs_review": True,
            "review_reason": "no_catalog_anchor",
        }

    species_raw = match.group("species")
    prefix = match.group("prefix")
    # A few filenames have an underscore between the prefix and the digits
    # (e.g. "NMW_12345") instead of none (e.g. "NMW12345") - tolerate both.
    catalog_raw = match.group("catalog_raw").lstrip("_")

    digits_match = LEADING_DIGITS_RE.match(catalog_raw)
    if not digits_match:
        # Prefix found ("_NMW"/"_MNW") but nothing numeric right after it,
        # e.g. "..._NoNumber_WEB". Species is still trustworthy; catalog
        # number isn't.
        return {
            "species": species_raw.replace("_", " ").strip(),
            "species_raw": species_raw,
            "catalog_number": None,
            "extra": catalog_raw,
            "needs_review": True,
            "review_reason": "no_digits_after_prefix",
        }

    digits, remainder = digits_match.groups()
    return {
        "species": species_raw.replace("_", " ").strip(),
        "species_raw": species_raw,
        "catalog_number": f"{prefix}{digits}",
        "extra": remainder.lstrip("_"),
        "needs_review": prefix != "NMW",
        "review_reason": "catalog_prefix_typo" if prefix != "NMW" else "",
    }


def read_sheet(path):
    with open(path, newline="") as f:
        return [{k: (v or "").strip() for k, v in r.items()} for r in csv.DictReader(f, delimiter="\t")]


def respell(species, spelling):
    """Species with its genus (first word) replaced via ``spelling`` {old: new}."""
    genus = genus_of(species)
    return spelling[genus] + species[len(genus):] if genus in spelling else species


def conflict_names(cell):
    """Species names of a catalog_conflicts.tsv "names (images)" cell, e.g.
    "Rutilus (from S plotizza) (1) | Scardinius plotizza (2)" -> both names."""
    return tuple(re.sub(r"\s*\(\d+\)$", "", part.strip()) for part in cell.split(" | "))


# The number a needs_review decision's "<number>" stands for: the first run of
# >= 4 digits after "_", optionally behind a (mistyped or foreign) museum
# prefix, e.g. "_MNW12345", "_MW12345", "_12345_", "_NRM(Stockholm)12345".
CATALOG_NUMBER_RE = re.compile(r"_(?:NMW|MNW|MW|NRM(?:\([^)]*\))?)?_?(\d{4,})")


def apply_decisions(rows, decisions_dir, sha):
    """Apply the three decision sheets + the duplicate rule to labels rows (dicts
    as written by main(), species not None for parsed rows) in place. ``sha``:
    {file_name: sha256}. Returns a Counter of what was done."""
    done = Counter()
    for r in rows:
        r["label_change"], r["label_drop"] = [], ""

    # 1. genus spellings
    spelling = {}
    for d in read_sheet(Path(decisions_dir) / "genus_spelling.tsv"):
        if d["decision"] not in ("accept", ""):
            raise SystemExit(f"genus_spelling.tsv: unknown decision {d['decision']!r} for {d['from_genus']}")
        if d["decision"] == "accept":
            spelling[d["from_genus"]] = d["to_genus"]
    for r in rows:
        if r["species"] and genus_of(r["species"]) in spelling:
            old = genus_of(r["species"])
            r["species"] = respell(r["species"], spelling)
            r["label_change"].append(f"genus {old} -> {spelling[old]}")
            done["genus_respelled"] += 1

    # 2. catalog conflicts: recompute the sets of names per catalog number, as on the sheet
    by_cat = defaultdict(list)
    for r in rows:
        if r["catalog_number"] and r["species"]:
            by_cat[r["catalog_number"]].append(r)
    sets = defaultdict(list)  # sorted names -> catalogs
    for cat, rs in by_cat.items():
        if len({genus_of(r["species"]) for r in rs}) > 1:
            sets[tuple(sorted({r["species"] for r in rs}))].append(cat)
    renames, drops, local_renames = {}, [], []
    sheet = read_sheet(Path(decisions_dir) / "catalog_conflicts.tsv")
    unmatched = set(sets) - {conflict_names(d["names (images)"]) for d in sheet}
    if unmatched:
        raise SystemExit(f"catalog conflicts not on the sheet: {sorted(unmatched)}")
    for d in sheet:
        names = conflict_names(d["names (images)"])
        cats = sets.get(names, [])
        n_img = sum(len(by_cat[c]) for c in cats)
        if (len(cats), n_img) != (int(d["catalogs"]), int(d["images"])):
            raise SystemExit(f"catalog_conflicts.tsv {names}: sheet says {d['catalogs']} catalogs / {d['images']} "
                             f"images, labels give {len(cats)} / {n_img}")
        m = re.fullmatch(r"rename (in catalog )?(.+) -> (.+)", d["decision"])
        if m:
            if m.group(2) not in names:
                raise SystemExit(f"catalog_conflicts.tsv: rename of {m.group(2)!r}, not one of {names}")
            if m.group(1):
                local_renames.append((m.group(2), m.group(3), cats))
            else:
                renames[m.group(2)] = m.group(3)
        elif d["decision"].startswith("drop "):
            name = d["decision"][len("drop "):]
            if name not in names:
                raise SystemExit(f"catalog_conflicts.tsv: drop of {name!r}, not one of {names}")
            drops.append((name, cats))
        elif d["decision"] != "keep":
            raise SystemExit(f"catalog_conflicts.tsv: unknown decision {d['decision']!r} for {names}")
    for name, cats in drops:
        for c in cats:
            for r in by_cat[c]:
                if r["species"] == name:
                    r["label_drop"] = f"catalog conflict: {name} in a catalog of another genus"
                    done["dropped_catalog_conflict"] += 1
    for old, new, cats in local_renames:
        for c in cats:
            for r in by_cat[c]:
                if r["species"] == old:
                    r["species"] = new
                    r["label_change"].append(f"rename in catalog {old} -> {new}")
                    done["renamed"] += 1
    for r in rows:
        if r["species"] in renames:
            r["label_change"].append(f"rename {r['species']} -> {renames[r['species']]}")
            r["species"] = renames[r["species"]]
            done["renamed"] += 1

    # 3. needs_review rows, keyed by the image's SHA-256
    by_sha = defaultdict(list)
    for r in rows:
        by_sha[sha[r["file_name"]]].append(r)
    for d in read_sheet(Path(decisions_dir) / "needs_review.tsv"):
        if len(by_sha.get(d["sha256"], [])) != 1:
            raise SystemExit(f"needs_review.tsv: {len(by_sha.get(d['sha256'], []))} images with sha256 {d['sha256']}")
        r = by_sha[d["sha256"]][0]
        dec = d["decision"]
        m = re.fullmatch(r"set catalog (NMW)<number>|keep, own specimen group ([A-Z]+)<number>", dec)
        if m:
            num = CATALOG_NUMBER_RE.search(Path(r["file_name"]).stem)
            if not num:
                raise SystemExit(f"needs_review.tsv: no catalog number in the file name of {d['sha256']}")
            r["catalog_number"] = f"{m.group(1) or m.group(2)}{num.group(1)}"
        elif dec == "drop":
            r["label_drop"] = f"needs_review: {d['note']}"
            done["dropped_needs_review"] += 1
        else:
            raise SystemExit(f"needs_review.tsv: unknown decision {dec!r}")
        r["needs_review"], r["review_reason"] = False, f"resolved: {dec}"
        r["label_change"].append(f"resolved: {dec}")
        done["needs_review_resolved"] += 1

    # 4. byte-identical copies: keep one per set
    by_sha = defaultdict(list)
    for r in rows:
        if not r["label_drop"]:
            by_sha[sha[r["file_name"]]].append(r)
    for copies in by_sha.values():
        if len(copies) < 2:
            continue
        done["duplicate_sets"] += 1
        done["duplicate_sets_genus_differs"] += len({genus_of(r["species"]) for r in copies}) > 1
        named = [r for r in copies if r["species_raw"] and r["species_raw"].replace("_", " ").strip() == r["species"]]
        keep = min(named or copies, key=lambda r: r["file_name"])
        for r in copies:
            if r is not keep:
                r["label_drop"] = "byte-identical copy of an image kept under another file name"
                done["dropped_duplicate"] += 1

    for r in rows:
        r["label_change"] = "; ".join(r["label_change"])
    return done


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-dir", type=Path, default=None,
                        help="Directory of input images to parse (searched recursively). "
                             "If not provided, you'll be prompted to choose from available datasets.")
    parser.add_argument("--out-csv", type=Path, default="/workspace/data/processed/labels.csv",
                        help="Where to write the labels CSV.")
    parser.add_argument("--decisions", type=Path, default=None,
                        help="Folder with the label decision sheets (genus_spelling.tsv, "
                             "catalog_conflicts.tsv, needs_review.tsv) to apply.")
    parser.add_argument("--hash-cache", type=Path, default=Path("data/processed/sha256_cache.json"),
                        help="SHA-256 cache shared with split_specimens.py (used with --decisions).")
    args = parser.parse_args()

    if args.raw_dir is None:
        default_raw = Path("/workspace/data/raw")
        args.raw_dir = choose_directory(default_raw)
    # absolute, so the hash cache keys match split_specimens.py's
    args.raw_dir = args.raw_dir.resolve()

    images = list(iter_images(args.raw_dir))
    print(f"Found {len(images)} image(s) under {args.raw_dir}")

    rows = []
    for path in images:
        rel_path = path.relative_to(args.raw_dir)
        # Top-level subfolder name (e.g. "full_body", "Roentgen") if the
        # image is nested one or more levels deep, else "" for an image
        # sitting directly under raw-dir. This is photo type, not
        # species/origin - NHM_datensatz separates full-body shots from
        # X-ray ("Röntgen") shots this way.
        photo_type = rel_path.parts[0] if len(rel_path.parts) > 1 else ""

        parsed = parse_filename(path.stem)
        rows.append({
            "file_name": str(rel_path),
            "photo_type": photo_type,
            **parsed,
        })

    columns = ["file_name", "photo_type", "species", "species_raw", "catalog_number",
               "extra", "needs_review", "review_reason"]
    if args.decisions is not None:
        cache = json.loads(args.hash_cache.read_text()) if args.hash_cache.exists() else {}
        n_cached = len(cache)
        try:
            sha = {str(p.relative_to(args.raw_dir)): file_sha256(p, cache) for p in images}
        finally:
            if len(cache) != n_cached:
                args.hash_cache.parent.mkdir(parents=True, exist_ok=True)
                args.hash_cache.write_text(json.dumps(cache))
        done = apply_decisions(rows, args.decisions, sha)
        columns += ["label_change", "label_drop"]
        print(f"Applied the decisions in {args.decisions} ({len(cache) - n_cached} images newly hashed):")
        for key in ("genus_respelled", "renamed", "needs_review_resolved", "dropped_catalog_conflict",
                    "dropped_needs_review", "duplicate_sets", "duplicate_sets_genus_differs", "dropped_duplicate"):
            print(f"  {key}: {done[key]}")

    df = pd.DataFrame(rows, columns=columns)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)

    n_review = int(df["needs_review"].sum())
    n_species = df["species"].nunique(dropna=True)
    print(f"Wrote {len(df)} row(s) to {args.out_csv}")
    print(f"  {len(df) - n_review} parsed cleanly, {n_review} flagged for manual review")
    print(f"  {n_species} distinct species value(s)")
    print("  Rows by photo_type:")
    for photo_type, count in df["photo_type"].value_counts(dropna=False).items():
        label = photo_type if photo_type else "(directly under raw-dir)"
        print(f"    {label}: {count}")
    if n_review:
        print("  Flagged rows by reason:")
        for reason, count in df.loc[df["needs_review"], "review_reason"].value_counts().items():
            print(f"    {reason}: {count}")
        print("  Sample flagged filenames (first 20):")
        for name in df.loc[df["needs_review"], "file_name"].head(20):
            print(f"    - {name}")


if __name__ == "__main__":
    main()
