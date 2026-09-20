"""Parse species labels out of NHM Wien filenames into a structured CSV.

Expected filename convention (per user, not yet confirmed with Neo/
supervisor - see tasks/clean-segmented-dataset-with-structured-labels.md
step 3):

    <species_name>_NMW<catalog_number>_<extra stuff>.<ext>

``species_name`` may itself contain underscores between words (e.g.
``Salmo_trutta``). ``NMW<catalog_number>`` is the museum's specimen catalog
number - but against the real NHM_datensatz export, "catalog_number" is not
just a plain integer: syntype/paratype series use ranges ("NMW1290-3"),
comma lists ("NMW13736-41,44-49"), and parenthetical sub-indices glued on
with no separator ("NMW95284(65895_2)"). A few filenames also use "MNW"
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
cross-check labels.csv against the image set (see task step 9).

Run inside the container, e.g.:

    docker compose run --rm vit-project python scripts/extract_labels.py
    docker compose run --rm vit-project python scripts/extract_labels.py \
        --raw-dir /workspace/data/raw/nmw_specimens \
        --out-csv /workspace/data/processed/labels.csv
"""

import argparse
import re
from pathlib import Path

import pandas as pd

# Pillow can decode all of these; segment_fish.py filters the same way.
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}

# <species>_NMW<anything> - species is greedy since it may contain
# underscores itself; the literal "_NMW"/"_MNW" anchors the split point.
# Catalog-number *structure* (digits vs ranges vs parens) is handled
# separately below, since it varies too much to fold into one regex.
FILENAME_RE = re.compile(r"^(?P<species>.+)_(?P<prefix>NMW|MNW)(?P<catalog_raw>.*)$")

# Leading digit run of catalog_raw, e.g. "95284" out of "95284(65895_2)".
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
    # (e.g. "NMW_90955") instead of none (e.g. "NMW90955") - tolerate both.
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


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-dir", type=Path, default=None,
                        help="Directory of input images to parse (searched recursively). "
                             "If not provided, you'll be prompted to choose from available datasets.")
    parser.add_argument("--out-csv", type=Path, default="/workspace/data/processed/labels.csv",
                        help="Where to write the labels CSV.")
    args = parser.parse_args()

    if args.raw_dir is None:
        default_raw = Path("/workspace/data/raw")
        args.raw_dir = choose_directory(default_raw)

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

    df = pd.DataFrame(rows, columns=[
        "file_name", "photo_type", "species", "species_raw", "catalog_number",
        "extra", "needs_review", "review_reason",
    ])

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
