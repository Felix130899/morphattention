"""Train / val / test split by specimen (NMW catalog number), not by image.

The same specimen appears in several images (left, right, head, X-ray), so a
per-image split would test the model on fish it has already seen. This
assigns whole specimens to one split, shared by both domains: a specimen's
photos and X-rays always land in the same split.

Specimen groups. The catalog number is read from the file name, right after
"_NMW" / "_MNW" (an optional "_" before the digits is tolerated). A tail
attached *without* an underscore is a catalog range or list and is expanded:
"NMW12345-67" = 12345..12367, "NMW12340-45,48-53" = 12340..12345 +
12348..12353, "NMW12345-NMW12350" = 12345..12350. Short numbers replace the
last digits of the previous one. A "-" tail that gives no forward range of
at most 200 ("NMW54321-2222", "NMW4321-10") is a sub-index written with "-"
and is ignored, so no catalog number is invented. A tail *after* an underscore
("NMW12345_1-2") is an index inside the lot and is ignored, as are
parentheses ("NMW12345(678)"). Images whose numbers overlap are merged
(union-find), so a series photo "NMW1234-6" and a single photo "NMW1235" are
one specimen group. Byte-identical images (same SHA-256, e.g. one X-ray
exported under an old and a new species name) are merged too, even when
their numbers differ, so a copy can never land in another split. One NMW number is a lot (jar) that can hold several fish;
the whole lot stays in one split.

Only usable images count: status "ok" in the run's annotations.jsonl.
Images without a catalog number can't be assigned to a specimen and are
dropped (listed in dropped_no_catalog.txt).

Classes are genera (the word before any "(Subgenus)" in the species). A genus
is *evaluated* in a domain if it has at least ``--min-specimens`` specimen
groups with usable images in that domain. Each group's genus is the most
common genus among its usable images (ties: alphabetical), since a few
groups carry synonyms. Per genus, groups are shuffled (seeded by seed +
genus, so adding a genus leaves the others unchanged) and picked greedily
for test, then val, until every evaluated domain has
max(1, round(frac * specimens)) of them; preferring groups that don't push an
already filled domain over its target. The rest is train. Groups of a genus
evaluated in no domain are "train_only". Rows keep their own label genus and
a ``genus_evaluated`` flag for their domain.

Output (``--out``, in gitignored data/ - it holds catalog numbers):
    split.csv          one row per usable image: file_name, domain, specimen,
                       split, genus, genus_evaluated
    specimens.csv      one row per specimen group: specimen, catalog_numbers,
                       genus, images per domain, split
    balance.csv        per domain and genus: evaluated, specimens and images
                       per split
    dropped_no_catalog.txt   usable images without a catalog number
    duplicates.csv     byte-identical usable images: sha256, file_name, species,
                       specimen, split (not dropped here: which copy's label is
                       right is a naming decision)
    run_config.json    inputs (+ sha256 of labels.csv), settings, counts, git state

    python scripts/split_specimens.py \\
        --labels data/processed/labels.csv \\
        --segmented data/processed/segmented/B_merged_2026-10-03 \\
        --out data/processed/split/B_merged_2026-10-03_seed0

SHA-256 sums are cached in ``--hash-cache`` by path, size and modification
time (shared with build_manifest.py), so only new or changed files are hashed.
"""

import argparse
import collections
import csv
import datetime
import hashlib
import json
import os
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from segment_fish import git_state  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
SPLITS = ("train", "val", "test")
MAX_RANGE = 200  # longest catalog range accepted; the data's longest is 86
CATALOG_RE = re.compile(r"_(?:NMW|MNW)_?(\d+)((?:[-,](?:NMW|MNW)?\d+)*)")


def catalog_numbers(file_name):
    """Catalog numbers of one image as a set of ints (empty if none), plus the
    ignored "-" tail items that give no forward range (reversed or too long)."""
    stem = Path(file_name).stem
    m = CATALOG_RE.search(stem)
    if not m:
        return set(), []
    prev = int(m.group(1))
    numbers, odd = {prev}, []
    for sep, digits in re.findall(r"([-,])(?:NMW|MNW)?(\d+)", m.group(2)):
        if len(digits) < len(str(prev)):
            num = int(str(prev)[:len(str(prev)) - len(digits)] + digits)
        else:
            num = int(digits)
        if sep == ",":
            numbers.add(num)
        elif prev < num <= prev + MAX_RANGE:
            numbers.update(range(prev, num + 1))
        else:
            odd.append(m.group(2))  # the rest of the tail belongs to the sub-index too
            break
        prev = num
    return numbers, odd


def genus_of(species):
    m = re.match(r"[^\s(]+", str(species))
    return m.group(0) if m else ""


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def raw_dirs_of(segmented, domains):
    """{domain: raw image folder} from each domain run's run_config.json."""
    return {d: resolve(json.loads((Path(segmented) / d / "run_config.json").read_text())["settings"]["raw_dir"])
            for d in domains}


def file_sha256(path, cache):
    """SHA-256 of a file's bytes, reusing ``cache`` {path: [size, mtime_ns, sha256]}."""
    st = os.stat(path)
    hit = cache.get(str(path))
    if hit and hit[0] == st.st_size and hit[1] == st.st_mtime_ns:
        return hit[2]
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    cache[str(path)] = [st.st_size, st.st_mtime_ns, h.hexdigest()]
    return h.hexdigest()


def hash_images(names, raw_dirs, cache_path):
    """{labels file_name: sha256}, through the JSON cache at ``cache_path``."""
    cache = json.loads(Path(cache_path).read_text()) if Path(cache_path).exists() else {}
    n_cached = len(cache)
    try:
        return {n: file_sha256(raw_dirs[n.split("/", 1)[0]] / n.split("/", 1)[1], cache) for n in names}
    finally:
        if len(cache) != n_cached:
            Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
            Path(cache_path).write_text(json.dumps(cache))


def load_status(segmented, domains):
    """{labels file_name ("<domain>/<name>"): (status, reason, n_instances)}."""
    status = {}
    for domain in domains:
        run = Path(segmented) / domain
        skipped = {}
        if (run / "skipped.txt").exists():
            for line in (run / "skipped.txt").read_text().splitlines():
                name, _, reason = line.partition("\t")
                skipped[name] = reason
        with open(run / "annotations.jsonl") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                name = rec["file_name"]
                reason = rec.get("reason") or skipped.get(name, "")
                status[f"{domain}/{name}"] = (rec["status"], reason, len(rec.get("instances") or []))
    return status


def build_groups(numbers_per_image, same_bytes=()):
    """{image: specimen id}. Images sharing any catalog number are one group,
    and so are the images of each list in ``same_bytes``; ids are S00001... in
    order of the group's smallest number."""
    parent = {}

    def find(x):
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for nums in numbers_per_image.values():
        nums = sorted(nums)
        for x in nums[1:]:
            parent[find(x)] = find(nums[0])
    for images in same_bytes:
        for img in images[1:]:
            parent[find(min(numbers_per_image[img]))] = find(min(numbers_per_image[images[0]]))
    roots = {}
    for nums in numbers_per_image.values():
        root = find(min(nums))
        roots[root] = min(roots.get(root, root), *nums)
    ids = {root: f"S{k:05d}" for k, root in enumerate(sorted(roots, key=roots.get), start=1)}
    return {img: ids[find(min(nums))] for img, nums in numbers_per_image.items()}


def majority(values):
    counts = collections.Counter(values)
    return min(counts, key=lambda v: (-counts[v], v))


def assign_splits(groups, domains, seed, val_frac, test_frac, min_specimens):
    """groups: {specimen: {"genus": g, "domains": set}} -> ({specimen: split},
    {domain: set of evaluated genera})."""
    by_genus = collections.defaultdict(list)
    for sid, g in groups.items():
        by_genus[g["genus"]].append(sid)
    split, evaluated = {}, {d: set() for d in domains}
    for genus in sorted(by_genus):
        sids = sorted(by_genus[genus])
        n = {d: sum(d in groups[s]["domains"] for s in sids) for d in domains}
        eval_d = [d for d in domains if n[d] >= min_specimens]
        for d in eval_d:
            evaluated[d].add(genus)
        if not eval_d:
            split.update({s: "train_only" for s in sids})
            continue
        pool = sids[:]
        random.Random(f"{seed}/{genus}").shuffle(pool)
        for name, frac in (("test", test_frac), ("val", val_frac)):
            need = {d: max(1, round(frac * n[d])) for d in eval_d}
            while any(need.values()):
                helps = [s for s in pool if any(need[d] for d in eval_d if d in groups[s]["domains"])]
                if not helps:
                    break
                clean = [s for s in helps if all(need[d] for d in eval_d if d in groups[s]["domains"])]
                pick = (clean or helps)[0]
                pool.remove(pick)
                split[pick] = name
                for d in eval_d:
                    if d in groups[pick]["domains"] and need[d]:
                        need[d] -= 1
        split.update({s: "train" for s in pool})
    return split, evaluated


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--labels", type=Path, default=Path("data/processed/labels.csv"))
    parser.add_argument("--segmented", type=Path, required=True,
                        help="Curated run with one subfolder per domain (full_body, Röntgen).")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--test-frac", type=float, default=0.15)
    parser.add_argument("--min-specimens", type=int, default=5,
                        help="Specimen groups a genus needs in a domain to be evaluated there (default 5).")
    parser.add_argument("--hash-cache", type=Path, default=Path("data/processed/sha256_cache.json"))
    args = parser.parse_args()
    if args.min_specimens < 3:
        parser.error("--min-specimens must be >= 3 (one each for train, val, test)")

    with open(args.labels, newline="") as f:
        labels = list(csv.DictReader(f))
    domains = sorted({r["photo_type"] for r in labels})
    status = load_status(args.segmented, domains)
    missing = [r["file_name"] for r in labels if r["file_name"] not in status]
    if missing:
        raise SystemExit(f"{len(missing)} labels.csv rows have no record in {args.segmented}, "
                         f"e.g. {missing[:2]}")

    usable = [r for r in labels if status[r["file_name"]][0] == "ok"]
    numbers, odd_tails, no_catalog = {}, 0, []
    for r in usable:
        nums, odd = catalog_numbers(r["file_name"])
        odd_tails += bool(odd)
        if nums:
            numbers[r["file_name"]] = nums
        else:
            no_catalog.append(r["file_name"])
    rows = [r for r in usable if r["file_name"] in numbers]
    sha = hash_images([r["file_name"] for r in rows], raw_dirs_of(args.segmented, domains), args.hash_cache)
    by_sha = collections.defaultdict(list)
    for r in rows:
        by_sha[sha[r["file_name"]]].append(r["file_name"])
    same_bytes = [names for names in by_sha.values() if len(names) > 1]
    specimen = build_groups(numbers, same_bytes)

    members = collections.defaultdict(list)
    for r in rows:
        members[specimen[r["file_name"]]].append(r)
    groups = {sid: {"genus": majority(genus_of(r["species"]) for r in rs),
                    "domains": {r["photo_type"] for r in rs}} for sid, rs in members.items()}
    split, evaluated = assign_splits(groups, domains, args.seed, args.val_frac, args.test_frac,
                                     args.min_specimens)

    # The same, counting the images the mask review dropped as if they were usable:
    # which genera lost their evaluated status through the drops?
    dropped_rows = [r for r in labels if status[r["file_name"]][0] == "dropped"]
    with_drops = {**numbers, **{r["file_name"]: catalog_numbers(r["file_name"])[0] for r in dropped_rows
                                if catalog_numbers(r["file_name"])[0]}}
    spec_d = build_groups(with_drops, same_bytes)
    mem_d = collections.defaultdict(list)
    for r in rows + [r for r in dropped_rows if r["file_name"] in with_drops]:
        mem_d[spec_d[r["file_name"]]].append(r)
    groups_d = {sid: {"genus": majority(genus_of(r["species"]) for r in rs),
                      "domains": {r["photo_type"] for r in rs}} for sid, rs in mem_d.items()}
    _, evaluated_d = assign_splits(groups_d, domains, args.seed, args.val_frac, args.test_frac,
                                   args.min_specimens)

    args.out.mkdir(parents=True, exist_ok=True)
    with open(args.out / "split.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file_name", "domain", "specimen", "split", "genus", "genus_evaluated"])
        for r in rows:
            sid, g = specimen[r["file_name"]], genus_of(r["species"])
            w.writerow([r["file_name"], r["photo_type"], sid, split[sid], g, g in evaluated[r["photo_type"]]])
    with open(args.out / "specimens.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["specimen", "catalog_numbers", "genus", *[f"images_{d}" for d in domains], "split"])
        for sid in sorted(members):
            nums = sorted(set().union(*(numbers[r["file_name"]] for r in members[sid])))
            per_d = collections.Counter(r["photo_type"] for r in members[sid])
            w.writerow([sid, ";".join(f"NMW{x}" for x in nums), groups[sid]["genus"],
                        *[per_d[d] for d in domains], split[sid]])
    (args.out / "dropped_no_catalog.txt").write_text("".join(n + "\n" for n in no_catalog))
    species_of = {r["file_name"]: r["species"] for r in rows}
    with open(args.out / "duplicates.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sha256", "file_name", "species", "specimen", "split"])
        for names in sorted(same_bytes):
            for n in names:
                w.writerow([sha[n], n, species_of[n], specimen[n], split[specimen[n]]])

    balance = collections.defaultdict(lambda: collections.Counter())
    for r in rows:
        sid, d = specimen[r["file_name"]], r["photo_type"]
        g = groups[sid]["genus"]
        balance[(d, g)][f"images_{split[sid]}"] += 1
    for sid, g in groups.items():
        for d in g["domains"]:
            balance[(d, g["genus"])][f"specimens_{split[sid]}"] += 1
    all_splits = (*SPLITS, "train_only")
    cols = [f"{k}_{s}" for k in ("specimens", "images") for s in all_splits]
    with open(args.out / "balance.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["domain", "genus", "evaluated", *cols])
        for (d, g) in sorted(balance):
            w.writerow([d, g, g in evaluated[d], *[balance[(d, g)][c] for c in cols]])

    summary = {}
    for d in domains:
        dr = [r for r in rows if r["photo_type"] == d]
        img = collections.Counter(split[specimen[r["file_name"]]] for r in dr)
        ev = collections.Counter(split[specimen[r["file_name"]]] for r in dr
                                 if genus_of(r["species"]) in evaluated[d])
        spec = collections.Counter(split[s] for s, g in groups.items() if d in g["domains"])
        summary[d] = {"usable_images": len(dr), "evaluated_genera": len(evaluated[d]),
                      "images": dict(img), "images_of_evaluated_genera": dict(ev), "specimens": dict(spec),
                      "lost_evaluation_through_drops": sorted(evaluated_d[d] - evaluated[d])}
    label_genus_differs = sum(genus_of(r["species"]) != groups[specimen[r["file_name"]]]["genus"] for r in rows)
    counts = {"labels_rows": len(labels), "usable": len(usable), "no_catalog_dropped": len(no_catalog),
              "catalog_numbers": len(set().union(*numbers.values())),
              "specimen_groups": len(groups), "images_with_ignored_dash_tail": odd_tails,
              "rows_whose_genus_differs_from_their_group": label_genus_differs,
              "byte_identical_sets": len(same_bytes),
              "byte_identical_sets_with_different_species": sum(
                  len({species_of[n] for n in names}) > 1 for names in same_bytes)}
    (args.out / "run_config.json").write_text(json.dumps({
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "labels": str(args.labels), "labels_sha256": hashlib.sha256(args.labels.read_bytes()).hexdigest(),
        "segmented": str(args.segmented),
        "settings": {"seed": args.seed, "val_frac": args.val_frac, "test_frac": args.test_frac,
                     "min_specimens": args.min_specimens, "max_range": MAX_RANGE},
        "counts": counts, "per_domain": summary, **git_state()}, indent=1, ensure_ascii=False))

    print(f"{counts['usable']} usable images, {len(no_catalog)} without catalog number dropped; "
          f"{len(groups)} specimen groups ({odd_tails} images with a '-' sub-index tail ignored)")
    print(f"{label_genus_differs} images carry a genus other than their group's (synonyms)")
    print(f"{len(same_bytes)} sets of byte-identical images, kept in one group each "
          f"({counts['byte_identical_sets_with_different_species']} with different species names): duplicates.csv")
    for d, s in summary.items():
        print(f"\n{d}: {s['usable_images']} images, {s['evaluated_genera']} evaluated genera "
              f"(>= {args.min_specimens} specimens)")
        for name in (*SPLITS, "train_only"):
            print(f"  {name:10s} specimens {s['specimens'].get(name, 0):5d}   images {s['images'].get(name, 0):6d}"
                  f"   of evaluated genera {s['images_of_evaluated_genera'].get(name, 0):6d}")
        if s["lost_evaluation_through_drops"]:
            print(f"  below {args.min_specimens} specimens only because of the mask drops: "
                  + ", ".join(s["lost_evaluation_through_drops"]))
    print(f"\nWrote {args.out}/split.csv, specimens.csv, balance.csv, run_config.json")


if __name__ == "__main__":
    main()
