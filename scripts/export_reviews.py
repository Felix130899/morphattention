"""Export a review_masks.py review dir as a tracked record keyed by SHA-256.

Review verdicts are the source of most mask numbers in the thesis, but a
review dir names images by file name (which holds the NHM catalog number), so
it stays in gitignored data/. This writes the final verdicts (last decision
per image wins, as review_masks.py replays them) as a tab-separated table
keyed by the SHA-256 of each raw image, with no file names:

    # source: <review dir>          # header lines: where it came from and
    # mode: single | compare        # what was judged (run labels -> run dirs,
    # run <label>: <run dir>        # seed, population, sample size)
    # seed: 3 ...
    position  sha256  [run]  verdict  categories  [group]

``position`` is the image's place in the shown order (sample.json /
order.json), so a prefix of a random sample stays recognisable; empty when
the review had no stored order. ``group`` comes from a groups.csv next to the
decisions (e.g. fin_cut / ok in a dev compare). Images whose raw file no
longer exists (the "Kopie" duplicates deleted on 2026-10-01) are counted in
the header, not listed.

    python scripts/export_reviews.py \\
        --review-dir data/processed/segmented/D_ruleB_2026-10-09/Röntgen/review_random_seed3 \\
        --out pipeline/decisions/reviews/D_ruleB_Röntgen_random_seed3.tsv

The raw image folder is read from the reviewed run's run_config.json
(settings.raw_dir); give ``--raw-dir`` when the review dir is not inside a run.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from review_masks import load_decisions  # noqa: E402
from split_specimens import REPO_ROOT, file_sha256, resolve  # noqa: E402


def repo_path(path):
    """``path`` relative to the repo root if it is inside it (for the header)."""
    try:
        return str(Path(path).resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def review_info(review_dir):
    """(mode, {run label: run dir}, seed, population, n, [(run, file_name) in shown order])."""
    review_dir = Path(review_dir)
    lines = [json.loads(x) for x in (review_dir / "decisions.jsonl").read_text().splitlines() if x.strip()]
    mode = "compare" if lines and "run" in lines[0] else "single"
    runs, seed, population, n, order = {}, None, None, None, []
    if (review_dir / "sample.json").exists():
        s = json.loads((review_dir / "sample.json").read_text())
        runs = {"": s["run_dir"]}
        seed, population, n = s["seed"], s["population"], s["n"]
        order = [("", name) for name in s["sample"]]
    elif (review_dir / "order.json").exists():
        o = json.loads((review_dir / "order.json").read_text())
        runs, seed = o.get("runs", {}), o.get("seed")
        order = [tuple(item) if mode == "compare" else ("", item[1] if isinstance(item, list) else item)
                 for item in o["items"]]
    return mode, runs, seed, population, n, order


def raw_dir_of(review_dir, runs):
    """settings.raw_dir of the reviewed run(s); all runs of one review share it."""
    candidates = [Path(r) for r in runs.values()] + [Path(review_dir).parent]
    raw = {json.loads((resolve(c) / "run_config.json").read_text())["settings"]["raw_dir"]
           for c in candidates if (resolve(c) / "run_config.json").exists()}
    if len(raw) != 1:
        raise SystemExit(f"cannot tell the raw image folder of {review_dir} ({sorted(raw)}); give --raw-dir")
    return resolve(raw.pop())


def export(review_dir, raw_dir, cache):
    """(header lines, rows) for one review dir; rows = dicts in shown order."""
    mode, runs, seed, population, n, order = review_info(review_dir)
    decisions = load_decisions(Path(review_dir), compare=mode == "compare")
    groups = {}
    if (Path(review_dir) / "groups.csv").exists():
        with open(Path(review_dir) / "groups.csv", newline="") as f:
            groups = {r["file_name"]: r["group"] for r in csv.DictReader(f)}
    position = {key if mode == "compare" else key[1]: k for k, key in enumerate(order)}
    rows, missing = [], 0
    for key, d in decisions.items():
        run, name = key if mode == "compare" else ("", key)
        path = Path(raw_dir) / name
        if not path.exists():
            missing += 1
            continue
        rows.append({"position": position.get(key, ""), "sha256": file_sha256(path, cache), "run": run,
                     "verdict": d["verdict"], "categories": ",".join(d["categories"]),
                     "group": groups.get(name, "")})
    rows.sort(key=lambda r: (r["position"] == "", r["position"] if r["position"] != "" else 0, r["sha256"], r["run"]))
    header = [f"source: {repo_path(review_dir)}", f"mode: {mode}"]
    header += [f"run {label or '(single)'}: {repo_path(resolve(path))}" for label, path in sorted(runs.items())]
    header += [f"{k}: {v}" for k, v in (("seed", seed), ("population", population), ("sample_n", n),
                                         ("judged", len(rows)), ("raw_file_missing", missing)) if v is not None]
    return header, rows


def write_tsv(path, header, rows):
    cols = ["position", "sha256", *(["run"] if any(r["run"] for r in rows) else []), "verdict", "categories",
            *(["group"] if any(r["group"] for r in rows) else [])]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        f.write("".join(f"# {h}\n" for h in header))
        w = csv.DictWriter(f, fieldnames=cols, delimiter="\t", lineterminator="\n", extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--review-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, help="raw images of the reviewed run (default: from its run_config)")
    parser.add_argument("--hash-cache", type=Path, default=Path("data/processed/sha256_cache.json"))
    args = parser.parse_args()
    _, runs, *_ = review_info(args.review_dir)
    raw_dir = resolve(args.raw_dir) if args.raw_dir else raw_dir_of(args.review_dir, runs)
    cache = json.loads(args.hash_cache.read_text()) if args.hash_cache.exists() else {}
    n_cached = len(cache)
    try:
        header, rows = export(args.review_dir, raw_dir, cache)
    finally:
        if len(cache) != n_cached:
            args.hash_cache.write_text(json.dumps(cache))
    write_tsv(args.out, header, rows)
    print(f"{args.out}: {len(rows)} verdicts ({header[-1]})")


if __name__ == "__main__":
    main()
