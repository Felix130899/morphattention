"""Merge a targeted re-run back into a full segment_fish.py run, as a new run dir.

A targeted re-run (e.g. the bleed fallback over a few hundred images) only
holds the re-segmented images. This writes a complete run to ``--out``: every
record from ``--base``, except the images taken from ``--patch``, which
replace theirs. Neither input is changed, so the base run stays the "before"
for the thesis numbers.

Which images to take:
  * ``--take FILE``: one file name per line (``#`` comments, blank lines ignored)
  * ``--take-ok-from REVIEW_DIR:RUN``: every image a blind
    ``review_masks.py --compare`` review judged "ok" in run RUN (the name
    given to the patch run in ``--compare RUN=...``)
Both can be combined (union). Every taken image must be "ok" in the patch run.

``--drop FILE`` (optional): images excluded from the curated set, one per
line as ``file_name<TAB>reason``. Their record becomes
``{"status": "dropped", "reason": ...}``: no COCO annotation, no mask or
overlay in ``--out`` (the base run still has them), listed in dropped.txt.
A name can't be both taken and dropped.

``--drop-sha FILE`` (optional): the same, keyed by image content instead of
name (``sha256<TAB>reason`` lines after a ``sha256<TAB>reason`` header; the
tracked form in ``pipeline/decisions/``). The base run's raw images are
hashed (cached in ``--hash-cache``) to find them.

``--qa-rule B`` (optional): also drop every "ok" image with any instance
> 2 % outside its detector box, an inverted or unresolved bleed fallback, or
edge_uncertain_frac > 0.5 (X-rays; vault decisions/2026-10-08-xray-drop-rule-B).
Reason ``qa_rule_B: <which conditions>``. Images already dropped by a list
keep the list's reason.

Drop-only mode: ``--base`` + ``--drop`` without ``--patch`` writes the base
run minus the dropped images (e.g. to apply an earlier run's drops to a fresh
full run, so the numbers stay comparable).

Output (same layout as segment_fish.py, so review_masks.py etc. work on it):
    annotations.jsonl, coco/annotations.json, skipped.txt   rebuilt
    masks/, overlays/   hard links to the base/patch files (copies across filesystems)
    run_config.json     both inputs' configs, the taken list + its sha256, git state
    merged_from_patch.txt   the taken file names
    dropped.txt         file_name<TAB>reason of every dropped image

    python scripts/merge_runs.py \\
        --base data/processed/segmented/B_full_2026-10-01/full_body \\
        --patch data/processed/segmented/B_bleedfix/full_body \\
        --take-ok-from data/processed/segmented/B_bleedfix/review_compare_full_body:bleedfix \\
        --out data/processed/segmented/B_merged_2026-10-03/full_body

    python scripts/merge_runs.py \\
        --base data/processed/segmented/D_full_2026-10-05/full_body \\
        --drop data/processed/segmented/B_merged_2026-10-03/full_body/dropped.txt \\
        --out data/processed/segmented/D_merged_2026-10-05/full_body

    python scripts/merge_runs.py \\
        --base data/processed/segmented/D_full_2026-10-05/Röntgen \\
        --drop-sha pipeline/decisions/mask_drops_Röntgen.tsv --qa-rule B \\
        --out data/processed/segmented/D_final/Röntgen
"""

import argparse
import datetime
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from review_masks import load_decisions  # noqa: E402
from segment_fish import git_state, parse_image_list, print_summary, write_outputs  # noqa: E402
from split_specimens import file_sha256, resolve  # noqa: E402

QA_RULES = ("B",)


def read_records(run_dir):
    """{file_name: record} from a run's annotations.jsonl, read-only (unlike
    segment_fish.load_done, which repairs the file in place). A run with a
    half-written line is refused: finish or --resume it first."""
    records = {}
    path = Path(run_dir) / "annotations.jsonl"
    for n, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            raise SystemExit(f"{path} line {n} is not valid JSON - is that run still going or crashed?")
        records[rec["file_name"]] = rec
    return records


def ok_names_from_review(review_dir, run):
    """File names judged "ok" for ``run`` in a compare-mode review dir."""
    decisions = load_decisions(Path(review_dir), compare=True)
    runs = {r for r, _ in decisions}
    if run not in runs:
        raise SystemExit(f"run {run!r} not in {review_dir} (runs judged there: {sorted(runs)})")
    return sorted(name for (r, name), d in decisions.items() if r == run and d["verdict"] == "ok")


def merge_records(base, patch, take):
    """Base records (file order kept) with every name in ``take`` replaced by the patch record.

    base/patch: {file_name: record}. Fails if a taken name is missing from
    either run or not "ok" in the patch.
    """
    problems = [f"{n}: not in base run" for n in take if n not in base]
    problems += [f"{n}: not in patch run" for n in take if n not in patch]
    problems += [f"{n}: patch status {patch[n]['status']}" for n in take
                 if n in patch and patch[n]["status"] != "ok"]
    if problems:
        raise SystemExit(f"{len(problems)} problem(s) with the taken images:\n  " + "\n  ".join(problems))
    take = set(take)
    return {name: (patch[name] if name in take else rec) for name, rec in base.items()}


def read_drop_list(text):
    """{file_name: reason} from ``file_name<TAB>reason`` lines (reason optional;
    ``#`` comment lines and blank lines ignored)."""
    drops = {}
    for line in text.splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        name, _, reason = line.partition("\t")
        drops[name.strip()] = reason.strip() or "dropped"
    return drops


def read_sha_drop_list(text):
    """{sha256: reason} from a ``sha256<TAB>reason`` table (``#`` comment lines,
    the header line and blank lines ignored)."""
    drops = {}
    for line in text.splitlines():
        if not line.strip() or line.startswith("#") or line.startswith("sha256\t"):
            continue
        sha, _, reason = line.partition("\t")
        drops[sha.strip()] = reason.strip() or "dropped"
    return drops


def names_by_sha(records, raw_dir, cache):
    """{sha256: [file_name, ...]} of a run's records, hashing the raw images."""
    out = {}
    for name in records:
        out.setdefault(file_sha256(Path(raw_dir) / name, cache), []).append(name)
    return out


def qa_rule_b(rec):
    """Rule B's reason for an "ok" record, or None if no instance trips it."""
    if rec["status"] != "ok":
        return None
    why = set()
    for inst in rec["instances"]:
        if (inst.get("outside_box_frac") or 0) > 0.02:
            why.add("outside_box")
        if inst.get("bleed_fallback") in ("inverse", "unresolved"):
            why.add(inst["bleed_fallback"])
        if inst.get("edge_uncertain_frac", 0) > 0.5:
            why.add("edge_uncertain")
    return "qa_rule_B: " + ",".join(sorted(why)) if why else None


def apply_drops(merged, drops, take):
    """Mark every dropped image's record as dropped. Fails on unknown names or
    names that are also taken from the patch."""
    problems = [f"{n}: not in base run" for n in drops if n not in merged]
    problems += [f"{n}: also in the take list" for n in drops if n in set(take)]
    if problems:
        raise SystemExit(f"{len(problems)} problem(s) with the dropped images:\n  " + "\n  ".join(problems))
    return {name: ({"file_name": name, "status": "dropped", "reason": drops[name]} if name in drops else rec)
            for name, rec in merged.items()}


def link_or_copy(src, dst):
    try:
        os.link(src, dst)
    except OSError:  # other filesystem, or links unsupported
        shutil.copy2(src, dst)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", type=Path, required=True, help="full segment_fish.py run dir")
    parser.add_argument("--patch", type=Path, help="targeted re-run dir (leave out for drop-only mode)")
    parser.add_argument("--take", type=Path, help="file with the names to take from --patch")
    parser.add_argument("--take-ok-from", metavar="REVIEW_DIR:RUN",
                        help="take every image judged ok for RUN in this compare-mode review dir")
    parser.add_argument("--drop", type=Path, help="file_name<TAB>reason lines: exclude these images")
    parser.add_argument("--drop-sha", type=Path, help="sha256<TAB>reason table: exclude these images")
    parser.add_argument("--qa-rule", choices=QA_RULES, help="also drop the images this QA rule flags")
    parser.add_argument("--hash-cache", type=Path, default=Path("data/processed/sha256_cache.json"))
    parser.add_argument("--out", type=Path, required=True, help="new run dir (must not exist)")
    args = parser.parse_args()
    if args.patch is None:
        if args.take is not None or args.take_ok_from is not None:
            parser.error("--take/--take-ok-from need --patch")
        if args.drop is None and args.drop_sha is None and args.qa_rule is None:
            parser.error("give --patch (with --take and/or --take-ok-from) and/or --drop / --drop-sha / --qa-rule")
    elif args.take is None and args.take_ok_from is None:
        parser.error("give --take and/or --take-ok-from")
    if args.out.exists():
        raise SystemExit(f"{args.out} already exists - choose a new --out")

    take = set()
    if args.take is not None:
        take |= set(parse_image_list(args.take.read_text()))
    if args.take_ok_from is not None:
        review_dir, _, run = args.take_ok_from.rpartition(":")
        take |= set(ok_names_from_review(review_dir, run))
    take = sorted(take)

    base = read_records(args.base)
    patch = read_records(args.patch) if args.patch is not None else {}
    merged = merge_records(base, patch, take)
    drops = read_drop_list(args.drop.read_text()) if args.drop is not None else {}
    n_sha_drops = 0
    if args.drop_sha is not None:
        raw_dir = resolve(json.loads((args.base / "run_config.json").read_text())["settings"]["raw_dir"])
        cache = json.loads(args.hash_cache.read_text()) if args.hash_cache.exists() else {}
        n_cached = len(cache)
        try:
            by_sha = names_by_sha(base, raw_dir, cache)
        finally:
            if len(cache) != n_cached:
                args.hash_cache.write_text(json.dumps(cache))
        sha_drops = read_sha_drop_list(args.drop_sha.read_text())
        missing = [s for s in sha_drops if s not in by_sha]
        if missing:
            raise SystemExit(f"{len(missing)} sha256 value(s) of {args.drop_sha} match no image of {args.base}")
        for s, reason in sha_drops.items():
            for name in by_sha[s]:
                drops.setdefault(name, reason)
        n_sha_drops = len(sha_drops)
    n_rule_drops = 0
    if args.qa_rule == "B":
        taken = set(take)
        for name, rec in merged.items():
            reason = qa_rule_b(rec) if name not in drops and name not in taken else None
            if reason:
                drops[name] = reason
                n_rule_drops += 1
    merged = apply_drops(merged, drops, take)

    for d in ("masks", "overlays", "coco"):
        (args.out / d).mkdir(parents=True)
    taken = set(take)
    for name, rec in merged.items():
        if rec["status"] != "ok":
            continue
        src_run = args.patch if name in taken else args.base
        stem = Path(name).stem
        for sub, ext in (("masks", ".png"), ("overlays", ".jpg")):
            link_or_copy(src_run / sub / f"{stem}{ext}", args.out / sub / f"{stem}{ext}")
    (args.out / "annotations.jsonl").write_text("".join(json.dumps(r) + "\n" for r in merged.values()))
    coco, skipped = write_outputs(list(merged.values()), args.out)
    (args.out / "merged_from_patch.txt").write_text("".join(n + "\n" for n in take))
    (args.out / "dropped.txt").write_text("".join(f"{n}\t{r}\n" for n, r in sorted(drops.items())))
    cfg = {
        "merged": {
            "created": datetime.datetime.now().isoformat(timespec="seconds"), **git_state(),
            "base": str(args.base), "patch": str(args.patch) if args.patch else None,
            "take": str(args.take) if args.take else None, "take_ok_from": args.take_ok_from,
            "n_taken": len(take), "taken_sha256": hashlib.sha256("\n".join(take).encode()).hexdigest(),
            "drop": str(args.drop) if args.drop else None,
            "drop_sha": str(args.drop_sha) if args.drop_sha else None,
            "drop_sha_sha256": hashlib.sha256(args.drop_sha.read_bytes()).hexdigest() if args.drop_sha else None,
            "n_drop_sha": n_sha_drops, "qa_rule": args.qa_rule, "n_qa_rule_dropped": n_rule_drops,
            "n_dropped": len(drops),
        },
        "base_config": json.loads((args.base / "run_config.json").read_text()),
        "patch_config": json.loads((args.patch / "run_config.json").read_text()) if args.patch else None,
        # review_masks.py reads settings.raw_dir; both runs share it.
        "settings": {"raw_dir": json.loads((args.base / "run_config.json").read_text())["settings"]["raw_dir"]},
    }
    (args.out / "run_config.json").write_text(json.dumps(cfg, indent=2))
    print(f"Took {len(take)} image(s) from {args.patch or '(no patch)'}, dropped {len(drops)}, "
          f"{len(merged) - len(take) - len(drops)} unchanged from {args.base}.")
    print_summary(coco, skipped)


if __name__ == "__main__":
    main()
