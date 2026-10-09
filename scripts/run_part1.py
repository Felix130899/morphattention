"""Run part 1 of the pipeline from one config file: raw images -> manifest.

Steps (``--steps``, default all, always in this order):

    labels    extract_labels.py --decisions <decisions>/labels -> labels_<name>.csv
    segment   segment_fish.py per domain -> segmented/<name>_full/<domain>
              (always with --resume: a finished run is only re-checked)
    merge     merge_runs.py per domain: --drop-sha <decisions>/mask_drops_<domain>.tsv
              and the domain's --qa-rule -> segmented/<name>/<domain>
              (skipped when that dir exists; delete it to redo the drops)
    split     split_specimens.py -> split/<name>_seed<seed>
    manifest  build_manifest.py -> manifest/<name>_seed<seed>

The random mask review and the error report are the human step between
"segment" and "merge" and are not run here; their outcome comes back as
decision files. Every step writes its own run_config.json; the effective config
(after --name) is written to segmented/<name>/part1_config.yaml.

    python scripts/run_part1.py
    python scripts/run_part1.py --config pipeline/config.yaml --steps split,manifest --dry-run
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"
STEPS = ("labels", "segment", "merge", "split", "manifest")


def cli_flags(options):
    """segment_fish.py flags from a {setting: value} dict: True -> --x, False -> --no-x
    (booleans) , lists -> --x a b, None -> left out."""
    flags = []
    for key, value in options.items():
        flag = "--" + key.replace("_", "-")
        if value is None:
            continue
        if value is True:
            flags.append(flag)
        elif value is False:
            flags.append("--no-" + key.replace("_", "-"))
        elif isinstance(value, (list, tuple)):
            flags += [flag, *map(str, value)]
        else:
            flags += [flag, str(value)]
    return flags


def paths(cfg):
    p = Path(cfg["processed_dir"])
    name, seed = cfg["name"], cfg["split"]["seed"]
    return {"labels": p / f"labels_{name}.csv", "full": p / "segmented" / f"{name}_full",
            "curated": p / "segmented" / name, "split": p / "split" / f"{name}_seed{seed}",
            "manifest": p / "manifest" / f"{name}_seed{seed}"}


def commands(cfg, steps):
    """[(step, argv or None, skip reason or None)] for the chosen steps, in pipeline order."""
    out, py = [], sys.executable
    p, dec, raw = paths(cfg), Path(cfg["decisions_dir"]), Path(cfg["raw_dir"])
    cache = ["--hash-cache", str(cfg["hash_cache"])]
    if "labels" in steps:
        out.append(("labels", [py, str(SCRIPTS / "extract_labels.py"), "--raw-dir", str(raw),
                               "--decisions", str(dec / "labels"), "--out-csv", str(p["labels"]), *cache], None))
    for domain, d in cfg["domains"].items():
        if "segment" in steps:
            out.append((f"segment {domain}", [py, str(SCRIPTS / "segment_fish.py"), "--raw-dir", str(raw / domain),
                                              "--out-dir", str(p["full"]), "--resume",
                                              *cli_flags(d.get("segment") or {})], None))
    for domain, d in cfg["domains"].items():
        if "merge" in steps:
            out_dir = p["curated"] / domain
            argv = [py, str(SCRIPTS / "merge_runs.py"), "--base", str(p["full"] / domain),
                    "--drop-sha", str(dec / f"mask_drops_{domain}.tsv"), *cache, "--out", str(out_dir)]
            if d.get("qa_rule"):
                argv += ["--qa-rule", d["qa_rule"]]
            out.append((f"merge {domain}", argv, f"{out_dir} exists" if out_dir.exists() else None))
    s = cfg["split"]
    if "split" in steps:
        out.append(("split", [py, str(SCRIPTS / "split_specimens.py"), "--labels", str(p["labels"]),
                              "--segmented", str(p["curated"]), "--out", str(p["split"]),
                              "--seed", str(s["seed"]), "--val-frac", str(s["val_frac"]),
                              "--test-frac", str(s["test_frac"]), "--min-specimens", str(s["min_specimens"]),
                              *cache], None))
    if "manifest" in steps:
        out.append(("manifest", [py, str(SCRIPTS / "build_manifest.py"), "--labels", str(p["labels"]),
                                 "--segmented", str(p["curated"]), "--split-dir", str(p["split"]),
                                 "--out", str(p["manifest"]), *cache], None))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "pipeline" / "config.yaml")
    parser.add_argument("--steps", default=",".join(STEPS), help=f"comma-separated subset of {','.join(STEPS)}")
    parser.add_argument("--name", help="override the run name from the config")
    parser.add_argument("--dry-run", action="store_true", help="print the commands, run nothing")
    args = parser.parse_args()
    steps = [s.strip() for s in args.steps.split(",") if s.strip()]
    unknown = [s for s in steps if s not in STEPS]
    if unknown:
        parser.error(f"unknown step(s) {unknown}; choose from {STEPS}")
    cfg = yaml.safe_load(args.config.read_text())
    if args.name:
        cfg["name"] = args.name

    os.chdir(REPO_ROOT)  # config paths are relative to the repo root
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, [str(REPO_ROOT / "src"),
                                                                       os.environ.get("PYTHONPATH")]))}
    for step, argv, skip in commands(cfg, steps):
        print(f"=== {step}: {' '.join(argv)}", flush=True)
        if skip:
            print(f"    skipped: {skip}", flush=True)
            continue
        if not args.dry_run:
            subprocess.run(argv, check=True, env=env)
    if not args.dry_run and any(s in steps for s in ("merge", "split", "manifest")):
        curated = paths(cfg)["curated"]
        if curated.exists():
            (curated / "part1_config.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False))


if __name__ == "__main__":
    main()
