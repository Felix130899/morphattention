"""Tests for scripts/pipeline/import_labels.py (own labels CSV) and the split + manifest on its output.

Runs under pytest or directly: python tests/test_import_labels.py
"""

import csv
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(Path(__file__).resolve().parents[1] / "scripts" / d) for d in ("", "pipeline", "review")]

ENV = {**os.environ, "PYTHONPATH": str(REPO / "src")}


def run(script, *args):
    return subprocess.run([sys.executable, str(REPO / "scripts" / "pipeline" / script), *map(str, args)],
                          capture_output=True, text=True, env=ENV)


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def make_dataset(tmp):
    """raw/{photos,xrays} with 8 specimens of two species; F7 photo is a byte copy of F6's;
    returns the labels rows (file_name, specimen, species, origin)."""
    rows = []
    for k in range(8):
        for domain in ("photos", "xrays"):
            name = f"{domain}/F{k}_{domain}.jpg"
            (tmp / "raw" / domain).mkdir(parents=True, exist_ok=True)
            data = b"photo-F6" if (domain, k) in (("photos", 6), ("photos", 7)) else f"{name}".encode()
            (tmp / "raw" / name).write_bytes(data)
            rows.append({"file_name": name, "specimen": f"F{k}", "species": "Salmo trutta" if k < 4 else "Esox lucius",
                         "origin": "lake" if k % 2 else "river"})
    rows[2]["species"] = "Salmo truta"  # misspelling, fixed by value_renames.tsv
    return rows


def write_sheets(dec, renames=(), fixes=()):
    (dec / "labels").mkdir(parents=True, exist_ok=True)
    with open(dec / "labels" / "value_renames.tsv", "w") as f:
        f.write("column\tfrom\tto\timages\tnote\tdecision\n" + "".join("\t".join(r) + "\n" for r in renames))
    with open(dec / "labels" / "image_fixes.tsv", "w") as f:
        f.write("sha256\tnote\tdecision\n" + "".join("\t".join(r) + "\n" for r in fixes))


def write_runs(tmp, rows):
    """A fake curated segment_fish.py run per domain: every image ok, a mask each."""
    for domain in ("photos", "xrays"):
        run_dir = tmp / "seg" / domain
        (run_dir / "masks").mkdir(parents=True)
        (run_dir / "run_config.json").write_text(json.dumps({"settings": {"raw_dir": str(tmp / "raw" / domain)}}))
        names = [r["file_name"].split("/", 1)[1] for r in rows if r["file_name"].startswith(domain + "/")]
        (run_dir / "annotations.jsonl").write_text(
            "".join(json.dumps({"file_name": n, "status": "ok", "instances": [{}]}) + "\n" for n in names))
        for n in names:
            (run_dir / "masks" / f"{Path(n).stem}.png").write_bytes(b"png")


def sha(path):
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_csv_and_images_must_match():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        rows = make_dataset(tmp)
        (tmp / "raw" / "loose.jpg").write_bytes(b"x")
        write_csv(tmp / "labels.csv", rows[1:] + [rows[3], {**rows[0], "file_name": "photos/gone.jpg"}])
        res = run("import_labels.py", "--raw-dir", tmp / "raw", "--csv", tmp / "labels.csv", "--out-csv", tmp / "out.csv",
                  "--hash-cache", tmp / "cache.json")
        assert res.returncode != 0 and not (tmp / "out.csv").exists()
        for part in ("1 file name(s) listed twice", "1 row(s) without an image", "photos/gone.jpg",
                     "2 image(s) without a row", "photos/F0_photos.jpg", "directly under"):
            assert part in res.stderr, (part, res.stderr)
        write_csv(tmp / "labels.csv", [{k: v for k, v in r.items() if k != "specimen"} for r in rows])
        res = run("import_labels.py", "--raw-dir", tmp / "raw", "--csv", tmp / "labels.csv", "--out-csv", tmp / "out.csv")
        assert res.returncode != 0 and "no 'specimen' column" in res.stderr


def test_import_split_manifest_with_own_labels():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        rows = make_dataset(tmp)
        rows[9]["specimen"] = ""  # F4 X-ray: no specimen id -> left out of the split
        write_csv(tmp / "labels.csv", rows)
        write_sheets(tmp / "dec", renames=[("species", "Salmo truta", "Salmo trutta", "1", "misspelling", "accept")],
                     fixes=[(sha(tmp / "raw/xrays/F5_xrays.jpg"), "wrong origin", "set origin river; set species Esox lucius"),
                            (sha(tmp / "raw/xrays/F3_xrays.jpg"), "blurred", "drop")])
        cache = tmp / "cache.json"
        res = run("import_labels.py", "--raw-dir", tmp / "raw", "--csv", tmp / "labels.csv", "--decisions", tmp / "dec" / "labels",
                  "--out-csv", tmp / "labels_out.csv", "--hash-cache", cache)
        assert res.returncode == 0, res.stderr
        out = {r["file_name"]: r for r in read_csv(tmp / "labels_out.csv")}
        assert list(next(iter(out.values()))) == ["file_name", "photo_type", "specimen_id", "species", "origin",
                                                  "label_change", "label_drop"]
        assert len(out) == 16 and out["photos/F1_photos.jpg"]["species"] == "Salmo trutta"
        assert out["photos/F1_photos.jpg"]["label_change"] == "species Salmo truta -> Salmo trutta"
        f5 = out["xrays/F5_xrays.jpg"]
        assert (f5["origin"], f5["species"], f5["label_change"]) == ("river", "Esox lucius", "set origin river; set species Esox lucius")
        assert out["xrays/F3_xrays.jpg"]["label_drop"] == "image fix: blurred"
        assert out["photos/F7_photos.jpg"]["label_drop"].startswith("byte-identical") and not out["photos/F6_photos.jpg"]["label_drop"]
        assert out["photos/F0_photos.jpg"]["photo_type"] == "photos" and out["photos/F0_photos.jpg"]["specimen_id"] == "F0"

        write_runs(tmp, rows)
        res = run("split_specimens.py", "--labels", tmp / "labels_out.csv", "--segmented", tmp / "seg", "--out", tmp / "split",
                  "--class", "origin", "--min-specimens", "3", "--hash-cache", cache)
        assert res.returncode == 0, res.stderr
        split = {r["file_name"]: r for r in read_csv(tmp / "split" / "split.csv")}
        # 16 images - F3 X-ray (fix) - F7 photo (copy) - F4 X-ray (no specimen id)
        assert len(split) == 13 and "xrays/F4_xrays.jpg" not in split
        assert (tmp / "split" / "dropped_no_catalog.txt").read_text() == "xrays/F4_xrays.jpg\n"
        assert list(next(iter(split.values())))[-2:] == ["origin", "origin_evaluated"]
        by_spec = {}
        for r in split.values():
            by_spec.setdefault(r["specimen"], set()).add(r["split"])
        assert all(len(v) == 1 for v in by_spec.values())  # one split per specimen
        # F7's photo is a byte copy of F6's: dropped in the label step, so it can't sit in two splits
        assert "photos/F7_photos.jpg" not in split and split["xrays/F7_xrays.jpg"]["specimen"] != split["photos/F6_photos.jpg"]["specimen"]
        specimens = read_csv(tmp / "split" / "specimens.csv")
        assert list(specimens[0])[:3] == ["specimen", "specimen_ids", "origin"]
        assert sorted(r["specimen_ids"] for r in specimens) == [f"F{k}" for k in range(8)]  # F3, F4 keep their photos
        cfg = json.loads((tmp / "split" / "run_config.json").read_text())
        assert cfg["specimens_from"] == "labels column specimen_id" and cfg["counts"]["per_image_specimens"] == 0

        res = run("build_manifest.py", "--labels", tmp / "labels_out.csv", "--segmented", tmp / "seg",
                  "--split-dir", tmp / "split", "--out", tmp / "manifest", "--hash-cache", cache)
        assert res.returncode == 0, res.stderr
        man = {r["file_name"]: r for r in read_csv(tmp / "manifest" / "manifest.csv")}
        assert list(next(iter(man.values()))) == ["file_name", "sha256", "domain", "specimen_id", "species", "origin",
                                                  "specimen", "mask_path", "n_instances", "status", "reason", "split",
                                                  "origin_evaluated"]
        assert man["xrays/F4_xrays.jpg"]["reason"] == "no specimen id" and man["xrays/F3_xrays.jpg"]["reason"] == "image fix: blurred"
        assert sum(r["status"] == "usable" for r in man.values()) == 13

        res = run("split_specimens.py", "--labels", tmp / "labels_out.csv", "--segmented", tmp / "seg", "--out", tmp / "split_g",
                  "--class", "genus", "--min-specimens", "3", "--hash-cache", cache)  # derived from the species column
        assert res.returncode == 0, res.stderr
        assert {r["genus"] for r in read_csv(tmp / "split_g" / "split.csv")} == {"Salmo", "Esox"}


def test_specimen_none_makes_every_image_its_own_specimen():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        rows = [{k: v for k, v in r.items() if k != "specimen"} for r in make_dataset(tmp)]
        write_csv(tmp / "labels.csv", rows)
        cache = tmp / "cache.json"
        res = run("import_labels.py", "--raw-dir", tmp / "raw", "--csv", tmp / "labels.csv", "--specimen", "none",
                  "--out-csv", tmp / "labels_out.csv", "--hash-cache", cache)
        assert res.returncode == 0 and "WARNING" in res.stdout
        out = read_csv(tmp / "labels_out.csv")
        assert all(r["specimen_id"] == "image:" + sha(tmp / "raw" / r["file_name"]) for r in out)
        write_runs(tmp, rows)
        res = run("split_specimens.py", "--labels", tmp / "labels_out.csv", "--segmented", tmp / "seg", "--out", tmp / "split",
                  "--class", "species", "--min-specimens", "3", "--hash-cache", cache)
        assert res.returncode == 0, res.stderr
        assert "WARNING: 15 images have no specimen information" in res.stdout
        assert json.loads((tmp / "split" / "run_config.json").read_text())["counts"]["per_image_specimens"] == 15


def test_fix_sheets_refuse_what_they_cannot_apply():
    import import_labels as il
    rows = [{"file_name": "p/a.jpg", "species": "A b", "label_change": [], "label_drop": ""}]
    with tempfile.TemporaryDirectory() as tmp:
        dec = Path(tmp)
        for renames, fixes, msg in [([("species", "X y", "A b", "", "", "accept")], [], "no image has species"),
                                    ([("colour", "A b", "C d", "", "", "accept")], [], "not a label column"),
                                    ([("species", "A b", "C d", "", "", "maybe")], [], "unknown decision"),
                                    ([], [("nosuchsha", "", "drop")], "no image with sha256"),
                                    ([], [("h0", "", "drop; set species C d")], "unknown decision"),
                                    ([], [("h0", "", "set colour red")], "unknown decision")]:
            write_sheets(dec, renames, fixes)
            try:
                il.apply_fixes([dict(r, label_change=[]) for r in rows], ["species"], dec / "labels", {"p/a.jpg": "h0"})
            except SystemExit as e:
                assert msg in str(e), (msg, str(e))
                continue
            raise AssertionError(f"accepted {renames} {fixes}")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
