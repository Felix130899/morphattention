"""Tests for scripts/merge_runs.py on tiny fake runs (no model, no real images).

Runs under pytest or directly: python tests/test_merge_runs.py
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import merge_runs as mr  # noqa: E402

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "merge_runs.py"


def rec(name, tag, status="ok"):
    if status != "ok":
        return {"file_name": name, "status": status, "reason": "no_detection"}
    return {"file_name": name, "status": "ok", "width": 10, "height": 10, "qa": {"flags": [tag]},
            "instances": [{"segmentation": [[0, 0, 5, 0, 5, 5]], "bbox": [0, 0, 5, 5], "area": 25.0}]}


def fake_run(root, records):
    root.mkdir(parents=True)
    for d in ("masks", "overlays"):
        (root / d).mkdir()
    for r in records:
        if r["status"] == "ok":
            stem = Path(r["file_name"]).stem
            (root / "masks" / f"{stem}.png").write_text(r["qa"]["flags"][0])
            (root / "overlays" / f"{stem}.jpg").write_text(r["qa"]["flags"][0])
    (root / "annotations.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    (root / "run_config.json").write_text(json.dumps({"settings": {"raw_dir": "raw"}, "sessions": []}))
    return root


def test_merge_replaces_only_taken_and_keeps_order():
    base = {n: rec(n, "base") for n in ("c.jpg", "a.jpg", "b.jpg")}
    patch = {n: rec(n, "patch") for n in ("a.jpg", "b.jpg")}
    out = mr.merge_records(base, patch, ["a.jpg"])
    assert list(out) == ["c.jpg", "a.jpg", "b.jpg"]
    assert [r["qa"]["flags"][0] for r in out.values()] == ["base", "patch", "base"]


def test_merge_refuses_unknown_or_skipped_patch_images():
    base = {"a.jpg": rec("a.jpg", "base"), "b.jpg": rec("b.jpg", "base")}
    patch = {"a.jpg": rec("a.jpg", "patch", status="skipped")}
    for take in (["a.jpg"], ["b.jpg"], ["z.jpg"]):
        try:
            mr.merge_records(base, patch, take)
        except SystemExit:
            continue
        raise AssertionError(f"take {take} was accepted")


def test_ok_names_from_compare_review():
    with tempfile.TemporaryDirectory() as d:
        lines = [{"item": 0, "run": "new", "file_name": "a.jpg", "verdict": "fix", "categories": ["bleed"]},
                 {"item": 0, "run": "new", "file_name": "a.jpg", "verdict": "ok", "categories": []},
                 {"item": 1, "run": "old", "file_name": "b.jpg", "verdict": "ok", "categories": []},
                 {"item": 2, "run": "new", "file_name": "b.jpg", "verdict": "fix", "categories": ["bleed"]}]
        (Path(d) / "decisions.jsonl").write_text("".join(json.dumps(x) + "\n" for x in lines))
        assert mr.ok_names_from_review(d, "new") == ["a.jpg"]  # last line wins
        try:
            mr.ok_names_from_review(d, "nope")
        except SystemExit:
            return
        raise AssertionError("unknown run accepted")


def test_cli_end_to_end_leaves_inputs_untouched():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        base = fake_run(d / "base", [rec("a.jpg", "base"), rec("b.jpg", "base"), rec("s.jpg", "", "skipped")])
        patch = fake_run(d / "patch", [rec("a.jpg", "patch")])
        before = {p: p.read_bytes() for p in list(base.rglob("*")) + list(patch.rglob("*")) if p.is_file()}
        (d / "take.txt").write_text("# fixed\na.jpg\n")
        subprocess.run([sys.executable, str(SCRIPT), "--base", str(base), "--patch", str(patch),
                        "--take", str(d / "take.txt"), "--out", str(d / "out")], check=True, capture_output=True)
        out = d / "out"
        assert (out / "masks" / "a.png").read_text() == "patch"
        assert (out / "masks" / "b.png").read_text() == "base"
        coco = json.loads((out / "coco" / "annotations.json").read_text())
        assert [i["file_name"] for i in coco["images"]] == ["a.jpg", "b.jpg"]
        assert (out / "skipped.txt").read_text().startswith("s.jpg")
        assert (out / "merged_from_patch.txt").read_text() == "a.jpg\n"
        assert json.loads((out / "run_config.json").read_text())["merged"]["n_taken"] == 1
        assert all(p.read_bytes() == b for p, b in before.items())  # inputs unchanged


if __name__ == "__main__":
    tests = [(name, fn) for name, fn in list(globals().items()) if name.startswith("test_")]
    for name, fn in tests:
        fn()
        print(f"  ok  {name}")
    print(f"All {len(tests)} merge_runs tests passed.")
