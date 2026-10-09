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



def test_drop_list_and_drops():
    drops = mr.read_drop_list("# header\na.jpg\tafter fallback: bleed\nb.jpg\n\n")
    assert drops == {"a.jpg": "after fallback: bleed", "b.jpg": "dropped"}
    merged = {n: rec(n, "base") for n in ("a.jpg", "b.jpg", "c.jpg")}
    out = mr.apply_drops(merged, {"a.jpg": "bleed"}, take=["c.jpg"])
    assert out["a.jpg"] == {"file_name": "a.jpg", "status": "dropped", "reason": "bleed"}
    assert out["b.jpg"]["status"] == "ok"
    for bad in ({"z.jpg": "x"}, {"c.jpg": "x"}):  # unknown / also taken
        try:
            mr.apply_drops(merged, bad, take=["c.jpg"])
        except SystemExit:
            continue
        raise AssertionError(f"drop {bad} was accepted")


def test_cli_drop_excludes_from_coco_and_masks():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        base = fake_run(d / "base", [rec("a.jpg", "base"), rec("b.jpg", "base"), rec("c.jpg", "base")])
        patch = fake_run(d / "patch", [rec("a.jpg", "patch")])
        (d / "take.txt").write_text("a.jpg\n")
        (d / "drop.txt").write_text("b.jpg\tstill bleeds\n")
        subprocess.run([sys.executable, str(SCRIPT), "--base", str(base), "--patch", str(patch),
                        "--take", str(d / "take.txt"), "--drop", str(d / "drop.txt"), "--out", str(d / "out")],
                       check=True, capture_output=True)
        out = d / "out"
        coco = json.loads((out / "coco" / "annotations.json").read_text())
        assert [i["file_name"] for i in coco["images"]] == ["a.jpg", "c.jpg"]
        assert not (out / "masks" / "b.png").exists() and (base / "masks" / "b.png").exists()
        assert (out / "dropped.txt").read_text() == "b.jpg\tstill bleeds\n"
        assert (out / "skipped.txt").read_text() == ""


def test_cli_drop_only_mode_needs_no_patch():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        base = fake_run(d / "base", [rec("a.jpg", "base"), rec("b.jpg", "base")])
        (d / "drop.txt").write_text("b.jpg\tbleed\n")
        subprocess.run([sys.executable, str(SCRIPT), "--base", str(base), "--drop", str(d / "drop.txt"),
                        "--out", str(d / "out")], check=True, capture_output=True)
        out = d / "out"
        coco = json.loads((out / "coco" / "annotations.json").read_text())
        assert [i["file_name"] for i in coco["images"]] == ["a.jpg"]
        assert (out / "masks" / "a.png").read_text() == "base" and not (out / "masks" / "b.png").exists()
        assert (out / "dropped.txt").read_text() == "b.jpg\tbleed\n"
        cfg = json.loads((out / "run_config.json").read_text())
        assert cfg["merged"]["patch"] is None and cfg["patch_config"] is None and cfg["merged"]["n_taken"] == 0
        # without --patch: --take is refused, and --drop is required
        (d / "take.txt").write_text("a.jpg\n")
        for extra in (["--drop", str(d / "drop.txt"), "--take", str(d / "take.txt")], []):
            r = subprocess.run([sys.executable, str(SCRIPT), "--base", str(base), "--out", str(d / "out2"), *extra],
                               capture_output=True)
            assert r.returncode != 0 and not (d / "out2").exists(), extra


def test_qa_rule_b():
    def inst(**kw):
        return {"outside_box_frac": 0.0, "bleed_fallback": "none", "edge_uncertain_frac": 0.1, **kw}
    ok = {"status": "ok", "instances": [inst(), inst()]}
    assert mr.qa_rule_b(ok) is None
    assert mr.qa_rule_b({"status": "ok", "instances": [inst(), inst(outside_box_frac=0.03)]}) == "qa_rule_B: outside_box"
    assert mr.qa_rule_b({"status": "ok", "instances": [inst(bleed_fallback="inverse", edge_uncertain_frac=0.6)]}) == (
        "qa_rule_B: edge_uncertain,inverse")
    assert mr.qa_rule_b({"status": "ok", "instances": [inst(bleed_fallback="unresolved")]}) == "qa_rule_B: unresolved"
    assert mr.qa_rule_b({"status": "ok", "instances": [inst(bleed_fallback="candidate", outside_box_frac=0.02)]}) is None
    assert mr.qa_rule_b({"status": "dropped", "reason": "x"}) is None


def test_cli_drop_sha_and_qa_rule():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        flagged = rec("c.jpg", "base")
        flagged["instances"][0]["bleed_fallback"] = "inverse"
        both = rec("b.jpg", "base")
        both["instances"][0]["edge_uncertain_frac"] = 0.9
        base = fake_run(d / "base", [rec("a.jpg", "base"), both, flagged])
        raw = d / "raw"
        raw.mkdir()
        for k, n in enumerate(["a.jpg", "b.jpg", "c.jpg"]):
            (raw / n).write_bytes(bytes([k]))
        (base / "run_config.json").write_text(json.dumps({"settings": {"raw_dir": str(raw)}, "sessions": []}))
        sha_b = mr.file_sha256(raw / "b.jpg", {})
        (d / "drops.tsv").write_text(f"# reviewed\nsha256\treason\n{sha_b}\tfirst review: bleed\n")
        subprocess.run([sys.executable, str(SCRIPT), "--base", str(base), "--drop-sha", str(d / "drops.tsv"),
                        "--qa-rule", "B", "--hash-cache", str(d / "cache.json"), "--out", str(d / "out")],
                       check=True, capture_output=True)
        # the list's reason wins over the rule for b; c is dropped by the rule alone
        assert (d / "out" / "dropped.txt").read_text() == "b.jpg\tfirst review: bleed\nc.jpg\tqa_rule_B: inverse\n"
        cfg = json.loads((d / "out" / "run_config.json").read_text())["merged"]
        assert (cfg["n_drop_sha"], cfg["n_qa_rule_dropped"], cfg["n_dropped"]) == (1, 1, 2)
        # a sha256 that matches no image is refused
        (d / "bad.tsv").write_text("sha256\treason\n" + "0" * 64 + "\tx\n")
        r = subprocess.run([sys.executable, str(SCRIPT), "--base", str(base), "--drop-sha", str(d / "bad.tsv"),
                            "--hash-cache", str(d / "cache.json"), "--out", str(d / "out3")], capture_output=True)
        assert r.returncode != 0 and not (d / "out3").exists()


if __name__ == "__main__":
    tests =[(name, fn) for name, fn in list(globals().items()) if name.startswith("test_")]
    for name, fn in tests:
        fn()
        print(f"  ok  {name}")
    print(f"All {len(tests)} merge_runs tests passed.")
