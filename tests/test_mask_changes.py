"""Tests for scripts/mask_changes.py on tiny fake runs (no model, no real images).

Runs under pytest or directly: python tests/test_mask_changes.py
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import mask_changes as mc  # noqa: E402

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "mask_changes.py"


def fake_run(root, masks, records=None, skipped=()):
    """A run dir with one 10x10 0/255 mask per name (``masks``: {name: bool array})."""
    (root / "masks").mkdir(parents=True)
    records = records or {}
    lines = []
    for name, m in masks.items():
        Image.fromarray(np.where(m, 255, 0).astype(np.uint8)).save(root / "masks" / f"{Path(name).stem}.png")
        lines.append(records.get(name, {"file_name": name, "status": "ok", "qa": {}, "instances": [{}]}))
    lines += [{"file_name": n, "status": "skipped", "reason": "no_detection"} for n in skipped]
    (root / "annotations.jsonl").write_text("".join(json.dumps(r) + "\n" for r in lines))
    return root


def box(y0, y1, x0, x1):
    m = np.zeros((10, 10), bool)
    m[y0:y1, x0:x1] = True
    return m


def test_mask_iou():
    assert mc.mask_iou(box(0, 5, 0, 5), box(0, 5, 0, 5)) == 1.0
    assert mc.mask_iou(box(0, 5, 0, 4), box(0, 5, 2, 6)) == 10 / 30
    assert mc.mask_iou(np.zeros((3, 3), bool), np.zeros((3, 3), bool)) == 1.0
    try:
        mc.mask_iou(np.zeros((3, 3), bool), np.zeros((3, 4), bool))
    except ValueError:
        return
    raise AssertionError("different sizes accepted")


def test_rules_fired():
    rec = {"qa": {"wrong_region_dropped": 1},
           "instances": [{"bleed_fallback": "none", "wrong_region": "reprompt", "far_pieces_dropped": 0,
                          "holes_filled_px": 12},
                         {"bleed_fallback": "inverse", "wrong_region": "none", "far_pieces_dropped": 2,
                          "holes_filled_px": 0}]}
    assert mc.rules_fired(rec) == ["bleed_inverse", "far_pieces_dropped", "holes_filled",
                                   "wrong_region_dropped", "wrong_region_reprompt"]
    assert mc.rules_fired({"qa": {}, "instances": [{"bleed_fallback": "none", "wrong_region": "none"}]}) == []
    assert mc.rules_fired({"status": "ok", "qa": {}, "instances": [{}]}) == []  # B records lack the D fields
    fins = [{"fin_extension": "extended", "fin_growth": 0.05}, {"fin_extension": "guarded", "fin_growth": 0.9},
            {"fin_extension": "extended", "fin_growth": 0.0}, {"fin_extension": "skipped", "fin_growth": 0.0}]
    assert mc.rules_fired({"qa": {}, "instances": fins}) == ["fin_extended", "fin_guarded"]


def test_cli_writes_lists_counts_and_seeded_pick():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        same, holes = box(0, 5, 0, 5), box(0, 5, 0, 5)
        holes_rec = {"file_name": "h.jpg", "status": "ok", "qa": {}, "instances": [{"holes_filled_px": 3}]}
        before = fake_run(d / "B", {"s.jpg": same, "h.jpg": box(0, 5, 0, 4), "c.jpg": box(0, 2, 0, 2),
                                    "x.jpg": same}, skipped=["n.jpg"])
        after = fake_run(d / "D", {"s.jpg": same, "h.jpg": holes, "c.jpg": box(5, 9, 5, 9), "n.jpg": same},
                         records={"h.jpg": holes_rec}, skipped=["x.jpg"])
        before_files = {p: p.read_bytes() for p in before.rglob("*") if p.is_file()}
        r = subprocess.run([sys.executable, str(SCRIPT), "--before", str(before), "--after", str(after),
                            "--name", "B", "--pick", "1", "--seed", "0", "--workers", "2"],
                           check=True, capture_output=True, text=True)
        summary = json.loads((after / "changed_vs_B.json").read_text())
        assert json.loads(r.stdout) == summary
        assert (summary["compared"], summary["changed"], summary["changed_without_rule"]) == (3, 2, 1)
        assert (summary["ok_only_before"], summary["ok_only_after"]) == (1, 1)  # x.jpg / n.jpg
        assert summary["rule_images"] == {"holes_filled": 1} == summary["rule_images_changed"]
        rows = (after / "changed_vs_B.tsv").read_text().splitlines()
        assert rows[0].startswith("file_name\tiou") and "h.jpg\t0.8000\t20\t25\tholes_filled" in rows
        changed = [n for n in (after / "changed_vs_B.txt").read_text().splitlines() if not n.startswith("#")]
        assert changed == ["c.jpg", "h.jpg"]
        expected = ["c.jpg", "h.jpg"]
        mc.random.Random(0).shuffle(expected)
        picked = (after / "changed_vs_B_pick1_seed0.txt").read_text().splitlines()[1:]
        assert picked == expected[:1]
        assert all(p.read_bytes() == b for p, b in before_files.items())  # --before untouched
        r = subprocess.run([sys.executable, str(SCRIPT), "--before", str(before), "--after", str(after),
                            "--pick", "3"], capture_output=True, text=True)
        assert r.returncode != 0 and "only 2 changed" in r.stderr


if __name__ == "__main__":
    tests = [(name, fn) for name, fn in list(globals().items()) if name.startswith("test_")]
    for name, fn in tests:
        fn()
        print(f"  ok  {name}")
    print(f"All {len(tests)} mask_changes tests passed.")
