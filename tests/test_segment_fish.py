"""Unit tests for the per-instance rules in scripts/segment_fish.py.

Synthetic masks only - no model is loaded, so this runs in seconds.
Requires PYTHONPATH=/workspace/src (set in the Dockerfile). Runs under
pytest or directly: python tests/test_segment_fish.py
"""

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import segment_fish as sf  # noqa: E402


def rect_mask(h, w, x0, y0, x1, y1):
    m = np.zeros((h, w), bool)
    m[y0:y1, x0:x1] = True
    return m


def test_best_candidate_is_highest_iou_not_index_0():
    cands = np.zeros((2, 3, 4, 4), bool)
    cands[0, 2, 0, 0] = True  # instance 0: best is candidate 2
    cands[1, 1, 1, 1] = True  # instance 1: best is candidate 1
    iou = np.array([[0.5, 0.6, 0.9], [0.2, 0.8, 0.7]])
    masks, chosen = sf.best_candidates(cands, iou)
    assert chosen.tolist() == [2, 1]
    assert masks[0, 0, 0] and masks[1, 1, 1]


def test_head_box_inside_body_is_dropped():
    body = rect_mask(100, 100, 10, 10, 90, 40)
    head = rect_mask(100, 100, 10, 10, 30, 40)
    masks = np.stack([body, head])
    boxes = np.array([[10, 10, 90, 40], [10, 10, 30, 40]], float)
    kept, dropped = sf.remove_duplicates(masks, boxes, np.array([0.6, 0.4]), np.zeros(2, bool), 0.8)
    assert kept == [0] and dropped == [(1, 0)]


def test_group_box_does_not_swallow_single_fish():
    fish_a = rect_mask(100, 100, 10, 10, 40, 30)
    fish_b = rect_mask(100, 100, 60, 10, 90, 30)
    group = fish_a | fish_b
    masks = np.stack([fish_a, fish_b, group])
    boxes = np.array([[10, 10, 40, 30], [60, 10, 90, 30], [10, 10, 90, 30]], float)
    kept, dropped = sf.remove_duplicates(masks, boxes, np.array([0.5, 0.5, 0.38]), np.zeros(3, bool), 0.8)
    assert kept == [0, 1] and [i for i, _ in dropped] == [2]


def test_touching_neighbour_under_bleeding_mask_is_kept():
    # SAM mask of A bled over all of B, but B's own box is not inside A's box.
    a_bleeding = rect_mask(100, 100, 10, 10, 90, 30)
    b = rect_mask(100, 100, 60, 10, 90, 30)
    boxes = np.array([[10, 10, 58, 30], [60, 10, 90, 30]], float)
    kept, dropped = sf.remove_duplicates(np.stack([a_bleeding, b]), boxes, np.array([0.7, 0.5]),
                                         np.zeros(2, bool), 0.8)
    assert kept == [0, 1] and dropped == []


def test_giant_is_never_dropped_and_never_drops_others():
    giant = np.ones((100, 100), bool)
    fish = rect_mask(100, 100, 10, 10, 40, 30)
    boxes = np.array([[0, 0, 100, 100], [10, 10, 40, 30]], float)
    kept, dropped = sf.remove_duplicates(np.stack([giant, fish]), boxes, np.array([0.9, 0.5]),
                                         np.array([True, False]), 0.8)
    assert kept == [0, 1] and dropped == []


def test_contested_pixel_goes_to_box_owner_over_higher_score():
    a = rect_mask(20, 40, 0, 0, 30, 20)   # bleeds into x 20-30
    b = rect_mask(20, 40, 20, 0, 40, 20)
    boxes = np.array([[0, 0, 20, 20], [20, 0, 40, 20]], float)
    out, contested = sf.resolve_overlaps(np.stack([a, b]), boxes, np.array([0.9, 0.4]))
    assert contested == 20 * 10
    assert not (out[0] & out[1]).any()
    assert out[1][:, 21:30].all() and not out[0][:, 21:30].any()


def test_contested_pixel_in_both_boxes_goes_to_higher_score():
    a = rect_mask(20, 20, 0, 0, 20, 20)
    b = rect_mask(20, 20, 5, 5, 15, 15)
    boxes = np.array([[0, 0, 20, 20], [0, 0, 20, 20]], float)
    out, _ = sf.resolve_overlaps(np.stack([a, b]), boxes, np.array([0.4, 0.9]))
    assert out[1][5:15, 5:15].all() and not out[0][5:15, 5:15].any()


def test_size_flags():
    assert sf.size_flags(0.001, 0.005, 0.9) == ["tiny"]
    assert sf.size_flags(0.95, 0.005, 0.9) == ["giant"]
    assert sf.size_flags(0.3, 0.005, 0.9) == []


def test_coco_geometry_maps_back_to_original_resolution():
    m = rect_mask(50, 50, 2, 2, 12, 12)  # 10x10 px at working res
    seg, bbox, area = sf.mask_to_coco(m, 2.0, 2.0)
    assert bbox == [4.0, 4.0, 20.0, 20.0]
    assert area == 400.0
    xs, ys = seg[0][0::2], seg[0][1::2]
    assert min(xs) == 4.0 and min(ys) == 4.0


def test_load_done_drops_truncated_last_line():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "annotations.jsonl"
        p.write_text(json.dumps({"file_name": "a.jpg", "status": "ok"}) + "\n" + '{"file_name": "b.jp')
        done = sf.load_done(p)
        assert list(done) == ["a.jpg"]
        assert p.read_text().endswith("}\n")  # clean for appending


def test_load_done_refuses_corrupt_middle_line():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "annotations.jsonl"
        p.write_text('{"file_name": "a.jp\n' + json.dumps({"file_name": "b.jpg"}) + "\n")
        try:
            sf.load_done(p)
        except SystemExit:
            return
        raise AssertionError("corrupt middle line was accepted")


def test_resume_with_changed_settings_is_refused():
    with tempfile.TemporaryDirectory() as d:
        run_dir = Path(d)
        sf.prepare_run_config(run_dir, {"tiny_area_frac": 0.005}, resume=False)
        sf.prepare_run_config(run_dir, {"tiny_area_frac": 0.005}, resume=True)  # same: fine
        assert len(json.loads((run_dir / "run_config.json").read_text())["sessions"]) == 2
        try:
            sf.prepare_run_config(run_dir, {"tiny_area_frac": 0.01}, resume=True)
        except SystemExit:
            return
        raise AssertionError("resume with different settings was accepted")


if __name__ == "__main__":
    tests = [(name, fn) for name, fn in list(globals().items()) if name.startswith("test_")]
    for name, fn in tests:
        fn()
        print(f"  ok  {name}")
    print(f"All {len(tests)} segment_fish tests passed.")
