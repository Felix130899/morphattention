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


def random_logits(n=2, h=12, w=16, seed=0):
    return np.random.default_rng(seed).normal(0, 3, (n, 3, h, w)).astype(np.float32)


def test_default_candidate_choice_reproduces_best_candidates():
    # Defaults (score, threshold 0) must give the old masks bit for bit.
    logits = random_logits()
    iou = np.array([[0.5, 0.6, 0.9], [0.2, 0.8, 0.7]])
    masks, chosen, areas = sf.choose_candidates(logits, iou)
    old_masks, old_chosen = sf.best_candidates(logits > 0, iou)
    assert np.array_equal(masks, old_masks) and np.array_equal(chosen, old_chosen)
    assert areas.tolist() == (logits > 0).reshape(2, 3, -1).sum(axis=2).tolist()


def test_largest_candidate_and_threshold():
    logits = np.full((1, 3, 10, 10), -5.0, np.float32)
    logits[0, 0, 2:8, 2:8] = 5.0            # body, 36 px, best IoU
    logits[0, 1, 2:8, 2:8] = 5.0
    logits[0, 1, 2:8, 8:10] = -1.0          # body + faint fin (12 px at logit -1)
    logits[0, 2, 0:2, 0:2] = 5.0            # 4 px
    iou = np.array([[0.9, 0.5, 0.1]])
    # At threshold 0 the fin is outside; candidates 0 and 1 tie -> lowest index.
    _, chosen, areas = sf.choose_candidates(logits, iou, 0.0, "largest")
    assert areas.tolist() == [[36, 36, 4]] and chosen.tolist() == [0]
    # At -2 the fin is in candidate 1, which is now the largest.
    masks, chosen, areas = sf.choose_candidates(logits, iou, -2.0, "largest")
    assert areas.tolist() == [[36, 48, 4]] and chosen.tolist() == [1]
    assert masks[0, 2:8, 8:10].all()
    # "score" ignores area.
    _, chosen, _ = sf.choose_candidates(logits, iou, -2.0, "score")
    assert chosen.tolist() == [0]


def test_threshold_is_strict_like_hf():
    logits = np.zeros((1, 3, 2, 2), np.float32)
    logits[0, :, 0, 0] = 0.5
    masks, _, _ = sf.choose_candidates(logits, np.array([[1.0, 0, 0]]), 0.0)
    assert masks[0].sum() == 1  # logit == threshold is outside


def test_margin_off_is_identity():
    masks = np.stack([rect_mask(20, 20, 2, 2, 8, 8), rect_mask(20, 20, 12, 12, 18, 18)])
    out, r = sf.grow_margins(masks, 0.0)
    assert np.array_equal(out, masks) and r.tolist() == [0, 0]


def test_margin_radius_and_euclidean_growth():
    m = rect_mask(60, 60, 20, 20, 40, 40)  # 400 px -> sqrt 20
    out, r = sf.grow_margins(m[None], 0.1)  # r = round(2.0) = 2
    assert r.tolist() == [2]
    g = out[0]
    assert g[18:42, 20:40].all() and g[20:40, 18:42].all()  # 2 px on every side
    assert not g[17, 30] and not g[30, 42]                    # but not 3
    assert g[19, 19] and not g[18, 19] and not g[18, 18]     # round corner: sqrt(2) <= 2 < sqrt(5)
    tiny = rect_mask(10, 10, 5, 5, 6, 6)
    _, r = sf.grow_margins(tiny[None], 0.01)
    assert r.tolist() == [1]  # never below 1 px


def test_margin_never_steals_from_neighbour_and_nearer_wins():
    # Two fish 4 px apart (columns 10-13 are background), big margins.
    a = rect_mask(20, 30, 0, 5, 10, 15)
    b = rect_mask(20, 30, 14, 5, 24, 15)
    out, r = sf.grow_margins(np.stack([a, b]), 0.5)
    assert (r >= 4).all()
    assert out[0][a].all() and out[1][b].all()          # nothing lost
    assert not out[0][b].any() and not out[1][a].any()  # nothing stolen
    assert not (out[0] & out[1]).any()                  # still disjoint
    # Gap columns: 10, 11 nearer A; 12, 13 nearer B (11 vs 12: dists 2 vs 3).
    assert out[0][5:15, 10:12].all() and out[1][5:15, 12:14].all()


def test_margin_tie_goes_to_lower_index():
    a = rect_mask(10, 21, 0, 0, 10, 10)
    b = rect_mask(10, 21, 11, 0, 21, 10)
    out, _ = sf.grow_margins(np.stack([a, b]), 0.5)  # column 10 equidistant
    assert out[0][:, 10].all() and not out[1][:, 10].any()


def test_edge_uncertain_frac_flags_soft_fin():
    m = rect_mask(100, 100, 30, 30, 70, 70)  # 1600 px -> ring w = max(1, round(0.02 * 40)) = 1
    ring = 4 * 40  # 1-px ring; diagonal corners are sqrt(2) > 1 away
    logits = np.full((100, 100), -4.0, np.float32)  # background just below the band
    logits[m] = 5.0
    assert sf.edge_uncertain_frac(m, logits, 0.0) == 0.0  # crisp edge
    logits[30:70, 70:75] = -1.0  # soft "fin" right of the body
    assert sf.edge_uncertain_frac(m, logits, 0.0) == 40 / ring
    # The band moves with the threshold: at -1.5 it is [-4.5, -1.5], which
    # holds the -4 background ring but not the -1 fin.
    assert sf.edge_uncertain_frac(m, logits, -1.5) == 120 / ring
    # Pixels of another instance are not part of the ring.
    neighbour = rect_mask(100, 100, 70, 0, 100, 100)
    assert sf.edge_uncertain_frac(m, logits, 0.0, blocked=neighbour) == 0.0
    # A mask filling the image has no ring.
    full = np.ones((10, 10), bool)
    assert sf.edge_uncertain_frac(full, np.zeros((10, 10), np.float32), 0.0) == 0.0


def test_edge_ring_width_scales_with_size():
    m = rect_mask(400, 400, 100, 100, 300, 300)  # sqrt 200 -> w = 4
    logits = np.full((400, 400), -10.0, np.float32)
    logits[m] = 5.0
    logits[100:300, 300:304] = -1.0  # band exactly 4 px wide right of the body
    frac = sf.edge_uncertain_frac(m, logits, 0.0)
    assert frac == 4 * 200 / (4 * 4 * 200 + 4 * 8)  # ring: 4 sides x 4 px + 8 px per round corner
    logits[100:300, 304] = -1.0      # a 5th column is outside the ring
    assert sf.edge_uncertain_frac(m, logits, 0.0) == frac


def test_border_and_components():
    m = rect_mask(20, 20, 5, 5, 10, 10)
    assert not sf.touches_border(m) and sf.n_components(m) == 1
    m[0, 15] = True
    assert sf.touches_border(m) and sf.n_components(m) == 2
    d = np.zeros((5, 5), bool)
    d[1, 1] = d[2, 2] = True  # diagonal neighbours: one 8-connected component
    assert sf.n_components(d) == 1


def test_drop_small_components_keeps_largest_and_big_pieces():
    m = rect_mask(50, 50, 5, 5, 25, 25)    # 400 px fish
    m[40:44, 40:44] = True                 # 16 px piece (~3.8 %)
    m[1, 48] = True                        # 1 px speck
    assert (sf.drop_small_components(m, 0.0) == m).all()
    out = sf.drop_small_components(m, 0.01)
    assert out[40:44, 40:44].all() and not out[1, 48] and out[5:25, 5:25].all()
    out = sf.drop_small_components(m, 0.5)  # only the largest piece survives
    assert out.sum() == 400
    speck = np.zeros((10, 10), bool)
    speck[3, 3] = True
    assert (sf.drop_small_components(speck, 0.9) == speck).all()  # never empties a mask


def test_parse_image_list_skips_comments_blanks_and_repeats():
    text = "# dev set v1\n\n  a.jpg  \nsub/b.jpg\n# c.jpg\na.jpg\n"
    assert sf.parse_image_list(text) == ["a.jpg", "sub/b.jpg"]


def test_select_images_fails_loudly_on_unknown_names():
    rel = {Path("/r/a.jpg"): "a.jpg", Path("/r/sub/b.jpg"): "sub/b.jpg", Path("/r/c.jpg"): "c.jpg"}
    assert list(sf.select_images(rel, ["sub/b.jpg", "a.jpg"]).values()) == ["a.jpg", "sub/b.jpg"]
    try:
        sf.select_images(rel, ["a.jpg", "nope.jpg", "b.jpg"])
    except SystemExit as e:
        assert "nope.jpg" in str(e) and "b.jpg" in str(e)  # every offender listed
        return
    raise AssertionError("unknown image names were accepted")


def default_args(**kw):
    from types import SimpleNamespace

    a = dict(raw_dir="raw", prompt="center", max_side=2048, dedup_containment=0.8, tiny_area_frac=0.001,
             giant_area_frac=0.9, candidate="score", mask_threshold=0.0, margin_frac=0.0, min_component_frac=0.0,
             image_list=None, bleed_outside_frac=0.0, bleed_fill=(0.2, 0.9), bleed_bg_border=0.3,
             bleed_max_box_edge=0.1, wrong_region_fix=False, keep_near_frac=0.0, fill_holes_frac=0.0)
    a.update(kw)
    return SimpleNamespace(**a)


def test_new_settings_are_recorded_and_list_content_is_hashed():
    s = sf.run_settings(default_args(candidate="largest", mask_threshold=-2.0, margin_frac=0.03,
                                     image_list=Path("dev.txt")), ["b.jpg", "a.jpg"])
    assert s["candidate"] == "largest" and s["mask_threshold"] == -2.0 and s["margin_frac"] == 0.03
    assert s["image_list"] == str(Path("dev.txt").resolve())
    same = sf.run_settings(default_args(image_list=Path("dev.txt")), ["a.jpg", "b.jpg"])
    other = sf.run_settings(default_args(image_list=Path("dev.txt")), ["a.jpg", "c.jpg"])
    assert s["image_list_sha256"] == same["image_list_sha256"] != other["image_list_sha256"]


def test_resume_of_pre_option_run_accepts_defaults_only():
    with tempfile.TemporaryDirectory() as d:
        run_dir = Path(d)
        old = sf.run_settings(default_args())
        for k in sf.LEGACY_DEFAULTS:
            del old[k]  # a run_config.json written before the new options
        (run_dir / "run_config.json").write_text(json.dumps({"settings": old, "sessions": [{}]}))
        sf.prepare_run_config(run_dir, sf.run_settings(default_args()), resume=True)
        cfg = json.loads((run_dir / "run_config.json").read_text())
        assert cfg["settings"]["candidate"] == "score"  # written back explicitly
        for changed in (dict(candidate="largest"), dict(mask_threshold=-2.0), dict(margin_frac=0.03),
                        dict(min_component_frac=0.01), dict(bleed_outside_frac=0.02),
                        dict(wrong_region_fix=True), dict(keep_near_frac=0.02), dict(fill_holes_frac=0.02),
                        dict(image_list=Path("dev.txt"))):
            try:
                sf.prepare_run_config(run_dir, sf.run_settings(default_args(**changed),
                                                               ["a.jpg"] if "image_list" in changed else None),
                                      resume=True)
            except SystemExit:
                continue
            raise AssertionError(f"resume with {changed} was accepted")


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


# --- bleed fallback -------------------------------------------------------
# 100x100 image, detector box (20, 20)-(80, 60): a fish-shaped mask inside it,
# a bleed that fills the box and spills into the background, and the
# background itself (everything but the fish).
BOX = np.array([20, 20, 80, 60], float)
FISH = rect_mask(100, 100, 25, 30, 75, 50)   # fills 1000/2400 of the box
BLEED = rect_mask(100, 100, 0, 0, 100, 100)
BACKGROUND = BLEED & ~FISH


def bleed_args(**kw):
    return default_args(bleed_outside_frac=0.02, **kw)


def test_box_fit():
    assert sf.box_fit(FISH, BOX) == (0.0, 1000 / 2400)
    out, fill = sf.box_fit(BLEED, BOX)
    assert fill == 1.0 and abs(out - (1 - 2400 / 10000)) < 1e-9


def test_bleed_fallback_off_or_not_triggered_keeps_choice():
    cands = np.stack([BLEED, FISH, BACKGROUND])
    m, src, k, inv = sf.bleed_fallback(cands, 0, BOX, default_args())  # off
    assert src == "none" and k == 0 and (m == BLEED).all()
    m, src, k, inv = sf.bleed_fallback(cands, 1, BOX, bleed_args())  # fish chosen: inside the box
    assert src == "none" and k == 1


def test_bleed_fallback_takes_the_candidate_inside_the_box():
    cands = np.stack([BLEED, FISH, BACKGROUND])
    m, src, k, inv = sf.bleed_fallback(cands, 0, BOX, bleed_args())
    assert src == "candidate" and k == 1 and not inv and (m == FISH).all()


def test_bleed_fallback_prefers_the_largest_valid_candidate():
    with_fins = rect_mask(100, 100, 22, 25, 78, 55)  # fills 1680/2400 = 0.7
    cands = np.stack([BLEED, FISH, with_fins])
    _, src, k, _ = sf.bleed_fallback(cands, 0, BOX, bleed_args())
    assert src == "candidate" and k == 2


def test_bleed_fallback_rejects_box_shaped_and_tiny_candidates():
    box_blob = rect_mask(100, 100, 20, 20, 80, 60)   # fills the box: bleed inside it
    eye = rect_mask(100, 100, 30, 35, 35, 40)        # fills 25/2400
    cands = np.stack([BLEED, box_blob, eye])
    m, src, k, _ = sf.bleed_fallback(cands, 0, BOX, bleed_args())
    assert src == "unresolved" and k == 0 and (m == BLEED).all()


def test_bleed_fallback_inverts_a_background_candidate():
    eye = rect_mask(100, 100, 30, 35, 35, 40)
    cands = np.stack([BLEED, eye, BACKGROUND])  # SAM never returned the fish itself
    m, src, k, inv = sf.bleed_fallback(cands, 0, BOX, bleed_args())
    assert src == "inverse" and k == 2 and inv and (m == FISH).all()


def test_bleed_fallback_inverse_drops_text_specks_and_needs_a_background_candidate():
    eye = rect_mask(100, 100, 30, 35, 35, 40)
    text = rect_mask(100, 100, 70, 55, 72, 57)  # a printed letter inside the box
    cands = np.stack([BLEED, eye, BACKGROUND & ~text])
    m, src, _, _ = sf.bleed_fallback(cands, 0, BOX, bleed_args())
    assert src == "inverse" and (m == FISH).all()
    # Not background-like (covers little of the frame): never inverted.
    m, src, _, _ = sf.bleed_fallback(cands, 0, BOX, bleed_args(bleed_bg_border=1.01))
    assert src == "unresolved"



def test_box_edge_frac():
    assert sf.box_edge_frac(FISH, BOX) == 0.0
    assert sf.box_edge_frac(BLEED, BOX) == 1.0


def test_inverse_pieces_drops_border_pieces_but_keeps_inner_ones():
    corner = rect_mask(100, 100, 0, 0, 10, 10)     # canvas corner: touches the border
    fin = rect_mask(100, 100, 80, 35, 85, 45)      # detached piece inside the image
    out = sf.inverse_pieces(FISH | corner | fin, 0.01)
    assert (out == (FISH | fin)).all()


def test_bleed_fallback_rejects_inverse_running_along_the_box_edge():
    eye = rect_mask(100, 100, 30, 35, 35, 40)
    in_box_bg = rect_mask(100, 100, 20, 20, 80, 60) & ~FISH  # background left inside the box
    inv_source = BLEED & ~in_box_bg                           # its inverse = in-box background
    cands = np.stack([BLEED, eye, inv_source])
    _, src, _, _ = sf.bleed_fallback(cands, 0, BOX, bleed_args())
    assert src == "unresolved"


# --- wrong-region fix -----------------------------------------------------
# A box that spans the whole 100x100 image (close-up / X-ray canvas): the
# background around the fish and the canvas corners never leave the box.
FULL_BOX = np.array([0, 0, 100, 100], float)
BIG_FISH = rect_mask(100, 100, 10, 30, 90, 70)
AROUND = ~BIG_FISH
CORNERS = rect_mask(100, 100, 0, 0, 40, 40) | rect_mask(100, 100, 60, 60, 100, 100)  # ~40 % of the outline


def test_wrong_region_detects_inverse_and_canvas_corners_but_not_the_fish():
    assert sf.box_center_cover(BIG_FISH, FULL_BOX) == 1.0 and sf.box_center_cover(AROUND, FULL_BOX) == 0.0
    assert not sf.is_wrong_region(BIG_FISH, FULL_BOX)
    assert sf.is_wrong_region(AROUND, FULL_BOX) and sf.is_wrong_region(CORNERS, FULL_BOX)
    curled = rect_mask(100, 100, 10, 10, 90, 20) | rect_mask(100, 100, 10, 80, 90, 90)  # middle empty
    assert not sf.is_wrong_region(curled, FULL_BOX)  # ... but it touches the outline only at its tips


def test_reprompt_points_put_the_positive_on_the_fish_and_negatives_in_the_wrong_region():
    points, labels = sf.reprompt_points(CORNERS, FULL_BOX)
    assert labels == [1, 0, 0] and points[0] == [50.0, 50.0]
    assert all(CORNERS[int(y), int(x)] for x, y in points[1:])
    shifted = AROUND | rect_mask(100, 100, 40, 40, 60, 60)  # centre itself in the wrong region
    (x, y), *_ = sf.reprompt_points(shifted, FULL_BOX)[0]
    assert not shifted[int(y), int(x)]


def fake_sam(candidates, iou):
    """Stand-in for SAM in wrong_region_fix(): logits +5 inside each candidate, -5 outside."""
    def call(model, processor, image, **kw):
        assert kw["input_labels"][0][0][0] == 1 and len(kw["input_points"][0][0]) == len(kw["input_labels"][0][0])
        return np.where(np.stack(candidates), 5.0, -5.0)[None].astype(np.float32), np.array([iou], np.float32)
    return call


def test_wrong_region_fix_takes_the_best_valid_reprompt_candidate():
    real = sf.sam_logits
    try:
        fish_and_canvas = BIG_FISH | rect_mask(100, 100, 0, 0, 100, 10)  # runs along the outline
        sf.sam_logits = fake_sam([CORNERS, fish_and_canvas, BIG_FISH], [0.99, 0.98, 0.9])
        mask, _, iou, areas, k, status = sf.wrong_region_fix(CORNERS, FULL_BOX, None, (None, None),
                                                             bleed_args(mask_threshold=0.0))
        assert status == "reprompt" and k == 2 and (mask == BIG_FISH).all()
        assert areas.tolist() == [int(CORNERS.sum()), int(fish_and_canvas.sum()), int(BIG_FISH.sum())]
        # Nothing valid: dropped next to other fish, kept (unresolved) when alone.
        sf.sam_logits = fake_sam([CORNERS, AROUND, CORNERS], [0.9, 0.9, 0.9])
        assert sf.wrong_region_fix(CORNERS, FULL_BOX, None, (None, None), bleed_args(), 500)[-1] == "dropped"
        assert sf.wrong_region_fix(CORNERS, FULL_BOX, None, (None, None), bleed_args(), 0)[-1] == "unresolved"
        # A valid but tiny candidate next to a big fish is a spurious detection.
        speck = rect_mask(100, 100, 45, 45, 55, 55)
        sf.sam_logits = fake_sam([speck, speck, speck], [0.9, 0.9, 0.9])
        small_box = np.array([40, 40, 60, 60], float)
        assert sf.wrong_region_fix(CORNERS, small_box, None, (None, None), bleed_args(), 10_000)[-1] == "dropped"
    finally:
        sf.sam_logits = real


# --- clean-up ----------------------------------------------------------------
def test_keep_near_pieces_drops_far_small_pieces_only():
    body = rect_mask(300, 300, 50, 100, 250, 200)     # 20000 px: r = round(0.02 * sqrt(20000)) = 3
    fin = rect_mask(300, 300, 252, 120, 256, 180)     # 3 px from the body: stays
    label = rect_mask(300, 300, 60, 280, 100, 285)    # far, 1 % of the body: dropped
    half = rect_mask(300, 300, 50, 215, 250, 240)     # far but 25 % of the body: stays
    out, n = sf.keep_near_pieces(body | fin | label, 0.02)
    assert (out == (body | fin)).all() and n == 1
    out, n = sf.keep_near_pieces(body | half | label, 0.02)
    assert (out == (body | half)).all() and n == 1
    chain = rect_mask(300, 300, 258, 120, 262, 180)   # 3 px from the fin, 9 px from the body
    assert (sf.keep_near_pieces(body | fin | chain, 0.02)[0] == (body | fin | chain)).all()
    assert (sf.keep_near_pieces(body | label, 0.0)[0] == (body | label)).all()


def test_fill_small_holes_fills_small_holes_not_big_ones_or_other_fish():
    ring = rect_mask(100, 100, 10, 10, 90, 90) & ~rect_mask(100, 100, 20, 20, 80, 80)  # big hole
    speckled = rect_mask(100, 100, 10, 10, 50, 50)
    speckled[20, 20] = speckled[30, 31] = False
    out, added = sf.fill_small_holes(speckled, 0.02)
    assert (out == rect_mask(100, 100, 10, 10, 50, 50)).all() and added == 2
    assert (sf.fill_small_holes(ring, 0.02)[0] == ring).all()
    other = np.zeros((100, 100), bool)
    other[30, 31] = True  # another fish's pixel inside the hole
    out, added = sf.fill_small_holes(speckled, 0.02, blocked=other)
    assert added == 1 and not out[30, 31]


if __name__ == "__main__":
    tests = [(name, fn) for name, fn in list(globals().items()) if name.startswith("test_")]
    for name, fn in tests:
        fn()
        print(f"  ok  {name}")
    print(f"All {len(tests)} segment_fish tests passed.")
