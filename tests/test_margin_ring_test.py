"""Unit tests for the variant construction in scripts/margin_ring_test.py.

Synthetic images only - no model is loaded, so this runs in seconds.
Requires PYTHONPATH=/workspace/src (set in the Dockerfile). Runs under
pytest or directly: python tests/test_margin_ring_test.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import margin_ring_test as rt  # noqa: E402


def fish_scene():
    """200x300 image with random background and one 40x100 'fish' at (100..200, 80..120)."""
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, (200, 300, 3), dtype=np.uint8)
    mask = np.zeros((1, 200, 300), bool)
    mask[0, 80:120, 100:200] = True
    return img, mask


def test_variants_partition_the_pixels_as_documented():
    img, masks = fish_scene()
    v, stats = rt.make_variants(img, masks, 0.1)  # r = round(0.1 * sqrt(4000)) = 6
    assert stats["margin_px"] == 6 and stats["fish_px"] == 4000
    assert stats["ring_px"] > 0 and stats["outer_px"] > 0
    # crop variants share one shape; full/background keep the whole image
    crop_shapes = {v[k].shape for k in ("fish", "fish_ring", "ring", "ring_shape", "outer", "outer_shape")}
    assert len(crop_shapes) == 1 and v["full"].shape == v["background"].shape == img.shape
    # the background variant hides the fish and its margin
    assert (v["background"][80:120, 100:200] == rt.GREY).all()
    assert (v["background"][0, 0] == img[0, 0]).all()


def test_shape_twins_carry_no_image_content():
    img, masks = fish_scene()
    v, _ = rt.make_variants(img, masks, 0.1)
    for k in ("ring_shape", "outer_shape"):
        assert set(np.unique(v[k])) <= {rt.GREY, rt.INK}
    # the twin paints exactly the pixels its variant shows
    for k in ("ring", "outer"):
        shown = (v[k] != rt.GREY).any(axis=2)
        painted = (v[k + "_shape"] == rt.INK).all(axis=2)
        assert not (shown & ~painted).any()


def test_outer_band_keeps_a_gap_and_never_touches_the_fish():
    from scipy.ndimage import distance_transform_edt

    img, masks = fish_scene()
    img[:] = 7  # flat background so shown pixels are easy to locate
    img[masks[0]] = 200
    v, stats = rt.make_variants(img, masks, 0.1)
    r = stats["margin_px"]
    assert not (v["outer"] == 200).any()              # no fish pixel in the band
    assert not (v["ring"] == 200).all(axis=2).any()   # nor in the margin ring
    # straight fish edges: the crop starts 3r above/left of the fish (r margin + r gap + r band)
    oy, ox = 80 - 3 * r, 100 - 3 * r
    ys, xs = np.nonzero((v["outer"] != rt.GREY).any(axis=2))
    dist_to_fish = distance_transform_edt(~masks[0])[ys + oy, xs + ox]
    assert len(ys) and dist_to_fish.min() > 2 * r - 1   # margin + gap stay clear
    assert dist_to_fish.max() <= 3 * r + 1


def test_rasterised_instances_are_disjoint():
    record = {"width": 100, "height": 50, "instances": [
        {"segmentation": [[10, 10, 60, 10, 60, 40, 10, 40]]},
        {"segmentation": [[50, 10, 90, 10, 90, 40, 50, 40]]},   # overlaps the first
        {"segmentation": [[0, 0, 1, 0]]},                        # too short: ignored
    ]}
    masks = rt.load_record_masks(record, (100, 50))
    assert masks.shape[0] == 2
    assert not (masks[0] & masks[1]).any()


def test_to_input_shape_and_padding():
    x = rt.to_input(np.full((30, 90, 3), 255, np.uint8))
    assert x.shape == (3, rt.INPUT_SIDE, rt.INPUT_SIDE) and x.dtype == np.float32
    grey = (rt.GREY / 255.0 - rt.IMAGENET_MEAN[0]) / rt.IMAGENET_STD[0]
    assert abs(x[0, 0, 0] - grey) < 1e-5  # top padding is grey


def test_exact_mcnemar():
    a = np.array([True] * 10 + [False] * 10)
    b = np.array([False] * 10 + [False] * 10)
    a_only, b_only, p = rt.exact_mcnemar(a, b)
    assert (a_only, b_only) == (10, 0) and p < 0.01
    assert rt.exact_mcnemar(a, a)[2] == 1.0


if __name__ == "__main__":
    tests = [(name, fn) for name, fn in list(globals().items()) if name.startswith("test_")]
    for name, fn in tests:
        fn()
        print(f"  ok  {name}")
    print(f"All {len(tests)} margin_ring_test tests passed.")
