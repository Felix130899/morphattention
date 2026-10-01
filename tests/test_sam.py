"""Smoke test for the SAM ViT-L integration.

Verifies the model loads and runs inference, and that no network access is
attempted while doing so (run scripts/download_sam.py first to populate the
local cache). Requires PYTHONPATH=/workspace/src (set in the Dockerfile).
"""

import socket
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from data.models.sam import load_sam, segment

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


def _block_network():
    def blocked_connect(self, *a, **kw):
        raise RuntimeError("Network access attempted during offline SAM inference!")

    socket.socket.connect = blocked_connect


def test_sam_offline_inference():
    _block_network()

    model, processor = load_sam()

    rng = np.random.default_rng(0)
    image = Image.fromarray((rng.random((600, 800, 3)) * 255).astype("uint8"))
    masks, iou_scores = segment(model, processor, image, input_points=[[[400, 300]]])

    assert masks[0].shape[-2:] == (600, 800)
    assert iou_scores.numel() == 3


def test_segment_fish_logits_reproduce_segment_masks():
    """segment_fish's default path (logits > 0, highest IoU) == the old segment() masks."""
    import segment_fish as sf

    _block_network()
    model, processor = load_sam()
    rng = np.random.default_rng(1)
    image = Image.fromarray((rng.random((480, 700, 3)) * 255).astype("uint8"))
    boxes = [[[50, 60, 400, 300], [300, 200, 650, 450]]]
    masks, iou = segment(model, processor, image, input_boxes=boxes)
    logits, iou2 = sf.sam_logits(model, processor, image, input_boxes=boxes)
    assert logits.dtype == np.float32 and logits.shape == (2, 3, 480, 700)
    assert np.array_equal(iou[0].numpy(), iou2)
    assert np.array_equal(masks[0].numpy().astype(bool), logits > 0.0)
    old, old_chosen = sf.best_candidates(masks[0].numpy().astype(bool), iou[0].numpy())
    new, new_chosen, _ = sf.choose_candidates(logits, iou2)
    assert np.array_equal(old, new) and np.array_equal(old_chosen, new_chosen)


if __name__ == "__main__":
    test_sam_offline_inference()
    test_segment_fish_logits_reproduce_segment_masks()
    print("SAM offline inference tests passed.")
