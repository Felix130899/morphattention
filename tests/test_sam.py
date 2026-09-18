"""Smoke test for the SAM ViT-L integration.

Verifies the model loads and runs inference, and that no network access is
attempted while doing so (run scripts/download_sam.py first to populate the
local cache). Requires PYTHONPATH=/workspace/src (set in the Dockerfile).
"""

import socket

import numpy as np
from PIL import Image

from data.models.sam import load_sam, segment


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


if __name__ == "__main__":
    test_sam_offline_inference()
    print("SAM offline inference test passed.")
