"""Smoke test for the Grounding DINO integration.

Verifies the model loads and runs inference, and that no network access is
attempted while doing so (run scripts/download_dino.py first to populate the
local cache). Requires PYTHONPATH=/workspace/src (set in the Dockerfile).
"""

import socket

import numpy as np
from PIL import Image

from data.models.dino import load_dino, detect


def _block_network():
    def blocked_connect(self, *a, **kw):
        raise RuntimeError("Network access attempted during offline Grounding DINO inference!")

    socket.socket.connect = blocked_connect


def test_dino_offline_inference():
    _block_network()

    model, processor = load_dino()

    rng = np.random.default_rng(0)
    image = Image.fromarray((rng.random((600, 800, 3)) * 255).astype("uint8"))
    boxes, scores = detect(model, processor, image, text_prompt="fish.")

    # Random noise shouldn't contain a fish, but inference must run cleanly
    # offline and return well-formed (possibly empty) results.
    assert boxes.shape[-1] == 4
    assert boxes.shape[0] == scores.shape[0]


if __name__ == "__main__":
    test_dino_offline_inference()
    print("Grounding DINO offline inference test passed.")
