"""DINOv2 (self-supervised ViT) loading helpers, used as a frozen feature
extractor (e.g. scripts/margin_ring_test.py).

Run scripts/download_dinov2.py once after building the image to cache the
checkpoint under data/model_cache/. load_dinov2() below forces Hugging Face
Hub offline mode before loading, so this code path can only ever use that
local cache and can never reach out to the network - no image data or
metadata can leave the container. Mirrors src/data/models/sam.py.
"""

import os

import torch

CHECKPOINT = "facebook/dinov2-base"


def load_dinov2(device=None):
    """Load DINOv2 ViT-B/14 (no processor: callers normalise themselves) onto the given device."""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from transformers import Dinov2Model

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    return Dinov2Model.from_pretrained(CHECKPOINT).to(device).eval()
