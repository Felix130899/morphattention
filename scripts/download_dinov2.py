"""One-time download of the DINOv2 ViT-B/14 checkpoint into the local cache.

Run this once after building the image (it needs network access to fetch
public model weights from the Hugging Face Hub):

    docker compose run --rm vit-project python scripts/download_dinov2.py

The cache lives under data/model_cache/ (HF_HOME, set in the Dockerfile),
which is bind-mounted from the host and gitignored, so the weights persist
across container rebuilds without ever being committed or baked into the
image. After this, src/data/models/dinov2.py loads the model fully offline.
"""

import os

from transformers import Dinov2Model

from data.models.dinov2 import CHECKPOINT

if __name__ == "__main__":
    print(f"Downloading {CHECKPOINT} into {os.environ.get('HF_HOME')} ...")
    Dinov2Model.from_pretrained(CHECKPOINT)
    print("Done. DINOv2 ViT-B/14 is now cached locally and ready for offline use.")
