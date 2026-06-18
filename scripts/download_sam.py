"""One-time download of the SAM ViT-L checkpoint into the local cache.

Run this once after building the image (it needs network access to fetch
public model weights from the Hugging Face Hub - no image data is involved):

    docker compose run --rm vit-project python scripts/download_sam.py

The cache lives under data/model_cache/ (HF_HOME, set in the Dockerfile),
which is bind-mounted from the host and gitignored, so the weights persist
across container rebuilds without ever being committed or baked into the
image. After this, src/data/models/sam.py loads the model fully offline.
"""

import os

from transformers import SamModel, SamProcessor

from data.models.sam import CHECKPOINT

if __name__ == "__main__":
    print(f"Downloading {CHECKPOINT} into {os.environ.get('HF_HOME')} ...")
    SamModel.from_pretrained(CHECKPOINT)
    SamProcessor.from_pretrained(CHECKPOINT)
    print("Done. SAM ViT-L is now cached locally and ready for offline use.")
