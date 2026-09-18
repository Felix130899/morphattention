"""One-time download of the Grounding DINO checkpoint into the local cache.

Mirrors scripts/download_sam.py: run this once after building the image (it
needs network access to fetch public model weights from the Hugging Face
Hub):

    docker compose run --rm vit-project python scripts/download_dino.py

The cache lives under data/model_cache/ (HF_HOME, set in the Dockerfile),
which is bind-mounted from the host and gitignored, so the weights persist
across container rebuilds without ever being committed or baked into the
image. After this, src/data/models/dino.py loads the model fully offline.
"""

import os

from transformers import AutoProcessor, GroundingDinoForObjectDetection

from data.models.dino import CHECKPOINT

if __name__ == "__main__":
    print(f"Downloading {CHECKPOINT} into {os.environ.get('HF_HOME')} ...")
    GroundingDinoForObjectDetection.from_pretrained(CHECKPOINT)
    AutoProcessor.from_pretrained(CHECKPOINT)
    print("Done. Grounding DINO is now cached locally and ready for offline use.")
