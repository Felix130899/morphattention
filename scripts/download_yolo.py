"""One-time download of a YOLO detector checkpoint into the local cache.

Mirrors scripts/download_sam.py: run this once (with network access) after
building the image, and the weights land under data/model_cache/ - which is
bind-mounted from the host and gitignored, so they persist across rebuilds
without being committed or baked into the image. After this, segment_fish.py
can run the detector fully offline from that cached .pt.

    # default (yolov8n.pt - a COCO model, has NO fish class; mechanism only):
    docker compose run --rm vit-project python scripts/download_yolo.py

    # a real fish detector you sourced (Roboflow Universe, your own, ...):
    docker compose run --rm vit-project python scripts/download_yolo.py \
        --weights https://example.com/fish_yolov8.pt

The downloaded file is copied to data/model_cache/<name>.pt; pass that path
to segment_fish.py via --yolo-weights.
"""

import argparse
import os
import shutil
from pathlib import Path

from ultralytics import YOLO

CACHE = Path(os.environ.get("YOLO_CACHE", "/workspace/data/model_cache"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", default="yolov8n.pt",
                        help="A YOLO checkpoint name, local path, or URL to fetch.")
    args = parser.parse_args()

    CACHE.mkdir(parents=True, exist_ok=True)
    dest = CACHE / Path(args.weights).name

    print(f"Fetching {args.weights} ...")
    model = YOLO(args.weights)  # downloads if it's a name/URL, loads if local

    src = Path(model.ckpt_path)  # the actual file ultralytics loaded from
    if src.resolve() != dest.resolve():
        shutil.copy(src, dest)
    print(f"Done. Cached at {dest} - pass it to segment_fish.py via --yolo-weights.")
