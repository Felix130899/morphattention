"""Grounding DINO (zero-shot text-prompted object detection) loading and
inference helpers.

Run scripts/download_dino.py once after building the image to cache the
checkpoint under data/model_cache/. load_dino() below forces Hugging Face Hub
offline mode before loading, so this code path can only ever use that local
cache and can never reach out to the network - no image data or metadata can
leave the container during detection. Mirrors src/data/models/sam.py.
"""

import os

import torch

CHECKPOINT = "IDEA-Research/grounding-dino-tiny"


def load_dino(device=None):
    """Load Grounding DINO and its processor onto the given (or best available) device."""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from transformers import AutoProcessor, GroundingDinoForObjectDetection

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = GroundingDinoForObjectDetection.from_pretrained(CHECKPOINT).to(device).eval()
    processor = AutoProcessor.from_pretrained(CHECKPOINT)
    return model, processor


def detect(model, processor, image, text_prompt="fish.", box_threshold=0.35, text_threshold=0.25):
    """Zero-shot detect ``text_prompt`` in a PIL image, returning boxes in pixel coords.

    text_prompt: lowercase, each concept ending in a period, e.g. "fish." -
    that's the format Grounding DINO was trained on.

    Returns (boxes, scores): boxes is an (N, 4) float tensor of [x0, y0, x1, y1]
    in the original image's pixel coordinates; scores is an (N,) float tensor.
    N is 0 if nothing cleared box_threshold.
    """
    device = next(model.parameters()).device
    inputs = processor(images=image, text=text_prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        target_sizes=[image.size[::-1]],  # (height, width)
    )[0]
    return results["boxes"].cpu(), results["scores"].cpu()
