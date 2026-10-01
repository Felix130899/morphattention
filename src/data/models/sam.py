"""Segment Anything (SAM ViT-L) loading and inference helpers.

Run scripts/download_sam.py once after building the image to cache the
checkpoint under data/model_cache/. load_sam() below forces Hugging Face Hub offline mode
before loading, so this code path can only ever use that local cache and
can never reach out to the network - no image data or metadata can leave
the container during segmentation.
"""

import os

import torch

CHECKPOINT = "facebook/sam-vit-large"


def load_sam(device=None):
    """Load SAM ViT-L and its processor onto the given (or best available) device."""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from transformers import SamModel, SamProcessor

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = SamModel.from_pretrained(CHECKPOINT).to(device).eval()
    processor = SamProcessor.from_pretrained(CHECKPOINT)
    return model, processor


def segment(model, processor, image, input_points=None, input_boxes=None):
    """Run point- and/or box-prompted segmentation on a PIL image.

    input_points: list of point lists per object, e.g. [[[x, y]]] for one
    point prompt on one object, in pixel coordinates of ``image``.
    input_boxes: one list containing the image's boxes, e.g.
    [[[x0, y0, x1, y1], [x0, y0, x1, y1]]] for two objects - one box per
    object (from a detector like Grounding DINO or YOLO), in pixel
    coordinates of ``image``. Boxes are a stronger prompt than a single
    point since they constrain the object's extent.
    """
    device = next(model.parameters()).device
    inputs = processor(
        image, input_points=input_points, input_boxes=input_boxes, return_tensors="pt"
    ).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    masks = processor.image_processor.post_process_masks(
        outputs.pred_masks.cpu(),
        inputs["original_sizes"].cpu(),
        inputs["reshaped_input_sizes"].cpu(),
    )
    return masks, outputs.iou_scores.cpu()
