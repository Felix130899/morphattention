# morphattention
Explainability analysis of ViT attention maps on fish morphological features.

docker build -t morphattention-env .

docker image (to check if it is there)

morph

## Project structure

| Folder | Purpose |
|---|---|
| `data/` | Raw, processed and annotated fish images (not tracked by git) |
| `src/` | All source code: models, XAI methods, and utilities |
| `notebooks/` | Exploratory analysis and result walkthroughs |
| `experiments/` | Configs and outputs for each experimental run |
| `figures/` | Final plots exported for the thesis |
| `tests/` | Sanity checks for core evaluation logic |

## SAM ViT-L (Segment Anything)

`src/data/models/sam.py` wraps `facebook/sam-vit-large` (via `transformers.SamModel`/`SamProcessor`):

```python
from data.models.sam import load_sam, segment

model, processor = load_sam()
masks, iou_scores = segment(model, processor, image, input_points=[[[x, y]]])
```

**Setup (once per machine, after `docker compose build`):**

```bash
docker compose run --rm vit-project python scripts/download_sam.py
```

This downloads the checkpoint (~1.2GB) into `data/model_cache/` — bind-mounted
from the host and gitignored, so it's never committed and never baked into
the image, but persists across container rebuilds. The image itself stays
small and safe to push to GitHub.

`load_sam()` forces Hugging Face Hub offline mode before loading the model,
so segmentation can only ever use that local cache — no image data or
metadata can leave the container while running SAM. `tests/test_sam.py`
enforces this with a hard socket-level block during the smoke test.

## Batch fish segmentation

`scripts/segment_fish.py` runs SAM over a chosen directory of images and writes
three parallel output trees, so you can pick whichever fits later work:

| Folder | Contents |
|---|---|
| `masks/` | binary PNG per image (0 = background, 255 = fish) |
| `overlays/` | original image with a translucent red mask — for eyeballing quality |
| `coco/annotations.json` | one COCO file (polygons + bbox + area) for the run |

You select which folder to process with `--raw-dir` (**required**) — images
live in per-category subfolders of `data/raw/`, e.g. `data/raw/trout/`. It
reads any common format (jpg, jpeg, png, tif/tiff, bmp, webp) recursively, so
no pre-conversion is needed. Outputs nest under the selected folder's name —
`data/processed/segmented/<raw-dir name>/{masks,overlays,coco}/` — so
processing several folders never overwrites earlier runs. Prompts are
Grounding DINO boxes ("fish."); the defaults are the thesis setting D, except
`--bleed-max-box-edge 0.3` for X-rays. The whole of part 1 (labels → masks →
drops → split → manifest) runs with `python scripts/run_part1.py`, settings in
`pipeline/config.yaml`.

```bash
docker compose run --rm vit-project python scripts/segment_fish.py \
    --raw-dir /workspace/data/raw/NHM_datensatz/full_body
```