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