# morphattention
Explainability analysis of ViT attention maps on fish morphological features.

## Setup (Docker)

Needs Docker with Compose v2 or later (the `docker compose` plugin; it is numbered v5 since late 2025) and, for the GPU, the NVIDIA Container Toolkit on the host.
In the cloned repo:

```bash
printf 'USER_ID=%s\nGROUP_ID=%s\n' "$(id -u)" "$(id -g)" > .env    # files written in the container stay yours
cp docker-compose.override.example.yml docker-compose.override.yml  # your own mounts, see below
docker compose build
docker compose run --rm vit-project python -c "import torch; print(torch.cuda.is_available())"
# model weights, once (the only step that needs the network; afterwards everything runs offline)
docker compose run --rm vit-project python scripts/download_sam.py
docker compose run --rm vit-project python scripts/download_dino.py
docker compose run --rm vit-project python -m pytest                # all tests
```

In VS Code, "Dev Containers: Reopen in Container" uses the same files and creates the override from the example
when it is missing.

**Your own setup** goes into `docker-compose.override.yml` (gitignored), never into `docker-compose.yml`:
- images stored elsewhere: mount them under `data/raw/`, e.g.
  `- /mnt/storage/fish_images:/workspace/data/raw/my_dataset:ro`
- other personal mounts (editor or Claude Code config, notes)
- no NVIDIA GPU: `deploy: !reset null` (Compose ≥ 2.24); SAM + DINO then run on the CPU, slowly

`docker compose config` shows what Compose will actually use, both files merged.

**Versions:** `requirements.txt` pins the direct dependencies; `constraints.txt` pins everything pip installs
on top of the base image (`pytorch/pytorch:2.2.1-cuda12.1-cudnn8-runtime`, Python 3.10). To add a package, add
it pinned to `requirements.txt`, rebuild, and copy its new sub-dependencies from `pip freeze` into
`constraints.txt`.

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
drops → split → manifest) runs with `python scripts/run_part1.py`: the default
`pipeline/config.yaml` is a neutral start for your own dataset (every mask fix off,
decisions from a copy of `pipeline/decisions_template/`), the thesis run is
`python scripts/run_part1.py --config pipeline/config_thesis_nhm.yaml`.

```bash
docker compose run --rm vit-project python scripts/segment_fish.py \
    --raw-dir /workspace/data/raw/NHM_datensatz/full_body
```