# Progress Log

## 2026-06-19 — Batch fish segmentation pipeline

**Goal:** make SAM usable over a whole dataset (thousands of images, mixed
formats, named by species/origin and *not* bounding-box labelled) without
hand-clicking a prompt per image, while keeping the clean offline / clean-image
guarantees the SAM integration established.

**What was built:**
- `scripts/segment_fish.py` — processes a chosen folder (`--raw-dir`,
  **required**; images live in per-category subfolders of `data/raw/`, e.g.
  `data/raw/trout/`), searched recursively, reads any common format
  (jpg/jpeg/png/tif/tiff/bmp/webp via PIL, no pre-conversion), derives point
  prompt(s) per image, runs `segment()`, and writes three parallel output trees
  under `data/processed/segmented/<raw-dir name>/` (nested by folder name so
  processing several folders never overwrites earlier runs):
  - `masks/<stem>.png` — binary mask (0 = bg, 255 = fish)
  - `overlays/<stem>.png` — translucent red mask on the original, for QA
  - `coco/annotations.json` — one COCO file (polygons + bbox + area) per run
- Two prompting modes (`--prompt`): `center` (single centre point — fits the
  single-fish specimen photos, no extra setup) and `yolo` (one mask per
  detected fish; needs a fish-trained checkpoint — stock YOLO has no fish
  class).
- `scripts/download_yolo.py` — one-time YOLO weights fetch mirroring
  `download_sam.py`; caches the `.pt` into `data/model_cache/` (gitignored,
  bind-mounted, never baked into the image), so the detector then loads offline.
- `requirements.txt` — added `ultralytics` (unpinned, matching torch/timm).

**Not yet done / next steps:**
- **Not yet run on real images** — `data/raw` is still empty; user will try the
  pipeline on real fish photos next. Check the `overlays/` output first to
  confirm prompt placement before trusting the masks.
- No fish-trained YOLO checkpoint sourced yet. Default `yolov8n.pt` is a COCO
  model (no fish class), present only to prove the download/cache mechanism.
  Candidates noted: Roboflow Universe fish YOLOv8 (easiest drop-in),
  Fishial.ai (segmentation + recognition), FathomNet (in-habitat), or training
  a small custom YOLOv8.
- `ultralytics` has no hard offline switch like HF's `*_OFFLINE` env vars and
  may emit anonymous telemetry; loading a local `.pt` won't leak images, but
  `yolo settings sync=False` would close that gap if the same airtight
  guarantee is wanted.

## 2026-06-18 — SAM ViT-L integration

**Goal:** add `facebook/sam-vit-large` (Segment Anything) to the container as
a clean, reusable component, after testing it works, without bloating the
Docker image or putting any image data at risk of leaving the machine.

**What was found:**
- `requirements.txt` had `transformers` unpinned, which had resolved to
  `5.5.4`. That version refuses to load PyTorch at all (needs torch≥2.4), so
  *no* Hugging Face model — SAM or otherwise — could have been loaded before
  this fix, even though `transformers` itself imported without error.
- `datasets` (also unpinned) pulls in `numpy>=2`, which silently breaks
  `torch` ↔ `numpy` interop (tensors fail to convert, with only a warning,
  not a hard error). Pinned `numpy<2` to prevent this.

**What was built:**
- `src/data/models/sam.py` — `load_sam()` / `segment()` helpers wrapping
  `transformers.SamModel` / `SamProcessor`. `load_sam()` forces
  `HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1` before loading, so this code
  path can never reach the network regardless of how it's invoked — the
  guarantee lives at the call site, not just in container env vars.
- `scripts/download_sam.py` — explicit one-time download step (needs network
  to fetch public weights, no image data involved). Run once after building
  the image.
- Cache lives at `data/model_cache/` — already gitignored, bind-mounted from
  the host via `docker-compose.yml`'s `.:/workspace` volume. This means
  weights persist across container rebuilds without ever being committed or
  baked into the image, keeping the image small enough to push to GitHub.
- `tests/test_sam.py` — smoke test that loads SAM, runs point-prompted
  segmentation on a synthetic image, and hard-blocks outbound sockets at the
  Python `socket` layer during the run, so the offline guarantee is actually
  verified, not just assumed.
- `requirements.txt` pinned: `transformers==4.46.3`, `tokenizers==0.20.3`,
  `numpy<2`.

**Verified (inside the running container, GPU available):**
- `transformers==4.46.3` imports and works fine with the base image's
  `torch==2.2.1`.
- SAM ViT-L loads, runs on CUDA, and produces correct-shaped masks + IoU
  scores on a synthetic test image.
- With the local cache populated and a hard socket block active, inference
  still succeeds — confirming zero network access during segmentation.
- Reinstalling from the updated `requirements.txt` reproduces a working
  `numpy`/`torch`/`opencv`/`transformers` environment.

**Not yet done / next steps:**
- Could not run `docker build` itself from inside this session (no Docker
  daemon available inside the already-running container) — every Dockerfile
  step was instead replayed manually in this exact environment. Should do a
  real `docker compose build` + `docker compose run --rm vit-project python
  scripts/download_sam.py` on the host to confirm end-to-end before relying
  on it.
- No real fish images yet (`data/raw` is empty) — SAM has only been
  exercised on a synthetic random image. Worth a quick sanity check on a
  real fish photo once available.
- `src/data/models/explainability/utils/` exists but is empty — presumably
  where SAM output feeds into the attention-map analysis next.

## 2026-06-19 — Segmentation testing with center mode

**What was tested:**
- Ran `segment_fish.py` with `--prompt center` on real fish images from
  `first_test_dataset`.
- Segmentation completed successfully — masks, overlays, and COCO annotations
  were all generated and saved correctly.

**Findings:**
- **Success** — the pipeline works end-to-end with no errors.
- **Quality issue** — the segmentation quality is poor. The masks do not
  accurately capture the fish boundaries; they appear too loose or incomplete.
  This is expected since `center` mode uses only a single point at image centre,
  which often doesn't provide enough prompt specificity for accurate SAM output.

**Next steps:**
- Train or source a fish-specific YOLOv8 model to enable `--prompt yolo` mode.
  This will detect individual fish and use each detection centre as a point
  prompt, providing better coverage and accuracy per fish.
- Once a trained fish YOLO is available, re-segment with `--prompt yolo` to
  compare quality improvements.
- Grounding DINO takes the text prompt "fish" and gives you boxes zero-shot
- The reason to do detector→SAM at all is that you don't have masks yet — SAM is how you cheaply generate them. Once you've   got masks, a YOLO-seg model is your fast deployment model and you drop SAM entirely.
- Boxes from Grounding DINO ("fish") or a fish-pretrained YOLO.
Box-prompt SAM 2.1 → masks.
Human review and cleanup. This is the step people skip and regret. Box-prompted SAM is reliable but not perfect on fins, occlusion, and shadows — fix those, because everything downstream inherits these errors.
Train YOLO-seg (or Mask R-CNN) on the cleaned masks.
Loop: use the trained model to label the next batch, correct only its mistakes, retrain. This active-learning loop is where the real scaling happens.
- Grounding DINO or fish-pretrained YOLO for boxes → SAM 2.1 (use video mode if your data is video) → manual cleanup → train YOLO11-seg → iterate.
