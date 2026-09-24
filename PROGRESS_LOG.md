# Progress Log

## 2026-09-24 — Per-instance masks in segment_fish.py

**Goal:** one mask per detected fish instead of one unioned mask per image,
so the masks can become YOLO-seg training data. Design settled question by
question beforehand: `vault/thesis-log/decisions/2026-09-24-per-instance-mask-output.md`.

**Built (`scripts/segment_fish.py`, rewritten):**
- COCO `annotations.json` is the source of truth, one annotation per
  instance with detector box + score, all 3 SAM IoU scores and the chosen
  candidate index, and `tiny`/`giant` flags. Image-level `qa` block with
  flags (`multi_instance`, `duplicates_removed`, `overlap_resolved`,
  `empty_masks_dropped`, `tiny`, `giant`) for triage filtering.
- SAM candidate = highest predicted IoU. The old "index 0 is best" comment
  was wrong: transformers returns the 3 candidates unsorted.
- Duplicate removal by mask + box containment (higher detector score wins;
  giants never dropped), contested pixels go to the box owner, empty masks
  dropped and counted.
- Crash-safe: `annotations.jsonl` appended per image, `--resume`,
  `run_config.json` with settings + git commit per session; a resume with
  different settings is refused.
- Works at `--max-side 2048` (SAM sees 1024 px anyway); COCO geometry and
  the combined mask PNG are mapped back to original resolution. Overlays
  are now JPEG at working resolution.
- Unreadable files are logged to `skipped.txt` instead of crashing.
- `yolo` mode now uses detector boxes as SAM box prompts (was box centres).
- `tests/test_segment_fish.py`: 12 unit tests on synthetic masks.

**Dataset facts found on the way (`NHM_datensatz`):** median 7.7 MP, max
151 MP (25 images above Pillow's 89 MP guard; limit lifted); 2 corrupt
JPEGs in `Röntgen/` (`Alburnus_mentoides_NMW55510_SYNTYPE_RW_WEB.jpg`,
`Phoxinellus_pseudalepidotus_NMW51103_1-7_paratypes_RW1622_WEB.jpg`), also
unreadable by OpenCV; all 8-bit; no EXIF rotation.

**Pilot (`data/processed/segmented/pilot_per_instance/`, 11 photos + 8
X-rays, separate runs, `--prompt dino`, default thresholds):**
- Instance counts match the catalog series in the filenames in every
  multi-specimen image checked (e.g. `1-35` → 35, `8-14` X-ray → 7).
  Ruler and handwritten label in the 35-fish tray were not detected.
- Fins included on the checked photos; X-ray masks follow the full body
  outline (soft tissue + fins), not just the skeleton.
- The best SAM candidate was index 0 for only 27/62 photo instances and
  0/18 X-ray instances, so the old code took a worse mask for most fish.
- ~1 s/image incl. model load (~5 h for the full dataset); ~0.35 MB
  output per image (~6 GB total).
- `tiny` threshold (0.5%) is miscalibrated: 22/35 normal fish in the tray
  image were flagged.
- Duplicate removal and overlap resolution never triggered in this pilot;
  they are covered by unit tests only so far.
- Species name + catalog number are printed into many `_WEB` images
  (3 of 4 overlays viewed). Clever Hans risk for the ViT, see vault log.

## 2026-09-20 — GPU passthrough re-verified inside rebuilt dev container (real model load)

**Goal:** close out the CUDA/NVML item for good by checking from *inside*
the actual rebuilt dev container (VS Code "Rebuild Container", picking up
the fixed `docker-compose.yml`) — not just a one-off `docker compose run`
container as in the entry below — and by loading a real model rather than
only checking the `torch.cuda.is_available()` flag.

**Ran:** `nvidia-smi` (RTX 4070 Ti, driver 595.84); `torch.cuda.is_available()`
→ `True`, `get_device_name(0)` → `NVIDIA GeForce RTX 4070 Ti`;
`load_sam()` from `src/data/models/sam.py` (checkpoint already cached
under `data/model_cache/`, no network needed) followed by
`next(model.parameters()).device` and a point-prompted `segment()` call on
one `first_test_dataset` image.

**Result:** model loads onto `cuda:0` (not just the availability flag),
load took 2.72s, inference took 0.85s, mask shape correct
(`[1, 3, 224, 224]`). No prior CPU-only timing numbers exist in this log to
compare against (the earlier CUDA-unavailable session only reported
`is_available() == False`, not actual CPU inference times), so this is a
GPU-only timing reference for future comparison, not a speedup measurement.

**Not yet done / next steps:** none for GPU — item stays closed, now
confirmed from inside the real dev container rather than a one-off
container.

## 2026-09-20 — GPU passthrough verified on real host; docker-compose.yml confirmed working

**Goal:** verify, on the actual dev machine (not the sandboxed session from
earlier today), whether the CUDA/NVML failure was a real driver problem or
an artifact of that sandbox, and confirm the reconstructed
`docker-compose.yml` (gitignored per
`vault/thesis-log/decisions/2026-09-19-gitignore-docker-compose.md`)
actually works.

**Ran on host:** `nvidia-smi` — works, driver 595.84, CUDA 13.2, RTX 4070
Ti. `dpkg -l | grep nvidia-container` — `nvidia-container-toolkit` 1.20.0-1
already installed; `docker info` already lists the `nvidia` runtime, so no
toolkit install/config step was needed.

**Ran:** `docker compose config` (mounts to `~/.claude` and
`~/Obsidian_Linux` both resolve and exist), `docker compose build` (clean
build, no errors), then
`docker compose run --rm vit-project python -c "import torch; print(torch.cuda.is_available())"`
→ `True`, device count 1, `NVIDIA GeForce RTX 4070 Ti`.

**Result:** the NVML failure was specific to the earlier sandboxed session
(no real GPU passthrough there), not a problem with this repo or machine.
`docker-compose.yml` needed **no changes** — the reconstruction from
earlier today (build args, `vit-project` service name, both bind mounts,
`deploy.resources.reservations.devices` GPU block) was correct as written
and is now confirmed working end-to-end on the real host.

**Not yet done / next steps:** none for GPU — this line item is closed.

## 2026-09-20 — NHM_datensatz arrives; extract_labels.py built; CUDA diagnosed

**Goal:** turn the real NHM Wien export into a structured label file (steps
3-4 of `tasks/clean-segmented-dataset-with-structured-labels.md`), now that
`data/raw/NHM_datensatz/` has landed (16,975 images).

**Dataset:** `data/raw/NHM_datensatz/` has two top-level subfolders,
`full_body/` (11,766 images) and `Röntgen/` (5,209 X-ray images), nothing
sitting directly at the root. Filenames follow
`<species>_NMW<catalog_number>_<free text>.<ext>`, but `catalog_number` is
not a plain integer in practice: syntype series use ranges (`NMW1290-3`),
comma lists (`NMW13736-41,44-49`), and parenthetical sub-indices glued on
directly (`NMW95284(65895_2)`). A few filenames use `MNW` instead of `NMW`
(a typo in the source data, not corrected here).

**Built `scripts/extract_labels.py`:** walks a raw-dir, splits each
filename on the literal `_NMW`/`_MNW` anchor (not on a digit-only shape),
peels a leading digit run off the remainder as `catalog_number`, and keeps
everything after that verbatim in `extra` rather than guessing at origin
(origin's encoding is still unconfirmed — see Decision below). Also adds
`photo_type` from the image's top-level subfolder name (`full_body` /
`Röntgen`). Writes `data/processed/labels.csv` with columns `file_name,
photo_type, species, species_raw, catalog_number, extra, needs_review,
review_reason`.

**Iteration:** the first version anchored on `\d+` immediately after
`_NMW`, which only handled clean single-number cases — 579/16,975 rows
flagged. Loosening the split to anchor on the prefix literal instead of
digit shape, and letting the remainder absorb ranges/lists/parens/typos,
dropped that to 9/16,975. Those 9 are genuine one-off source anomalies, not
parser bugs: 3x `MNW` typo (same specimen, 3 X-ray angles), 2x `NoNumber`
(no catalog number was ever assigned), 1x a different museum's accession
format (`NRM(Stockholm)51830`), 1x a bare number with no `NMW`/`MNW` prefix
at all, 1x a literature citation instead of a catalog number
(`Holly_1928c_Fig.1`), 1x a second, different typo missing the `N`
(`MW49112`). Each is tagged with a `review_reason` in the CSV.

**Decision:** kept a flat image folder + CSV mapping (no physical
per-species folders), and deliberately left catalog-number substructure and
origin unparsed rather than guessed — see
`vault/thesis-log/decisions/2026-09-20-label-structure.md`. Step 3 of the
task file (confirming the naming convention with Neo/supervisor) is still
open, specifically for where/whether origin is encoded.

**Result:** 16,966/16,975 rows (99.9%) parsed cleanly; 2,291 distinct
`species` values; `labels.csv` exists and is ready for step 6 (run
`segment_fish.py --prompt dino` over `NHM_datensatz`) once per-instance
masks (step 2) are implemented.

**CUDA investigated (user asked why masks are slow):** `load_sam()` /
`load_dino()` already auto-select `cuda` when `torch.cuda.is_available()`
— no code change needed there. In this container,
`torch.cuda.is_available()` is `False` despite the image being CUDA-built
(`torch==2.2.1+cu12.1`) and `/dev/nvidia0`, `/dev/nvidiactl`,
`/dev/nvidia-uvm` all present; `nvidia-smi` fails with "Failed to
initialize NVML: Unknown Error". Device files are passed through but NVML
still can't init, which points at a host/container NVIDIA driver-library
mismatch (the userspace driver libs nvidia-container-toolkit normally
injects aren't lining up with the host's kernel module) — a container
launch/host config issue, not something fixable by editing repo code.
`docker-compose.yml` is gitignored and wasn't present in this session to
inspect its GPU device reservation.

**Not yet done / next steps:**
- Diagnose GPU passthrough at the host/container-launch level.
- Manual spot-check of `labels.csv`'s `extra` column and the 9
  `needs_review` rows.
- Per-instance mask output (step 2, still not done) before running
  segmentation on the real dataset.

## 2026-09-20 — DINO box-prompting smoke test

**Goal:** confirm `--prompt dino` (Grounding DINO zero-shot detection ->
box-prompted SAM) runs end-to-end on real images and produces plausible
masks, per step 1 of `tasks/clean-segmented-dataset-with-structured-labels.md`.
This was an implementation smoke test, not a center-vs-dino quality
comparison (DINO boxes are already known/assumed to beat a single center
point as a SAM prompt).

**Ran:** `scripts/segment_fish.py --prompt dino` and, for reference,
`--prompt center`, both over `data/raw/first_test_dataset/` (6 WhatsApp
photos, pre-resized to 224x224 — not yet real full-resolution NHM Wien
museum images).

**Result:**
- DINO: 6/6 images produced a detection, 0 skips. One mask per image (still
  unioned, not per-instance — that split is step 2, not done yet).
- Center: 6/6 images processed, 0 skips, for comparison.
- Overlay QA looks correct; DINO's SAM prompt is visibly tighter/wider than
  center's, e.g. image 1: DINO bbox width 221px/area 12610px vs center's
  188px/area 10721px, consistent across all 6 images.

**Not yet done / next steps:**
- Per-instance mask output (step 2) — `to_binary()` still unions all
  detections into one mask per image.
- NHM Wien naming convention confirmation and `scripts/extract_labels.py`
  (steps 3-4) — blocked on real specimen images arriving.
- Full run on real (non-resized, multi-fish-per-photo) museum images once
  available.

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
- Grounding DINO (slow, heavy, zero-shot text prompt "fish")
- Feed boxes into SAM (box-prompted, replaces center-point mode)
- Human clean-up (fix fins / occlusion / shadows)
- Train YOLO-seg on the cleaned masks, drop DINO + SAM
