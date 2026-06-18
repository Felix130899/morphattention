# Progress Log

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
