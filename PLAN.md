# masterthesis_fish_VIT — Plan

Goal: Determine whether a ViT can classify fish specimen images (by species
and by geographic origin) using biologically meaningful morphological
features rather than spurious technical artifacts (Clever Hans predictors),
and validate its attention focus against expert anatomical knowledge.
Constraints: Deadline 2026-12-31. Local-only processing, no cloud/external
APIs. Only Neo + supervisor access; weights never published. NHM attribution
required in public outputs. Both species-ID and origin-ID tasks are in
scope, run sequentially, to compare where the approach is strong vs. limited.

## Milestone: Clean segmented dataset with structured labels
Goal: Fix the known SAM mask-quality problem (center-point prompting is too
loose) by wiring in DINO box-prompting, and get the existing NHM Wien labels
into a structured format covering both species and origin per image.
Done when: All specimen images in `data/raw` have DINO+SAM masks that pass
overlay QA, and a single structured label file (species + origin per image)
exists and matches the image set.

## Milestone: Architecture selection & baseline classifiers
Goal: Compare candidate ViT architectures (e.g. supervised ViT-B/16 vs.
DINOv2 features + linear probe) and train baseline classifiers for both the
species-ID and origin-ID tasks on the segmented dataset.
Done when: Each candidate architecture has a trained model per task with
logged accuracy/F1, and one architecture is chosen and documented for the
attention-analysis phase.

## Milestone: Attention-based Clever Hans investigation
Goal: Extract and visualize ViT attention maps for the chosen model(s) on
both tasks, and quantify how much each prediction relies on non-biological
artifacts (background, padding, borders) vs. the segmented fish region.
Done when: Attention visualizations exist for a representative sample from
both tasks, with a quantitative artifact-reliance metric (e.g. attention
mass inside vs. outside the segmentation mask) computed and compared
between the two tasks.

## Milestone: Expert anatomical validation
Goal: Compare the model's attention focus against expert-defined anatomical
landmarks to assess whether classification is driven by biologically
meaningful regions.
Done when: A documented comparison (qualitative overlay and/or a
landmark-overlap metric) exists between attention hotspots and
expert-annotated anatomical regions, for both tasks.

## Milestone: Synthesis & thesis writing
Goal: Consolidate the quantitative (classification performance) and
qualitative (Clever Hans / expert validation) findings into the thesis
document, answering both research questions from the expose.
Done when: Full thesis draft covers results for both classification tasks
and is submission-ready ahead of the 2026-12-31 deadline.



"Written for: a Claude Code session running inside the rebuilt container, with no memory of this conversation.

Context: I rebuilt the dev container (VS Code "Rebuild Container") to pick up a fixed docker-compose.yml at the repo root. Previously the container you were running in was built from a stale vit-env:latest image (3 months old) that lacked working GPU passthrough — torch.cuda.is_available() was False and nvidia-smi failed with an NVML error. That was diagnosed on the host (outside any container) as a sandbox/stale-image issue, not a real driver problem: nvidia-smi works fine on the bare host (driver 595.84, RTX 4070 Ti), nvidia-container-toolkit was already installed, and a fresh docker compose build + docker compose run --rm vit-project python -c "import torch; print(torch.cuda.is_available())" returned True from a one-off container built off the current docker-compose.yml.

This is now documented in CLAUDE.md (Current state / Next steps) and today's PROGRESS_LOG.md entries (2026-09-20, "GPU passthrough verified on real host...").

Please verify, from inside this container (the one you're running in right now, post-rebuild):

nvidia-smi — should show the RTX 4070 Ti.
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))" — should print True and the GPU name.
Actually load a real model (e.g. whatever load_sam() / load_dino() in src/data/models/sam.py / dino.py use) and confirm it lands on cuda:0, not just that the flag is True — e.g. check next(model.parameters()).device or run a small inference and time it against the earlier CPU numbers if any exist in PROGRESS_LOG.md.
If all of that checks out, this closes out the CUDA/NVML item — no further CLAUDE.md edit needed beyond what's already there unless you find something new (e.g. a specific op that still falls back to CPU).
If anything in steps 1–3 fails or looks different from what's described above (e.g. torch.cuda.is_available() is False again inside this new container), stop and report back — that would mean the rebuild didn't actually pick up the new image/compose file, which is worth flagging rather than re-diagnosing from scratch."