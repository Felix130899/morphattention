## Name
masterthesis_fish_VIT

## Purpose
XAI on ViT attention maps for fish morphology (NHM Wien imagery)

## Non-negotiable constraints
- Local-only processing, no cloud/external APIs
- Only Neo + supervisor access; weights never published
- NHM attribution required in public outputs

## Current state
- SAM ViT-L + Grounding DINO work on GPU inside the dev container (`--prompt dino` mode exists)
- NHM dataset in `data/raw/NHM_datensatz/` (16,975 images); `labels.csv` parsed (9 rows need review, origin unconfirmed)
- `segment_fish.py` still unions all detections into one mask per image
- Labels: flat image folder + `labels.csv`, no per-species folders (vault/thesis-log/decisions/2026-09-20-label-structure.md)

## Next steps
1. Per-instance masks in `scripts/segment_fish.py` (one mask per DINO box instead of `to_binary()` union)
2. Run DINO+SAM on a stratified sample of `NHM_datensatz` (mixed species, multi-fish, Röntgen)
3. Write mask-policy decision, then triage/fix masks in CVAT (self-hosted) or X-AnyLabeling
4. Train YOLO-seg on 300–500 curated images, evaluate on held-out test set
- Full checklist: vault/thesis-log/tasks/clean-segmented-dataset-with-structured-labels.md

## Detailed log
See vault/thesis-log/ (symlinked, see below)

## Update protocol (read this before editing this file)
- "Current State" and "Next Steps": OVERWRITE, never append.
- Max 5 bullets each. Hitting the cap means something belongs in
  vault/thesis-log/ instead — link it, don't inline it.
- Only include state changes that are true as of THIS session.
- A "Next Steps" item only counts if it's concrete enough to start
  without re-deriving context (not "improve model" — "add SamPredictor
  fallback for tray images with >1 fish per mask").
- Never touch "Purpose" or "Non-negotiable constraints" unless Neo
  explicitly asks.