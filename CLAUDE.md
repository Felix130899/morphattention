## Name
masterthesis_fish_VIT

## Purpose
XAI on ViT attention maps for fish morphology (NHM Wien imagery)

## Non-negotiable constraints
- Local-only processing, no cloud/external APIs
- Only Neo + supervisor access; weights never published
- NHM attribution required in public outputs

## Current state
- Real NHM Wien export arrived: `data/raw/NHM_datensatz/` (16,975 images,
  split `full_body/` vs `Röntgen/` subfolders, no images at the root).
- `scripts/extract_labels.py` built and run: splits filenames on the
  literal `_NMW`/`_MNW` anchor (catalog number itself has ranges/lists/
  parens, so it's not digit-parsed) into `data/processed/labels.csv`
  (species, catalog_number, extra, photo_type, needs_review). 16,966/16,975
  rows parsed cleanly; see PROGRESS_LOG.md 2026-09-20 entry.
- Label-structure decision: one flat image folder + CSV mapping, no
  physical per-species folders; origin left unparsed in `extra` since the
  convention isn't confirmed yet — see
  `vault/thesis-log/decisions/2026-09-20-label-structure.md`.
- Segmentation still unions multiple detections into one mask per image
  (per-instance split not implemented yet — needed for multi-fish photos).

## Next steps
1. Resolve the 9 `needs_review` rows in `labels.csv` by hand and spot-check
   the `extra` column for a sample of `full_body`/`Röntgen` rows.
2. Change `to_binary()`/mask-writing in `scripts/segment_fish.py` to emit
   one mask per detected instance instead of unioning them.
3. Run `scripts/segment_fish.py --prompt dino` over
   `data/raw/NHM_datensatz/` once step 2 (per-instance masks) is done.

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