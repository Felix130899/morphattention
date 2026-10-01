## Name
masterthesis_fish_VIT

## Purpose
XAI on ViT attention maps for fish morphology (NHM Wien imagery)

## Non-negotiable constraints
- Local-only processing, no cloud/external APIs
- Only Neo + supervisor access; weights never published
- NHM attribution required in public outputs

## Current state
- Masks: decided setting = B (`--mask-threshold -1 --min-component-frac 0.01`) + 3 % margin applied when building training images; blind dev-set review: fix rate 39 → 23 % photos, 36 → 22 % X-rays, cut fins nearly gone (vault/thesis-log/decisions/2026-10-01-fin-fix-setting.md + results HTML next to it)
- Margin ring test passed (`scripts/margin_ring_test.py`, DINOv2 linear probe): margin adds no genus info beyond the outline; background alone predicts genus at 71 % / 61 % → masking is necessary
- `segment_fish.py`: `--candidate/--mask-threshold/--margin-frac/--min-component-frac/--image-list` + per-instance QA features; `review_masks.py`: categories (keys 1–6) + blind `--compare` mode + `--summary`. Mask policy: vault/thesis-log/decisions/2026-10-01-mask-policy.md
- Old full run (2026-09-24, `data/processed/segmented/{full_body,Röntgen}/`) is superseded; code checkpoint before the fin fix: git tag `pre-fin-fix-2026-10-01`
- NHM dataset: 16,942 images (Kopie duplicates removed), `labels.csv` matches (9 rows need review, origin unconfirmed); split train/test by NMW specimen number

## Next steps
1. Full re-run with setting B (no margin) over `full_body` and `Röntgen` into a new run dir under `data/processed/segmented/` (pattern: `data/processed/devset_fin_fix/run_devset.sh`, ~2 h)
2. Look at C's remaining non-fin errors (photos: 9 wrong_object, several catfish `_HOLOTYPE_…_SL…` images, + 4 missed_fish; X-rays: 9 bleed); lists in the results HTML
3. Random review sample per domain (~1,000 each, excluding `data/processed/devset_fin_fix/` images) with `review_masks.py --image-list`, then train the QA model (QA features + DINOv2 crop embedding) and rank all masks
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