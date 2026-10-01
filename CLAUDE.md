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
- Margin ring test passed (`scripts/margin_ring_test.py`, DINOv2 linear probe): margin adds no genus info beyond the outline; background alone predicts genus at 71 % / 61 % → masking is necessary. Robustness check without bleed images: unchanged. Record: vault/thesis-log/experiments/2026-10-01-margin-ring-test.md; frozen archive `data/archive/2026-10-01_margin_ring_test/`; git tag `margin-ring-test-2026-10-01`
- `segment_fish.py`: `--candidate/--mask-threshold/--margin-frac/--min-component-frac/--image-list` + per-instance QA features; `review_masks.py`: categories (keys 1–6) + blind `--compare` mode + `--summary`. Mask policy: vault/thesis-log/decisions/2026-10-01-mask-policy.md
- Full re-run with setting B done (2026-10-01): `data/processed/segmented/B_full_2026-10-01/` (11,731 photos, 5,207 X-rays); old run (2026-09-24) superseded; checkpoint before the fin fix: git tag `pre-fin-fix-2026-10-01`
- NHM dataset: 16,942 images (Kopie duplicates removed), `labels.csv` matches (9 rows need review, origin unconfirmed); split train/test by NMW specimen number

## Next steps
1. Review the suspected whole-background bleed (masks incl. printed names): `python scripts/review_masks.py --run-dir data/processed/segmented/B_full_2026-10-01/<full_body|Röntgen> --image-list <run-dir>/suspect_bleed.txt` (197 photos, 392 X-rays; from `scripts/flag_bleed.py`)
2. Then fix them in the generator (e.g. fall back to a smaller SAM candidate when masks cover > 70 % + touch the border) and re-run only those images; also look at C's non-fin errors (photos: 9 wrong_object, catfish `_HOLOTYPE_…_SL…` series; lists in the results HTML)
3. Random review sample per domain (~1,000 each, excluding the dev set) with `review_masks.py --image-list`, then train the QA model and rank all masks
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