## Name
masterthesis_fish_VIT

## Purpose
XAI on ViT attention maps for fish morphology (NHM Wien imagery)

## Non-negotiable constraints
- Local-only processing, no cloud/external APIs
- Only Neo + supervisor access; weights never published
- NHM attribution required in public outputs

## Current state
- Masks: setting B (`--mask-threshold -1 --min-component-frac 0.01`) + 3 % margin at training-image time; fin fix cut the fix rate 39 → 23 % photos, 36 → 22 % X-rays; margin ring test passed, background alone predicts genus at 71 % / 61 % → masking necessary. Full B run: `data/processed/segmented/B_full_2026-10-01/` (11,731 photos, 5,207 X-rays)
- Bleed review done (2026-10-03): all 589 suspects reviewed, bleed confirmed in 72 photos (0.6 %) + 379 X-rays (7.3 %); verdicts in `B_full_2026-10-01/<domain>/review/`
- Bleed fallback in `segment_fish.py` (`--bleed-outside-frac 0.02`, `--bleed-max-box-edge` 0.1 photos / 0.3 X-rays; commit `151a802`). Run over the 589 suspects → `data/processed/segmented/B_bleedfix/`: 60/72 photo + 278/379 X-ray bleeds replaced, 0/91 OK masks touched; `<domain>/changed.txt` (374 total), `<domain>/unresolved.txt` (90). Not yet blind-reviewed, not merged
- `scripts/merge_runs.py`: merges a targeted re-run into a NEW full run dir (inputs untouched), `--take-ok-from <review-dir>:<run>`
- Thesis docs: vault/thesis-log/experiments/2026-10-03-bleed-fallback.md (full record) + experiments/mask-quality-history.md (all mask stages, failure catalogue, numbers); figures `B_bleedfix/figures/`. NHM dataset 16,942 images, `labels.csv` matches (9 rows need review)

## Next steps
1. Neo: blind compare of the 374 changed masks. Why: thresholds were tuned on these images, the blind verdict decides what gets merged and gives the thesis numbers. How: `python scripts/review_masks.py --compare B=data/processed/segmented/B_full_2026-10-01/<domain> --compare bleedfix=data/processed/segmented/B_bleedfix/<domain> --image-list data/processed/segmented/B_bleedfix/<domain>/changed.txt --review-dir data/processed/segmented/B_bleedfix/review_compare_<domain> --seed 0`, then `--summary --review-dir …`
2. Then: `scripts/merge_runs.py --base …/B_full_2026-10-01/<domain> --patch …/B_bleedfix/<domain> --take-ok-from …/B_bleedfix/review_compare_<domain>:bleedfix --out data/processed/segmented/B_merged_2026-10-03/<domain>`; fill in §5 of the bleed-fallback record + mask-quality-history
3. CVAT list: `unresolved.txt` + bleed the trigger missed (6 photos, 27 X-rays) + replacements judged fix
4. Afterwards: remaining non-bleed errors (missed_fish, merged), train/test split by NMW specimen number, random review sample per domain → QA model
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