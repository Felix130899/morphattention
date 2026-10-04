## Name
masterthesis_fish_VIT

## Purpose
XAI on ViT attention maps for fish morphology (NHM Wien imagery)

## Non-negotiable constraints
- Local-only processing, no cloud/external APIs
- Only Neo + supervisor access; weights never published
- NHM attribution required in public outputs

## Current state
- **Current mask set: `data/processed/segmented/B_merged_2026-10-03/{full_body,Röntgen}`** = B run + bleed-fallback masks judged OK, bad suspects dropped (no CVAT): 11,693 photos + 5,031 X-rays usable, 38 + 176 dropped (`dropped.txt`). Margin (3 %) is applied when building training images. History + numbers: vault/thesis-log/experiments/mask-quality-history.md
- **Split + manifest done (2026-10-04)**: `scripts/split_specimens.py` → `data/processed/split/B_merged_2026-10-03_seed0/`, `scripts/build_manifest.py` → `data/processed/manifest/B_merged_2026-10-03_seed0/manifest.csv` (all 16,942 images; gitignored, holds catalog numbers). Genus classes, ≥ 5 specimens per domain evaluated (197 photo / 153 X-ray genera), 70/15/15 by specimen group shared across domains, 0 leaks. Decision: vault/thesis-log/decisions/2026-10-04-specimen-split.md
- Random review samples drawn (`review_masks.py --sample 300 --seed 0`, blind): `B_merged_2026-10-03/<domain>/review_random_seed0/`, not yet reviewed
- Label problems found by the split: 41 byte-identical image sets (18 with different species names, `split/.../duplicates.csv`), 103 catalog numbers with > 1 species name (synonyms, typos, NomMus). 6 images without catalog number dropped
- Scope (2026-10-03): part 1 = raw → labels → masks → random review + error report → curated set → split → manifest, one command, human decisions as text files keyed by image SHA-256; QA model + YOLO-seg scrapped. Plan: vault/thesis-log/tasks/pipeline-part-1-cleanup.md

## Next steps
1. Review the 300 random masks per domain (Neo): `python scripts/review_masks.py --run-dir data/processed/segmented/B_merged_2026-10-03/<domain> --sample 300 --seed 0`, then `--summary --review-dir .../<domain>/review_random_seed0` for the rate + Wilson CI
2. Error-rate report (chart page): % wrong masks per domain with Wilson CI, per error category, + which setting fixes each category. Too high → change setting, rerun, blind compare, review a fresh sample
3. Decide which copy of each byte-identical pair to keep (`duplicates.csv`, mostly old vs. current species name) and normalise the species vocabulary (synonyms/typos per catalog number); then rerun `split_specimens.py` + `build_manifest.py`
4. Then close part 1: vault/thesis-log/tasks/pipeline-part-1-cleanup.md. Full checklist: vault/thesis-log/tasks/clean-segmented-dataset-with-structured-labels.md

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