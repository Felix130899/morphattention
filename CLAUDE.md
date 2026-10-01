## Name
masterthesis_fish_VIT

## Purpose
XAI on ViT attention maps for fish morphology (NHM Wien imagery)

## Non-negotiable constraints
- Local-only processing, no cloud/external APIs
- Only Neo + supervisor access; weights never published
- NHM attribution required in public outputs

## Current state
- `segment_fish.py` writes one mask per fish (COCO + QA flags, `--resume`, `run_config.json`); design: vault/thesis-log/decisions/2026-09-24-per-instance-mask-output.md
- First full run done (2026-09-24, `data/processed/segmented/{full_body,Röntgen}/`): 11,764 photos → 20,461 masks, 5,207 X-rays, 2 unreadable. Masks cut fins (main failure in 1,841 alphabetical photo reviews, ~10 % bad), so the run will be redone; those old reviews are not QA-model training labels
- Decided: no manual review of all 17k masks. Fix generator → re-run → QA model trained on a random reviewed sample ranks masks → review top 5–10 % + 300 random per domain. Mask policy so far: ≤ ~5 % fin-edge loss OK, bleed into shadow/tray/text always bad
- Species names printed into many `_WEB` images → masked-image training + raw-image Clever Hans baseline (vault/thesis-log/decisions/2026-09-24-text-in-images-clever-hans.md)
- NHM dataset in `data/raw/NHM_datensatz/` (16,975 images, 33 "Kopie" files, 3 checked byte-identical); `labels.csv` parsed (9 rows need review, origin unconfirmed); train/test split must be by NMW specimen number

## Next steps
1. Hash-check all 33 "Kopie" files against their originals, delete the identical ones (before the re-run, so outputs need no cleanup)
2. Write the mask-policy decision in vault/thesis-log/decisions/ (agreed fin/bleed rule above; open: margin yes/no + width, barbels, tags/pins, X-ray outline)
3. Fin fix on a fixed dev set (~50 photos from `full_body/review/fix.txt` + ~30 X-rays): largest-area SAM candidate vs. `best_candidates` argmax, logit threshold −1…−3, size-relative margin; write QA features (candidate areas, area ratio, edge logit band, border touch, components) into COCO; then re-run everything into a new run dir
4. Extend `scripts/review_masks.py` (random sample + categories fin_cut/bleed/merged/partial/wrong_object/missed_fish), review ~1,000 per domain, train QA model (features + DINOv2 crop embedding), rank all masks
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