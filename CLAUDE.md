## Name
masterthesis_fish_VIT

## Purpose
XAI on ViT attention maps for fish morphology (NHM Wien imagery)

## Non-negotiable constraints
- Local-only processing, no cloud/external APIs
- Only Neo + supervisor access; weights never published
- NHM attribution required in public outputs

## Current state
- **Current mask set: `data/processed/segmented/B_merged_2026-10-03/{full_body,Röntgen}`** = B run + bleed-fallback masks judged OK, bad suspects dropped (Neo's decision, no CVAT): 11,693 photos + 5,031 X-rays usable, 38 + 176 dropped (`dropped.txt`, status "dropped" in annotations.jsonl, no mask). Margin (3 %) is applied when building training images
- Bleed fallback (`segment_fish.py --bleed-outside-frac 0.02 --bleed-max-box-edge` 0.1 photos / 0.3 X-rays): blind compare on 374 changed masks: fix rate 100 → 18 % photos, 99.6 → 26 % X-rays, 0 worse. Remaining large-coverage suspects (86 + 6) are all human-verified OK
- Earlier: fin fix cut the fix rate 39 → 23 % / 36 → 22 %; margin ring test passed; background alone predicts genus at 71 % / 61 % → masking necessary
- Thesis docs: vault/thesis-log/experiments/mask-quality-history.md (all mask stages + numbers), 2026-10-03-bleed-fallback.md (+ -results.html); `scripts/merge_runs.py` (`--take-ok-from`, `--drop`)
- NHM dataset 16,942 images, `labels.csv` matches (9 rows need review). Drops are not random: Coregonus −22 X-rays, Barbus −18; 6 singleton genera lose their only image in a domain

## Next steps
1. Train/test split by NMW specimen number (not by image) on `B_merged_2026-10-03`, separate test sets per domain; check class balance after the drops
2. Extend `review_masks.py` with a random-sample mode, then review ~1,000 random masks per domain from `B_merged_2026-10-03` (masks < 70 % coverage were never reviewed)
3. QA model on those labels (stored QA features incl. new `outside_box_frac` + DINOv2 crop embedding) → rank all masks, review the worst 5–10 %
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