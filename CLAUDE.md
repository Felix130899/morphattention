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
- Pilot (11 photos + 8 X-rays, `data/processed/segmented/pilot_per_instance/`): works on both; `--tiny-area-frac` 0.005 too high for trays
- Species names printed into many `_WEB` images: Clever Hans risk (vault/thesis-log/log/2026-09-24-per-instance-masks.md)
- NHM dataset in `data/raw/NHM_datensatz/` (16,975 images, 2 corrupt X-ray JPEGs); `labels.csv` parsed (9 rows need review, origin unconfirmed)

## Next steps
1. Run `segment_fish.py --prompt dino` on a stratified sample, `full_body/` and `Röntgen/` as separate runs; tune `--tiny-area-frac`, check dedup/overlap on real data
2. Write mask-policy decision (photos + X-rays), then triage/fix masks in CVAT (self-hosted) or X-AnyLabeling
3. Train YOLO-seg on 300–500 curated images, separate test sets per domain, joint model vs. two specialists
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