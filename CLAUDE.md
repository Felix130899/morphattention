## Name
masterthesis_fish_VIT

## Purpose
XAI on ViT attention maps for fish morphology (NHM Wien imagery)

## Non-negotiable constraints
- Local-only processing, no cloud/external APIs
- Only Neo + supervisor access; weights never published
- NHM attribution required in public outputs

## Current state
- **Mask sets**: split + manifest are still built on `data/processed/segmented/B_merged_2026-10-03/` (11,693 photos + 5,031 X-rays usable, 38 + 176 dropped). **Photos: setting D accepted 2026-10-08** (`D_merged_2026-10-05/full_body`, 4.0 % wrong, CI 2.3–6.9; vs. B on the 549 changed photos 66 fixed / 0 broken of 100 reviewed): vault/thesis-log/decisions/2026-10-08-photo-masks-D-accepted.md. X-rays on D fail (14.7 %, fin_cut 25 / 300). History: vault/thesis-log/experiments/mask-quality-history.md
- **Setting E = D + X-ray fin extension (2026-10-08, proposed)**: `segment_fish.py --fin-threshold -6 --fin-max-growth 0.3` (commit `0b26e63`; off = D bit-identical): connected parts of the same SAM candidate above logit −6, added pieces on the image border dropped, > 30 % growth → keep D. Dev list (seed-1 25 fin_cut + 256 ok), area vs. D: fin_cut median +4.9 %, ok max +9.4 % (plain t−2: fin_cut +1.1 %, useless; plain t−6: 4 plate fills). Full X-ray run → `data/processed/segmented/E_full_2026-10-08/` (started 12:43 UTC, ~80 min). Record + numbers: vault/thesis-log/decisions/2026-10-08-xray-fin-extension.md
- **Split + manifest (2026-10-04, on B)**: `scripts/split_specimens.py` → `data/processed/split/B_merged_2026-10-03_seed0/`, `scripts/build_manifest.py` → `data/processed/manifest/B_merged_2026-10-03_seed0/manifest.csv` (gitignored, holds catalog numbers). Genus classes, ≥ 5 specimens per domain, 70/15/15 by specimen group, 0 leaks. Decision: vault/thesis-log/decisions/2026-10-04-specimen-split.md
- Label problems found by the split: 41 byte-identical image sets (18 with different species names, `split/.../duplicates.csv`), 103 catalog numbers with > 1 species name (synonyms, typos, NomMus). 6 images without catalog number dropped
- Scope (2026-10-03): part 1 = raw → labels → masks → random review + error report → curated set → split → manifest, one command, human decisions as text files keyed by image SHA-256. Plan: vault/thesis-log/tasks/pipeline-part-1-cleanup.md

## Next steps
1. Neo: blind compare D vs. E on the dev list (162 overlays, ~8 min), command in decision note §4 (`review_masks.py --compare D=... --compare E=data/processed/segmented/dev_D_fin6_border_2026-10-08/Röntgen --image-list data/processed/segmented/review_compare_DE_Röntgen_dev/image_list.txt --review-dir data/processed/segmented/review_compare_DE_Röntgen_dev --seed 0`), then `compare_report.py` (no `--baseline`: dev list is not random)
2. When `E_full_2026-10-08/full_run.log` says finished: `merge_runs.py --base E_full_2026-10-08/Röntgen --drop B_merged_2026-10-03/Röntgen/dropped.txt --out E_merged_<DATE>/Röntgen` (photos = D_merged full_body); draw seed-2 sample (300, exclude seed-0 + seed-1 names) with `review_masks.draw_sample`; Neo: blind compare **B vs. E** on it (600 overlays, ~30 min) = the like-for-like "better than B" number + E's rate for the ≤ 10 % rule (decision note §5; `compare_report.py` page text assumes changed images, needs a random-sample mode)
3. If E ≤ 10 %: accept (fill decision §4–5), rerun `split_specimens.py` + `build_manifest.py` on `E_merged_<DATE>`
4. Decide which copy of each byte-identical pair to keep (`duplicates.csv`) and normalise the species vocabulary; then rerun split + manifest
5. Close part 1: vault/thesis-log/tasks/pipeline-part-1-cleanup.md, checklist vault/thesis-log/tasks/clean-segmented-dataset-with-structured-labels.md

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