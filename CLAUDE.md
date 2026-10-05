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
- **Setting D, candidate replacement for B (2026-10-05)**: `data/processed/segmented/D_merged_2026-10-05/` = D full run (`segment_fish.py` setting D, commit `ec26ce9`) + B's 38 + 176 drops. Blind compare on 100 random changed images/domain: 131 fixed, 0 broken. Fresh random sample (seed 1, 300/domain, seed-0 images excluded): photos **4.0 %** (CI 2.3–6.9) accepted, X-rays **14.7 %** (CI 11.1–19.1) **not accepted** (rule ≤ 10 %): fin_cut 25/300, all on masks identical to B; this review was stricter than seed 0 (B: 7.0 / 10.0 %). Details: vault/thesis-log/tasks/full-run-setting-D.md §4–5, experiments/mask-quality-history.md stage 6
- Label problems found by the split: 41 byte-identical image sets (18 with different species names, `split/.../duplicates.csv`), 103 catalog numbers with > 1 species name (synonyms, typos, NomMus). 6 images without catalog number dropped
- Scope (2026-10-03): part 1 = raw → labels → masks → random review + error report → curated set → split → manifest, one command, human decisions as text files keyed by image SHA-256; QA model + YOLO-seg scrapped. Plan: vault/thesis-log/tasks/pipeline-part-1-cleanup.md

## Next steps
1. X-ray fin fix, Neo picks option A or B first (vault/thesis-log/tasks/full-run-setting-D.md §5). Option A: dev list = the 25 fin_cut + 256 ok X-rays of `D_merged_2026-10-05/Röntgen/review_random_seed1` → `segment_fish.py` setting D + `--mask-threshold -2` on that list → blind compare vs. D → if better, X-ray-only rerun (~1.3 h) → fresh sample seed 2 with `--exclude` = seed-0 + seed-1 sample names → `error_report.py`. Photos on D (4.0 %) can be accepted on their own; then split + manifest on D
2. Decide which copy of each byte-identical pair to keep (`duplicates.csv`, mostly old vs. current species name) and normalise the species vocabulary (synonyms/typos per catalog number); then rerun `split_specimens.py` + `build_manifest.py`
3. Then close part 1: vault/thesis-log/tasks/pipeline-part-1-cleanup.md. Full checklist: vault/thesis-log/tasks/clean-segmented-dataset-with-structured-labels.md

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