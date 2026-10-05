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
- **Mask error rate accepted (rule ≤ 10 %, Neo 2026-10-04)**: blind random sample 300/domain, seed 0: photos 7.0 % (Wilson CI 4.6–10.5), X-rays 10.0 % (7.1–13.9, exactly on the limit); bleed 2.7 / 3.3 %. Report vault/thesis-log/experiments/2026-10-04-mask-error-rate.html (`scripts/error_report.py`), decision decisions/2026-10-04-mask-error-acceptance.md. **Then Neo chose to fix the errors with setting D** (`segment_fish.py`, commit `ec26ce9`; dev estimate photos ~4 %, X-rays ~4.3 %; record vault/thesis-log/experiments/2026-10-04-random-review-fixes.md). **D full run done 2026-10-05** → `data/processed/segmented/D_merged_2026-10-05/` (same 38 + 176 drops as B, same counts); masks changed vs. B: 549 photos (4.7 %), 372 X-rays (7.4 %) (`changed_vs_B.*`). **Blind compare B vs. D done (Neo, 2026-10-05)**: on 100 random changed images/domain wrong 84 → 18 % (photos), 97 → 32 % (X-rays), 0 broken, page vault/thesis-log/experiments/2026-10-05-compare-B-vs-D.html (`scripts/compare_report.py`). **Fresh random sample on D (seed 1, Neo 2026-10-05)**: photos **4.0 %** (CI 2.3–6.9, accepted), X-rays **14.7 %** (CI 11.1–19.1, **not accepted**), driven by fin_cut 25/300 on masks identical to B (D didn't touch them; review likely stricter than seed 0). Page vault/thesis-log/experiments/2026-10-05-mask-error-rate-D.html, analysis in tasks/full-run-setting-D.md §5. B is still the current set
- Label problems found by the split: 41 byte-identical image sets (18 with different species names, `split/.../duplicates.csv`), 103 catalog numbers with > 1 species name (synonyms, typos, NomMus). 6 images without catalog number dropped
- Scope (2026-10-03): part 1 = raw → labels → masks → random review + error report → curated set → split → manifest, one command, human decisions as text files keyed by image SHA-256; QA model + YOLO-seg scrapped. Plan: vault/thesis-log/tasks/pipeline-part-1-cleanup.md

## Next steps
1. Decide the X-ray fin_cut problem (Neo): photos D pass; X-rays fail at 14.7 % on fin_cut that D doesn't fix. Option: try `--mask-threshold -2` for X-rays on a dev list (the 25 fin_cut images of seed 1 + OK ones), blind compare, rerun X-rays only, then a fresh sample (seed 2, exclude seed 0 + seed 1). Options + numbers: vault/thesis-log/tasks/full-run-setting-D.md §5
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