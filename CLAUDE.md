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
- **Setting E = D + X-ray fin extension: rejected (2026-10-08)**. `segment_fish.py --fin-threshold` (commit `0b26e63`, default off). Seed-2 random sample (300 X-rays, seed-0 + seed-1 excluded), B and E judged blind in one session: **12.7 % vs. 12.7 %**; the extension broke 22 masks (bleed) for ≈ 11 fixed; D's rules on the same sample 11 fixed / 0 broken (p < 0.001). D's X-rays inferred 8.0–9.3 % on seed 2 vs. 14.7 % on seed 1 (p = 0.03), pooled 11.8 % (CI 9.5–14.7). Record: vault/thesis-log/decisions/2026-10-08-xray-fin-extension.md
- **Split + manifest (2026-10-04, on B)**: `scripts/split_specimens.py` → `data/processed/split/B_merged_2026-10-03_seed0/`, `scripts/build_manifest.py` → `data/processed/manifest/B_merged_2026-10-03_seed0/manifest.csv` (gitignored, holds catalog numbers). Genus classes, ≥ 5 specimens per domain, 70/15/15 by specimen group, 0 leaks. Decision: vault/thesis-log/decisions/2026-10-04-specimen-split.md
- Label problems found by the split: 41 byte-identical image sets (18 with different species names, `split/.../duplicates.csv`), 103 catalog numbers with > 1 species name (synonyms, typos, NomMus). 6 images without catalog number dropped
- Scope (2026-10-03): part 1 = raw → labels → masks → random review + error report → curated set → split → manifest, one command, human decisions as text files keyed by image SHA-256. Plan: vault/thesis-log/tasks/pipeline-part-1-cleanup.md

## Next steps
1. Neo: go through the label proposal sheets in `data/processed/label_decisions/` (gitignored; rebuilt by `make_proposals.py` there) and edit the `decision` column where you disagree: `genus_spelling.csv` (32 genus spellings → one class each), `catalog_conflicts.csv` (26 name sets across genera: rename / keep / drop), `needs_review.csv` (9 rows). Species-only conflicts (61 catalogs, same genus) are left as is: classes are genera. With the proposals, 0 of 41 duplicate sets disagree on genus → keep one copy per set automatically
2. Neo + supervisor: vault/thesis-log/tasks/2026-10-08-supervisor-questions-part-1.md (which number decides the X-rays: seed 1 14.7 %, seed 2 ≈ 9.0 %, pooled 11.8 %; is origin encoded anywhere)
3. Claude, after 1: apply the decision sheets in the label step (global genus/species map, per-catalog drops, needs_review fixes, one copy per duplicate set), then rerun `split_specimens.py` + `build_manifest.py` on `D_merged_2026-10-05` (or the X-ray set 2 decides)
4. Close part 1: vault/thesis-log/tasks/pipeline-part-1-cleanup.md (rebuild section to be rewritten around D: one full run + drops), checklist vault/thesis-log/tasks/clean-segmented-dataset-with-structured-labels.md

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