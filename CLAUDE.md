## Name
masterthesis_fish_VIT

## Purpose
XAI on ViT attention maps for fish morphology (NHM Wien imagery)

## Non-negotiable constraints
- Local-only processing, no cloud/external APIs
- Only Neo + supervisor access; weights never published
- NHM attribution required in public outputs

## Current state
- **Masks**: photos = setting D accepted (`data/processed/segmented/D_merged_2026-10-05/full_body`, 4.0 %). X-rays = setting D, pooled 70 / 600 = 11.7 % wrong (seed 1 14.7 %, seed 2 8.7 %); fin fix (setting E, `--fin-threshold`, default off) **rejected**: vault/thesis-log/decisions/2026-10-08-xray-fin-extension.md. History: vault/thesis-log/experiments/mask-quality-history.md
- **X-ray route decided (Neo, 2026-10-08)**: drop rule B (any instance with `outside_box_frac` > 0.02, `bleed_fallback` inverse/unresolved, or `edge_uncertain_frac` > 0.5 → 429 of 5,031 X-rays, 5 genera below 5 specimens) + a fresh seed-3 sample that **alone** decides at ≤ 10 %; pass rule written before the sample: vault/thesis-log/decisions/2026-10-08-xray-drop-rule-B.md
- **Label decisions done (Neo, 2026-10-08), not yet applied in code**: `data/processed/label_decisions/` (gitignored): `genus_spelling.csv` (30 / 32 accepted; decision column empty = rejected), `catalog_conflicts.csv` (26 sets: renames apply to the species name everywhere, keep/drop per catalog), `needs_review.csv` (9 rows). Species-only conflicts (61 catalogs) left as is: classes are genera. With these fixes 0 / 41 duplicate sets disagree on genus. Origin: not in this dataset (a later live-fish dataset carries it)
- **Split + manifest** are still the 2026-10-04 ones on B (`data/processed/split/B_merged_2026-10-03_seed0/`, `data/processed/manifest/B_merged_2026-10-03_seed0/manifest.csv`; gitignored, hold catalog numbers): genus classes, ≥ 5 specimens per domain, 70/15/15 by specimen group. Decision: vault/thesis-log/decisions/2026-10-04-specimen-split.md
- Scope (2026-10-03): part 1 = raw → labels → masks → random review + error report → curated set → split → manifest, one command, human decisions as text files keyed by image SHA-256. Plan: vault/thesis-log/tasks/pipeline-part-1-cleanup.md

## Next steps
1. Claude: build the rule-B drop list from `D_merged_2026-10-05/Röntgen/annotations.jsonl` (reason `qa_rule_B`), write the seed-3 exclude list (seed 0 + 1 from `data/processed/segmented/exclude_Röntgen_seed0+seed1.txt` + seed 2 from `E_merged_2026-10-08/Röntgen/sample_seed2/sample.json`), `merge_runs.py --base D_merged_2026-10-05/Röntgen --drop <rule-B list>` → new X-ray dir; Neo: `review_masks.py --run-dir <that dir> --sample 300 --seed 3 --exclude <list>` (~15 min) → `--summary` + `error_report.py`; apply the pass rule exactly as written
2. Claude: apply the label decision sheets in the label step (global genus spelling + species renames, per-catalog drops, needs_review fixes, one copy per duplicate set: the one whose file name carries the final name, else alphabetically first)
3. After 1 + 2: rerun `split_specimens.py` + `build_manifest.py` on the final set (photos D_merged, X-rays D minus rule B)
4. Close part 1: vault/thesis-log/tasks/pipeline-part-1-cleanup.md (rewrite its rebuild section around D: one full run + drops; delete E runs ~6 GB after Neo OKs), checklist vault/thesis-log/tasks/clean-segmented-dataset-with-structured-labels.md

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