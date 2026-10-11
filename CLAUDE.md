## Name
masterthesis_fish_VIT

## Purpose
XAI on ViT attention maps for fish morphology (NHM Wien imagery)

## Non-negotiable constraints
- Local-only processing, no cloud/external APIs
- Only Neo + supervisor access; weights never published
- NHM attribution required in public outputs

## Current state
- **Masks**: photos = setting D accepted (4.0 %). X-rays = D minus drop rule B (429 dropped, 4,602 ok): seed-3 sample **11.0 %** (CI 7.9–15.0), above 10 %; left: fin_cut 28 / 300, bleed 5. Result: vault/thesis-log/decisions/2026-10-08-xray-drop-rule-B.md; history: vault/thesis-log/experiments/mask-quality-history.md
- **Supervisor answer pending** (talk week of 2026-10-12 earliest; Neo expects "accept 11 %"): question in vault/thesis-log/tasks/2026-10-08-supervisor-questions-part-1.md §3; what accepting needs + the dataset it gives: vault/thesis-log/decisions/2026-10-09-xray-accept-11-percent.md (draft, pending note at the bottom)
- **Final set built on that assumption** (gitignored, each with `PENDING_SUPERVISOR.txt`): `data/processed/segmented/D_final_2026-10-09/` (links: `full_body` → D_merged_2026-10-05, `Röntgen` → D_ruleB_2026-10-09), labels `data/processed/labels_final.csv` (47 label drops in `label_drop`), split `data/processed/split/D_final_2026-10-09_seed0/`, manifest `data/processed/manifest/D_final_2026-10-09_seed0/manifest.csv`: photos 11,660 / 202 genera, X-rays 4,591 / 152 genera (rule B costs 7)
- **Cleanup phases 1, 2, 3a–3d done (3a–3d: 2026-10-11)**, plan: vault/thesis-log/tasks/pipeline-part-1-cleanup.md (phase 3 = Neo's vault/thesis-log/tasks/clean_up_task_brainstorm_neo.md: part 1 must run on other datasets). Decisions + review verdicts tracked in `pipeline/decisions/` (SHA-256 keyed, no catalog numbers; repo is public). Two configs: `pipeline/config.yaml` = neutral default (mask fixes off, standard review categories, decisions from a copy of `pipeline/decisions_template/`), thesis = `--config pipeline/config_thesis_nhm.yaml` (setting D, the six thesis categories, rule B). Review categories + setting advice live in the config; every review dir has `review.json` (categories, reviewers; the 15 old ones backfilled 2026-10-11); error + compare report have fixed sections and cold wording. Split class from the config (`split.class`: genus / binomial / any labels column; specimen ids independent of the class). Docker: tracked standard compose + gitignored override, pinned versions, image `7cdf8ae34ecb`; `python -m pytest` 124 passed
- Scope (2026-10-03): part 1 = raw → labels → masks → random review + error report → curated set → split → manifest, one command, human decisions as text files keyed by image SHA-256

## Next steps
1. Neo: supervisor talk on §3. After it: accepted → delete the 3 `PENDING_SUPERVISOR.txt` (segmented/split/manifest `D_final_2026-10-09*`) and the pending note in vault/thesis-log/decisions/2026-10-09-xray-accept-11-percent.md, set its status accepted; otherwise → rebuild as that doc's last section says
2. Claude, cleanup phase 3e–3g (details in the plan): 3e own labels CSV route (incl. an own `specimen` column instead of NMW catalog grouping) + NHM parser as the documented dirty-labels example; 3f sort `scripts/`; 3g README incl. review-loop how-to (review_masks → error_report → change one setting → compare_report → fresh sample)
3. Claude + Neo, cleanup phase 4 (before it, Neo decides the 3 usable D_final images without species: leave out or set species by a label decision, plan 3d): full rebuild `python scripts/run_part1.py --config pipeline/config_thesis_nhm.yaml --name rebuild_<date>` (~4 h GPU, GPU must be free) and compare against `D_final_2026-10-09`; then Neo OKs the delete list in the plan
4. Claude: update vault/thesis-log/PLAN.md (origin-ID not in the NHM set, moves to the live-fish dataset; part-1 "done when" = genus labels)
5. Milestone 2 (baselines) not before Neo says so (Neo, 2026-10-09)

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