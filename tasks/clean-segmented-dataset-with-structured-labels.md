# Task: Clean segmented dataset with structured labels

From PLAN.md milestone: **Clean segmented dataset with structured labels**
Goal: Fix the known SAM mask-quality problem (center-point prompting is too
loose) by wiring in DINO box-prompting, and get the existing NHM Wien labels
into a structured format covering both species and origin per image.
Done when: All specimen images in `data/raw` have DINO+SAM masks that pass
overlay QA, and a single structured label file (species + origin per image)
exists and matches the image set.

## Purpose beyond this milestone (why quality matters here)

The masks produced here aren't just for this milestone's own sake — they're
meant to become pseudo-label training data for a YOLO model, so DINO+SAM can
eventually be dropped from the pipeline (see `PROGRESS_LOG.md`'s 2026-06-19
entry). That means mask errors (e.g. multiple fish merged into one blob) and
label errors don't just affect this dataset — they get baked into the next
model trained on top of it. Curating the actual YOLO training set from these
outputs is done manually by the user, not automated here.

## Note on current state (checked against the code, not assumed)

Grounding DINO box-prompting is already implemented — `scripts/segment_fish.py`
has a working `--prompt dino` mode and `src/data/models/sam.py`'s `segment()`
already accepts `input_boxes`. It has not been run/validated on real fish
images yet (`data/raw/processed/annotations/` is empty; only test WhatsApp
photos exist under `data/raw/first_test_dataset/`). DINO is already known/
assumed to beat `--prompt center` — step 1 below is an **implementation
smoke test** (does the pipeline run end-to-end and produce plausible
output), not a comparative quality judgment against center-mode.

Labels are currently only embedded in NHM Wien's filename/folder naming
convention (per user), and the real specimen images aren't in the repo yet
— only the label-extraction tooling can be built now; running it end-to-end
is blocked on those images arriving. Some records only have a species name;
others have species + exact origin — the label format must tolerate both,
not treat a missing origin as an error.

Museum photos can contain more than one specimen per image, so segmentation
must produce one mask per detected fish instance, not a single unioned mask
per image (the current `to_binary()` in `segment_fish.py` unions all
detections into one combined mask — this needs to change). Labels are
recorded at the image level (from the filename/folder), so all instance
masks from the same image inherit that image's species/origin label unless
evidence says otherwise.

**Accepted risk:** milestone 4/5 steps below are blocked on real NHM Wien
images arriving and the naming convention being confirmed with Neo/
supervisor. No contingency plan is built in for this — the user considers
this low-risk and will personally drive it to completion before moving to
the next milestone.

## Steps

1. **DONE (2026-09-20, see PROGRESS_LOG.md).** Implement/smoke-test
   `--prompt dino` end-to-end on `data/raw/first_test_dataset/`: confirm it
   runs without error, produces a plausible mask per detected fish, and
   captures each box's DINO confidence score (already available via the
   existing `--dino-box-threshold` flag, which stays user-adjustable). This
   is a pipeline check, not a center-vs-dino quality comparison.
2. Change segmentation output to one mask per detected instance instead of a
   single unioned mask per image — needed for real museum photos with
   multiple specimens. Verify on any test image where DINO returns more than
   one box.
3. **PARTIALLY DONE (2026-09-20).** Species + catalog-number convention
   confirmed empirically from `data/raw/NHM_datensatz/`'s real filenames:
   `<species>_NMW<catalog_number>_<free text>`, where `catalog_number` isn't
   a plain integer (ranges/lists/parenthetical sub-indices; a few `MNW`
   typos). Still **open**: whether/where origin is encoded — nothing in the
   observed filenames looked like an origin field, but this needs an actual
   confirm with Neo/supervisor, not just inference from the data. See
   `vault/thesis-log/decisions/2026-09-20-label-structure.md`.
4. **DONE, but scope changed (2026-09-20).** `scripts/extract_labels.py`
   exists and walks a raw-dir, but writes `file_name`, `photo_type`,
   `species`, `species_raw`, `catalog_number`, `extra`, `needs_review`,
   `review_reason` — no `origin` column yet, since step 3 hasn't confirmed
   how (or whether) it's encoded. The catalog-number tail and any would-be
   origin text both land in `extra` verbatim rather than being guessed at.
   Revisit this step's column list once step 3 closes.
5. Build and maintain a species/origin vocabulary list (e.g.
   `data/processed/label_vocabulary.json`) covering the known valid values,
   for automated validation in step 7. Not started.
6. **PARTIALLY DONE.** Real NHM Wien images are in place
   (`data/raw/NHM_datensatz/`) and `scripts/extract_labels.py` has been run,
   producing `data/processed/labels.csv` (16,966/16,975 rows parsed
   cleanly). `scripts/segment_fish.py --prompt dino` has NOT been run over
   this dataset yet — blocked on step 2 (per-instance masks) first.
7. Automated label validation: check every parsed `species`/`origin` value
   against the vocabulary list from step 5, flag any row missing the
   required `species` field, and flag statistical outliers (values far
   rarer than expected) for manual review. This catches systematic parser
   bugs across the whole dataset — it cannot catch a value that's
   internally consistent but factually wrong.
8. Manual spot-check: the user reviews a sample of `labels.csv` rows against
   the real filenames/folders by hand, to catch factually-wrong-but-
   consistent labels that the automated check in step 7 can't see. This is
   done before the dataset is trusted for downstream training.
9. Cross-check `labels.csv` against the segmentation run's
   `coco/annotations.json` image list — every segmented image must have a
   matching label row and vice versa. Skipped images (DINO found zero
   detections) are logged individually by filename, with a total skip count
   reported at the end of the run — not silently dropped.

## Files touched

- `scripts/segment_fish.py` (modify `to_binary`/mask-writing so each
  detected instance gets its own mask file instead of being unioned)
- `scripts/extract_labels.py` (new)
- `data/processed/labels.csv` (generated output; stays gitignored under the
  existing `data/` rule)
- `data/processed/label_vocabulary.json` (new — known-valid species/origin
  values for automated validation)
- `PROGRESS_LOG.md` (append the DINO pipeline smoke-test results from step 1)
- `data/raw/<category>/` (real NHM images added locally, not committed)
- `data/processed/segmented/<dataset>/{masks,overlays,coco}/` (regenerated
  via `segment_fish.py --prompt dino`, now with per-instance masks)

## Acceptance criteria

- `--prompt dino` runs end-to-end on `data/raw/first_test_dataset/` without
  error, producing one plausible mask per detected fish instance with its
  DINO confidence score captured; `--dino-box-threshold` remains adjustable.
- On any test image with more than one DINO detection, segmentation
  produces separate per-instance masks, not one merged mask.
- `scripts/extract_labels.py` produces `labels.csv` with `species` populated
  for every row, and `origin` populated only where the naming convention
  actually encodes it (never a fabricated placeholder).
- Automated validation (step 7) runs against `label_vocabulary.json` and
  reports: any row failing vocabulary match, any row missing `species`, and
  any statistical outlier values — as a reviewable list, not just a
  pass/fail.
- The user has manually spot-checked a sample of `labels.csv` rows against
  real filenames before the dataset is considered ready for the next
  milestone.
- Skipped images (zero DINO detections) are logged individually by
  filename, with a total count reported at the end of the run.
- PLAN.md's milestone done-criteria are met: real specimen images in
  `data/raw` have DINO+SAM per-instance masks passing overlay QA, and
  `labels.csv` exists, passes automated validation, and matches the image
  set.
