# morphattention
Explainability analysis of ViT attention maps on fish morphological features (master's thesis, images of the
NHM Wien Fish Collection). This repository holds **part 1** of the pipeline: raw images to a curated, split dataset
with reviewed segmentation masks. It runs on other datasets than the thesis's by editing a config and a few
decision files, without code changes. Later milestones (baselines, attention analysis) are not built yet.

Everything runs locally. The only network access is the one-time model weight download; models then load offline,
the review page listens on `127.0.0.1` only, and the HTML reports fetch nothing. Images, outputs and weights live in
the gitignored `data/`.

## What part 1 does

```
raw images -> labels -> masks -> random review + error report (human, until accepted) -> drops -> split -> manifest.csv
```

| Step | What happens | Script |
|---|---|---|
| labels | one row per image, from your CSV or parsed from NHM file names; label fixes from decision files | `scripts/pipeline/import_labels.py`, `extract_labels.py` |
| segment | Grounding DINO finds each object from a text prompt (`"fish."`), SAM masks it; masks, overlays, COCO file, QA fields per instance | `scripts/pipeline/segment_fish.py` |
| review | a random sample per domain is judged in the browser; the error report gives the error rate with a confidence interval | `scripts/review/` |
| merge | leaves out the images the review judged bad (and, optionally, a QA drop rule) -> curated set | `scripts/pipeline/merge_runs.py` |
| split | train / val / test by specimen: all images of one specimen land in the same split | `scripts/pipeline/split_specimens.py` |
| manifest | `manifest.csv`, the training input | `scripts/pipeline/build_manifest.py` |

`scripts/run_part1.py` runs every step except the review, in this order, from one config file.

`manifest.csv` has one row per raw image, unusable ones included, so you can see why an image is not used:

| Column | Content |
|---|---|
| `file_name`, `sha256`, `domain` | path below `raw_dir`, SHA-256 of the image bytes, first folder of the path |
| label columns, `<class>` | own CSV: `specimen_id` and your label columns. NHM: `species`, `genus`, `catalog_number`, `needs_review`. Plus the split's class if derived (`binomial`, or `genus` from your `species`) |
| `specimen`, `mask_path`, `n_instances` | specimen group of the split; mask PNG in the curated run (0 = background, 255 = object); number of object masks |
| `status`, `reason` | `usable` / `dropped` / `skipped`, and why (label decision, mask review, no detection, no class value, ...) |
| `split`, `<class>_evaluated` | `train` / `val` / `test` / `train_only` / `none`; whether the class has enough specimens to be evaluated in this domain |

## Setup (Docker)

Needs Docker with Compose v2 or later (the `docker compose` plugin; it is numbered v5 since late 2025) and, for the GPU, the NVIDIA Container Toolkit on the host.
In the cloned repo:

```bash
printf 'USER_ID=%s\nGROUP_ID=%s\n' "$(id -u)" "$(id -g)" > .env    # files written in the container stay yours
cp docker-compose.override.example.yml docker-compose.override.yml  # your own mounts, see below
docker compose build
docker compose run --rm vit-project python -c "import torch; print(torch.cuda.is_available())"
# model weights, once (the only step that needs the network; afterwards everything runs offline)
docker compose run --rm vit-project python scripts/setup/download_sam.py
docker compose run --rm vit-project python scripts/setup/download_dino.py
docker compose run --rm vit-project python -m pytest                # all tests
```

In VS Code, "Dev Containers: Reopen in Container" uses the same files and creates the override from the example
when it is missing.

**Your own setup** goes into `docker-compose.override.yml` (gitignored), never into `docker-compose.yml`:
- images stored elsewhere: mount them under `data/raw/`, e.g.
  `- /mnt/storage/fish_images:/workspace/data/raw/my_dataset:ro`
- other personal mounts (editor or Claude Code config, notes)
- no NVIDIA GPU: `deploy: !reset null` (Compose ≥ 2.24); SAM + DINO then run on the CPU, slowly

`docker compose config` shows what Compose will actually use, both files merged.

**Versions:** `requirements.txt` pins the direct dependencies; `constraints.txt` pins everything pip installs
on top of the base image (`pytorch/pytorch:2.2.1-cuda12.1-cudnn8-runtime`, Python 3.10). To add a package, add
it pinned to `requirements.txt`, rebuild, and copy its new sub-dependencies from `pip freeze` into
`constraints.txt`.

All commands below run in the container, from the repo root (`/workspace`): in the dev container's terminal, or
prefixed with `docker compose run --rm vit-project`. Config paths are relative to the repo root.

## Your data

Put the images under one folder, one subfolder per **domain** (a kind of image that gets its own settings, review
and error rate, e.g. photos and X-rays). Below a domain folder any depth is fine; formats jpg, jpeg, png, tif/tiff,
bmp, webp.

```
data/raw/my_dataset/        raw_dir in the config
├── photos/                 domain "photos"
│   ├── IMG_0001.jpg
│   └── ...
├── xrays/                  domain "xrays"
└── labels.csv              your labels (labels.source: csv)
```

Two routes for the labels (`labels.source` in the config):

| `labels.source` | Use it for | Specimens from | Label fixes in `<decisions_dir>/labels/` |
|---|---|---|---|
| `csv` (default) | your own dataset: a CSV with one row per image | your specimen column (`labels.specimen`, or `none`) | `value_renames.tsv`, `image_fixes.tsv` |
| `nhm_filenames` | NHM Wien file names `Genus_species_NMW<number>_….jpg` (the thesis) | NMW catalog number in the file name | `genus_spelling.tsv`, `catalog_conflicts.tsv`, `needs_review.tsv` |

CSV format, example and decision sheets:
[`pipeline/decisions_template/README.md`](pipeline/decisions_template/README.md#own-labels-labelssource-csv).
In short: `file_name` relative to `raw_dir`, a specimen column (images of one physical object stay in one split),
any label columns; messy values are fixed in the decision sheets, not in the CSV, so every change stays visible.
The split's class (`split.class`) is `genus` or `binomial` (from a `species` column) or any label column as written.

## Run part 1

1. Copy the empty decision files and make one `mask_drops_<domain>.tsv` per domain:

   ```bash
   cp -r pipeline/decisions_template pipeline/decisions_my_dataset
   mv pipeline/decisions_my_dataset/mask_drops_DOMAIN.tsv pipeline/decisions_my_dataset/mask_drops_photos.tsv
   cp pipeline/decisions_my_dataset/mask_drops_photos.tsv pipeline/decisions_my_dataset/mask_drops_xrays.tsv
   ```

   Header only = no decisions. Images are keyed by the SHA-256 of their bytes, never by file name.

2. Edit `pipeline/config.yaml`. It is the neutral start: plain Grounding DINO + SAM, every mask fix off, no drop
   rule, and every key is explained in its comments. Set at least `name`, `raw_dir`, `decisions_dir`, the `labels`
   block, one entry per domain under `domains`, and `split.class`. The `review` block holds the acceptance limit
   (`max_rate`, default 0.10) and the error categories of the review (a standard set: `bleed`, `wrong_object`,
   `merged`, `partial`, `missed_object`, `missing_part`; add or split your own, at most 9). Set `attribution`
   for the footer of the reports.

3. Run:

   ```bash
   python scripts/run_part1.py --dry-run                    # check the inputs, print the commands, run nothing
   python scripts/run_part1.py --steps labels,segment       # labels + masks; then the review loop below
   python scripts/run_part1.py --steps merge,split,manifest # once the review is accepted
   ```

   `--steps` takes any subset of `labels,segment,merge,split,manifest` (always run in that order; default all),
   `--name NAME` replaces the config's run name (every output path contains it), `--config FILE` picks another
   config (default `pipeline/config.yaml`), `--dry-run` prints the commands and missing inputs and runs nothing.

Before anything runs, `run_part1.py` checks the inputs of the chosen steps: `raw_dir` and every domain folder, the
labels CSV, the label sheets of your labels route, and `mask_drops_<domain>.tsv` per domain if `merge` runs. If
one is missing it lists them and stops, so no run silently uses someone else's decisions.

- `segment` needs the GPU (the thesis's ~17,000 images take about 4 h on one GPU). It always runs with `--resume`,
  so an interrupted run continues; a rerun under the same name with changed settings is refused, so a changed
  setting needs a new `--name`.
- `merge` is skipped when `segmented/<name>/<domain>` exists; delete that folder to apply changed drops. A domain's
  `qa_rule` drops images by their QA fields there (`null` = none; `B` = the thesis X-ray rule, see
  `scripts/pipeline/merge_runs.py`).

## The review loop

Masks are never all correct. The loop estimates the share of wrong masks per domain from a random sample and
changes settings until that share is at most `review.max_rate`. The scripts make each round a few commands; the
judging and the choice of setting stay with you. Every script documents its flags in its docstring
(`python <script> --help`). The examples use the domain `photos`; repeat the per-domain commands for every domain.

**1. Segment** with the neutral settings: `python scripts/run_part1.py --steps labels,segment` writes
`data/processed/segmented/my_dataset_full/<domain>/`.

**2. Random review** of a seeded sample:

```bash
python scripts/review/review_masks.py --run-dir data/processed/segmented/my_dataset_full/photos \
    --sample 300 --seed 1 --reviewer Ana
```

Open `http://localhost:8765` (`--port` to change). The page shows one overlay at a time, blind (no file name, QA
flags or object count). Keys: `1`–`9` toggle an error category (the config's `review.categories`, in that order),
`f` fix (the mask is wrong), `x` drop (the image itself is unusable), `→`/Space next (an image left without a
verdict counts as ok), `←` back, `u` clear, `n` next unjudged, hold `r` for the raw image. Every keypress is saved
to `<run-dir>/review_random_seed1/` and a restart resumes; resuming with a changed category list is refused.
The server binds `127.0.0.1` inside the container: in the VS Code dev container the port is forwarded to your
browser; with plain `docker compose run` add `network_mode: host` to your override (Linux).

**3. Error report:**

```bash
python scripts/review/error_report.py \
    --review-dir data/processed/segmented/my_dataset_full/photos/review_random_seed1 \
    --review-dir data/processed/segmented/my_dataset_full/xrays/review_random_seed1 \
    --out data/processed/reports/my_dataset_seed1.html
```

One offline HTML page with, per domain: the error rate (fix + drop) / judged with a 95 % Wilson interval, accepted
or not (the point estimate decides: rate ≤ `review.max_rate`), the count per category, and the table **Settings per
category**: which segment setting lowers each category, which way to turn it, what it costs, and the value this run
used. `python scripts/review/review_masks.py --summary --review-dir <review dir>` prints the numbers in the terminal.

**4. Not accepted: change one setting.** Take the largest category, look it up in the report's settings table and
change **one** value in the config's `segment` block (or in one domain's `segment`). Rerun under a new name; the
labels step is cheap and the split later reads `labels_<name>.csv`:

```bash
python scripts/run_part1.py --steps labels,segment --name my_dataset_v2
```

**5. Which masks changed:**

```bash
python scripts/review/mask_changes.py --name v1 \
    --before data/processed/segmented/my_dataset_full/photos \
    --after data/processed/segmented/my_dataset_v2_full/photos --pick 100 --seed 0
```

Writes into `--after`: `changed_vs_v1.{tsv,txt,json}` (IoU per image and which fixes fired, the changed images,
counts) and `changed_vs_v1_pick100_seed0.txt` (100 random changed ones; with fewer it stops: rerun without `--pick`
and use the `.txt`).

**6. Blind compare** of the old and the new run on those images, then the compare report:

```bash
python scripts/review/review_masks.py \
    --compare v1=data/processed/segmented/my_dataset_full/photos \
    --compare v2=data/processed/segmented/my_dataset_v2_full/photos \
    --image-list data/processed/segmented/my_dataset_v2_full/photos/changed_vs_v1_pick100_seed0.txt \
    --review-dir data/processed/reviews/compare_v1_v2_photos --seed 0 --reviewer Ana
python scripts/review/compare_report.py --before v1 --after v2 \
    --review-dir data/processed/reviews/compare_v1_v2_photos \
    --baseline data/processed/segmented/my_dataset_full/photos/review_random_seed1 \
    --out data/processed/reports/compare_v1_v2.html
```

Each image is shown once per run, shuffled, with only "item k / N" on the page; the review dir must not be inside a
run dir. Use the same short names in `mask_changes.py --name`, `--compare` and `--before`/`--after` (the report
reads `changed_vs_<before>.json`). Per domain it shows the wrong rate before and after, images fixed / broken /
still wrong (McNemar test), categories gone / kept / new, and with `--baseline` (the old run's random review dir)
the expected error rate of the new run.

**7. If the new run is better** (more fixed than broken, no category made worse than you accept): draw a fresh
random sample of the new run with a **new seed** and make its error report:

```bash
python scripts/review/review_masks.py --run-dir data/processed/segmented/my_dataset_v2_full/photos \
    --sample 300 --seed 2 --reviewer Ana
python scripts/review/error_report.py \
    --review-dir data/processed/segmented/my_dataset_v2_full/photos/review_random_seed2 \
    --out data/processed/reports/my_dataset_v2_seed2.html
```

Still not accepted: back to step 4 with the next setting, compared against this run. If the new run was worse,
undo the setting and try the next one from the table.

**8. Accepted: drop the reviewed-bad images and build the set.** Export the accepted sample's verdicts keyed by
SHA-256 (a record without file names), append its `fix` and `drop` images as drop rows, then run the rest of part 1
under the accepted run's name. Keep the config at that run's settings: `merge` saves it as
`segmented/<name>/part1_config.yaml`.

```bash
python scripts/review/export_reviews.py \
    --review-dir data/processed/segmented/my_dataset_v2_full/photos/review_random_seed2 \
    --out pipeline/decisions_my_dataset/reviews/my_dataset_v2_photos_seed2.tsv
awk -F'\t' '!/^#/ && ($3=="fix" || $3=="drop") {print $2 "\trandom review seed 2: " $3 " " $4}' \
    pipeline/decisions_my_dataset/reviews/my_dataset_v2_photos_seed2.tsv \
    >> pipeline/decisions_my_dataset/mask_drops_photos.tsv
python scripts/run_part1.py --name my_dataset_v2 --steps merge,split,manifest
```

Why the loop works this way:
- **One setting at a time.** Every fix has a cost (the report's cost column: a lower `mask_threshold` keeps thin
  parts but lets in more background). With two changes in one run you cannot tell which one fixed or broke what.
- **Blind compare on changed images.** Unchanged masks (IoU ≥ 0.99) look the same in both runs, so ~100 changed
  images show the difference. No run name, file name or QA flag on the page and a shuffled order keep you from
  favouring the new run (overlays with typical QA flags can still be recognisable).
- **New seed for the acceptance sample.** The setting was picked to fix what the old sample showed, so the old
  sample would rate the new run too well. A new seed draws other images; to keep the tuning images out entirely,
  add `--exclude FILE` (one file name per line, e.g. the `sample` list of the old `sample.json`).
- **Sample size.** 300 images give an interval of roughly ±3.5 percentage points around 10 %.

Each domain is reviewed and accepted on its own. A new `--name` re-segments every domain; a domain whose settings
did not change gets the same masks (up to the pixel differences below), so its accepted review carries over. With
a `qa_rule`, review the merged run `segmented/<name>/<domain>` (run `--steps merge` first), since the rule's drops
are not in the training set. Its random review lands inside that folder unless you pass `--review-dir` elsewhere,
so export the verdicts (step 8) before you delete the folder for the final merge.

## Outputs

Under `processed_dir` (`data/processed/`, gitignored; outputs contain your file names):

| Path | Step | Content |
|---|---|---|
| `labels_<name>.csv` | labels | one row per image: labels, `label_change`, `label_drop` |
| `segmented/<name>_full/<domain>/` | segment | `masks/` (PNG per image, 0/255, original size), `overlays/`, `coco/annotations.json`, `annotations.jsonl` (QA fields per instance), `skipped.txt`, `run_config.json`; the random review dirs land here too |
| `segmented/<name>/<domain>/` | merge | curated run, same layout, without the dropped images (`dropped.txt` with reasons; masks are hard links) |
| `segmented/<name>/part1_config.yaml` | merge, split, manifest | the effective config of the run |
| `split/<name>_seed<seed>/` | split | `split.csv`, `specimens.csv`, `balance.csv`, `dropped_no_catalog.txt`, `dropped_no_class.txt`, `duplicates.csv` |
| `manifest/<name>_seed<seed>/manifest.csv` | manifest | the training input |
| `sha256_cache.json` | all | cached image hashes |

`segment`, `merge`, `split` and `manifest` also write a `run_config.json` with their settings, inputs and git state.

## Reproducibility

The same raw images, config and decision files give the same curated set, split and manifest with one command:
every human decision is a file the pipeline reads, keyed by image content, so renaming images does not break it,
and package versions are pinned. GPU inference can differ by a few pixels between machines (GPU model, driver), so
masks are not guaranteed to be bit-identical across machines.

## The thesis run

```bash
python scripts/run_part1.py --config pipeline/config_thesis_nhm.yaml
```

Needs the NHM Wien images (not public) under `data/raw/NHM_datensatz/`, domains `full_body` (photos) and `Röntgen`
(X-rays). Differences from the neutral config: labels parsed from the NHM file names (`nhm_filenames`); setting D
(every mask fix on, values in the config), X-rays with `bleed_max_box_edge: 0.3` and drop rule B; the six thesis
error categories `fin_cut`, `bleed`, `merged`, `partial`, `missed_fish`, `wrong_object`; the author's decisions
in `pipeline/decisions/` (mask drops, label sheets, and in `reviews/` the exported verdicts behind the thesis error
rates; see [`pipeline/decisions/README.md`](pipeline/decisions/README.md)). Review tools on thesis runs need
`--config pipeline/config_thesis_nhm.yaml` for its categories.

## Repository layout

| Path | Content |
|---|---|
| `scripts/run_part1.py` | part 1 in one command |
| `scripts/pipeline/` | the steps: `extract_labels.py`, `import_labels.py`, `segment_fish.py`, `merge_runs.py`, `split_specimens.py`, `build_manifest.py` |
| `scripts/review/` | review tools: `review_masks.py`, `error_report.py`, `compare_report.py`, `mask_changes.py`, `export_reviews.py`, `flag_bleed.py` (lists suspected background bleed for a targeted review) |
| `scripts/setup/` | model downloads: `download_sam.py`, `download_dino.py`, `download_dinov2.py` (DINOv2, for later milestones) |
| `pipeline/` | `config.yaml` (neutral), `config_thesis_nhm.yaml`, `decisions_template/` (empty decision files), `decisions/` (thesis decisions) |
| `src/data/models/` | model wrappers (SAM, Grounding DINO, DINOv2), loading offline from `data/model_cache/` |
| `experiments/<date>-<name>/` | finished experiment scripts, kept as records; not pipeline steps |
| `tests/` | pytest suite |
| `Dockerfile`, `docker-compose*.yml`, `.devcontainer/`, `requirements.txt`, `constraints.txt` | container and pinned versions |
| `data/` | gitignored: raw images, outputs, model weights |
| `vault/thesis-log` | link to the author's private notes (decisions, experiment logs) that docstrings cite; not available in a clone |

## Tests

`python -m pytest` in the container runs all tests. `tests/test_sam.py` and `tests/test_dino.py` load the
downloaded weights with the network blocked, so download the models first.

## Attribution

Images used in the thesis: Naturhistorisches Museum Wien (NHM Wien), Fish Collection. The images, and the masks,
labels and model weights derived from them, are not part of this repository. Segmentation uses SAM
(`facebook/sam-vit-large`) and Grounding DINO (`IDEA-Research/grounding-dino-tiny`), downloaded from the Hugging
Face Hub under their own licenses.

## License

GNU General Public License v3.0, see [`LICENSE`](LICENSE). It covers the code in this repository.
