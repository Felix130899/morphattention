# Decision files: empty template

Copy this folder for your own dataset and point `decisions_dir` in your config at the copy:

```bash
cp -r pipeline/decisions_template pipeline/decisions_my_dataset
mv pipeline/decisions_my_dataset/mask_drops_DOMAIN.tsv pipeline/decisions_my_dataset/mask_drops_photos.tsv
```

One `mask_drops_<domain>.tsv` per domain in the config. A file with only its header means "no decisions";
a missing file stops `scripts/run_part1.py` before anything runs, so no run silently uses someone else's
decisions. The filled-in thesis files (NHM Wien) are in `pipeline/decisions/`, as an example.

Images are keyed by the **SHA-256 of the image file's bytes** (`sha256sum <file>`), not by file name, so the
files stay valid when images are renamed and file names (which may hold catalog numbers) stay out of git.

| File | Read by | Columns read | One row = |
|---|---|---|---|
| `mask_drops_<domain>.tsv` | `merge_runs.py --drop-sha` (step "merge") | `sha256`, `reason` | an image whose mask failed the review and is left out of the curated set |
| `labels/genus_spelling.tsv` | `extract_labels.py --decisions` (step "labels") | `from_genus`, `to_genus`, `decision` | two spellings of a genus; `accept` = rename `from_genus` to `to_genus`, empty = keep apart |
| `labels/catalog_conflicts.tsv` | same | `names (images)`, `catalogs`, `images`, `decision` | a set of species names found under catalog numbers that carry more than one genus |
| `labels/needs_review.tsv` | same | `sha256`, `note`, `decision` | an image whose file name the parser could not read cleanly |

The other columns (`images_from`, `proposal`, `confidence`, …) are for the person deciding and are not read.
`mask_drops_*.tsv` may have `#` comment lines; the label sheets may not.

## Label sheets

- **catalog_conflicts.tsv** must list every conflict in the data: if `extract_labels.py` finds a set of names
  that is not on the sheet, it stops and prints the names. `names (images)` is those names, sorted, each with
  its image count, joined by ` | `, e.g. `Alestes adolfi (3) | Rutilus rutilus (1)`; `catalogs` (how many
  catalog numbers) and `images` (how many images) must match the data.
  Decisions: `keep`, `rename A -> B` (everywhere), `rename in catalog A -> B` (only in these catalogs),
  `drop A` (in these catalogs).
- **needs_review.tsv** decisions: `drop`, `set catalog NMW<number>`, `keep, own specimen group XYZ<number>`;
  `<number>` is filled in from the file name at run time.

The label parser and these two sheets are specific to file names of the form `Genus_species_NMW12345….jpg`
(NHM Wien). Support for other datasets' labels is planned (own labels CSV); until then they need
`scripts/extract_labels.py` adapted.
