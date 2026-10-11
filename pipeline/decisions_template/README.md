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

Which label sheets are read depends on `labels.source` in the config; the other ones can stay empty.

| File | Read by | Columns read | One row = |
|---|---|---|---|
| `mask_drops_<domain>.tsv` | `merge_runs.py --drop-sha` (step "merge") | `sha256`, `reason` | an image whose mask failed the review and is left out of the curated set |
| `labels/value_renames.tsv` | `import_labels.py --decisions` (step "labels", `source: csv`) | `column`, `from`, `to`, `decision` | a value of one of your label columns to replace (spelling, synonym); `accept` = replace, empty = keep apart |
| `labels/image_fixes.tsv` | same | `sha256`, `note`, `decision` | one image whose labels are wrong: `drop`, or `set <column> <value>` |
| `labels/genus_spelling.tsv` | `extract_labels.py --decisions` (step "labels", `source: nhm_filenames`) | `from_genus`, `to_genus`, `decision` | two spellings of a genus; `accept` = rename `from_genus` to `to_genus`, empty = keep apart |
| `labels/catalog_conflicts.tsv` | same | `names (images)`, `catalogs`, `images`, `decision` | a set of species names found under catalog numbers that carry more than one genus |
| `labels/needs_review.tsv` | same | `sha256`, `note`, `decision` | an image whose file name the parser could not read cleanly |

The other columns (`images`, `images_from`, `proposal`, `confidence`, …) are for the person deciding and are not
read. `mask_drops_*.tsv` may have `#` comment lines; the label sheets may not. Every applied fix is written to the
labels' `label_change` column, every dropped image to `label_drop` with the reason.

## Own labels (`labels.source: csv`)

Your labels CSV (`labels.path`), comma-separated, UTF-8, one row per image:

```csv
file_name,specimen,species,origin
photos/IMG_0001.jpg,F-17,Salmo trutta,lake
photos/IMG_0002.jpg,F-17,Salmo trutta,lake
xrays/F-17_lateral.png,F-17,Salmo trutta,lake
photos/IMG_0003.jpg,F-18,Salmo truta,river
```

- `file_name` is relative to `raw_dir`; the first folder is the domain. Every image under `raw_dir` needs exactly
  one row and every row an image, otherwise `scripts/pipeline/import_labels.py` stops and lists the mismatches.
- The specimen column (`labels.specimen`, here `specimen`) says which images show the same physical object; they
  always land in the same split. Without specimen information set `specimen: none`: every image is then its own
  specimen, and the split warns that images of one specimen can be in train and test.
- All other columns are your labels; any of them can be the class (`split.class`).

Fix messy labels here, not in the CSV, so every change stays visible:

- **value_renames.tsv**: `column  from  to  images  note  decision`, e.g.
  `species  Salmo truta  Salmo trutta  1  misspelling  accept`. `from` must occur in that column.
- **image_fixes.tsv**: `sha256  note  decision`, decision `drop` or `set <column> <value>`, several joined with
  `; `, e.g. `set species Salmo trutta; set origin lake`. Byte-identical copies all get the fix.

Byte-identical images (same SHA-256) are kept once, under the alphabetically first file name.

## NHM file names (`labels.source: nhm_filenames`)

The thesis route: `scripts/pipeline/extract_labels.py` parses `Genus_species_NMW12345….jpg` (NHM Wien) and groups specimens
by NMW catalog number. A worked example of messy labels fixed transparently; the filled-in sheets are in
`pipeline/decisions/labels/`.

- **catalog_conflicts.tsv** must list every conflict in the data: if `extract_labels.py` finds a set of names
  that is not on the sheet, it stops and prints the names. `names (images)` is those names, sorted, each with
  its image count, joined by ` | `, e.g. `Alestes adolfi (3) | Rutilus rutilus (1)`; `catalogs` (how many
  catalog numbers) and `images` (how many images) must match the data.
  Decisions: `keep`, `rename A -> B` (everywhere), `rename in catalog A -> B` (only in these catalogs),
  `drop A` (in these catalogs).
- **needs_review.tsv** decisions: `drop`, `set catalog NMW<number>`, `keep, own specimen group XYZ<number>`,
  `set species <name>` (when the parser found no species); combine with `; `, e.g.
  `set catalog NMW<number>; set species Cottus gobio`;
  `<number>` is filled in from the file name at run time.
