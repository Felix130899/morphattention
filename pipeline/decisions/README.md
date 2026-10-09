# Human decisions the pipeline reads

Every decision a person made about the dataset, as small tab-separated files, so a rerun from the raw images
gives the same curated set without asking anyone again. Images are keyed by the **SHA-256 of the image file's
bytes**, never by file name: NHM file names contain catalog numbers, which stay out of this public repo.
`<number>` in a decision is filled in from the file name at run time.

| File | Read by | Content |
|---|---|---|
| `mask_drops_full_body.tsv`, `mask_drops_Röntgen.tsv` | `scripts/merge_runs.py --drop-sha` | images dropped after the mask reviews, with the reason |
| `labels/genus_spelling.tsv` | `scripts/extract_labels.py --decisions` | genus spellings merged (decision `accept`; empty = kept apart) |
| `labels/catalog_conflicts.tsv` | same | catalogs whose images carry different genera: `rename A -> B` (everywhere), `rename in catalog A -> B`, `drop A` (in those catalogs), `keep` |
| `labels/needs_review.tsv` | same | file names the parser could not read cleanly: `set catalog NMW<number>`, `keep, own specimen group XYZ<number>`, `drop` |

`reviews/*.tsv` (written by `scripts/export_reviews.py`) are the **records** behind the mask numbers: the final
verdict per image of every review session (random samples seed 0–3, blind compares, first reviews), in the order
shown, with a header saying which runs were judged. The pipeline does not read them; they are the tracked source of
the error rates in the thesis.

Rules that are code, not lists: X-ray drop rule B (`merge_runs.py --qa-rule B`) and one copy per set of
byte-identical images (`extract_labels.py`). Reasons and history: `vault/thesis-log/` (not in this repo).
