# 2026-10-01: dev set for comparing segmentation settings

Finished experiment record, not a pipeline step. `make_devset.py` picked a stratified dev set (100 photos + 50
X-rays, seed 0, at most one image per specimen) on which segmentation settings were compared blind with
`scripts/review/review_masks.py --compare` (the fin fix, setting C). It is not a random sample, so it was kept out
of the later random reviews.

Thesis notes (Neo's vault, not in this repo): decisions/2026-10-01-fin-fix-setting, decisions/2026-10-01-mask-policy.
Its input paths are the thesis data's (NHM Wien); for a new dataset the review loop in the main README replaces it.
