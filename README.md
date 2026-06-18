# morphattention
Explainability analysis of ViT attention maps on fish morphological features.

docker build -t morphattention-env .

docker image (to check if it is there)

morph

## Project structure

| Folder | Purpose |
|---|---|
| `data/` | Raw, processed and annotated fish images (not tracked by git) |
| `src/` | All source code: models, XAI methods, and utilities |
| `notebooks/` | Exploratory analysis and result walkthroughs |
| `experiments/` | Configs and outputs for each experimental run |
| `figures/` | Final plots exported for the thesis |
| `tests/` | Sanity checks for core evaluation logic |