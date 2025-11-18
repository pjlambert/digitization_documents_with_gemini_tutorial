# digitization-tutorial

Reproducible digitization workflow (R/Python scripts + manuscript). This repo demonstrates a pipeline to preprocess images/PDFs, run binary and structured OCR/transcription, and extract images and structured data, together with a LaTeX manuscript describing the approach.

## Repository structure

- `scripts/` — Workflow scripts:
  - `00_auth.sh` — Authenticate with Google Cloud (gcloud ADC)
  - `01_load_light_preprocess_image.R` — Ingest + light preprocessing (R)
  - `02_binary_classify.py` — Binary classification (Python)
  - `03_pure_transcription.py` — OCR/transcription (Python)
  - `04_struct_transcription.py` — Structured transcription (Python)
  - `05_struct_transcription_and_classify.py` — Combined pipeline (Python)
  - `06_image_extraction.py` — Image extraction (Python)
- `raw_data/` — Source PDFs and inputs (tracked with Git LFS)
- `int_data/` — Intermediate artifacts (images, JSON, CSV). `int_data/output/` contains derived CSVs (ignored by Git by default).
- `manuscript/` — LaTeX slides/manuscript. Build artifacts are ignored; compiled PDF is ignored by default.

## Prerequisites
- Git, Git LFS: https://git-scm.com/ and https://git-lfs.com/
- GitHub CLI (optional but recommended): https://cli.github.com/
- Google Cloud SDK (for scripts/00_auth.sh): https://cloud.google.com/sdk/docs/install
- R (and packages used by `01_load_light_preprocess_image.R`)
- Python 3.x and packages used by the Python scripts

## Quickstart
1) Authenticate with Google Cloud (if using the GCP features):
   ```bash
   bash scripts/00_auth.sh
   ```
2) Ensure Git LFS is installed locally:
   ```bash
   git lfs install
   ```
3) Run the pipeline components as needed, for example:
   ```bash
   Rscript scripts/01_load_light_preprocess_image.R
   python scripts/02_binary_classify.py
   python scripts/03_pure_transcription.py
   python scripts/04_struct_transcription.py
   python scripts/05_struct_transcription_and_classify.py
   python scripts/06_image_extraction.py
   ```

## Notes on data and LFS
- Large binaries (`*.pdf`, `*.png`, `*.jpg`, `*.jpeg`, `*.tif`, `*.tiff`) are tracked via Git LFS.
- Ensure you have the right to redistribute any files placed in `raw_data/`. If redistribution is restricted, consider excluding such files and adding instructions to acquire them.

## License
MIT — see `LICENSE`.
