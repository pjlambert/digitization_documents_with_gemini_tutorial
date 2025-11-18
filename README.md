# Tutorial for Digitization with Gemini

Reproducible digitization workflow (R/Python scripts + slides). This repo demonstrates a pipeline to preprocess images/PDFs, run binary and structured OCR/transcription, and extract images and structured data, together with LaTeX slides describing the approach.

Affiliation: University of Warwick’s CAGE (Competitive Advantage in the Global Economy) Institute — https://warwick.ac.uk/fac/soc/economics/research/centres/cage/

Download the slides (PDF): [Direct download](https://github.com/pjlambert/digitization_documents_with_gemini_tutorial/raw/main/slides/digitization_tutorial_slides.pdf?download=1) • [View on GitHub](https://github.com/pjlambert/digitization_documents_with_gemini_tutorial/blob/main/slides/digitization_tutorial_slides.pdf)

## Repository structure

- `scripts/` — Workflow scripts:
  - `00_auth.sh` — Authenticate with Google Cloud (gcloud ADC)
  - `01_load_light_preprocess_image.R` — Ingest + light preprocessing (R)
  - `02_binary_classify.py` — Binary classification (Python)
  - `03_pure_transcription.py` — OCR/transcription (Python)
  - `04_struct_transcription.py` — Structured transcription (Python)
  - `05_struct_transcription_and_classify.py` — Combined pipeline (Python)
  - `06_image_extraction.py` — Image extraction (Python)

- `int_data/` — Intermediate artifacts (images, JSON, CSV). `int_data/output/` contains derived CSVs (ignored by Git by default).
- `slides/` — LaTeX slides. Built PDF is committed: `slides/digitization_tutorial_slides.pdf` (tracked with Git LFS). Other LaTeX build artifacts remain ignored.



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


## License
MIT — see `LICENSE`.
