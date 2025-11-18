# Tutorial for Digitization with Gemini

Reproducible digitization workflow (R/Python scripts + slides). This repo demonstrates a pipeline to preprocess images/PDFs, run binary and structured OCR/transcription, and extract images and structured data, together with LaTeX slides describing the approach.

Affiliation: University of Warwick’s [CAGE](https://warwick.ac.uk/fac/soc/economics/research/centres/cage/)

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

### Google Cloud setup (Vertex AI + Gemini)
1) Create a Google Cloud account and project
   - Sign up: https://cloud.google.com/
   - Create a new project in the Console: https://console.cloud.google.com/projectcreate
   - Make sure billing is enabled for the project
2) Enable the Vertex AI API
   - In the Console, go to APIs & Services → Enable APIs and Services
   - Search for and enable “Vertex AI API”
   - Console link: https://console.cloud.google.com/vertex-ai
3) Choose a region and set environment variables
   - Recommended region: us-central1 (wide model availability)
   - Export vars (these are used by scripts and ADC):
     ```bash
     export GOOGLE_CLOUD_PROJECT="your-project-id"
     export GOOGLE_CLOUD_LOCATION="us-central1"
     ```
4) Install Google Cloud SDK and authenticate
   - Install gcloud: https://cloud.google.com/sdk/docs/install
   - Initialize: `gcloud init`
   - Application Default Credentials (ADC) for local development:
     ```bash
     gcloud auth application-default login
     ```
   - Or simply run the helper script from this repo:
     ```bash
     bash scripts/00_auth.sh
     ```
5) (If needed) Permissions
   - Your user may need the "Vertex AI User" role on the project to call Vertex AI
   - Grant via IAM: https://console.cloud.google.com/iam-admin/iam

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
