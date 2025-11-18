#!/bin/bash
# ===============================================================================
# STEP 0: AUTHENTICATE WITH GOOGLE CLOUD
# ===============================================================================
#
# This script authenticates your local machine with Google Cloud Platform (GCP)
# to enable access to services like Cloud Storage and Vertex AI.
#
# PREREQUISITES:
# - Google Cloud SDK (gcloud) must be installed
#   Install from: https://cloud.google.com/sdk/docs/install
# - You must have a Google Cloud project set up
# - You must have appropriate permissions on the project
#
# USAGE:
#   bash scripts/00_auth.sh
#
# This command will:
# - Open a browser window for you to sign in with your Google account
# - Create application default credentials that other scripts will use
# - Store credentials locally at: ~/.config/gcloud/application_default_credentials.json
#
# You only need to run this once per machine, or whenever:
# - You switch to a different Google Cloud project
# - Your credentials expire
# - You're setting up a new development environment
#
# NEXT STEP:
# After authentication succeeds, run: Rscript scripts/01_load_image.R
#
# ===============================================================================

echo "Authenticating with Google Cloud..."
gcloud auth application-default login
