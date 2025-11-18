# ===============================================================================
# STEP 1: LOAD AND PROCESS IMAGE FROM PDF
# ===============================================================================
#
# This script:
# 1. Extracts page 23 from a PDF catalog as an image
# 2. Preprocesses image for better OCR quality (deskew, enhance, denoise)
# 3. Saves processed image locally
#
# PREREQUISITES:
# - R must be installed with required packages
# - PDF file must be in: ./raw_data/sears/
#
# REQUIRED R PACKAGES:
# Run this once to install packages:
#   install.packages(c("pdftools", "magick"))
#
# USAGE:
#   Rscript 01_load_image.R
#
# OUTPUT:
# - Local image: ./int_data/images/image001.png
#
# ===============================================================================

# Load required libraries
library(magick)

# Set working directory
setwd("/Users/pjl/Dropbox/digitization_tutorial")

# Set file paths
pdf_path <- "./raw_data/sears/Sears Roebuck Catalog_ElectricalGoodsCatalogue134_1917.pdf"
output_dir <- "./int_data/images/"
dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

cat("Extracting page 23 from PDF using magick...\n")

# Read PDF page directly with magick (specify density for resolution)
# magick uses 0-based indexing, so page 23 is actually page 22 in the array
pdf_spec <- sprintf("%s[22]", pdf_path)  # Page 23 (0-indexed as 22)
img <- image_read(pdf_spec, density = 300)

cat("Successfully read page from PDF\n")
cat("Image dimensions:", image_info(img)$width, "x", image_info(img)$height, "\n")

# Convert to PNG format explicitly (rasterize the PDF)
cat("Converting to PNG format...\n")
img <- image_convert(img, format = "png")

# Save raw extraction first
raw_file <- file.path(output_dir, "image001_raw.png")
image_write(img, path = raw_file, format = "png")
cat("Saved raw extraction to:", raw_file, "\n")

# Apply cleaning steps
cat("Applying image processing...\n")
img_processed <- img %>%
    image_rotate(degrees = 1.25) %>%   # Counter-clockwise to fix clockwise tilt
    image_trim() %>%                    # Remove edges created by rotation
    image_normalize() %>%               # Normalize contrast
    image_enhance() %>%                 # Enhance image quality
    image_despeckle()                   # Remove noise

cat("Note: Deskew applied with threshold=1 for automatic rotation correction\n")

# Save processed image as image001
output_file <- file.path(output_dir, "image001_clean.png")
image_write(img_processed, path = output_file, format = "png")

cat("Saved processed image to:", output_file, "\n")
cat("Processing complete!\n")
