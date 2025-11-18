#!/usr/bin/env python3
"""
BINARY CLASSIFICATION: Does the image contain phone prices?

This script processes a single image using Gemini Pro 2.5 to perform
binary classification - determining whether the image contains prices of phones.

PREREQUISITES:
- Must have run: bash 00_auth.sh (GCP authentication)
- Image must exist at: int_data/images/image001_clean.png

REQUIRED PYTHON PACKAGES:
  pip install google-genai

USAGE:
  python 02_binary_classify.py

OUTPUT:
- Console output with classification result (Yes/No)
- JSON file with detailed response: int_data/binary_classification_result.json
"""

import os
import json
import time
from datetime import datetime
from pathlib import Path
from google import genai
from google.genai.types import HttpOptions

# Set the working directory
os.chdir('/Users/pjl/Dropbox/digitization_tutorial')

# Configuration
PROJECT_ID = "applied-economics-ai"
LOCATION = "us-central1"
MODEL_NAME = "gemini-2.5-pro"

# Image path
IMAGE_PATH = "int_data/images/image001_clean.png"

# Output directory
OUTPUT_DIR = "./int_data"
OUTPUT_CSV_DIR = "./int_data/output"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_CSV_DIR, exist_ok=True)

# Initialize Gen AI client for Vertex AI
os.environ['GOOGLE_CLOUD_PROJECT'] = PROJECT_ID
os.environ['GOOGLE_CLOUD_LOCATION'] = LOCATION
os.environ['GOOGLE_GENAI_USE_VERTEXAI'] = 'True'

client = genai.Client(http_options=HttpOptions(api_version="v1"))

print("\n" + "="*70)
print("BINARY CLASSIFICATION: PHONE PRICES DETECTION")
print("="*70)
print(f"Model: {MODEL_NAME}")
print(f"Image: {IMAGE_PATH}")
print(f"Project: {PROJECT_ID}")
print(f"Location: {LOCATION}")
print()

# Define the classification prompt
prompt = """Analyze this image and perform a binary classification task.

Question: Does this image contain prices of phones (mobile phones, cell phones, telephones)?

Look for:
- Any images or illustrations of phones/telephones
- Price information associated with those phones
- Price tags, cost listings, or monetary values near phone products

Answer with a clear YES or NO, followed by a brief explanation.

Format your response as:
CLASSIFICATION: [YES or NO]
EXPLANATION: [Brief explanation of what you see and why you classified it this way]
"""

# Check if image exists
if not os.path.exists(IMAGE_PATH):
    print(f"❌ Error: Image not found at {IMAGE_PATH}")
    exit(1)

print(f"✓ Image found: {IMAGE_PATH}")
print(f"\n⏳ Processing image with Gemini Pro 2.5...")

# Upload the image file
start_time = time.time()

try:
    # Read image file
    from google.genai import types
    with open(IMAGE_PATH, 'rb') as f:
        image_bytes = f.read()
    print(f"✓ Image loaded")
    
    # Generate content with the image using Part
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=[
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(text=prompt),
                    types.Part.from_bytes(data=image_bytes, mime_type="image/png")
                ]
            )
        ]
    )
    
    elapsed_time = time.time() - start_time
    
    # Extract response text
    response_text = response.text
    
    # Parse classification result
    classification = "UNKNOWN"
    if "CLASSIFICATION:" in response_text:
        for line in response_text.split('\n'):
            if line.strip().startswith("CLASSIFICATION:"):
                classification = line.split(":", 1)[1].strip().upper()
                if "YES" in classification:
                    classification = "YES"
                elif "NO" in classification:
                    classification = "NO"
                break
    
    # Display results
    print(f"\n✓ Processing completed in {elapsed_time:.2f}s")
    print("\n" + "="*70)
    print("CLASSIFICATION RESULT")
    print("="*70)
    print(f"\n{response_text}\n")
    print("="*70)
    
    # Prepare output data
    result_data = {
        "timestamp": datetime.now().isoformat(),
        "model": MODEL_NAME,
        "image_path": IMAGE_PATH,
        "classification": classification,
        "full_response": response_text,
        "processing_time_seconds": elapsed_time,
        "project_id": PROJECT_ID,
        "location": LOCATION
    }
    
    # Save to JSON file
    output_path = os.path.join(OUTPUT_DIR, "binary_classification_result.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Results saved to: {output_path}")
    
    # Save to CSV file
    import csv
    csv_path = os.path.join(OUTPUT_CSV_DIR, "binary_classification.csv")
    
    # Extract explanation from response
    explanation = ""
    if "EXPLANATION:" in response_text:
        for line in response_text.split('\n'):
            if line.strip().startswith("EXPLANATION:"):
                explanation = line.split(":", 1)[1].strip()
                break
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['timestamp', 'model', 'image_path', 'classification', 'explanation', 'processing_time_seconds'])
        writer.writerow([
            result_data['timestamp'],
            result_data['model'],
            result_data['image_path'],
            result_data['classification'],
            explanation,
            result_data['processing_time_seconds']
        ])
    
    print(f"✓ CSV results saved to: {csv_path}")
    
except Exception as e:
    elapsed_time = time.time() - start_time
    print(f"\n❌ Error after {elapsed_time:.2f}s: {str(e)}")
    
    # Save error to JSON
    error_data = {
        "timestamp": datetime.now().isoformat(),
        "model": MODEL_NAME,
        "image_path": IMAGE_PATH,
        "status": "error",
        "error_message": str(e),
        "processing_time_seconds": elapsed_time
    }
    
    output_path = os.path.join(OUTPUT_DIR, "binary_classification_result.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(error_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Error details saved to: {output_path}")
    exit(1)

print("\n" + "="*70)
print("CLASSIFICATION COMPLETE")
print("="*70)
