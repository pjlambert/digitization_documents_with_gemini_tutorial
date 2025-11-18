#!/usr/bin/env python3
"""
PURE TRANSCRIPTION: Extract all text from an image

This script processes a single image using Gemini Pro 2.5 to perform
pure text transcription - extracting all visible text as one continuous block.

PREREQUISITES:
- Must have run: bash 00_auth.sh (GCP authentication)
- Image must exist at: int_data/images/image001_clean.png

REQUIRED PYTHON PACKAGES:
  pip install google-genai

USAGE:
  python 03_pure_transcription.py

OUTPUT:
- Console output with transcribed text
- JSON file with detailed response: int_data/pure_transcription_result.json
- CSV file: int_data/output/pure_transcription.csv
"""

import os
import json
import time
import csv
from datetime import datetime
from pathlib import Path
from google import genai
from google.genai import types
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
print("PURE TRANSCRIPTION: TEXT EXTRACTION")
print("="*70)
print(f"Model: {MODEL_NAME}")
print(f"Image: {IMAGE_PATH}")
print(f"Project: {PROJECT_ID}")
print(f"Location: {LOCATION}")
print()

# Define the transcription prompt
prompt = """Please transcribe all visible text from this image.

Extract every piece of text you can see, including:
- All headings and titles
- Product descriptions
- Prices and measurements
- Item numbers
- Any small text or fine print
- Headers and footers
- Page numbers
- Company names or branding

Output the text as one continuous block, preserving the reading order as much as possible.
Include all text exactly as it appears, maintaining original spelling, punctuation, and formatting."""

# Check if image exists
if not os.path.exists(IMAGE_PATH):
    print(f"❌ Error: Image not found at {IMAGE_PATH}")
    exit(1)

print(f"✓ Image found: {IMAGE_PATH}")
print(f"\n⏳ Processing image with Gemini Pro 2.5...")

start_time = time.time()

try:
    # Read image file
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
    transcribed_text = response.text
    
    # Display results
    print(f"\n✓ Processing completed in {elapsed_time:.2f}s")
    print("\n" + "="*70)
    print("TRANSCRIBED TEXT")
    print("="*70)
    print(f"\n{transcribed_text}\n")
    print("="*70)
    
    # Prepare output data
    result_data = {
        "timestamp": datetime.now().isoformat(),
        "model": MODEL_NAME,
        "image_path": IMAGE_PATH,
        "transcribed_text": transcribed_text,
        "text_length": len(transcribed_text),
        "processing_time_seconds": elapsed_time,
        "project_id": PROJECT_ID,
        "location": LOCATION
    }
    
    # Save to JSON file
    output_path = os.path.join(OUTPUT_DIR, "pure_transcription_result.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Results saved to: {output_path}")
    
    # Save to CSV file
    csv_path = os.path.join(OUTPUT_CSV_DIR, "pure_transcription.csv")
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['timestamp', 'model', 'image_path', 'transcribed_text', 'text_length', 'processing_time_seconds'])
        writer.writerow([
            result_data['timestamp'],
            result_data['model'],
            result_data['image_path'],
            result_data['transcribed_text'],
            result_data['text_length'],
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
    
    output_path = os.path.join(OUTPUT_DIR, "pure_transcription_result.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(error_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Error details saved to: {output_path}")
    exit(1)

print("\n" + "="*70)
print("TRANSCRIPTION COMPLETE")
print("="*70)
