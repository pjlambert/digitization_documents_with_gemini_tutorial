#!/usr/bin/env python3
"""
STRUCTURED TRANSCRIPTION: Extract products with structured fields

This script processes a single image using Gemini Pro 2.5 to extract
product information in a structured format with separate fields for
product name, description, and price.

PREREQUISITES:
- Must have run: bash 00_auth.sh (GCP authentication)
- Image must exist at: int_data/images/image001_clean.png

REQUIRED PYTHON PACKAGES:
  pip install google-genai pydantic

USAGE:
  python 04_struct_transcription.py

OUTPUT:
- Console output with structured product data
- JSON file: int_data/structured_transcription_result.json
- CSV file: int_data/output/structured_transcription.csv
"""

import os
import json
import time
import csv
from datetime import datetime
from pathlib import Path
from typing import Optional, List
from pydantic import BaseModel, Field
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
print("STRUCTURED TRANSCRIPTION: PRODUCT EXTRACTION")
print("="*70)
print(f"Model: {MODEL_NAME}")
print(f"Image: {IMAGE_PATH}")
print(f"Project: {PROJECT_ID}")
print(f"Location: {LOCATION}")
print()

# Define structured output schema using Pydantic
class Product(BaseModel):
    """A single product with structured fields"""
    product_name: str = Field(
        description="The product name or title, extracted verbatim from the page. This should be the main heading or title for this product."
    )
    description: str = Field(
        description="The full product description, extracted verbatim from the page. Include all descriptive text about this product, maintaining original wording."
    )
    price: Optional[str] = Field(
        default=None,
        description="The price as shown on the page, in exact format (e.g., '$10.60', '$3.75 a Pair', '60c'). If multiple prices, include all. If no price is visible, leave as null."
    )

class ProductList(BaseModel):
    """List of all products on the page"""
    products: List[Product] = Field(
        description="All products found on this page. Extract each distinct product separately."
    )

# Define the extraction prompt
prompt = """Extract all products from this image in a structured format.

For each product you identify:
1. **Product Name**: Extract the main heading or title verbatim (e.g., "Hand Telephone for Two-Station Line", "Wall Telephone for Two-Station Line")
2. **Description**: Extract the complete product description verbatim - all the text that describes this product, including features, specifications, and any additional details
3. **Price**: Extract the price exactly as shown (e.g., "$10.60", "$6.35", "$3.75 a Pair")

IMPORTANT RULES:
- Extract ALL text VERBATIM - do not paraphrase, summarize, or modify the original text
- Each distinct product should be a separate entry
- If a product has multiple prices (e.g., different configurations), include all prices in the price field
- Maintain original spelling, punctuation, and formatting from the catalog
- Include item numbers if they appear in the text

Extract every product you can identify on this page."""

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
    
    # Generate content with structured output
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
        ],
        config={
            "response_mime_type": "application/json",
            "response_schema": ProductList,
        }
    )
    
    elapsed_time = time.time() - start_time
    
    # Get the parsed structured output
    product_list: ProductList = response.parsed
    
    # Display results
    print(f"\n✓ Processing completed in {elapsed_time:.2f}s")
    print(f"\n✓ Extracted {len(product_list.products)} products")
    
    print("\n" + "="*70)
    print("STRUCTURED PRODUCTS")
    print("="*70)
    
    for i, product in enumerate(product_list.products, 1):
        print(f"\n{'='*70}")
        print(f"PRODUCT #{i}")
        print(f"{'='*70}")
        print(f"\nName: {product.product_name}")
        print(f"\nDescription:\n{product.description}")
        if product.price:
            print(f"\nPrice: {product.price}")
        else:
            print(f"\nPrice: [Not specified]")
    
    print("\n" + "="*70)
    
    # Prepare output data for JSON
    result_data = {
        "timestamp": datetime.now().isoformat(),
        "model": MODEL_NAME,
        "image_path": IMAGE_PATH,
        "product_count": len(product_list.products),
        "products": [p.model_dump() for p in product_list.products],
        "processing_time_seconds": elapsed_time,
        "project_id": PROJECT_ID,
        "location": LOCATION
    }
    
    # Save to JSON file
    output_path = os.path.join(OUTPUT_DIR, "structured_transcription_result.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Results saved to: {output_path}")
    
    # Save to CSV file
    csv_path = os.path.join(OUTPUT_CSV_DIR, "structured_transcription.csv")
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['product_number', 'product_name', 'description', 'price', 'timestamp', 'model'])
        
        for i, product in enumerate(product_list.products, 1):
            writer.writerow([
                i,
                product.product_name,
                product.description,
                product.price if product.price else "",
                result_data['timestamp'],
                result_data['model']
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
    
    output_path = os.path.join(OUTPUT_DIR, "structured_transcription_result.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(error_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Error details saved to: {output_path}")
    exit(1)

print("\n" + "="*70)
print("STRUCTURED TRANSCRIPTION COMPLETE")
print("="*70)
