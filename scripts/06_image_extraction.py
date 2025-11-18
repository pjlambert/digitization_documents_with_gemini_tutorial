#!/usr/bin/env python3
"""
IMAGE EXTRACTION WITH OBJECT DETECTION

This script uses Gemini 2.5 Pro's object detection capabilities to identify 
and extract telephone product images from a catalog page. It uses proper
bounding box coordinates (normalized to 0-1000) and disables thinking mode
for better detection accuracy.

IMPROVEMENTS (v4 - Precision + Deduplication):
- Tighter boxing rules with explicit precision requirements
- Structured output schema for deterministic results
- Temperature=0 for consistent, tight bounding boxes
- NO rounding until final pixel conversion for maximum precision
- Two-pass refinement: coarse detection → zoom & rebox for tight alignment
- IoU-based deduplication to merge overlapping detections

PREREQUISITES:
- Must have run: bash 00_auth.sh (GCP authentication)
- Image must exist at: int_data/images/image001_clean.png
- PIL/Pillow installed for image manipulation

REQUIRED PYTHON PACKAGES:
  pip install google-genai Pillow

USAGE:
  python 06_image_extraction.py

OUTPUT:
- Individual PNG files for each extracted image: int_data/segmentation_files/image_*.png
- JSON metadata file: int_data/image_extraction_metadata.json
"""

import os
import json
import time
from datetime import datetime
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
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
SEGMENTATION_DIR = "./int_data/segmentation_files"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(SEGMENTATION_DIR, exist_ok=True)

# Initialize Gen AI client for Vertex AI
os.environ['GOOGLE_CLOUD_PROJECT'] = PROJECT_ID
os.environ['GOOGLE_CLOUD_LOCATION'] = LOCATION
os.environ['GOOGLE_GENAI_USE_VERTEXAI'] = 'True'

client = genai.Client(http_options=HttpOptions(api_version="v1"))

# Helper functions for two-pass refinement and deduplication
def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union (IoU) between two boxes.
    
    Args:
        box1, box2: tuples of (ymin, xmin, ymax, xmax)
    
    Returns:
        IoU value between 0 and 1
    """
    y1min, x1min, y1max, x1max = box1
    y2min, x2min, y2max, x2max = box2
    
    # Calculate intersection
    inter_ymin = max(y1min, y2min)
    inter_xmin = max(x1min, x2min)
    inter_ymax = min(y1max, y2max)
    inter_xmax = min(x1max, x2max)
    
    if inter_ymin >= inter_ymax or inter_xmin >= inter_xmax:
        return 0.0
    
    inter_area = (inter_ymax - inter_ymin) * (inter_xmax - inter_xmin)
    
    # Calculate union
    box1_area = (y1max - y1min) * (x1max - x1min)
    box2_area = (y2max - y2min) * (x2max - x2min)
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0.0

def deduplicate_boxes(detections, iou_threshold=0.5):
    """
    Remove duplicate/overlapping detections using Non-Maximum Suppression.
    
    Args:
        detections: list of detection dicts with 'box_2d', 'score', 'label'
        iou_threshold: IoU threshold for considering boxes as duplicates
    
    Returns:
        list of deduplicated detections
    """
    if len(detections) <= 1:
        return detections
    
    # Sort by confidence score (highest first)
    sorted_detections = sorted(detections, key=lambda x: x.get('score', 0.0), reverse=True)
    
    keep = []
    suppressed = [False] * len(sorted_detections)
    
    for i, det1 in enumerate(sorted_detections):
        if suppressed[i]:
            continue
        
        keep.append(det1)
        box1 = tuple(det1['box_2d'])
        
        # Suppress overlapping lower-confidence boxes
        for j in range(i + 1, len(sorted_detections)):
            if suppressed[j]:
                continue
            
            box2 = tuple(sorted_detections[j]['box_2d'])
            iou = calculate_iou(box1, box2)
            
            if iou > iou_threshold:
                suppressed[j] = True
    
    return keep

def expand_box(b, w, h, margin=0.10):
    """
    Expand a bounding box by a margin percentage to provide context for refinement.
    Keep float precision throughout.
    
    Args:
        b: tuple of (ymin, xmin, ymax, xmax) as floats
        w: image width
        h: image height
        margin: expansion margin as fraction of box size (default 0.10 = 10%)
    
    Returns:
        tuple of (ymin, xmin, ymax, xmax) expanded and clipped to image bounds
    """
    ymin, xmin, ymax, xmax = b
    box_h, box_w = ymax - ymin, xmax - xmin
    dy, dx = box_h * margin, box_w * margin
    return (
        max(0.0, ymin - dy),
        max(0.0, xmin - dx),
        min(float(h), ymax + dy),
        min(float(w), xmax + dx),
    )

def refine_box(crop_img):
    """
    Refine a bounding box by running a second model call on a cropped region.
    This zoom-and-rebox approach helps get tighter alignment on crowded pages.
    
    Args:
        crop_img: PIL Image of the cropped region
    
    Returns:
        list of [ymin, xmin, ymax, xmax] normalized to 0-1000
    """
    refine_prompt = """Return ONE tight bounding box for the telephone device in this crop.
Exclude text/background. Box tight to outermost device pixels.
Format: [ymin, xmin, ymax, xmax] normalized to 0–1000 with decimal precision.
Respond with JSON only: {"box_2d":[...]}"""
    
    # Define simple schema for single box refinement
    refine_schema = types.Schema(
        type=types.Type.OBJECT,
        properties={
            "box_2d": types.Schema(
                type=types.Type.ARRAY,
                items=types.Schema(type=types.Type.NUMBER),
                min_items=4, max_items=4,
                description="[ymin, xmin, ymax, xmax] normalized to 0–1000"
            )
        },
        required=["box_2d"]
    )
    
    refine_resp = client.models.generate_content(
        model=MODEL_NAME,
        contents=[crop_img, refine_prompt],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=refine_schema,
            temperature=0
        )
    )
    rb = json.loads(refine_resp.text)["box_2d"]
    return [float(v) for v in rb]

print("\n" + "="*70)
print("IMAGE EXTRACTION WITH OBJECT DETECTION (v4)")
print("="*70)
print(f"Model: {MODEL_NAME}")
print(f"Image: {IMAGE_PATH}")
print(f"Project: {PROJECT_ID}")
print(f"Location: {LOCATION}")
print(f"Improvements: Precision + deduplication + two-pass refinement")
print()

# Define the improved prompt for object detection with tighter boxing rules
prompt = """Detect every standalone TELEPHONE HARDWARE product on this page.
Return only the device hardware (exclude people, scenes, captions, prices, borders).

CRITICAL: Detect each device ONCE only. Do not create multiple boxes for the same device.

Boxing rules (critical):
- Box is tight to the outermost pixels of the device silhouette. Do not include labels,
  shadows, page borders, or surrounding text. If uncertain, err slightly tighter.
- Coordinate system origin is the top-left corner of the page.
- Format: [ymin, xmin, ymax, xmax] normalized to 0–1000 with at least 1 decimal place.
- ymax > ymin and xmax > xmin. Return numbers only (no units).
- Add a short label for the device; add a confidence score in [0,1].

Output JSON array only, example:
[
  {"label":"Wall-mounted telephone","score":0.94,"box_2d":[123.4, 567.8, 245.6, 678.9]},
  ...
]"""

# Check if image exists
if not os.path.exists(IMAGE_PATH):
    print(f"❌ Error: Image not found at {IMAGE_PATH}")
    exit(1)

print(f"✓ Image found: {IMAGE_PATH}")

# Load the image to get dimensions
img = Image.open(IMAGE_PATH)
img_width, img_height = img.size
print(f"✓ Image dimensions: {img_width}x{img_height} pixels")

print(f"\n⏳ Pass 1: Detecting telephone devices (coarse boxes)...")

start_time = time.time()

try:
    # Define structured output schema for deterministic results
    schema = types.Schema(
        type=types.Type.ARRAY,
        items=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "label": types.Schema(type=types.Type.STRING),
                "score": types.Schema(type=types.Type.NUMBER),
                "box_2d": types.Schema(
                    type=types.Type.ARRAY,
                    items=types.Schema(type=types.Type.NUMBER),
                    min_items=4, max_items=4,
                    description="[ymin, xmin, ymax, xmax] normalized to 0–1000"
                )
            },
            required=["box_2d", "label"]
        )
    )
    
    # Configure for object detection with structured JSON output and temperature=0
    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=schema,
        temperature=0  # Deterministic and usually tighter boxes
    )
    
    # Generate content with object detection
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=[img, prompt],  # PIL images can be passed directly
        config=config
    )
    
    pass1_time = time.time() - start_time
    
    # Parse the JSON response
    detected_objects = json.loads(response.text)
    
    print(f"✓ Pass 1 completed in {pass1_time:.2f}s")
    print(f"✓ Detected {len(detected_objects)} telephone devices")
    
    # Deduplicate overlapping boxes
    print(f"\n⏳ Deduplicating overlapping boxes (IoU > 0.5)...")
    deduplicated = deduplicate_boxes(detected_objects, iou_threshold=0.5)
    
    if len(deduplicated) < len(detected_objects):
        print(f"✓ Removed {len(detected_objects) - len(deduplicated)} duplicate(s)")
        print(f"✓ {len(deduplicated)} unique devices remaining")
        detected_objects = deduplicated
    else:
        print(f"✓ No duplicates found")
    
    # Display initial detection results
    print("\n" + "="*70)
    print("PASS 1: COARSE DETECTIONS (after deduplication)")
    print("="*70)
    
    for i, obj in enumerate(detected_objects, 1):
        label = obj.get('label', 'Unknown')
        score = obj.get('score', 0.0)
        box = obj['box_2d']
        print(f"{i}. {label} (confidence: {score:.3f})")
    
    print("\n" + "="*70)
    print("PASS 2: REFINING BOUNDING BOXES")
    print("="*70)
    print("(Zoom & rebox each detection for tighter alignment)")
    
    # Extract and save each image with two-pass refinement
    extracted_files = []
    pass2_start = time.time()
    
    for i, obj in enumerate(detected_objects, 1):
        label = obj.get('label', 'Unknown')
        score = obj.get('score', 0.0)
        box = obj['box_2d']
        
        print(f"\n🔍 Refining image #{i}: {label}...")
        
        # STEP 1: Convert first-pass normalized coords to pixels (keep as float)
        ymin_n, xmin_n, ymax_n, xmax_n = [float(v) for v in box]
        
        # Keep float precision until final conversion
        ymin_f = ymin_n / 1000.0 * img_height
        xmin_f = xmin_n / 1000.0 * img_width
        ymax_f = ymax_n / 1000.0 * img_height
        xmax_f = xmax_n / 1000.0 * img_width
        
        # Ensure coordinates are within image bounds
        ymin_f = max(0.0, min(ymin_f, float(img_height)))
        xmin_f = max(0.0, min(xmin_f, float(img_width)))
        ymax_f = max(0.0, min(ymax_f, float(img_height)))
        xmax_f = max(0.0, min(xmax_f, float(img_width)))
        
        # Skip invalid boxes
        if ymin_f >= ymax_f or xmin_f >= xmax_f:
            print(f"   ⚠️  Skipping - invalid bounding box")
            continue
        
        # STEP 2: Expand box to provide context for refinement (keep as float)
        eymin_f, exmin_f, eymax_f, exmax_f = expand_box(
            (ymin_f, xmin_f, ymax_f, xmax_f), 
            img_width, img_height, 
            margin=0.10
        )
        
        # Convert to int only for cropping
        eymin, exmin, eymax, exmax = int(eymin_f), int(exmin_f), int(eymax_f), int(exmax_f)
        
        # STEP 3: Crop expanded region
        crop = img.crop((exmin, eymin, exmax, eymax))  # PIL: (left, top, right, bottom)
        crop_width, crop_height = crop.size
        
        print(f"   Crop size: {crop_width}x{crop_height} pixels")
        
        # STEP 4: Ask model for one tight box inside crop (normalized 0-1000)
        try:
            rymin_n, rxmin_n, rymax_n, rxmax_n = refine_box(crop)
            
            # STEP 5: Map refined normalized coords back to full-image pixels (keep as float)
            rymin_f = eymin_f + (rymin_n / 1000.0) * crop_height
            rxmin_f = exmin_f + (rxmin_n / 1000.0) * crop_width
            rymax_f = eymin_f + (rymax_n / 1000.0) * crop_height
            rxmax_f = exmin_f + (rxmax_n / 1000.0) * crop_width
            
            # Use refined coordinates
            ymin_f, xmin_f, ymax_f, xmax_f = rymin_f, rxmin_f, rymax_f, rxmax_f
            
            # Ensure refined coordinates are within image bounds
            ymin_f = max(0.0, min(ymin_f, float(img_height)))
            xmin_f = max(0.0, min(xmin_f, float(img_width)))
            ymax_f = max(0.0, min(ymax_f, float(img_height)))
            xmax_f = max(0.0, min(xmax_f, float(img_width)))
            
            # Skip if refinement produced invalid box
            if ymin_f >= ymax_f or xmin_f >= xmax_f:
                print(f"   ⚠️  Refinement produced invalid box, using original")
                # Fall back to original coordinates from pass 1
                ymin_f = ymin_n / 1000.0 * img_height
                xmin_f = xmin_n / 1000.0 * img_width
                ymax_f = ymax_n / 1000.0 * img_height
                xmax_f = xmax_n / 1000.0 * img_width
            else:
                print(f"   ✓ Refined successfully")
                
        except Exception as e:
            print(f"   ⚠️  Refinement failed: {str(e)}, using original box")
            # Fall back to original coordinates from pass 1
            ymin_f = ymin_n / 1000.0 * img_height
            xmin_f = xmin_n / 1000.0 * img_width
            ymax_f = ymax_n / 1000.0 * img_height
            xmax_f = xmax_n / 1000.0 * img_width
        
        # Expand final box by 10% for some breathing room
        final_ymin_f, final_xmin_f, final_ymax_f, final_xmax_f = expand_box(
            (ymin_f, xmin_f, ymax_f, xmax_f),
            img_width, img_height,
            margin=0.10  # 10% expansion
        )
        
        # Convert to int only for final cropping
        ymin_int = int(round(final_ymin_f))
        xmin_int = int(round(final_xmin_f))
        ymax_int = int(round(final_ymax_f))
        xmax_int = int(round(final_xmax_f))
        
        # Crop the final image using expanded coordinates
        final_cropped = img.crop((xmin_int, ymin_int, xmax_int, ymax_int))
        
        # Save the cropped image
        output_filename = f"image_{i:03d}.png"
        output_path = os.path.join(SEGMENTATION_DIR, output_filename)
        final_cropped.save(output_path)
        
        print(f"   📁 Saved: {output_filename}")
        print(f"   Size: {xmax_int-xmin_int}x{ymax_int-ymin_int} pixels")
        
        extracted_files.append({
            "image_id": i,
            "filename": output_filename,
            "label": label,
            "confidence_score": score,
            "bounding_box_original_normalized": box,  # Original from pass 1
            "bounding_box_refined_pixels": {
                "xmin": xmin_int,
                "ymin": ymin_int,
                "xmax": xmax_int,
                "ymax": ymax_int
            },
            "dimensions": {
                "width": xmax_int - xmin_int,
                "height": ymax_int - ymin_int
            }
        })
    
    pass2_time = time.time() - pass2_start
    total_time = time.time() - start_time
    
    print("\n" + "="*70)
    print("CREATING ANNOTATED IMAGE")
    print("="*70)
    
    # Create a copy of the image for annotation
    annotated_img = img.copy()
    draw = ImageDraw.Draw(annotated_img)
    
    # Try to load a font, fall back to default if not available
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 40)
    except:
        font = ImageFont.load_default()
    
    # Draw bounding boxes and labels
    colors = ['red', 'blue', 'green', 'yellow', 'cyan', 'magenta', 'orange', 'purple', 'pink']
    
    for i, file_info in enumerate(extracted_files):
        bbox = file_info['bounding_box_refined_pixels']
        label = file_info['label']
        score = file_info['confidence_score']
        
        # Get color for this box
        color = colors[i % len(colors)]
        
        # Draw rectangle
        draw.rectangle(
            [(bbox['xmin'], bbox['ymin']), (bbox['xmax'], bbox['ymax'])],
            outline=color,
            width=8
        )
        
        # Draw label background with confidence score
        label_text = f"{i}. {label} ({score:.2f})"
        
        # Get text bounding box
        try:
            bbox_text = draw.textbbox((bbox['xmin'], bbox['ymin']-45), label_text, font=font)
            text_width = bbox_text[2] - bbox_text[0]
            text_height = bbox_text[3] - bbox_text[1]
        except:
            # Fallback for older PIL versions
            text_width, text_height = draw.textsize(label_text, font=font)
        
        # Draw background rectangle for text
        draw.rectangle(
            [(bbox['xmin'], bbox['ymin']-50), (bbox['xmin']+text_width+10, bbox['ymin'])],
            fill=color
        )
        
        # Draw label text
        draw.text(
            (bbox['xmin']+5, bbox['ymin']-45),
            label_text,
            fill='white',
            font=font
        )
    
    # Save annotated image
    annotated_path = os.path.join(SEGMENTATION_DIR, "annotated_image.png")
    annotated_img.save(annotated_path)
    print(f"\n✓ Annotated image saved: {annotated_path}")
    
    # Save metadata
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "model": MODEL_NAME,
        "source_image": IMAGE_PATH,
        "source_dimensions": {
            "width": img_width,
            "height": img_height
        },
        "improvements": {
            "version": "v4 - Precision + Deduplication",
            "pass1": "Coarse detection on full page with tight box prompting",
            "deduplication": "IoU-based NMS to remove overlapping boxes (threshold=0.5)",
            "pass2": "Zoom & rebox on expanded crop for precise alignment",
            "tighter_boxes": "Explicit tight boxing rules with decimal precision",
            "structured_output": "Schema-enforced JSON with temperature=0",
            "coordinate_precision": "Float precision maintained until final pixel conversion",
            "expansion_margin": "10% for refinement context"
        },
        "coordinate_system": "Gemini object detection: [ymin, xmin, ymax, xmax] normalized to 0-1000",
        "thinking_disabled": True,
        "total_images_extracted": len(extracted_files),
        "processing_time": {
            "pass1_detection_seconds": pass1_time,
            "pass2_refinement_seconds": pass2_time,
            "total_seconds": total_time
        },
        "extracted_images": extracted_files,
        "project_id": PROJECT_ID,
        "location": LOCATION
    }
    
    metadata_path = os.path.join(OUTPUT_DIR, "image_extraction_metadata.json")
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Metadata saved: {metadata_path}")
    
    # Summary
    print("\n" + "="*70)
    print("EXTRACTION SUMMARY")
    print("="*70)
    print(f"Total telephone devices extracted: {len(extracted_files)}")
    print(f"Saved to directory: {SEGMENTATION_DIR}")
    print(f"Pass 1 (coarse detection): {pass1_time:.2f}s")
    print(f"Pass 2 (refinement): {pass2_time:.2f}s")
    print(f"Total processing time: {total_time:.2f}s")
    print(f"Method: Two-pass zoom & rebox with deduplication")
    
except Exception as e:
    elapsed_time = time.time() - start_time
    print(f"\n❌ Error after {elapsed_time:.2f}s: {str(e)}")
    import traceback
    traceback.print_exc()
    
    # Save error metadata
    error_data = {
        "timestamp": datetime.now().isoformat(),
        "model": MODEL_NAME,
        "source_image": IMAGE_PATH,
        "status": "error",
        "error_message": str(e),
        "processing_time_seconds": elapsed_time
    }
    
    metadata_path = os.path.join(OUTPUT_DIR, "image_extraction_metadata.json")
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(error_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Error details saved to: {metadata_path}")
    exit(1)

print("\n" + "="*70)
print("IMAGE EXTRACTION COMPLETE (v4)")
print("="*70)
