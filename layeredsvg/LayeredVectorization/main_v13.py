"""
Main V13 Pipeline: Ollama + SAM3 Text-Prompted Layered Vectorization

Three-phase interactive pipeline:
  Phase 1 (fast): Ollama Qwen2.5 VL detects object keywords
  Phase 2 (fast): SAM3 segments each keyword into mask layers
  Phase 3 (heavy): User confirms layers -> SDS + DiffVG vectorization

Usage:
    # Phase 1: Detect keywords
    keywords = detect_keywords_v13(image_path, progress_cb)

    # User edits keywords + confidence...

    # Phase 2: Segment keywords
    layers = segment_keywords_v13(image_path, keywords_with_conf, output_dir, progress_cb)

    # User reviews layers, picks which to keep...

    # Phase 3: Vectorize confirmed layers
    result = vectorize_confirmed_v13(device, args, output_dir, confirmed_layers, progress_cb)
"""

import torch
import torch.nn.functional as F
from PIL import Image
import argparse
import os
import sys
import gc
import time
import numpy as np
import re
import cv2
import shutil
import base64
import json
import requests

import pydiffvg
import yaml

from transformers import Sam3Model, Sam3Processor, Sam3TrackerModel, Sam3TrackerProcessor
from io import BytesIO
import traceback
from vectorize_layer_v11 import vectorize_single_layer
from scipy import ndimage
import cairosvg

# Resolution for SDXL
V13_RESOLUTION = 1024

# Ollama configuration
# OLLAMA_BASE_URL = "http://localhost:11434"
# OLLAMA_GENERATE_URL = f"{OLLAMA_BASE_URL}/api/generate"
# OLLAMA_MODEL = "qwen2.5vl:7b"

# SAM3 global model holders (loaded on demand, unloaded for VRAM management)
SAM3_MODEL = None
SAM3_PROCESSOR = None
SAM3_TRACKER = None

# ============================================================================
# Utility functions
# ============================================================================

def init_diffvg(device: torch.device,
                use_gpu: bool = torch.cuda.is_available(),
                print_timing: bool = False):
    """Initialize pydiffvg settings."""
    pydiffvg.set_device(device)
    pydiffvg.set_use_gpu(use_gpu)
    pydiffvg.set_print_timing(print_timing)


def load_config(file_path, args):
    """Load configuration from YAML file and set as attributes on args."""
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
        for key, value in config.items():
            setattr(args, key, value)
    return args


def dilate_mask(mask: np.ndarray, dilation_px: int) -> np.ndarray:
    """Dilate a binary mask by a specified number of pixels."""
    if dilation_px <= 0:
        return mask
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * dilation_px + 1, 2 * dilation_px + 1)
    )
    dilated = cv2.dilate(mask, kernel, iterations=1)
    return dilated


def clear_gpu_memory():
    """Clear GPU memory cache."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def image_to_base64_ollama(image):
    """Convert PIL Image to base64 string for Ollama."""
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


# ============================================================================
# Ollama VRAM Management
# ============================================================================

# def unload_ollama_model():
#     """Unload Ollama model from memory to free VRAM."""
#     print(f"  Unloading Ollama model: {OLLAMA_MODEL}...")
#     try:
#         payload = {
#             "model": OLLAMA_MODEL,
#             "prompt": "",
#             "keep_alive": 0
#         }
#         requests.post(OLLAMA_GENERATE_URL, json=payload, timeout=30)
#         time.sleep(3.0)
#         # Second request to be sure
#         try:
#             requests.post(OLLAMA_GENERATE_URL, json=payload, timeout=10)
#         except Exception:
#             pass
#         time.sleep(2.0)
#         clear_gpu_memory()
#         print(f"  Ollama model unloaded")
#         return True
#     except Exception as e:
#         print(f"  Error unloading Ollama model: {e}")
#         return False


# ============================================================================
# SAM3 Model Management
# ============================================================================

def load_sam3_model(device, tracker = False):
    """Load SAM3 model on demand."""
    global SAM3_MODEL, SAM3_PROCESSOR, SAM3_TRACKER
    if SAM3_MODEL is not None and SAM3_TRACKER == tracker:
        return True
    try:
        unload_sam3_model()
        print("  Loading SAM3 model (facebook/sam3)...")
        SAM3_TRACKER = tracker
        if tracker:
            SAM3_PROCESSOR = Sam3TrackerProcessor.from_pretrained("facebook/sam3")
            SAM3_MODEL = Sam3TrackerModel.from_pretrained("facebook/sam3").to(device)
        else:
            SAM3_PROCESSOR = Sam3Processor.from_pretrained("facebook/sam3")
            SAM3_MODEL = Sam3Model.from_pretrained("facebook/sam3").to(device)

        print("  SAM3 model loaded successfully!")
        return True
    except Exception as e:
        traceback.print_exc()
        print(f"  Error loading SAM3 model: {e}")
        return False


def unload_sam3_model():
    """Unload SAM3 model to free VRAM."""
    global SAM3_MODEL, SAM3_PROCESSOR, SAM3_TRACKER
    if SAM3_MODEL is not None:
        print("  Unloading SAM3 model...")
        try:
            SAM3_MODEL = SAM3_MODEL.cpu()
        except Exception:
            pass
        del SAM3_MODEL
        del SAM3_PROCESSOR
        SAM3_MODEL = None
        SAM3_PROCESSOR = None
        SAM3_TRACKER = None
        gc.collect()
        gc.collect()
        clear_gpu_memory()
        print("  SAM3 model unloaded")


# ============================================================================
# PHASE 1: Keyword Detection via Ollama (~30s)
# ============================================================================

def detect_keywords_v13(image_path, progress_cb=None):
    """
    Use Ollama Qwen2.5 VL to detect objects in the image as keywords.

    Args:
        image_path: Path to the input image
        progress_cb: function(progress_pct, message)

    Returns:
        list[str] - list of detected keyword strings
    """
    # print("=" * 60)
    # print("Phase 1: Detecting objects with Ollama Qwen2.5 VL")
    # print("=" * 60)

    # if progress_cb:
    #     progress_cb(5, "Preparing image for object detection...")

    # # Unload SAM3 if loaded (free VRAM for Ollama)
    # if SAM3_MODEL is not None:
    #     unload_sam3_model()

    # # Load and resize image for Ollama
    # image_pil = Image.open(image_path).convert('RGB')
    # # Resize to reasonable size for vision model (max 1024 on longest side)
    # w, h = image_pil.size
    # max_dim = 1024
    # if max(w, h) > max_dim:
    #     scale = max_dim / max(w, h)
    #     image_pil = image_pil.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    # img_b64 = image_to_base64_ollama(image_pil)

    # if progress_cb:
    #     progress_cb(20, "Analyzing image with AI vision...")

    # prompt = (
    #     "List all distinct objects and elements visible in this image. "
    #     "Output ONLY keywords, one per line. Be specific (e.g. 'red flowers' not just 'flowers', "
    #     "'tree branch' not just 'tree'). Do not include background, sky, or abstract concepts. "
    #     "Do not number the items or add explanations."
    # )

    # payload = {
    #     "model": OLLAMA_MODEL,
    #     "prompt": prompt,
    #     "images": [img_b64],
    #     "stream": False,
    #     "keep_alive": "10m"
    # }

    # try:
    #     print(f"  Calling Ollama ({OLLAMA_MODEL})...")
    #     #response = requests.post(OLLAMA_GENERATE_URL, json=payload, timeout=120)
    #     #response.raise_for_status()
    #     result = {
    #         'response': 'sitting girl, shoes, red bandana, green shirt, yellow pants'
    #     }
    #     raw_response = result.get("response", "")
    #     print(f"  Raw response:\n{raw_response}")
    # except requests.exceptions.ConnectionError:
    #     raise RuntimeError(
    #         "Cannot connect to Ollama. Make sure Ollama is running (ollama serve) "
    #         f"and {OLLAMA_MODEL} is pulled (ollama pull {OLLAMA_MODEL})."
    #     )
    # except Exception as e:
    #     raise RuntimeError(f"Ollama error: {str(e)}")

    # if progress_cb:
    #     progress_cb(80, "Parsing keywords...")

    # # Parse response into keyword list
    # keywords = []
    # for line in raw_response.strip().split('\n'):
    #     line = line.strip()
    #     # Remove numbering like "1.", "- ", "* ", etc.
    #     line = re.sub(r'^[\d]+[.\)]\s*', '', line)
    #     line = re.sub(r'^[-*•]\s*', '', line)
    #     line = line.strip()
    #     if line and len(line) > 1 and len(line) < 80:
    #         keywords.append(line.lower())

    # # Deduplicate while preserving order
    # seen = set()
    # unique_keywords = []
    # for kw in keywords:
    #     if kw not in seen:
    #         seen.add(kw)
    #         unique_keywords.append(kw)

    if progress_cb:
        progress_cb(100, f"Skip detection")

    print(f"  Skip detection")
    return ["the object"]

def normalize_box(box, ori_width, ori_height, target_size):
    bbox = [
        round(box[0] * target_size / ori_width),
        round(box[1] * target_size / ori_height),
        round(box[2] * target_size / ori_width),
        round(box[3] * target_size / ori_height),
    ]
    return bbox

def normalize_points(points, ori_width, ori_height, target_size):
    ppoints = []
    for x, y in points:
        ppoints.append([
            round(points[idx][0] * target_size / ori_width),
            round(points[idx][1] * target_size / ori_height),
        ])
    return ppoints

# ============================================================================
# PHASE 2: SAM3 Segmentation (~1-2 min)
# ============================================================================

def segment_keywords_v13(image_path, keywords_with_conf, output_dir, progress_cb=None):
    """
    Segment each keyword using SAM3 text-prompted instance segmentation.

    Args:
        image_path: Path to the input image
        keywords_with_conf: list of {"keyword": str, "confidence": float}
        output_dir: Directory to save masks and previews
        progress_cb: function(progress_pct, message)

    Returns:
        list of layer metadata dicts
    """
    print("=" * 60)
    print("Phase 2: SAM3 Text-Prompted Segmentation")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if progress_cb:
        progress_cb(5, "Preparing segmentation model...")

    # VRAM swap: unload Ollama, load SAM3
    # unload_ollama_model()
    clear_gpu_memory()

    if progress_cb:
        progress_cb(10, "Loading segmentation model...")

    # Load and resize image
    image_pil = Image.open(image_path).convert('RGB')
    orig_w, orig_h = image_pil.size

    # Resize to working resolution
    working_pil = image_pil.resize((V13_RESOLUTION, V13_RESOLUTION), Image.LANCZOS)
    working_np = np.array(working_pil)

    # Ensure output dirs exist
    layers_dir = os.path.join(output_dir, "layers")
    os.makedirs(layers_dir, exist_ok=True)

    # Save working image
    working_img_path = os.path.join(output_dir, "input_resized.png")
    working_pil.save(working_img_path)

    if progress_cb:
        progress_cb(15, "Segmenting keywords...")

    # Segment each keyword
    all_masks = []  # list of (keyword, confidence, mask_np, score)
    n_keywords = len(keywords_with_conf)

    for i, kw_conf in enumerate(keywords_with_conf):
        keyword = kw_conf.get('keyword')
        box = kw_conf.get('box')
        points = kw_conf.get('points')
        confidence = kw_conf.get('confidence', 0.2)

        if keyword == '':
            keyword = None  # reset if empty string

        if box == '':
            box = None  # reset if empty string

        if points == '':
            points = None  # reset if empty string

        if box is not None:
            box = normalize_box(box, orig_w, orig_h, V13_RESOLUTION)

        if points is not None:
            points = normalize_points(points, orig_w, orig_h, V13_RESOLUTION)

        name = keyword if keyword is not None else ('Box' if box is not None else 'Point')

        pct = 15 + int(60 * i / max(n_keywords, 1))
        if progress_cb:
            progress_cb(pct, f"Segmenting '{name}' ({i+1}/{n_keywords})...")

        print(f"  [{i+1}/{n_keywords}] Segmenting '{name}' (confidence={confidence})...")

        try:
            if keyword is not None or (box is None and points is None):
                if not load_sam3_model(device):
                    raise RuntimeError("Failed to load SAM3 model")

                model_inputs = SAM3_PROCESSOR(
                    images=working_pil,
                    text=keyword,
                    input_boxes=[[box]] if box else None,
                    input_boxes_labels=[[1]] if box else None,
                    return_tensors="pt"
                )
                # manually move every tensor to device
                model_inputs = {k: v.to(SAM3_MODEL.device) for k, v in model_inputs.items() if hasattr(v, "to")}

                with torch.no_grad():
                    inference_output = SAM3_MODEL(**model_inputs)

                # Get original_sizes from inputs if available
                target_sizes = None
                if hasattr(model_inputs, 'get') and model_inputs.get("original_sizes") is not None:
                    target_sizes = model_inputs.get("original_sizes").tolist()
                else:
                    target_sizes = [[V13_RESOLUTION, V13_RESOLUTION]]

                processed_results = SAM3_PROCESSOR.post_process_instance_segmentation(
                    inference_output,
                    threshold=confidence,
                    mask_threshold=0.5,
                    target_sizes=target_sizes
                )[0]
                raw_masks = processed_results['masks'].cpu().numpy()
                raw_scores = processed_results['scores'].cpu().numpy().tolist()

            else:
                if not load_sam3_model(device, tracker = True):
                    raise RuntimeError("Failed to load SAM3 model")

                labels = [1] * len(points) if points else None
                model_inputs = SAM3_PROCESSOR(
                    images=working_pil,
                    input_points=[[points]] if points else None,
                    input_labels=[[labels]] if points else None,
                    input_boxes=[[box]] if box else None,
                    return_tensors="pt"
                )
                # manually move every tensor to device
                model_inputs = {k: v.to(SAM3_MODEL.device) for k, v in model_inputs.items() if hasattr(v, "to")}

                with torch.no_grad():
                    inference_output = SAM3_MODEL(**model_inputs)

                # Get original_sizes from inputs if available
                target_sizes = None
                if hasattr(model_inputs, 'get') and model_inputs.get("original_sizes") is not None:
                    target_sizes = model_inputs.get("original_sizes").tolist()
                else:
                    target_sizes = [[V13_RESOLUTION, V13_RESOLUTION]]

                processed_results = {
                    "masks": SAM3_PROCESSOR.post_process_masks(inference_output.pred_masks.cpu(), target_sizes)[0]
                    "scores": inference_output.object_score_logits,
                }
                raw_masks = processed_results['masks'].cpu().numpy()
                raw_scores = [confidence] * raw_masks.shape[0]
                # raw_scores = processed_results['scores'].cpu().numpy().tolist()

            if (raw_masks.ndim == 3 or raw_masks.ndim == 4) and raw_masks.shape[0] > 0:
                # Combine masks into one per keyword, filtering by score
                combined_mask = np.zeros((V13_RESOLUTION, V13_RESOLUTION), dtype=np.uint8)
                best_score = 0.0
                kept = 0
                for m_idx in range(raw_masks.shape[0]):
                    score = raw_scores[m_idx] if m_idx < len(raw_scores) else 0.0
                    # Skip low-score masks to avoid capturing background
                    if score < confidence:
                        continue
                    mask_2d = raw_masks[m_idx] if raw_masks.ndim == 3 else raw_masks[m_idx][0]
                    # Resize if needed
                    if mask_2d.shape != (V13_RESOLUTION, V13_RESOLUTION):
                        mask_2d = np.array(
                            Image.fromarray(mask_2d.astype(np.uint8) * 255).resize(
                                (V13_RESOLUTION, V13_RESOLUTION), Image.NEAREST
                            )
                        ) > 127
                    combined_mask = np.logical_or(combined_mask, mask_2d).astype(np.uint8)
                    best_score = max(best_score, score)
                    kept += 1

                area = int(np.sum(combined_mask))
                if area > 0:
                    all_masks.append({
                        'keyword': keyword,
                        'confidence': confidence,
                        'mask': combined_mask * 255,  # 0 or 255
                        'score': best_score,
                        'area': area,
                    })
                    print(f"    -> {raw_masks.shape[0]} mask(s), kept {kept} (score>={confidence}), area={area} px, score={best_score:.3f}")
                else:
                    print(f"    -> No pixels after filtering (0/{raw_masks.shape[0]} masks passed score>={confidence})")
            else:
                print(f"    -> No masks found for '{name}'")

        except Exception as e:
            print(f"    -> Error segmenting '{name}': {e}")
            traceback.print_exc()

    if progress_cb:
        progress_cb(78, "Creating remaining layer...")

    # Create "remaining" layer for uncaptured pixels
    H, W = V13_RESOLUTION, V13_RESOLUTION
    all_detected = np.zeros((H, W), dtype=bool)
    for item in all_masks:
        all_detected = np.logical_or(all_detected, item['mask'] > 127)

    # Non-black mask (pixels that have actual content)
    brightness = np.max(working_np, axis=2)
    non_black_mask = brightness > 20  # threshold for "has content"

    remaining_mask = np.logical_and(non_black_mask, ~all_detected)
    remaining_area = int(np.sum(remaining_mask))
    remaining_pct = remaining_area / max(np.sum(non_black_mask), 1) * 100

    print(f"  Remaining uncovered area: {remaining_pct:.1f}% of non-black pixels ({remaining_area} px)")

    if remaining_pct > 1.0:
        all_masks.append({
            'keyword': '_remaining',
            'confidence': 0.0,
            'mask': remaining_mask.astype(np.uint8) * 255,
            'score': 0.0,
            'area': remaining_area,
            'is_remaining': True,
        })
        print(f"  Added '_remaining' layer")

    if progress_cb:
        progress_cb(85, "Saving layer previews...")

    # Save layer previews, masks, and metadata
    layers_info = []
    for idx, item in enumerate(all_masks):
        lid = idx
        keyword = item['keyword']
        mask = item['mask']
        mask_bool = mask > 127

        # Preview: image masked with alpha on checkerboard
        preview = working_np.copy()
        preview[~mask_bool] = 0  # transparent areas become black

        # Save RGBA preview (with alpha)
        preview_rgba = np.zeros((H, W, 4), dtype=np.uint8)
        preview_rgba[:, :, :3] = preview
        preview_rgba[:, :, 3] = mask

        preview_path = os.path.join(layers_dir, f"layer_{lid}_preview.png")
        mask_path = os.path.join(layers_dir, f"layer_{lid}_mask.png")

        scale = 300 / max(orig_w, orig_h)
        preview_rgba = cv2.resize(preview_rgba, (round(orig_w * scale), round(orig_h * scale)), interpolation=cv2.INTER_NEAREST)

        Image.fromarray(preview_rgba).save(preview_path)
        Image.fromarray(mask).save(mask_path)

        area_pct = item['area'] / (H * W) * 100

        layers_info.append({
            'layer_id': lid,
            'keyword': keyword,
            'confidence': item['confidence'],
            'area': item['area'],
            'area_pct': round(area_pct, 1),
            'score': item.get('score', 0.0),
            'is_remaining': item.get('is_remaining', False),
            'preview_url': f"layer_{lid}_preview.png",
            'mask_url': f"layer_{lid}_mask.png",
            'preview_path': os.path.abspath(preview_path),
            'mask_path': os.path.abspath(mask_path),
        })

    # Create merge preview (all layers composited)
    if progress_cb:
        progress_cb(92, "Creating merge preview...")

    merge_preview = np.ones((H, W, 3), dtype=np.uint8) * 200  # light gray background
    for item in all_masks:
        mask_bool = item['mask'] > 127
        merge_preview[mask_bool] = working_np[mask_bool]

    merge_preview_path = os.path.join(layers_dir, "merge_preview.png")

    merge_preview_resized = cv2.resize(
        merge_preview,
        (orig_w, orig_h),
        interpolation=cv2.INTER_NEAREST  # IMPORTANT for masks / sharp edges
    )

    Image.fromarray(merge_preview_resized).save(merge_preview_path)

    # Save metadata for Phase 3
    meta = {
        'layers': layers_info,
        'H_orig': orig_h,
        'W_orig': orig_w,
        'resolution': V13_RESOLUTION,
        'image_path': os.path.abspath(image_path),
    }
    meta_path = os.path.join(output_dir, "layers_meta.json")
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)

    if progress_cb:
        progress_cb(100, f"Segmented {len(layers_info)} layers")

    print(f"\n  Phase 2 complete: {len(layers_info)} layers saved")
    for info in layers_info:
        tag = " [remaining]" if info['is_remaining'] else ""
        print(f"    Layer {info['layer_id']}: '{info['keyword']}' "
              f"({info['area_pct']}% area, conf={info['confidence']}){tag}")

    return layers_info


# ============================================================================
# PHASE 3: Vectorize Confirmed Layers (heavy, ~5-40 min)
# ============================================================================

def vectorize_layer_with_v11_pipeline(
    layer_img: np.ndarray,
    layer_mask: np.ndarray,
    device: torch.device,
    args,
    layer_id: int,
    v11_output_name: str,
    run_sds: bool = True
) -> str:
    """Vectorize a single layer using the V11 custom pipeline (app3 quality)."""
    layer_output_dir = f"./workdir/{v11_output_name}_layer_{layer_id}"
    os.makedirs(layer_output_dir, exist_ok=True)

    print(f"    Vectorizing layer {layer_id} with app3 quality pipeline...")
    if not run_sds:
        print(f"    WARNING: SDS disabled - quality will be lower")

    svg_path = vectorize_single_layer(
        layer_img=layer_img,
        layer_mask=layer_mask,
        device=device,
        args=args,
        output_dir=layer_output_dir,
        layer_id=layer_id,
        run_sds=run_sds
    )

    if svg_path and os.path.exists(svg_path):
        print(f"    SVG created: {svg_path}")
    else:
        print(f"    WARNING: SVG not created for layer {layer_id}")

    return svg_path


def merge_layer_svgs(layer_svgs: list, output_path: str, canvas_size: tuple, embed_masks: bool = True) -> str:
    """Merge multiple layer SVGs into a single SVG with grouped layers.
    For v13: layers are ordered by list position (user-defined keyword order)."""
    W, H = canvas_size

    # v13: no depth sorting needed, layers are in keyword order
    # First layer = bottom, last = top (remaining at the bottom)
    sorted_svgs = sorted(layer_svgs, key=lambda x: (
        0 if x.get('is_remaining', False) else 1,  # Remaining at bottom
        x.get('layer_id', 0)
    ))

    print("\n  === SVG Stacking Order (bottom to top) ===")
    for i, layer in enumerate(sorted_svgs):
        kw = layer.get('keyword', f"layer_{layer['layer_id']}")
        remaining = " [REMAINING]" if layer.get('is_remaining', False) else ""
        print(f"  {i}: layer_{layer['layer_id']} '{kw}'{remaining}")
    print()

    svg_parts = [
        '<?xml version="1.0" encoding="utf-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" '
        f'width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
        '  <!-- Each <g> group is a separate layer that can be moved independently -->',
    ]

    mask_ids = set()
    if embed_masks:
        svg_parts.append('  <defs>')
        for layer_info in sorted_svgs:
            layer_id = layer_info['layer_id']
            mask_path = layer_info.get('mask_path')
            if not mask_path or not os.path.exists(mask_path):
                continue
            with open(mask_path, 'rb') as mask_file:
                mask_data = base64.b64encode(mask_file.read()).decode('ascii')
            svg_parts.append(f'    <mask id="mask_layer_{layer_id}" maskUnits="userSpaceOnUse">')
            svg_parts.append(
                f'      <image href="data:image/png;base64,{mask_data}" '
                f'xlink:href="data:image/png;base64,{mask_data}" '
                f'x="0" y="0" width="{W}" height="{H}" preserveAspectRatio="none" />'
            )
            svg_parts.append('    </mask>')
            mask_ids.add(layer_id)
        svg_parts.append('  </defs>')

    for layer_info in sorted_svgs:
        svg_path = layer_info['svg_path']
        layer_id = layer_info['layer_id']
        keyword = layer_info.get('keyword', f'layer_{layer_id}')
        mask_attr = ''
        if embed_masks and layer_id in mask_ids:
            mask_attr = f' mask="url(#mask_layer_{layer_id})"'

        if not os.path.exists(svg_path):
            print(f"  Warning: Layer SVG not found: {svg_path}")
            continue

        with open(svg_path, 'r') as f:
            layer_svg = f.read()

        match = re.search(r'<svg[^>]*>(.*?)</svg>', layer_svg, re.DOTALL)
        if match:
            layer_content = match.group(1)
            safe_id = re.sub(r'[^a-zA-Z0-9_-]', '_', keyword)
            svg_parts.append(f'  <g id="{safe_id}" data-keyword="{keyword}" data-layer-id="{layer_id}"{mask_attr}>')
            svg_parts.append(f'    {layer_content}')
            svg_parts.append(f'  </g>')

    svg_parts.append('</svg>')

    merged_svg = '\n'.join(svg_parts)
    with open(output_path, 'w') as f:
        f.write(merged_svg)

    return output_path


def vectorize_confirmed_v13(device, args, output_dir, confirmed_layers, progress_cb=None):
    """
    Phase 3: Vectorize confirmed layers using SDS + DiffVG pipeline.

    Args:
        device: torch device
        args: config args (loaded from YAML)
        output_dir: working directory (from Phase 2)
        confirmed_layers: list of layer IDs to vectorize
        progress_cb: function(progress_pct, message)

    Returns:
        dict with svg_path, layers, n_layers
    """
    print("=" * 60)
    print(f"Phase 3: Vectorizing {len(confirmed_layers)} confirmed layers")
    print("=" * 60)

    layers_dir = os.path.join(output_dir, "layers")

    # Unload SAM3 to free VRAM for vectorization
    unload_sam3_model()
    clear_gpu_memory()

    # Load metadata from Phase 2
    meta_path = os.path.join(output_dir, "layers_meta.json")
    with open(meta_path, 'r') as f:
        meta = json.load(f)

    H_orig = meta['H_orig']
    W_orig = meta['W_orig']

    # Load working image
    working_img_path = os.path.join(output_dir, "input_resized.png")
    working_img = np.array(Image.open(working_img_path).convert('RGB'))

    if progress_cb:
        progress_cb(5, "Loading confirmed layers...")

    # Reload confirmed layers from disk
    layers = []
    for layer_meta in meta['layers']:
        lid = layer_meta['layer_id']
        if lid not in confirmed_layers:
            continue

        mask = np.array(Image.open(layer_meta['mask_path']).convert('L'))
        mask_bool = mask > 127

        # Create layer image: object on white background
        layer_img = np.ones_like(working_img) * 255
        layer_img[mask_bool] = working_img[mask_bool]

        layers.append({
            'mask': mask,
            'original_mask': mask.copy(),
            'mask_bool': mask_bool,
            'image': layer_img,
            'area': layer_meta['area'],
            'layer_id': lid,
            'keyword': layer_meta['keyword'],
            'is_remaining': layer_meta.get('is_remaining', False),
        })

    if not layers:
        raise ValueError("No layers selected for vectorization")

    # Re-assign layer IDs 0..N-1
    for i, layer in enumerate(layers):
        layer['original_layer_id'] = layer['layer_id']
        layer['layer_id'] = i

    print(f"  Loaded {len(layers)} layers")
    for layer in layers:
        tag = " [REMAINING]" if layer['is_remaining'] else ""
        print(f"    Layer {layer['layer_id']} (orig {layer['original_layer_id']}): "
              f"'{layer['keyword']}' area={layer['area']}{tag}")

    # Apply mask dilation
    mask_dilation_px = getattr(args, 'mask_dilation_px', 3)
    if mask_dilation_px > 0:
        if progress_cb:
            progress_cb(8, "Applying mask dilation...")
        print(f"\n  Applying mask dilation ({mask_dilation_px}px)...")
        for layer in layers:
            layer['mask'] = dilate_mask(layer['mask'], mask_dilation_px)
            layer['mask_bool'] = layer['mask'] > 127
            layer['image'] = np.ones_like(working_img) * 255
            layer['image'][layer['mask_bool']] = working_img[layer['mask_bool']]

    # Gap filling
    H, W = layers[0]['image'].shape[:2]
    coverage_mask = np.zeros((H, W), dtype=np.uint8)
    for layer in layers:
        coverage_mask = np.maximum(coverage_mask, layer['mask'])

    uncovered = coverage_mask == 0
    uncovered_count = int(np.sum(uncovered))

    if uncovered_count > 0:
        if progress_cb:
            progress_cb(10, f"Filling {uncovered_count} gap pixels...")
        print(f"  Filling {uncovered_count} uncovered pixels...")

        min_distances = np.full((H, W), np.inf, dtype=np.float32)
        nearest_layer_idx = np.zeros((H, W), dtype=np.int32)

        for i, layer in enumerate(layers):
            mask_binary = (layer['mask'] > 127).astype(np.uint8)
            dist = ndimage.distance_transform_edt(1 - mask_binary)
            closer = dist < min_distances
            min_distances[closer] = dist[closer]
            nearest_layer_idx[closer] = i

        for i, layer in enumerate(layers):
            pixels_to_add = uncovered & (nearest_layer_idx == i)
            added = int(np.sum(pixels_to_add))
            if added > 0:
                layer['mask'][pixels_to_add] = 255
                layer['image'][pixels_to_add] = working_img[pixels_to_add]
                layer['area'] = int(np.sum(layer['mask'] > 127))
                print(f"    Layer {i}: added {added} gap pixels")

    # Save updated layer images and masks (post-dilation)
    for layer in layers:
        lid = layer['layer_id']
        Image.fromarray(layer['image']).save(
            os.path.join(layers_dir, f"layer_{lid}_preview.png"))
        Image.fromarray(layer['mask']).save(
            os.path.join(layers_dir, f"layer_{lid}_mask.png"))

    # Per-layer SDS + DiffVG vectorization
    print("\n" + "=" * 60)
    print("Vectorizing each layer with app3 quality (SDS + DiffVG)")
    print("=" * 60)

    skip_sds = getattr(args, 'skip_sds', False)
    layer_svgs = []

    for i, layer in enumerate(layers):
        layer_id = layer['layer_id']
        print(f"\n--- Layer {i+1}/{len(layers)} (id={layer_id}, '{layer['keyword']}') ---")

        if progress_cb:
            progress = 15 + int(75 * i / len(layers))
            progress_cb(progress, f"Vectorizing layer {i+1}/{len(layers)} ('{layer['keyword']}')...")

        try:
            svg_path = vectorize_layer_with_v11_pipeline(
                layer['image'],
                layer['mask'],
                device,
                args,
                layer_id,
                args.file_save_name,
                run_sds=not skip_sds
            )

            if svg_path and os.path.exists(svg_path):
                final_layer_svg = os.path.join(layers_dir, f"layer_{layer_id}.svg")
                shutil.copy(svg_path, final_layer_svg)

                layer_svgs.append({
                    'layer_id': layer_id,
                    'svg_path': final_layer_svg,
                    'keyword': layer['keyword'],
                    'area': layer['area'],
                    'is_remaining': layer.get('is_remaining', False),
                    'mask_path': os.path.join(layers_dir, f"layer_{layer_id}_mask.png"),
                })
            else:
                print(f"    Warning: SVG not created for layer {layer_id}")

        except Exception as e:
            print(f"    Error vectorizing layer {layer_id}: {e}")
            traceback.print_exc()

        torch.cuda.empty_cache()

    # Merge layer SVGs
    print("\n" + "=" * 60)
    print("Merging layer SVGs into editable output")
    print("=" * 60)

    if progress_cb:
        progress_cb(93, "Merging layers...")

    final_svg_path = os.path.join(output_dir, "final.svg")
    merge_layer_svgs(layer_svgs, final_svg_path, (V13_RESOLUTION, V13_RESOLUTION), embed_masks=True)

    # Full-size version
    fullsize_svg_path = os.path.join(output_dir, "final_fullsize.svg")

    def write_fullsize_svg(source_path, dest_path):
        with open(source_path, 'r') as f:
            svg_content = f.read()

        def replace_root_svg_attr(content, attr, new_value):
            match = re.search(r'<svg\s[^>]*>', content)
            if match:
                svg_tag = match.group(0)
                new_svg_tag = re.sub(rf'{attr}="[^"]*"', f'{attr}="{new_value}"', svg_tag)
                content = content[:match.start()] + new_svg_tag + content[match.end():]
            return content

        svg_content = replace_root_svg_attr(svg_content, 'width', W_orig)
        svg_content = replace_root_svg_attr(svg_content, 'height', H_orig)

        match = re.search(r'<svg\s[^>]*>', svg_content)
        if match:
            svg_tag = match.group(0)
            if 'viewBox=' in svg_tag:
                svg_tag = re.sub(r'viewBox="[^"]*"',
                                 f'viewBox="0 0 {V13_RESOLUTION} {V13_RESOLUTION}"', svg_tag)
            else:
                svg_tag = svg_tag[:-1] + f' viewBox="0 0 {V13_RESOLUTION} {V13_RESOLUTION}">'
            if 'preserveAspectRatio=' in svg_tag:
                svg_tag = re.sub(r'preserveAspectRatio="[^"]*"', 'preserveAspectRatio="none"', svg_tag)
            else:
                svg_tag = svg_tag[:-1] + ' preserveAspectRatio="none">'
            svg_content = svg_content[:match.start()] + svg_tag + svg_content[match.end():]

        with open(dest_path, 'w') as f:
            f.write(svg_content)

    write_fullsize_svg(final_svg_path, fullsize_svg_path)

    # Render PNG
    print("Rendering composite PNG...")
    try:
        png_path = os.path.join(output_dir, "final.png")
        cairosvg.svg2png(url=final_svg_path, write_to=png_path)

        fullsize_png_path = os.path.join(output_dir, "final_fullsize.png")
        cairosvg.svg2png(url=fullsize_svg_path, write_to=fullsize_png_path)
    except Exception as e:
        print(f"  Warning: Could not render PNG: {e}")

    if progress_cb:
        progress_cb(100, "Complete!")

    print(f"\nDone! {len(layer_svgs)} editable layers created")
    print(f"Output: {output_dir}")

    return {
        'svg_path': fullsize_svg_path,
        'png_path': fullsize_png_path,
        'layers': layer_svgs,
        'n_layers': len(layer_svgs),
    }
