"""
Main V11 Pipeline: App3 SVG Quality + App8 Layer Editability

Combines:
- App8's SAM + Depth Anything semantic layer decomposition (split image into objects)
- App3's DiffVG optimization (SDS + structural + visual optimization)
  BUT with a custom per-layer pipeline that SKIPS SAM (uses provided masks)

Result: Editable semantic layers with DiffVG-quality paths.
Each object can be moved independently without leaving holes.

Usage:
    python main_v11.py -timg input.png -fsn output_name
"""

import torch
import torch.nn.functional as F
from PIL import Image
import argparse
import os
import sys
import numpy as np
import re
import cv2
import shutil
import base64

import pydiffvg
import yaml


# Resolution for SDXL
V11_RESOLUTION = 1024


# ============================================================================
# V11-specific utility functions (independent from main_v3)
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
    """
    Dilate a binary mask by a specified number of pixels.
    This helps eliminate gaps between adjacent layers by making masks overlap slightly.

    Args:
        mask: Binary mask (0 or 255), shape (H, W)
        dilation_px: Number of pixels to dilate (e.g., 3 means expand by 3 pixels)

    Returns:
        Dilated mask (0 or 255), same shape
    """
    if dilation_px <= 0:
        return mask

    # Create a circular structuring element for smooth dilation
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * dilation_px + 1, 2 * dilation_px + 1)
    )

    # Dilate the mask
    dilated = cv2.dilate(mask, kernel, iterations=1)

    return dilated


def decompose_into_layers(
    image: np.ndarray,
    device: torch.device,
    moge_version: str = "v2",
    moge_resolution: str = "High",
    n_depth_clusters: int = 3,
    min_mask_area: int = 500,
    sam_checkpoint_dir: str = None,
    mask_dilation_px: int = 3,
    refine_masks: bool = True,
    refine_iterations: int = 3,
    use_samhq: bool = False,  # Standard SAM (HQ-SAM can over-filter masks)
    sam_checkpoint: str = None,
    background_method: str = "depth"  # "depth" (Depth Anything) or "area" (largest mask)
) -> list:
    """
    Decompose image into semantic layers using SAM + Depth Anything.

    This is the "app8" part - splitting the image into movable objects.

    Args:
        mask_dilation_px: Number of pixels to dilate each mask. This creates
            slight overlap between adjacent layers, eliminating visible gaps
            at layer boundaries. Default is 3 pixels.
        refine_masks: Whether to use SAMRefiner for smoother mask edges.
        refine_iterations: Number of SAMRefiner iterations (more = smoother).
        background_method: How to determine which layer is background:
            - "depth": Use Depth Anything (lowest depth = furthest = background)
            - "area": Use mask area (largest mask = background)
    """
    decomposer_error = None
    try:
        module_dir = os.path.dirname(os.path.abspath(__file__))
        repo_root = os.path.dirname(module_dir)
        if repo_root not in sys.path:
            sys.path.append(repo_root)

        from LayeredVectorization.semantic_decomposer_v11 import SemanticLayerDecomposer

        print("  Using semantic decomposer with Depth Anything...")
        decomposer = SemanticLayerDecomposer(
            moge_version=moge_version,
            moge_resolution=moge_resolution,
            sam_checkpoint=sam_checkpoint,
            min_mask_area=min_mask_area,
            device=str(device),
            refine_masks=refine_masks,
            refine_iterations=refine_iterations,
            use_samhq=use_samhq
        )
        v8_layers = decomposer.decompose(image, n_clusters=n_depth_clusters)

        # === LAYER DEPTH/AREA LOGGING (for strategy analysis) ===
        print("\n  === Layer Analysis (for reordering strategy) ===")
        print("  Layer ID | Area (px) | Depth      | Notes")
        print("  " + "-" * 50)
        for layer in v8_layers:
            print(f"  {layer.layer_id:8d} | {layer.area:9d} | {layer.depth:10.4f} |")
        print("  " + "-" * 50)

        # Identify background layer based on selected method
        background_layer_id = None

        if background_method == "area":
            # Use LARGEST mask as background (user's suggestion)
            # Rationale: background (sky, walls, ground) often has the largest area
            max_area = 0
            for layer in v8_layers:
                if layer.area > max_area:
                    max_area = layer.area
                    background_layer_id = layer.layer_id
            bg_layer = next((l for l in v8_layers if l.layer_id == background_layer_id), None)
            bg_depth = bg_layer.depth if bg_layer else 0
            print(f"  Background identified: layer {background_layer_id} (area={max_area}, depth={bg_depth:.4f})")
            print("  Strategy: LARGEST area = background")
        else:
            # Default: Use LOWEST depth as background
            # Depth Anything (V2-style): higher = closer, lower = further
            # So background (furthest from camera) has the LOWEST depth value
            min_depth = float('inf')
            for layer in v8_layers:
                if layer.depth < min_depth:
                    min_depth = layer.depth
                    background_layer_id = layer.layer_id
            bg_layer = next((l for l in v8_layers if l.layer_id == background_layer_id), None)
            bg_area = bg_layer.area if bg_layer else 0
            print(f"  Background identified: layer {background_layer_id} (depth={min_depth:.4f}, area={bg_area})")
            print("  Strategy: LOWEST depth = background (Depth Anything: lower = further)")
        print("  " + "=" * 50 + "\n")

        layers = []
        for layer in v8_layers:
            original_mask = layer.mask

            # Apply mask dilation to ALL layers to eliminate white edge gaps
            # Each layer extends slightly, and proper stacking order ensures no visible seams
            is_background = (layer.layer_id == background_layer_id)
            if mask_dilation_px > 0:
                mask = dilate_mask(original_mask, mask_dilation_px)
                layer_type = "BACKGROUND" if is_background else "foreground"
                print(f"    Layer {layer.layer_id}: {layer_type} - dilated mask by {mask_dilation_px}px")
            else:
                mask = original_mask
            mask_bool = mask > 127

            layer_img = np.ones_like(image) * 255
            layer_img[mask_bool] = image[mask_bool]

            x, y, w, h = cv2.boundingRect(mask.astype(np.uint8))

            layers.append({
                'mask': mask,
                'original_mask': original_mask,  # Keep original for reference
                'mask_bool': mask_bool,
                'depth': float(layer.depth),
                'image': layer_img,
                'area': int(layer.area),
                'bbox': [int(x), int(y), int(w), int(h)],
                'layer_id': int(layer.layer_id),
                'is_foreground': bool(layer.is_foreground),
                'is_background': is_background
            })

        del decomposer
        torch.cuda.empty_cache()

        return layers
    except Exception as e:
        decomposer_error = e

    if decomposer_error is not None:
        print(f"  Warning: app8 decomposer unavailable ({decomposer_error}); falling back to legacy SAM+DepthAnything.")

    # IMPORTANT: Use build functions directly instead of sam_model_registry
    # to avoid potential registry pollution from segment_anything_hq
    from segment_anything import SamAutomaticMaskGenerator
    from segment_anything.build_sam import build_sam_vit_h, build_sam_vit_l, build_sam_vit_b
    from depth_anything_wrapper import get_depth_estimator

    # Build functions map (bypasses potentially polluted registry)
    build_functions = {
        "vit_h": build_sam_vit_h,
        "vit_l": build_sam_vit_l,
        "vit_b": build_sam_vit_b,
    }

    H, W = image.shape[:2]

    # Step 1: Estimate depth with Depth Anything
    print("  Estimating depth with Depth Anything...")
    depth_estimator = get_depth_estimator(
        version="auto",
        resolution="low",
        device=str(device)
    )
    depth_map = depth_estimator.estimate(image)
    # Depth Anything output (V2-style): higher = closer, lower = further

    # Clean up depth estimator
    del depth_estimator
    torch.cuda.empty_cache()

    # Step 2: Run SAM segmentation
    print("  Running SAM segmentation...")

    if sam_checkpoint_dir is None:
        sam_checkpoint_dir = os.path.dirname(os.path.abspath(__file__))

    sam_checkpoint_paths = [
        os.path.join(sam_checkpoint_dir, "checkpoints", "sam_vit_h_4b8939.pth"),
        os.path.join(sam_checkpoint_dir, "checkpoints", "sam_vit_b_01ec64.pth"),
        "checkpoints/sam_vit_h_4b8939.pth",
        "checkpoints/sam_vit_b_01ec64.pth",
    ]

    sam_checkpoint = None
    sam_model_type = "vit_h"
    for path in sam_checkpoint_paths:
        if os.path.exists(path):
            sam_checkpoint = path
            if "vit_b" in path:
                sam_model_type = "vit_b"
            break

    if sam_checkpoint is None:
        raise RuntimeError(f"SAM checkpoint not found")

    if sam_model_type not in build_functions:
        raise ValueError(f"Unknown SAM model type: {sam_model_type}")

    print(f"  Using SAM: {sam_model_type} with checkpoint {sam_checkpoint}")
    sam = build_functions[sam_model_type](checkpoint=sam_checkpoint)
    sam = sam.to(device)

    sam_config = {
        # CRITICAL FIX: Increase sampling to capture fine details (tree branches)
        'points_per_side': 64,  # Was 32
        # Lower thresholds slightly to accept finer/marginal masks
        'pred_iou_thresh': 0.82, # Was 0.86
        'stability_score_thresh': 0.90, # Was 0.92
        'crop_n_layers': 1,
        'crop_n_points_downscale_factor': 2,
        'min_mask_region_area': 50, # Was 100
    }

    mask_generator = SamAutomaticMaskGenerator(model=sam, **sam_config)
    masks_data = mask_generator.generate(image)

    # Clean up SAM
    del sam
    del mask_generator
    torch.cuda.empty_cache()

    # Filter and sort masks
    masks_data = [m for m in masks_data if m['area'] >= min_mask_area]
    masks_data = sorted(masks_data, key=lambda x: x['area'], reverse=True)

    # -------------------------------------------------------------------------
    # Deduplication & Containment Step
    # Remove highly overlapping masks AND masks contained in others
    # This is AGGRESSIVE to avoid multiple masks for the same object (e.g., fish body + fish head + fish tail)
    # -------------------------------------------------------------------------
    print(f"  Filtering duplicate/overlapping masks...")

    def calculate_overlaps(mask1, mask2):
        """
        Calculate overlap metrics between two masks.

        Returns:
            iou: Intersection over Union
            containment_small_in_large: How much of the smaller mask is inside the larger
            overlap_ratio_small: Intersection / area of smaller mask
        """
        intersection = np.logical_and(mask1, mask2).sum()
        area1 = mask1.sum()
        area2 = mask2.sum()
        union = np.logical_or(mask1, mask2).sum()

        iou = intersection / union if union > 0 else 0

        # Containment of smaller in larger
        smaller_area = min(area1, area2)
        containment = intersection / smaller_area if smaller_area > 0 else 0

        return iou, containment

    keep_indices = []
    if len(masks_data) > 0:
        # Keep track of which indices to remove
        remove_indices = set()

        for i in range(len(masks_data)):
            if i in remove_indices:
                continue

            mask_i_arr = masks_data[i]['segmentation']
            area_i = masks_data[i]['area']

            for j in range(i + 1, len(masks_data)):
                if j in remove_indices:
                    continue

                mask_j_arr = masks_data[j]['segmentation']
                area_j = masks_data[j]['area']

                iou, containment = calculate_overlaps(mask_i_arr, mask_j_arr)

                # Check 1: High IoU (near-duplicates)
                if iou > 0.70:  # Match app8 threshold for consistent layer merging
                    remove_indices.add(j)
                    print(f"    Dropping mask {j} (IoU {iou:.2f} with {i})")
                    continue

                # Check 2: Significant Containment
                # If smaller mask is >65% inside larger mask, it's likely a sub-part
                # (e.g., fish head inside whole fish) - merge into parent
                if containment > 0.65:  # Match app8 threshold
                    # Remove the smaller one (keep the larger, more complete mask)
                    if area_j < area_i:
                        remove_indices.add(j)
                        print(f"    Dropping mask {j} ({containment:.0%} contained in {i})")
                    else:
                        # j is larger, but i contains most of j? Unusual, skip
                        pass

        keep_indices = [i for i in range(len(masks_data)) if i not in remove_indices]

    original_count = len(masks_data)
    masks_data = [masks_data[i] for i in keep_indices]
    print(f"  Kept {len(masks_data)} masks (removed {original_count - len(masks_data)} redundant)")

    # Calculate depth for each mask first
    mask_depths = []
    for i, mask_info in enumerate(masks_data):
        mask_bool = mask_info['segmentation']
        avg_depth = float(np.mean(depth_map[mask_bool]))
        mask_depths.append(avg_depth)

    # === LAYER DEPTH/AREA LOGGING (for strategy analysis) ===
    print("\n  === Layer Analysis (for reordering strategy) ===")
    print("  Mask ID | Area (px) | Depth      | Notes")
    print("  " + "-" * 50)
    for i, mask_info in enumerate(masks_data):
        print(f"  {i:8d} | {mask_info['area']:9d} | {mask_depths[i]:10.4f} |")
    print("  " + "-" * 50)

    # Identify background layer based on selected method
    background_idx = 0

    if background_method == "area":
        # Use LARGEST mask as background
        max_area = 0
        for i, mask_info in enumerate(masks_data):
            if mask_info['area'] > max_area:
                max_area = mask_info['area']
                background_idx = i
        bg_depth = mask_depths[background_idx] if background_idx < len(mask_depths) else 0
        print(f"  Background identified: mask {background_idx} (area={max_area}, depth={bg_depth:.4f})")
        print("  Strategy: LARGEST area = background")
    else:
        # Default: Use LOWEST depth as background
        # Depth Anything (V2-style): higher = closer, lower = further
        # So background (furthest from camera) has the LOWEST depth value
        min_depth = float('inf')
        for i, depth in enumerate(mask_depths):
            if depth < min_depth:
                min_depth = depth
                background_idx = i
        bg_area = masks_data[background_idx]['area'] if background_idx < len(masks_data) else 0
        print(f"  Background identified: mask {background_idx} (depth={min_depth:.4f}, area={bg_area})")
        print("  Strategy: LOWEST depth = background (Depth Anything: lower = further)")
    print("  " + "=" * 50 + "\n")

    # Create layers
    layers = []
    for i, mask_info in enumerate(masks_data):
        original_mask = mask_info['segmentation'].astype(np.uint8) * 255

        # Apply mask dilation to ALL layers to eliminate white edge gaps
        is_background = (i == background_idx)
        if mask_dilation_px > 0:
            mask = dilate_mask(original_mask, mask_dilation_px)
            layer_type = "BACKGROUND" if is_background else "foreground"
            print(f"    Layer {i}: {layer_type} - dilated mask by {mask_dilation_px}px")
        else:
            mask = original_mask
        mask_bool = mask > 127

        # Use original mask for depth calculation (more accurate)
        original_mask_bool = original_mask > 127
        avg_depth = float(np.mean(depth_map[original_mask_bool]))

        # Extract layer image using dilated mask (all layers dilated to eliminate gaps)
        layer_img = np.ones_like(image) * 255
        layer_img[mask_bool] = image[mask_bool]

        layers.append({
            'mask': mask,
            'original_mask': original_mask,  # Keep original for reference
            'mask_bool': mask_bool,
            'depth': avg_depth,
            'image': layer_img,
            'area': mask_info['area'],
            'bbox': mask_info['bbox'],
            'layer_id': i,
            'is_background': is_background
        })

    # Sort by depth (background first)
    layers = sorted(layers, key=lambda x: x['depth'])

    return layers


def vectorize_layer_with_v11_pipeline(
    layer_img: np.ndarray,
    layer_mask: np.ndarray,
    device: torch.device,
    args,
    layer_id: int,
    v11_output_name: str,
    run_sds: bool = True
) -> str:
    """
    Vectorize a single layer using the V10 custom pipeline (app3 quality).

    This runs the SAME quality pipeline as app3:
    - SDS simplification (SDXL-based)
    - DiffVG optimization (structural + visual)

    The KEY DIFFERENCE from app3's full pipeline:
    - Uses the provided mask directly (no SAM - already have masks from decomposition)
    - This avoids running SAM twice (once for decomposition, once for vectorization)
    """
    from vectorize_layer_v11 import vectorize_single_layer

    # Create output directory for this layer
    layer_output_dir = f"./workdir/{v11_output_name}_layer_{layer_id}"
    os.makedirs(layer_output_dir, exist_ok=True)

    print(f"    Vectorizing layer {layer_id} with app3 quality pipeline...")
    print(f"    Output dir: {layer_output_dir}")
    if not run_sds:
        print(f"    WARNING: SDS disabled - quality will be lower than app3")

    # Run the custom per-layer vectorization (SDS + DiffVG, NO SAM)
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
        # Debug: list contents
        if os.path.exists(layer_output_dir):
            print(f"    Contents of {layer_output_dir}:")
            for root, dirs, files in os.walk(layer_output_dir):
                for f in files:
                    print(f"      {os.path.join(root, f)}")

    return svg_path


def merge_layer_svgs(layer_svgs: list, output_path: str, canvas_size: tuple, embed_masks: bool = True) -> str:
    """
    Merge multiple layer SVGs into a single SVG with grouped layers.
    Each layer is a <g> group that can be moved independently.

    Stacking order (bottom to top in final SVG):
    1. Background layer (lowest depth) - rendered first (behind everything)
    2. Other layers sorted by depth ascending (lower = farther = behind)

    Note: Depth Anything V2-style: lower values = further from camera
    """
    W, H = canvas_size

    # Sort by: 1) Background first, 2) depth ASCENDING (lower = further = behind)
    # In SVG, elements defined first are rendered behind later elements
    layer_svgs = sorted(layer_svgs, key=lambda x: (
        0 if x.get('is_background', False) else 1,  # Background first
        x.get('depth', 0)  # Then by depth ASCENDING (lower = further = behind)
    ))

    # Debug print stacking order
    print("\n  === SVG Stacking Order (bottom to top) ===")
    print("  Position | Layer ID | Depth | Background?")
    print("  " + "-" * 45)
    for i, layer in enumerate(layer_svgs):
        bg_marker = "YES" if layer.get('is_background', False) else "no"
        depth = layer.get('depth', 0)
        depth_str = f"{depth:.4f}" if isinstance(depth, float) else str(depth)
        print(f"  {i:8d} | {layer['layer_id']:8d} | {depth_str:>10} | {bg_marker}")
    print("  " + "-" * 45)
    print("  (Lower position = rendered first = behind)")
    print("  " + "=" * 45 + "\n")

    svg_parts = [
        '<?xml version="1.0" encoding="utf-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" '
        f'width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
        '  <!-- Each <g> group is a separate layer that can be moved independently in vector editors -->',
    ]

    mask_ids = set()
    if embed_masks:
        # Embed masks so paths cannot bleed outside their segmentation region.
        svg_parts.append('  <defs>')
        for layer_info in layer_svgs:
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

    for layer_info in layer_svgs:
        svg_path = layer_info['svg_path']
        layer_id = layer_info['layer_id']
        depth = layer_info['depth']
        mask_attr = ''
        if embed_masks and layer_id in mask_ids:
            mask_attr = f' mask="url(#mask_layer_{layer_id})"'

        if not os.path.exists(svg_path):
            print(f"  Warning: Layer SVG not found: {svg_path}")
            continue

        with open(svg_path, 'r') as f:
            layer_svg = f.read()

        # Extract content between <svg> tags
        match = re.search(r'<svg[^>]*>(.*?)</svg>', layer_svg, re.DOTALL)
        if match:
            layer_content = match.group(1)
            svg_parts.append(f'  <g id="layer_{layer_id}" data-depth="{depth:.3f}"{mask_attr}>')
            svg_parts.append(f'    {layer_content}')
            svg_parts.append(f'  </g>')

    svg_parts.append('</svg>')

    merged_svg = '\n'.join(svg_parts)
    with open(output_path, 'w') as f:
        f.write(merged_svg)

    return output_path


def layered_vectorization_v11(device, args, progress_callback=None):
    """
    Main V11 pipeline:
    1. Decompose image into semantic layers (SAM + Depth Anything)
    2. Run full App3 pipeline on each layer (SDS + SAM + DiffVG)
    3. Merge layer SVGs into one editable SVG
    """
    # init_diffvg and load_config are now defined locally in this file

    # Load input image
    input_img = np.array(Image.open(args.target_image).convert('RGB'))
    H_orig, W_orig = input_img.shape[:2]

    # Resize to V11_RESOLUTION (match app3's stretch-to-square preprocessing)
    working_img = np.array(
        Image.fromarray(input_img).resize((V11_RESOLUTION, V11_RESOLUTION), Image.LANCZOS)
    )

    # Setup workdir
    workdir = f"./workdir/{args.file_save_name}"
    os.makedirs(workdir, exist_ok=True)
    output_dir = workdir
    layers_dir = os.path.join(output_dir, "layers")
    os.makedirs(layers_dir, exist_ok=True)

    Image.fromarray(working_img).save(os.path.join(output_dir, "input_resized.png"))

    # ========================================
    # Decomposition uses ORIGINAL image (not SDS-simplified)
    # ========================================
    # FIX: Global SDS was causing SAM to detect fewer layers because the
    # simplified image merges similar regions. Instead:
    # - Use original image for SAM decomposition (more layers detected)
    # - Run SDS per-layer during vectorization (better SVG quality)
    decomposition_input = working_img
    print("  Using original image for SAM decomposition (per-layer SDS will run during vectorization)")

    # ========================================
    # Step 1: Decompose into semantic layers
    # ========================================
    print("=" * 60)
    print("Step 1: Decomposing into semantic layers")
    print("=" * 60)
    if progress_callback:
        progress_callback(20, "Decomposing into layers with SAM + Depth Anything...")

    # Get SAM settings from config (under sam: section)
    sam_config = getattr(args, 'sam', {}) or {}
    refine_masks = sam_config.get('refine_masks', True)
    refine_iterations = sam_config.get('refine_iterations', 3)
    use_samhq = sam_config.get('use_samhq', False)  # Standard SAM by default
    sam_checkpoint = sam_config.get('sam_checkpoint', None)

    layers = decompose_into_layers(
        decomposition_input,
        device,
        moge_version=getattr(args, 'moge_version', 'v2'),
        moge_resolution=getattr(args, 'moge_resolution', 'High'),
        n_depth_clusters=getattr(args, 'n_depth_clusters', 3),
        min_mask_area=getattr(args, 'min_mask_area', 500),
        mask_dilation_px=getattr(args, 'mask_dilation_px', 3),
        refine_masks=refine_masks,
        refine_iterations=refine_iterations,
        use_samhq=use_samhq,
        sam_checkpoint=sam_checkpoint,
        background_method=getattr(args, 'background_method', 'depth')
    )

    print(f"  Found {len(layers)} semantic layers")

    # Clear CUDA cache after decomposition to free HQ-SAM/Depth Anything VRAM
    torch.cuda.empty_cache()

    # Layer limiting: keep the N largest layers, drop the rest
    # Gap filling will assign dropped pixels to their nearest kept layer
    # This is simpler and more predictable than complex merging strategies
    max_layers = getattr(args, 'max_layers', 10)
    if len(layers) > max_layers:
        print(f"  Keeping {max_layers} largest layers out of {len(layers)} (dropped pixels will be gap-filled)")

        # Sort by area descending, keep largest N
        layers_by_area = sorted(layers, key=lambda x: x['area'], reverse=True)
        kept_layers = layers_by_area[:max_layers]
        dropped_layers = layers_by_area[max_layers:]

        # Log what we're dropping
        total_dropped_area = sum(l['area'] for l in dropped_layers)
        print(f"    Dropping {len(dropped_layers)} small layers ({total_dropped_area} pixels total)")
        for dl in dropped_layers[:5]:  # Show first 5
            print(f"      - layer {dl['layer_id']}: area={dl['area']}, depth={dl['depth']:.3f}")
        if len(dropped_layers) > 5:
            print(f"      ... and {len(dropped_layers) - 5} more")

        # Re-sort kept layers by depth for proper stacking
        layers = sorted(kept_layers, key=lambda x: x['depth'])

        # Reassign layer IDs
        for i, layer in enumerate(layers):
            layer['layer_id'] = i

        print(f"  Kept {len(layers)} layers")

    # === GAP FILLING: Assign uncovered pixels to nearest layer ===
    # SAM sometimes misses small areas. Fill them by assigning to the spatially nearest layer.
    H, W = layers[0]['image'].shape[:2]
    coverage_mask = np.zeros((H, W), dtype=np.uint8)
    for layer in layers:
        coverage_mask = np.maximum(coverage_mask, layer['mask'])

    uncovered = coverage_mask == 0
    uncovered_count = np.sum(uncovered)

    if uncovered_count > 0:
        print(f"  Filling {uncovered_count} uncovered pixels by assigning to nearest layer...")

        # For each layer, compute distance transform from its mask boundary
        # Then assign uncovered pixels to the layer with minimum distance
        from scipy import ndimage

        min_distances = np.full((H, W), np.inf, dtype=np.float32)
        nearest_layer_idx = np.zeros((H, W), dtype=np.int32)

        for i, layer in enumerate(layers):
            # Distance from each pixel to the nearest pixel IN this layer's mask
            mask_binary = (layer['mask'] > 127).astype(np.uint8)
            # Invert: distance transform gives distance to nearest zero
            dist = ndimage.distance_transform_edt(1 - mask_binary)
            closer = dist < min_distances
            min_distances[closer] = dist[closer]
            nearest_layer_idx[closer] = i

        # Assign uncovered pixels to their nearest layer
        for i, layer in enumerate(layers):
            pixels_to_add = uncovered & (nearest_layer_idx == i)
            if np.sum(pixels_to_add) > 0:
                # Add these pixels to this layer's mask
                layer['mask'][pixels_to_add] = 255
                # Get colors from the original image for these pixels
                layer['image'][pixels_to_add] = decomposition_input[pixels_to_add]
                layer['area'] = np.sum(layer['mask'] > 127)
                print(f"    Layer {i}: added {np.sum(pixels_to_add)} gap pixels")

        print(f"  Gap filling complete")

    # === MERGE PREVIEW: Composite all layers to verify no blanks ===
    # This creates a preview image showing all merged layers stacked together
    # Any blank/missing areas will appear as white, making issues immediately visible
    print(f"  Creating merge preview...")

    # Start with white background
    H, W = layers[0]['image'].shape[:2]
    merge_preview = np.ones((H, W, 3), dtype=np.uint8) * 255

    # Stack layers from background (lowest depth) to foreground (highest depth)
    layers_by_depth = sorted(layers, key=lambda x: x['depth'])
    for layer in layers_by_depth:
        mask = layer['mask'] > 127
        merge_preview[mask] = layer['image'][mask]

    # Save merge preview
    merge_preview_path = os.path.join(layers_dir, "merge_preview.png")
    Image.fromarray(merge_preview).save(merge_preview_path)
    print(f"  Saved merge preview: {merge_preview_path}")
    print(f"  >>> Check this file to verify no blank areas exist <<<")

    # Also create a coverage map showing which pixels are covered by at least one layer
    coverage_map = np.zeros((H, W), dtype=np.uint8)
    for layer in layers:
        coverage_map = np.maximum(coverage_map, layer['mask'])

    # Areas with 0 coverage = blanks (will appear black)
    coverage_preview_path = os.path.join(layers_dir, "coverage_map.png")
    Image.fromarray(coverage_map).save(coverage_preview_path)

    # Count uncovered pixels
    uncovered_pixels = np.sum(coverage_map == 0)
    total_pixels = H * W
    coverage_pct = 100.0 * (total_pixels - uncovered_pixels) / total_pixels
    print(f"  Coverage: {coverage_pct:.2f}% ({uncovered_pixels} uncovered pixels)")
    if uncovered_pixels > 0:
        print(f"  WARNING: {uncovered_pixels} pixels not covered by any layer!")

    # Save layer previews
    for layer in layers:
        preview_path = os.path.join(layers_dir, f"layer_{layer['layer_id']}_preview.png")
        Image.fromarray(layer['image']).save(preview_path)
        mask_path = os.path.join(layers_dir, f"layer_{layer['layer_id']}_mask.png")
        Image.fromarray(layer['mask']).save(mask_path)

    # ========================================
    # Step 2: Vectorize each layer with app3 quality (SDS + DiffVG)
    # ========================================
    print("")
    print("=" * 60)
    print("Step 2: Vectorizing each layer with app3 quality (SDS + DiffVG)")
    print("=" * 60)

    # Check if SDS should be skipped
    skip_sds = getattr(args, 'skip_sds', False)

    layer_svgs = []

    for i, layer in enumerate(layers):
        layer_id = layer['layer_id']
        print(f"\n--- Layer {i+1}/{len(layers)} (id={layer_id}, depth={layer['depth']:.3f}) ---")

        if progress_callback:
            progress = 10 + int(80 * i / len(layers))
            progress_callback(progress, f"Vectorizing layer {i+1}/{len(layers)} with app3 quality...")

        try:
            svg_path = vectorize_layer_with_v11_pipeline(
                layer['image'],
                layer['mask'],
                device,
                args,
                layer_id,
                args.file_save_name,  # Pass the V10 output name for proper naming
                run_sds=not skip_sds  # Can be disabled for faster processing
            )

            if svg_path and os.path.exists(svg_path):
                # Copy to layers output dir
                final_layer_svg = os.path.join(layers_dir, f"layer_{layer_id}.svg")
                shutil.copy(svg_path, final_layer_svg)

                layer_svgs.append({
                    'layer_id': layer_id,
                    'svg_path': final_layer_svg,
                    'depth': layer['depth'],
                    'area': layer['area'],
                    'is_background': layer.get('is_background', False),
                    'mask_path': os.path.join(layers_dir, f"layer_{layer_id}_mask.png")
                })
                print(f"    Layer {layer_id} vectorized successfully")
            else:
                print(f"    Warning: SVG not created for layer {layer_id}")

        except Exception as e:
            print(f"    Error vectorizing layer {layer_id}: {e}")
            import traceback
            traceback.print_exc()

        # Clear CUDA cache between layers to prevent OOM accumulation
        torch.cuda.empty_cache()

    # ========================================
    # Step 3: Merge layer SVGs
    # ========================================
    print("")
    print("=" * 60)
    print("Step 3: Merging layer SVGs into editable output")
    print("=" * 60)

    if progress_callback:
        progress_callback(95, "Merging layers...")

    final_svg_path = os.path.join(output_dir, "final.svg")
    # CRITICAL FIX: Enable masks by default for final.svg to prevents white background occlusion
    merge_layer_svgs(layer_svgs, final_svg_path, (V11_RESOLUTION, V11_RESOLUTION), embed_masks=True)

    masked_svg_path = os.path.join(output_dir, "final_masked.svg")
    merge_layer_svgs(layer_svgs, masked_svg_path, (V11_RESOLUTION, V11_RESOLUTION), embed_masks=True)

    # Create full-size version
    fullsize_svg_path = os.path.join(output_dir, "final_fullsize.svg")
    def write_fullsize_svg(source_path: str, dest_path: str) -> None:
        with open(source_path, 'r') as f:
            svg_content = f.read()

        # CRITICAL FIX: Only replace width/height in the FIRST <svg> tag (root element)
        # The embedded mask <image> elements also have width/height that must NOT be changed
        # Use a function to replace only the first match in the root <svg> tag

        def replace_root_svg_attr(content, attr, new_value):
            """Replace attribute only in the first/root <svg> tag."""
            # Find the first <svg ...> tag
            match = re.search(r'<svg\s[^>]*>', content)
            if match:
                svg_tag = match.group(0)
                # Replace the attribute within this tag only
                new_svg_tag = re.sub(rf'{attr}="[^"]*"', f'{attr}="{new_value}"', svg_tag)
                # Replace just this tag in the content
                content = content[:match.start()] + new_svg_tag + content[match.end():]
            return content

        svg_content = replace_root_svg_attr(svg_content, 'width', W_orig)
        svg_content = replace_root_svg_attr(svg_content, 'height', H_orig)

        # Find the root <svg> tag and modify viewBox and preserveAspectRatio within it
        match = re.search(r'<svg\s[^>]*>', svg_content)
        if match:
            svg_tag = match.group(0)

            # Handle viewBox
            if 'viewBox=' in svg_tag:
                svg_tag = re.sub(r'viewBox="[^"]*"', f'viewBox="0 0 {V11_RESOLUTION} {V11_RESOLUTION}"', svg_tag)
            else:
                # Add viewBox before closing >
                svg_tag = svg_tag[:-1] + f' viewBox="0 0 {V11_RESOLUTION} {V11_RESOLUTION}">'

            # Handle preserveAspectRatio - must add/replace in root svg tag only
            if 'preserveAspectRatio=' in svg_tag:
                svg_tag = re.sub(r'preserveAspectRatio="[^"]*"', 'preserveAspectRatio="none"', svg_tag)
            else:
                # Add preserveAspectRatio before closing >
                svg_tag = svg_tag[:-1] + ' preserveAspectRatio="none">'

            # Replace the root svg tag in content
            svg_content = svg_content[:match.start()] + svg_tag + svg_content[match.end():]

        with open(dest_path, 'w') as f:
            f.write(svg_content)

    write_fullsize_svg(final_svg_path, fullsize_svg_path)

    masked_fullsize_svg_path = os.path.join(output_dir, "final_fullsize_masked.svg")
    if os.path.exists(masked_svg_path):
        write_fullsize_svg(masked_svg_path, masked_fullsize_svg_path)

    # Render PNG
    print("Rendering composite PNG...")
    try:
        import cairosvg
        png_path = os.path.join(output_dir, "final.png")
        cairosvg.svg2png(url=final_svg_path, write_to=png_path)

        fullsize_png_path = os.path.join(output_dir, "final_fullsize.png")
        cairosvg.svg2png(url=fullsize_svg_path, write_to=fullsize_png_path,
                        output_width=W_orig, output_height=H_orig)
    except Exception as e:
        print(f"  Warning: Could not render PNG: {e}")

    if progress_callback:
        progress_callback(100, "Complete!")

    print("")
    print("=" * 60)
    print(f"Done! {len(layer_svgs)} editable layers created")
    print(f"Output: {output_dir}")
    print("  - final.svg: All layers merged (editable groups)")
    print("  - layers/: Individual layer SVGs")
    print("=" * 60)

    return {
        'svg_path': final_svg_path,
        'layers': layer_svgs,
        'n_layers': len(layer_svgs)
    }


if __name__ == "__main__":
    # init_diffvg and load_config are defined locally in this file

    parser = argparse.ArgumentParser(description="V11: App3 Quality + App8 Editability")
    parser.add_argument("-c", "--config", type=str, default="./config/base_config_sdxl.yaml")
    parser.add_argument("-timg", "--target_image", type=str, required=True)
    parser.add_argument("-fsn", "--file_save_name", type=str, default="output")

    # Layer decomposition params
    # Legacy params (kept for backward compat, now uses Depth Anything internally)
    parser.add_argument("--moge_version", type=str, default="v2")
    parser.add_argument("--moge_resolution", type=str, default="High")
    parser.add_argument("--n_depth_clusters", type=int, default=3)
    parser.add_argument("--max_layers", type=int, default=10, help="Maximum number of layers (for safety)")
    parser.add_argument("--min_mask_area", type=int, default=500, help="Minimum mask area in pixels (larger = fewer/cleaner masks)")
    parser.add_argument("--mask_dilation_px", type=int, default=3, help="Dilate masks by N pixels to eliminate gaps between layers")
    parser.add_argument("--background_method", type=str, default="depth", choices=["depth", "area"],
                        help="How to detect background: 'depth' (Depth Anything) or 'area' (largest mask)")
    parser.add_argument("--skip_sds", action="store_true", help="Skip SDS simplification (faster but lower quality)")

    # VTracer params
    parser.add_argument("--vtracer_enable", type=bool, default=True)
    parser.add_argument("--staircase_area", type=float, default=1.5)
    parser.add_argument("--corner_angle", type=float, default=135.0)
    parser.add_argument("--simplify_error", type=float, default=2.0)
    parser.add_argument("--smooth_iterations", type=int, default=0)

    args = parser.parse_args()
    args = load_config(args.config, args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    init_diffvg(device=device)

    layered_vectorization_v11(device, args)
