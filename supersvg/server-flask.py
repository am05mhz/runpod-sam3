"""
Experimental version: SAM3 segmentation + SuperSVG vectorization

Pipeline:
1. Use SAM3 service (separate venv) to segment the image into semantic regions
2. Process each segment through SuperSVG model
3. Combine all segment SVGs
4. Run optimization process

Architecture:
- SAM3 runs as a separate service on port 5002 (Python 3.10+)
- This app runs on port 5001 (SuperSVG conda env with Python 3.7)
- Communication via HTTP API
- Models are loaded on-demand to manage VRAM (not at startup)
"""

import os
import re
import uuid
import gc
import base64
import requests
import torch
import numpy as np
from io import BytesIO
from PIL import Image
from flask import Flask, request, render_template, send_file, jsonify
from torchvision import transforms
from skimage.segmentation import slic
import pydiffvg
import cairosvg

from models.supersvg_coarse import SuperSVG_coarse

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'temp')
OUTPUT_FOLDER = os.path.join(os.path.dirname(__file__), 'output')
CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), 'ckpts', 'coarse-model.pt')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp', 'bmp'}

# SAM3 service configuration
SAM3_SERVICE_URL = "http://127.0.0.1:5002"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Global models - loaded on demand, NOT at startup
supersvg_model = None
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# SuperSVG config
WIDTH = 224
BATCH_SIZE = 64

# Quality settings (SLIC segments per layer)
QUALITY_SETTINGS = {
    'low': 500,
    'default': 1500,
    'high': 5000,
    'best': 10000
}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def clear_gpu_memory():
    """Clear GPU memory cache."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    print("GPU memory cleared")


def load_supersvg_model():
    """Load the SuperSVG coarse model on demand."""
    global supersvg_model
    if supersvg_model is None:
        print(f"Loading SuperSVG model from {CHECKPOINT_PATH}...")
        supersvg_model = SuperSVG_coarse(stroke_num=128, path_num=4, width=WIDTH, num_loss=True)
        state_dict = torch.load(CHECKPOINT_PATH, map_location=device)
        # Handle DDP state dict
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k.replace('module.', '') if k.startswith('module.') else k
            new_state_dict[name] = v
        supersvg_model.load_state_dict(new_state_dict)
        supersvg_model.to(device)
        supersvg_model.eval()
        print("SuperSVG model loaded successfully!")
    return supersvg_model


def unload_supersvg_model():
    """Unload SuperSVG model to free VRAM."""
    global supersvg_model
    if supersvg_model is not None:
        print("Unloading SuperSVG model...")
        del supersvg_model
        supersvg_model = None
        clear_gpu_memory()
        print("SuperSVG model unloaded")


def check_sam3_service():
    """Check if SAM3 service is available."""
    try:
        response = requests.get(f"{SAM3_SERVICE_URL}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"SAM3 service: {data}")
            return data
    except requests.exceptions.ConnectionError:
        print(f"SAM3 service not available at {SAM3_SERVICE_URL}")
    except Exception as e:
        print(f"Error checking SAM3 service: {e}")
    return None


def load_sam3_via_service():
    """Tell SAM3 service to load its model."""
    try:
        print("Requesting SAM3 service to load model (this may take a while)...")
        response = requests.post(f"{SAM3_SERVICE_URL}/load", timeout=300)  # 5 min timeout for model loading
        if response.status_code == 200:
            data = response.json()
            status = data.get('status')
            print(f"SAM3 load response: {status}")
            return status == 'loaded'
        else:
            print(f"SAM3 load failed with status code: {response.status_code}")
    except requests.exceptions.Timeout:
        print("SAM3 model loading timed out (>5 minutes)")
    except Exception as e:
        print(f"Error loading SAM3 via service: {e}")
    return False


def unload_sam3_via_service():
    """Tell SAM3 service to unload its model to free VRAM."""
    try:
        response = requests.post(f"{SAM3_SERVICE_URL}/unload", timeout=30)
        if response.status_code == 200:
            print("SAM3 model unloaded via service")
            return True
    except Exception as e:
        print(f"Error unloading SAM3 via service: {e}")
    return False


def image_to_base64(image_pil):
    """Convert PIL Image to base64 string."""
    buffered = BytesIO()
    image_pil.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')


def masks_from_base64(b64_string):
    """Convert base64 encoded numpy array back to masks."""
    buffer = BytesIO(base64.b64decode(b64_string))
    data = np.load(buffer)
    return data['masks']


def reindex_groups(groups):
    """Re-index shape groups to have sequential shape_ids matching their position."""
    reindexed = []
    for i, grp in enumerate(groups):
        reindexed.append(
            pydiffvg.ShapeGroup(
                shape_ids=torch.LongTensor([i]),
                fill_color=grp.fill_color
            )
        )
    return reindexed


def svg_to_png(svg_path, png_path):
    """Convert SVG file to PNG using CairoSVG."""
    cairosvg.svg2png(url=svg_path, write_to=png_path)


def optimize_svg_precision(svg_path, decimals=2):
    """Post-process SVG to reduce decimal precision for smaller file size."""
    with open(svg_path, 'r') as f:
        content = f.read()

    original_size = len(content)

    def round_number(match):
        num_str = match.group(0)
        try:
            num = float(num_str)
            rounded = round(num, decimals)
            if rounded == int(rounded):
                return str(int(rounded))
            else:
                return f"{rounded:.{decimals}f}".rstrip('0').rstrip('.')
        except ValueError:
            return num_str

    def process_d_attr(match):
        d_content = match.group(1)
        processed = re.sub(r'-?\d+\.\d+', round_number, d_content)
        return f'd="{processed}"'

    content = re.sub(r'd="([^"]*)"', process_d_attr, content)

    def process_fill(match):
        fill_content = match.group(1)
        processed = re.sub(r'-?\d+\.\d+', round_number, fill_content)
        return f'fill="{processed}"'

    content = re.sub(r'fill="([^"]*)"', process_fill, content)

    with open(svg_path, 'w') as f:
        f.write(content)

    new_size = len(content)
    reduction = ((original_size - new_size) / original_size) * 100
    print(f"SVG optimized: {original_size:,} -> {new_size:,} bytes ({reduction:.1f}% reduction)")


def generate_sam3_masks_via_service(image_pil, use_ollama=True, conf_thresh=0.3, num_rounds=1):
    """
    Generate automatic segmentation masks using SAM3 service.
    Calls the SAM3 service running in a separate environment.

    VRAM Management:
    - When use_ollama=True: SAM3 service handles loading/unloading internally
      (unloads SAM3 -> runs Ollama -> unloads Ollama -> loads SAM3)
    - When use_ollama=False: Load SAM3 on-demand here

    Args:
        image_pil: PIL Image
        use_ollama: If True, use Ollama for object detection + SAM3 segmentation
        conf_thresh: Confidence threshold for segmentation
        num_rounds: Number of Ollama detection rounds

    Returns:
        Tuple of (masks list, labels list)
    """
    # Check service health
    health = check_sam3_service()
    if health is None:
        raise RuntimeError("SAM3 service not available. Make sure it's running on port 5002.")

    # Only pre-load SAM3 if NOT using Ollama
    # When using Ollama, the /auto_segment endpoint manages model loading internally
    if not use_ollama and not health.get('model_loaded', False):
        print("Loading SAM3 model via service...")
        if not load_sam3_via_service():
            raise RuntimeError("Failed to load SAM3 model")

    # Convert image to base64
    image_b64 = image_to_base64(image_pil)

    try:
        if use_ollama:
            # Use Ollama + SAM3 auto-segmentation
            print(f"Calling SAM3 service with Ollama auto-detection (rounds={num_rounds})...")
            response = requests.post(
                f"{SAM3_SERVICE_URL}/auto_segment",
                json={
                    'image': image_b64,
                    'conf_thresh': conf_thresh,
                    'num_rounds': num_rounds
                },
                timeout=600  # 10 minute timeout for Ollama + SAM3
            )
        else:
            # Use text-based segmentation for generic objects
            print("Calling SAM3 service with text query...")
            response = requests.post(
                f"{SAM3_SERVICE_URL}/segment_text",
                json={
                    'image': image_b64,
                    'query': 'objects',
                    'conf_thresh': conf_thresh
                },
                timeout=300
            )

        if response.status_code != 200:
            error = response.json().get('error', 'Unknown error')
            raise RuntimeError(f"SAM3 service error: {error}")

        data = response.json()
        num_masks = data.get('num_masks', 0)
        labels = data.get('labels', [])
        print(f"SAM3 service returned {num_masks} masks")
        if labels:
            unique_labels = list(set(labels))
            print(f"  Labels: {', '.join(unique_labels)}")

        if num_masks == 0:
            return [], []

        # Decode masks
        masks_array = masks_from_base64(data['masks'])

        # Convert to list of 2D masks
        if masks_array.ndim == 3:
            masks = [masks_array[i] for i in range(masks_array.shape[0])]
        else:
            masks = []

        return masks, labels

    except requests.exceptions.Timeout:
        raise RuntimeError("SAM3 service timeout - image may be too large")
    except requests.exceptions.ConnectionError:
        raise RuntimeError("Cannot connect to SAM3 service")
    except Exception as e:
        raise RuntimeError(f"SAM3 service error: {str(e)}")


def group_masks_by_label(masks, labels):
    """
    Group masks by their label and combine masks of the same object type.

    Args:
        masks: List of binary masks
        labels: List of labels corresponding to each mask

    Returns:
        Dict mapping label -> combined mask
    """
    from collections import defaultdict

    label_masks = defaultdict(list)

    for mask, label in zip(masks, labels):
        label_masks[label].append(mask)

    # Combine masks for each label
    combined = {}
    for label, mask_list in label_masks.items():
        if len(mask_list) == 1:
            combined[label] = mask_list[0]
        else:
            # Combine all masks of same label into one
            combined_mask = mask_list[0].astype(np.uint8)
            for m in mask_list[1:]:
                combined_mask = np.logical_or(combined_mask, m).astype(np.uint8)
            combined[label] = combined_mask

    return combined


def process_slic_segments_batched(model, image_np, segments, pass_name="Pass"):
    """
    Process SLIC segments in batches for GPU efficiency.
    Ported from app.py for use within each SAM3 object layer.

    Args:
        model: SuperSVG model
        image_np: Image numpy array (H, W, 3), values 0-1
        segments: SLIC segmentation labels
        pass_name: Name for logging

    Returns:
        Tuple of (shapes list, groups list)
    """
    resize_to_model = transforms.Resize((WIDTH, WIDTH))
    to_tensor = transforms.ToTensor()
    num_control_points = [2] * model.path_num

    num_segments = segments.max() + 1
    shapes = []
    groups = []

    # Pre-compute segment metadata
    segment_data = []
    for seg_idx in range(num_segments):
        seg_mask_full = (segments == seg_idx)
        if not seg_mask_full.any():
            continue

        rows = np.any(seg_mask_full, axis=1)
        cols = np.any(seg_mask_full, axis=0)
        y1, y2 = np.where(rows)[0][[0, -1]]
        x1, x2 = np.where(cols)[0][[0, -1]]

        crop_h, crop_w = y2 - y1 + 1, x2 - x1 + 1
        if crop_h < 4 or crop_w < 4:
            continue

        segment_data.append({
            'x1': x1, 'y1': y1,
            'crop_w': crop_w, 'crop_h': crop_h,
            'crop_img': image_np[y1:y2+1, x1:x2+1],
            'crop_mask': seg_mask_full[y1:y2+1, x1:x2+1].astype(np.float32)
        })

    total_segments = len(segment_data)

    # Process in batches
    for batch_start in range(0, total_segments, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, total_segments)
        batch_data = segment_data[batch_start:batch_end]

        # Prepare batch tensors
        batch_inputs = []
        for seg in batch_data:
            crop_img_tensor = to_tensor(seg['crop_img'].astype(np.float32))
            crop_mask_tensor = torch.from_numpy(seg['crop_mask']).unsqueeze(0)

            crop_img_tensor = resize_to_model(crop_img_tensor)
            crop_mask_tensor = resize_to_model(crop_mask_tensor)

            crop_mask_tensor_bin = (crop_mask_tensor > 0.5).float()
            masked_input = crop_img_tensor * crop_mask_tensor_bin - (1 - crop_mask_tensor_bin)
            batch_inputs.append(masked_input)

        # Stack and move to GPU
        batch_tensor = torch.stack(batch_inputs).to(device)

        # Run batch inference
        with torch.no_grad():
            strokes_batch = model.encoder(batch_tensor)

        strokes_batch_cpu = strokes_batch.detach().cpu()

        for i, seg in enumerate(batch_data):
            strokes_cpu = strokes_batch_cpu[i]
            x1, y1 = seg['x1'], seg['y1']
            crop_w, crop_h = seg['crop_w'], seg['crop_h']

            for stroke_idx in range(strokes_cpu.size(0)):
                stroke = strokes_cpu[stroke_idx]
                visibility = float(stroke[-1].item())
                if visibility < 0.5:
                    continue

                points = stroke[:24].reshape(-1, 2).numpy().copy()
                points[:, 0] = points[:, 0] * crop_w + x1
                points[:, 1] = points[:, 1] * crop_h + y1

                rgb = np.clip(stroke[24:27].numpy(), 0.0, 1.0)
                rgba = np.array([rgb[0], rgb[1], rgb[2], 1.0], dtype=np.float32)

                shapes.append(
                    pydiffvg.Path(
                        num_control_points=torch.LongTensor(num_control_points),
                        points=torch.from_numpy(points).float(),
                        stroke_width=torch.tensor(0.0),
                        is_closed=True
                    )
                )
                groups.append(
                    pydiffvg.ShapeGroup(
                        shape_ids=torch.LongTensor([0]),
                        fill_color=torch.from_numpy(rgba).float()
                    )
                )

    return shapes, groups


def filter_shapes_by_mask(shapes, groups, mask):
    """
    Filter shapes: keep only those whose centroid falls within the object mask.
    Simple approach - if shape center is inside the mask, keep it. Otherwise delete.

    Args:
        shapes: List of pydiffvg shapes
        groups: List of pydiffvg shape groups
        mask: Binary mask (H, W) where True = inside object

    Returns:
        Tuple of (filtered_shapes, filtered_groups)
    """
    h, w = mask.shape
    filtered_shapes = []
    filtered_groups = []

    for shape, group in zip(shapes, groups):
        points = shape.points.numpy()

        # Calculate centroid of the shape
        centroid_x = np.mean(points[:, 0])
        centroid_y = np.mean(points[:, 1])

        # Clamp centroid to image bounds
        cx = int(max(0, min(centroid_x, w - 1)))
        cy = int(max(0, min(centroid_y, h - 1)))

        # Keep shape only if centroid is inside the object mask
        if mask[cy, cx]:
            filtered_shapes.append(shape)
            filtered_groups.append(group)

    return filtered_shapes, filtered_groups


def process_single_object_mask(model, image_np, mask, label, n_segments=1500):
    """
    Process a single object's mask through SuperSVG using 3-pass SLIC.
    Each object layer gets the same quality treatment as app.py's full image.

    The 3-pass approach with shifted grids fills boundary gaps:
    - Pass 1: Normal SLIC segmentation
    - Pass 2: Shifted grid (half superpixel size)
    - Pass 3: Another shift pattern (third superpixel size)

    IMPORTANT: Each pass is filtered by the object mask immediately after processing.
    This ensures only shapes belonging to THIS object are kept, removing any noise
    that extends outside the SAM3-detected boundary.

    Args:
        model: SuperSVG model
        image_np: Full image as numpy array (H, W, 3), values 0-1
        mask: Combined binary mask for this object (full image size)
        label: Object label name
        n_segments: Number of SLIC segments (quality setting)

    Returns:
        Tuple of (shapes list, groups list, num_shapes)
    """
    # Find bounding box of the mask
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)

    if not rows.any() or not cols.any():
        return [], [], 0

    y1, y2 = np.where(rows)[0][[0, -1]]
    x1, x2 = np.where(cols)[0][[0, -1]]

    crop_h, crop_w = y2 - y1 + 1, x2 - x1 + 1
    if crop_h < 10 or crop_w < 10:
        return [], [], 0

    # Crop image and mask to bounding box
    crop_img = image_np[y1:y2+1, x1:x2+1].copy()
    crop_mask = mask[y1:y2+1, x1:x2+1].astype(bool)

    # Apply mask - areas outside the object become black
    crop_img_masked = crop_img.copy()
    crop_img_masked[~crop_mask] = 0.0

    # Scale n_segments based on crop size relative to typical image
    # Smaller objects get proportionally fewer segments
    crop_area = crop_h * crop_w
    typical_area = 1000 * 1000  # Reference area
    area_ratio = crop_area / typical_area
    scaled_segments = max(50, int(n_segments * area_ratio))

    # Calculate shift based on superpixel size
    avg_superpixel_size = int(np.sqrt(crop_area / max(scaled_segments, 1)))

    total_before = 0
    total_after = 0

    # === PASS 1: Normal SLIC segmentation ===
    segments1 = slic(
        crop_img,
        n_segments=scaled_segments,
        sigma=1,
        compactness=30,
        start_label=0,
    )
    segments1[~crop_mask] = -1

    pass1_shapes, pass1_groups = process_slic_segments_batched(
        model, crop_img_masked, segments1, f"{label} Pass1"
    )
    # Filter Pass 1 by object mask
    total_before += len(pass1_shapes)
    pass1_shapes, pass1_groups = filter_shapes_by_mask(pass1_shapes, pass1_groups, crop_mask)
    total_after += len(pass1_shapes)

    # === PASS 2: Shifted grid ===
    SHIFT_X = max(avg_superpixel_size // 2, 5)
    SHIFT_Y = max(avg_superpixel_size // 2, 5)

    crop_padded = np.pad(crop_img, ((SHIFT_Y, 0), (SHIFT_X, 0), (0, 0)), mode='edge')

    segments2_padded = slic(
        crop_padded,
        n_segments=scaled_segments,
        sigma=1,
        compactness=30,
        start_label=0,
    )
    segments2 = segments2_padded[SHIFT_Y:, SHIFT_X:]
    segments2[~crop_mask] = -1

    pass2_shapes, pass2_groups = process_slic_segments_batched(
        model, crop_img_masked, segments2, f"{label} Pass2"
    )
    # Filter Pass 2 by object mask
    total_before += len(pass2_shapes)
    pass2_shapes, pass2_groups = filter_shapes_by_mask(pass2_shapes, pass2_groups, crop_mask)
    total_after += len(pass2_shapes)

    # === PASS 3: Another shifted grid ===
    SHIFT3_X = max(avg_superpixel_size // 3, 3)
    SHIFT3_Y = max(avg_superpixel_size // 3, 3)

    crop_padded3 = np.pad(crop_img, ((0, SHIFT3_Y), (0, SHIFT3_X), (0, 0)), mode='edge')
    segments3_padded = slic(
        crop_padded3,
        n_segments=scaled_segments,
        sigma=1,
        compactness=30,
        start_label=0,
    )
    segments3 = segments3_padded[:crop_h, :crop_w]
    segments3[~crop_mask] = -1

    pass3_shapes, pass3_groups = process_slic_segments_batched(
        model, crop_img_masked, segments3, f"{label} Pass3"
    )
    # Filter Pass 3 by object mask
    total_before += len(pass3_shapes)
    pass3_shapes, pass3_groups = filter_shapes_by_mask(pass3_shapes, pass3_groups, crop_mask)
    total_after += len(pass3_shapes)

    # Report filtering results
    if total_before > total_after:
        removed = total_before - total_after
        print(f"    Mask filter: removed {removed} shapes outside boundary ({total_before} -> {total_after})")

    # Combine all passes: Pass3 (bottom), Pass2, Pass1 (top)
    all_shapes = pass3_shapes + pass2_shapes + pass1_shapes
    all_groups = pass3_groups + pass2_groups + pass1_groups

    # Offset all points by the crop origin (y1, x1) to put them in full image coordinates
    for shape in all_shapes:
        points = shape.points.numpy()
        points[:, 0] += x1
        points[:, 1] += y1
        shape.points = torch.from_numpy(points).float()

    return all_shapes, all_groups, len(all_shapes)


def process_objects_as_layers(model, image_np, label_masks, n_segments=1500):
    """
    Process each object type as a separate layer using 3-pass SLIC.

    Args:
        model: SuperSVG model
        image_np: Full image as numpy array (H, W, 3), values 0-1
        label_masks: Dict mapping label -> combined mask
        n_segments: Number of SLIC segments per layer (quality setting)

    Returns:
        List of dicts with 'label', 'shapes', 'groups', 'count'
    """
    layers = []

    for label, mask in label_masks.items():
        print(f"  Processing layer: {label} (3-pass SLIC, {n_segments} base segments)")
        shapes, groups, count = process_single_object_mask(model, image_np, mask, label, n_segments)

        if count > 0:
            layers.append({
                'label': label,
                'shapes': shapes,
                'groups': groups,
                'count': count
            })
            print(f"    -> {count} shapes (3 passes combined)")
        else:
            print(f"    -> skipped (no shapes)")

    return layers


def save_svg_with_layers(svg_path, width, height, layers, bg_shape, bg_group):
    """
    Save SVG with named layer groups for each object.

    Args:
        svg_path: Output path
        width: SVG width
        height: SVG height
        layers: List of layer dicts with 'label', 'shapes', 'groups'
        bg_shape: Background shape
        bg_group: Background group
    """
    # First, save with pydiffvg to get proper path data
    all_shapes = [bg_shape]
    all_groups = [bg_group]

    layer_ranges = []  # Track which shapes belong to which layer
    current_idx = 1  # Start after background

    for layer in layers:
        start_idx = current_idx
        all_shapes.extend(layer['shapes'])
        all_groups.extend(layer['groups'])
        end_idx = current_idx + len(layer['shapes'])
        layer_ranges.append({
            'label': layer['label'],
            'start': start_idx,
            'end': end_idx
        })
        current_idx = end_idx

    # Save initial SVG
    pydiffvg.save_svg(svg_path, width, height, all_shapes, reindex_groups(all_groups))

    # Post-process to add layer groups
    add_layer_groups_to_svg(svg_path, layer_ranges)


def add_layer_groups_to_svg(svg_path, layer_ranges):
    """
    Post-process SVG to wrap paths in named <g> groups for layers.
    Uses Inkscape-compatible attributes for proper layer support in vector editors.

    Args:
        svg_path: Path to SVG file
        layer_ranges: List of dicts with 'label', 'start', 'end' indices
    """
    with open(svg_path, 'r') as f:
        content = f.read()

    # Find all <path> elements - handle both self-closing (/>) and regular (>) tags
    path_pattern = re.compile(r'(<path[^>]*(?:/>|>))', re.DOTALL)
    paths = path_pattern.findall(content)

    print(f"  Found {len(paths)} paths in SVG")

    if not paths:
        print("  Warning: No paths found in SVG!")
        return

    # Extract SVG header
    svg_start_match = re.search(r'(<svg[^>]*>)', content)
    if not svg_start_match:
        print("  Warning: Could not find SVG header!")
        return

    svg_header = svg_start_match.group(1)

    # Add Inkscape namespace if not present
    if 'xmlns:inkscape' not in svg_header:
        svg_header = svg_header.replace(
            '<svg',
            '<svg xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape"'
        )

    # Build new content with layered structure
    new_content = '<?xml version="1.0" encoding="UTF-8"?>\n'
    new_content += svg_header + '\n'
    new_content += '<defs/>\n'

    # Add background layer (first path)
    new_content += '<g id="background" inkscape:label="Background" inkscape:groupmode="layer">\n'
    new_content += f'  {paths[0]}\n'
    new_content += '</g>\n'

    # Group remaining paths by layer (in reverse order so first layer is on top)
    for layer_info in reversed(layer_ranges):
        label = layer_info['label']
        start = layer_info['start']
        end = layer_info['end']

        # Sanitize label for XML id
        safe_id = re.sub(r'[^a-zA-Z0-9_-]', '_', label)

        layer_paths = paths[start:end]
        if layer_paths:
            new_content += f'<g id="{safe_id}" inkscape:label="{label}" inkscape:groupmode="layer">\n'
            for path in layer_paths:
                new_content += f'  {path}\n'
            new_content += '</g>\n'
            print(f"  Layer '{label}': {len(layer_paths)} paths")

    new_content += '</svg>'

    with open(svg_path, 'w') as f:
        f.write(new_content)

    print(f"  SVG saved with {len(layer_ranges)} layers")


def mask_black_regions(image_np, threshold=0.08):
    """Create a mask for non-black regions of the image."""
    brightness = np.max(image_np, axis=2)
    non_black_mask = brightness > threshold
    return non_black_mask


def filter_shapes_in_black_region(shapes, groups, non_black_mask, threshold_ratio=0.95):
    """Filter out shapes that fall entirely within black regions."""
    filtered_shapes = []
    filtered_groups = []
    h, w = non_black_mask.shape

    for shape, group in zip(shapes, groups):
        points = shape.points.numpy()

        in_black = 0
        total = len(points)

        for pt in points:
            x, y = int(pt[0]), int(pt[1])
            x = max(0, min(x, w - 1))
            y = max(0, min(y, h - 1))
            if not non_black_mask[y, x]:
                in_black += 1

        black_ratio = in_black / total if total > 0 else 0
        if black_ratio < threshold_ratio:
            filtered_shapes.append(shape)
            filtered_groups.append(group)

    return filtered_shapes, filtered_groups


def process_image_sam3(image_path, output_svg_path, output_png_path, max_dim=4096, use_ollama=True, conf_thresh=0.3, num_rounds=1, quality='default'):
    """
    Process an image using SAM3 segmentation + SuperSVG vectorization.

    VRAM Management Pipeline:
    1. Load and resize image
    2. SAM3 service: Ollama detects objects -> unload Ollama -> SAM3 segments -> return masks
    3. Unload SAM3, load SuperSVG
    4. Group masks by object label, combine same-object masks
    5. Process each object as a separate SVG layer with 3-pass SLIC
    6. Save SVG with named layer groups

    This ensures only one large model is in VRAM at a time.
    Each detected object becomes its own layer in the SVG.
    Each layer uses 3-pass SLIC like app.py for quality.

    Args:
        quality: 'low' (500), 'default' (1500), 'high' (5000), 'best' (10000) segments per layer
    """
    n_segments = QUALITY_SETTINGS.get(quality, 1500)
    print(f"=== SAM3 + SuperSVG Pipeline (Layer Mode, Quality: {quality}/{n_segments} segments) ===")

    # Load and resize image
    image_pil = Image.open(image_path).convert('RGB')
    orig_w, orig_h = image_pil.size

    scale = max_dim / max(orig_w, orig_h)
    if scale > 1:
        scale = 1
    output_w = int(orig_w * scale)
    output_h = int(orig_h * scale)

    image_pil = image_pil.resize((output_w, output_h), Image.LANCZOS)
    image_np = np.array(image_pil) / 255.0

    print(f"Image size: {output_w}x{output_h}")

    # Detect black regions
    non_black_mask = mask_black_regions(image_np, threshold=0.08)
    black_ratio = 1.0 - (np.sum(non_black_mask) / non_black_mask.size)
    print(f"Black region: {black_ratio*100:.1f}% of image")

    # Step 1: Generate SAM3 masks via service
    print("\n--- Step 1: SAM3 Segmentation (via service) ---")
    masks, labels = generate_sam3_masks_via_service(
        image_pil,
        use_ollama=use_ollama,
        conf_thresh=conf_thresh,
        num_rounds=num_rounds
    )

    if len(masks) == 0:
        print("Warning: No masks generated by SAM3, falling back to full image processing")
        masks = [np.ones((output_h, output_w), dtype=np.uint8)]
        labels = ['full_image']

    # Step 2: Group masks by label
    print("\n--- Step 2: Grouping Masks by Object ---")
    label_masks = group_masks_by_label(masks, labels)
    print(f"Found {len(label_masks)} unique objects: {', '.join(label_masks.keys())}")

    # Step 2b: Create "remaining" mask for areas not covered by any detected object
    print("\n--- Step 2b: Creating 'remaining' mask for uncovered areas ---")
    all_detected_mask = np.zeros((output_h, output_w), dtype=bool)
    for mask in label_masks.values():
        all_detected_mask = np.logical_or(all_detected_mask, mask.astype(bool))

    # Remaining = non-black areas that weren't detected by SAM3
    remaining_mask = np.logical_and(non_black_mask, ~all_detected_mask)
    remaining_coverage = np.sum(remaining_mask) / np.sum(non_black_mask) * 100 if np.sum(non_black_mask) > 0 else 0
    print(f"Remaining uncovered area: {remaining_coverage:.1f}% of non-black regions")

    # Add remaining as a layer if it has significant coverage
    if remaining_coverage > 1.0:  # More than 1% uncovered
        label_masks['_remaining'] = remaining_mask.astype(np.uint8)
        print(f"Added '_remaining' layer for uncovered areas")

    # Step 3: Unload SAM3 and load SuperSVG (VRAM management)
    print("\n--- Step 3: VRAM Management - Swap Models ---")
    unload_sam3_via_service()
    clear_gpu_memory()

    print("\n--- Step 4: Loading SuperSVG ---")
    model = load_supersvg_model()

    # Step 5: Process each object as a layer with 3-pass SLIC
    print(f"\n--- Step 5: SuperSVG Processing (Per-Object Layers, 3-pass SLIC) ---")

    # Apply black mask to image for processing
    image_np_masked = image_np.copy()
    image_np_masked[~non_black_mask] = 0.0

    layers = process_objects_as_layers(model, image_np_masked, label_masks, n_segments)

    # Step 6: Create background
    print("\n--- Step 6: Creating Background Layer ---")

    bg_points = np.array([[0, 0], [output_w, 0], [output_w, output_h], [0, output_h]], dtype=np.float32)
    bg_shape = pydiffvg.Path(
        num_control_points=torch.LongTensor([0, 0, 0, 0]),
        points=torch.from_numpy(bg_points).float(),
        stroke_width=torch.tensor(0.0),
        is_closed=True
    )
    bg_group = pydiffvg.ShapeGroup(
        shape_ids=torch.LongTensor([0]),
        fill_color=torch.tensor([0.0, 0.0, 0.0, 1.0])
    )

    # Step 7: Save SVG with named layers
    print("\n--- Step 7: Saving SVG with Named Layers ---")

    total_shapes = sum(layer['count'] for layer in layers) + 1  # +1 for background

    if layers:
        save_svg_with_layers(output_svg_path, output_w, output_h, layers, bg_shape, bg_group)
    else:
        # No layers, just save background
        pydiffvg.save_svg(output_svg_path, output_w, output_h, [bg_shape], [bg_group])

    # Step 8: Optimize and render PNG
    print("\n--- Step 8: Optimizing and Rendering ---")
    optimize_svg_precision(output_svg_path, decimals=2)
    svg_to_png(output_svg_path, output_png_path)

    # Print layer summary
    print(f"\n=== Complete ===")
    print(f"Total shapes: {total_shapes}")
    print(f"Layers created:")
    print(f"  - background")
    for layer in layers:
        print(f"  - {layer['label']}: {layer['count']} shapes")


@app.route('/')
def index():
    return render_template('index2.html')


@app.route('/status', methods=['GET'])
def status():
    """Get status of models and services."""
    sam3_health = check_sam3_service()
    return jsonify({
        'supersvg_loaded': supersvg_model is not None,
        'sam3_service': sam3_health,
        'device': device
    })


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        # Options from form
        use_ollama = request.form.get('use_ollama', 'true').lower() == 'true'
        conf_thresh = float(request.form.get('conf_thresh', '0.3'))
        num_rounds = int(request.form.get('num_rounds', '1'))
        quality = request.form.get('quality', 'default')

        # Validate quality setting
        if quality not in QUALITY_SETTINGS:
            quality = 'default'

        # Generate unique filename
        unique_id = str(uuid.uuid4())[:8]
        ext = file.filename.rsplit('.', 1)[1].lower()
        input_filename = f"{unique_id}_input.{ext}"
        output_svg_filename = f"{unique_id}_output.svg"
        output_png_filename = f"{unique_id}_output.png"

        input_path = os.path.join(UPLOAD_FOLDER, input_filename)
        output_svg_path = os.path.join(OUTPUT_FOLDER, output_svg_filename)
        output_png_path = os.path.join(OUTPUT_FOLDER, output_png_filename)

        file.save(input_path)

        try:
            process_image_sam3(
                input_path, output_svg_path, output_png_path,
                use_ollama=use_ollama,
                conf_thresh=conf_thresh,
                num_rounds=num_rounds,
                quality=quality
            )

            return jsonify({
                'success': True,
                'svg_url': f'/output/{output_svg_filename}',
                'png_url': f'/output/{output_png_filename}',
                'svg_filename': output_svg_filename,
                'png_filename': output_png_filename
            })
        except Exception as e:
            import traceback
            traceback.print_exc()
            return jsonify({'error': str(e)}), 500

    return jsonify({'error': 'Invalid file type'}), 400


@app.route('/output/<filename>')
def serve_output(filename):
    return send_file(os.path.join(OUTPUT_FOLDER, filename))


@app.route('/download/<filename>')
def download_file(filename):
    return send_file(
        os.path.join(OUTPUT_FOLDER, filename),
        as_attachment=True,
        download_name=filename
    )


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='SuperSVG + SAM3 experimental inference')
    parser.add_argument('--input', type=str, help='Path to an input image. If provided, runs once and exits.')
    parser.add_argument('--output-dir', type=str, default=OUTPUT_FOLDER, help='Directory for CLI outputs')
    parser.add_argument('--max-dim', type=int, default=4096, help='Max dimension for resizing before vectorization')
    parser.add_argument('--no-ollama', action='store_true', help='Disable Ollama object detection')
    parser.add_argument('--conf-thresh', type=float, default=0.3, help='SAM3 confidence threshold')
    parser.add_argument('--num-rounds', type=int, default=1, help='Number of Ollama detection rounds')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Flask host')
    parser.add_argument('--port', type=int, default=5001, help='Flask port (default 5001)')
    parser.add_argument('--debug', action='store_true', help='Enable Flask debug mode')
    args = parser.parse_args()

    if args.input:
        os.makedirs(args.output_dir, exist_ok=True)
        base = os.path.splitext(os.path.basename(args.input))[0]
        svg_out = os.path.join(args.output_dir, f'{base}_sam3_output.svg')
        png_out = os.path.join(args.output_dir, f'{base}_sam3_output.png')
        print(f'Running SAM3+SuperSVG inference on {args.input}')
        process_image_sam3(
            args.input, svg_out, png_out,
            max_dim=args.max_dim,
            use_ollama=not args.no_ollama,
            conf_thresh=args.conf_thresh,
            num_rounds=args.num_rounds
        )
        print(f'Output: {svg_out}, {png_out}')
    else:
        # DO NOT pre-load models - they will be loaded on demand
        print("Starting app2.py server (models will be loaded on-demand)...")
        print(f"SAM3 service expected at: {SAM3_SERVICE_URL}")
        print(f"Device: {device}")

        # Just check if SAM3 service is reachable (don't load its model)
        health = check_sam3_service()
        if health:
            print(f"SAM3 service is reachable: {health}")
        else:
            print("WARNING: SAM3 service not reachable at port 5002")

        app.run(host=args.host, port=args.port, debug=args.debug)
