"""
V11 Per-Layer Vectorization Pipeline

This is a custom pipeline for vectorizing a single layer (object) that already
has a mask from the SAM + Depth Anything decomposition step.

This pipeline runs the full app3 process on each layer:
- SDS simplification on the layer (SDXL-based)
- SAM to detect internal structure (holes, shading regions, etc.)
- DiffVG optimization (structural + visual)

This gives app3's SVG quality for each semantic layer.
"""

import torch
import torch.nn.functional as F
from PIL import Image
import os
import numpy as np
from tqdm import tqdm

import pydiffvg

# V11-specific imports (fully independent from other app versions)
from utils.img_process_v11 import (
    init_svg_by_mask,
    layer_segmented_masks,
    get_struct_masks_by_area,
    color_fitting,
    add_visual_paths,
    remove_lowquality_paths,
    merge_path,
    sam_img_seq,
    init_struct_target_imgs,
    svg_to_img,
    rgba_to_rgb
)
from sds_image_simplicity_sdxl import sds_based_simplification


# ============================================================================
# V11-specific DiffVG initialization and optimizer (copied from main_v3)
# ============================================================================

def init_diffvg(device: torch.device,
                use_gpu: bool = torch.cuda.is_available(),
                print_timing: bool = False):
    """Initialize pydiffvg settings."""
    pydiffvg.set_device(device)
    pydiffvg.set_use_gpu(use_gpu)
    pydiffvg.set_print_timing(print_timing)


def init_optimizer(shapes, shape_groups,
                   is_train_stroke: bool = False,
                   is_train_color: bool = True,
                   is_opt_list=None,
                   lr_base: dict = None):
    """Initialize Adam optimizer for SVG parameters."""
    if is_opt_list is None:
        is_opt_list = []
    if lr_base is None:
        lr_base = {}

    points_vars = []
    color_vars = []
    stroke_width_vars = []
    stroke_color_vars = []

    if len(is_opt_list) == 0:
        is_opt_list = [1 for _ in range(len(shapes))]

    for i, path in enumerate(shapes):
        if is_opt_list[i] == 1:
            path.id = i
            path.points.requires_grad = True
            points_vars.append(path.points)
            if is_train_stroke:
                path.stroke_width.requires_grad = True
                stroke_width_vars.append(path.stroke_width)

    if is_train_color:
        for i, group in enumerate(shape_groups):
            if is_opt_list[i] == 1:
                group.fill_color.requires_grad = True
                color_vars.append(group.fill_color)
                if is_train_stroke:
                    group.stroke_color.requires_grad = True
                    stroke_color_vars.append(group.stroke_color)

    params = {'point': points_vars}
    if is_train_color:
        params['color'] = color_vars
    if is_train_stroke:
        params['stroke_width'] = stroke_width_vars
        params['stroke_color'] = stroke_color_vars

    learnable_params = [
        {'params': params[ki], 'lr': lr_base.get(ki, 0.01), '_id': str(ki)}
        for ki in sorted(params.keys())
    ]
    svg_optimizer = torch.optim.Adam(learnable_params, betas=(0.9, 0.9), eps=1e-6)
    return svg_optimizer


# ============================================================================
# V11-specific optimization functions (save every 10 iterations for disk space)
# ============================================================================

def exclude_loss(raster_img, scale=1):
    """Exclusion loss to prevent path overlap."""
    img = F.relu(178/255 - raster_img)
    loss = torch.sum(img) * scale
    return loss


def svg_optimize_img_struct_v11(device, shapes, shape_groups,
                                target_img: np.ndarray,
                                layerd_struct_masks: list,
                                file_save_path: str,
                                train_conf: dict,
                                base_lr_conf: dict):
    """
    V11 structural optimization - saves every 10 iterations to reduce disk usage.
    """
    struct_target_imgs, struct_colors_list = init_struct_target_imgs(layerd_struct_masks)
    struct_target_imgs = [x.to(device) for x in struct_target_imgs]
    struct_shape_groups_list = []
    for struct_colors in struct_colors_list:
        struct_shape_groups = []
        for i, color in enumerate(struct_colors):
            path_group = pydiffvg.ShapeGroup(
                shape_ids=torch.LongTensor([i]),
                fill_color=torch.FloatTensor(color + [1]),
                stroke_color=torch.FloatTensor([0, 0, 0, 1])
            )
            struct_shape_groups.append(path_group)
        struct_shape_groups_list.append(struct_shape_groups)

    transparent_shape_groups = []
    for i in range(len(shapes)):
        path_group = pydiffvg.ShapeGroup(
            shape_ids=torch.LongTensor([i]),
            fill_color=torch.FloatTensor([0, 0, 0, 0.3]),
            stroke_color=torch.FloatTensor([0, 0, 0, 0.3])
        )
        transparent_shape_groups.append(path_group)

    black_bg = torch.tensor([0., 0., 0.], requires_grad=False, device=device)
    white_bg = torch.tensor([1., 1., 1.], requires_grad=False, device=device)

    img_height, img_width = target_img.shape[:2]
    target_img_t = torch.tensor(target_img, device=device) / 255
    target_img_t = target_img_t.permute(2, 0, 1)

    svg_optimizer = init_optimizer(shapes, shape_groups,
                                   train_conf["is_train_stroke"],
                                   train_conf["is_train_struct_color"],
                                   lr_base=base_lr_conf)

    num_iters = train_conf["struct_opt_num_iters"]
    save_interval = 10  # V11: Save every 10 iterations

    with tqdm(total=num_iters, desc="Struct opt", unit="iter") as pbar:
        for i in range(num_iters):
            loss_struct = 0
            loss_exclude = 0
            shape_index = 0
            for struct_i, struct_target_img in enumerate(struct_target_imgs):
                shape_index += len(layerd_struct_masks[struct_i])
                struct_img = svg_to_img(img_width, img_height,
                                        shapes[shape_index - len(layerd_struct_masks[struct_i]):shape_index],
                                        struct_shape_groups_list[struct_i],
                                        device)
                struct_img = rgba_to_rgb(struct_img, device, black_bg)
                loss_struct += F.mse_loss(struct_img, struct_target_img)

                transparent_img = svg_to_img(img_width, img_height,
                                             shapes[shape_index - len(layerd_struct_masks[struct_i]):shape_index],
                                             transparent_shape_groups[:len(layerd_struct_masks[struct_i])],
                                             device)
                transparent_img = rgba_to_rgb(transparent_img, device, white_bg)
                loss_exclude += exclude_loss(transparent_img, scale=2e-7)

            img = svg_to_img(img_width, img_height, shapes, shape_groups, device)
            img = rgba_to_rgb(img, device, white_bg)
            loss_mse = F.mse_loss(img, target_img_t)

            loss = loss_mse * 0.02 + loss_exclude + loss_struct
            svg_optimizer.zero_grad()
            loss.backward()
            svg_optimizer.step()

            # Save every 10 iterations + final
            if i % save_interval == 0 or i == num_iters - 1:
                pydiffvg.save_svg(f"{file_save_path}/{i}.svg",
                                  img_width, img_height, shapes, shape_groups)
            pbar.update(1)
    return shapes, shape_groups


def svg_optimize_img_visual_v11(device, shapes, shape_groups,
                                target_img: np.ndarray,
                                file_save_path: str,
                                is_opt_list,
                                train_conf: dict,
                                base_lr_conf: dict,
                                count: int = 0,
                                struct_path_num: int = 0,
                                is_path_merging_phase: bool = False):
    """
    V11 visual optimization - saves every 10 iterations to reduce disk usage.
    """
    img_height, img_width = target_img.shape[:2]
    target_img_t = torch.tensor(target_img, device=device) / 255
    target_img_t = target_img_t.permute(2, 0, 1)

    transparent_shape_groups = []
    for i in range(len(shapes) - struct_path_num):
        path_group = pydiffvg.ShapeGroup(
            shape_ids=torch.LongTensor([i]),
            fill_color=torch.FloatTensor([0, 0, 0, 0.3]),
            stroke_color=torch.FloatTensor([0, 0, 0, 0.3])
        )
        transparent_shape_groups.append(path_group)

    svg_optimizer = init_optimizer(shapes, shape_groups,
                                   train_conf["is_train_stroke"],
                                   train_conf["is_train_visual_color"],
                                   is_opt_list,
                                   lr_base=base_lr_conf)
    num_iters = train_conf["visual_opt_num_iters"]
    if is_path_merging_phase:
        num_iters = 50

    save_interval = 10  # V11: Save every 10 iterations

    with tqdm(total=num_iters, desc="Visual opt", unit="iter") as pbar:
        for i in range(num_iters):
            img = svg_to_img(img_width, img_height, shapes, shape_groups, device)
            img = rgba_to_rgb(img, device)
            loss = F.mse_loss(img, target_img_t)
            svg_optimizer.zero_grad()
            loss.backward()
            svg_optimizer.step()

            # Save every 10 iterations + final
            # if i % save_interval == 0 or i == num_iters - 1:
            #     pydiffvg.save_svg(f"{file_save_path}/{count}.svg",
            #                       img_width, img_height, shapes, shape_groups)
            count += 1
            pbar.update(1)
    return shapes, shape_groups, count


def vectorize_single_layer(
    layer_img: np.ndarray,
    layer_mask: np.ndarray,
    device: torch.device,
    args,
    output_dir: str,
    layer_id: int,
    run_sds: bool = True
) -> str:
    """
    Vectorize a single layer using DiffVG (app3's quality pipeline).

    This pipeline runs the full app3 process:
    1. SDS simplification on the layer image (SDXL-based)
    2. SAM to detect internal structure (holes, shading, etc.)
    3. DiffVG structural optimization
    4. Color fitting
    5. Visual refinement
    6. Save final SVG

    Args:
        layer_img: Layer image (object on white background), shape (H, W, 3)
        layer_mask: Binary mask (0 or 255), shape (H, W)
        device: torch device
        args: Config args from load_config
        output_dir: Directory to save outputs
        layer_id: Layer identifier for naming
        run_sds: Whether to run SDS simplification (default True for app3 quality)

    Returns:
        Path to the generated SVG file
    """
    os.makedirs(output_dir, exist_ok=True)

    H, W = layer_img.shape[:2]

    # Get VTracer parameters
    vtracer_enable = getattr(args, 'vtracer_enable', True)
    staircase_area = getattr(args, 'staircase_area', 1.5)
    corner_angle = getattr(args, 'corner_angle', 135.0)
    simplify_error = getattr(args, 'simplify_error', 2.0)
    smooth_iterations = getattr(args, 'smooth_iterations', 0)

    # Create subdirectories
    simp_img_seq_save_path = os.path.join(output_dir, "simplified_image_sequence")
    struct_svgs_save_path = os.path.join(output_dir, "struct_svgs")
    visual_svgs_save_path = os.path.join(output_dir, "visual_svgs")

    for path in [simp_img_seq_save_path, struct_svgs_save_path, visual_svgs_save_path]:
        os.makedirs(path, exist_ok=True)

    # Save input for reference
    layer_img_path = os.path.join(output_dir, "input.png")
    Image.fromarray(layer_img).save(layer_img_path)
    mask_path = os.path.join(output_dir, "mask.png")
    Image.fromarray(layer_mask).save(mask_path)

    # ========================================
    # Step 1: SDS Simplification (SDXL-based, same as app3)
    # ========================================
    if run_sds:
        print(f"      [Layer {layer_id}] Running SDS simplification (SDXL)...")
        all_simp_img_seq_save_path = os.path.join(output_dir, "all_simplified_image_sequence")
        os.makedirs(all_simp_img_seq_save_path, exist_ok=True)

        simp_img_seq = sds_based_simplification(
            device,
            layer_img_path,
            args.simp_img_seq_indexs,
            simp_img_seq_save_path,
            all_simp_img_seq_save_path
        )
        target_img = simp_img_seq[0]
        # Clear CUDA cache after SDS to free VRAM for DiffVG optimization
        torch.cuda.empty_cache()
    else:
        # Skip SDS only if explicitly disabled (for debugging/speed)
        print(f"      [Layer {layer_id}] SDS disabled - using original image (lower quality)")
        target_img = layer_img.copy()
        Image.fromarray(target_img).save(os.path.join(simp_img_seq_save_path, "0.png"))

    img_height, img_width = target_img.shape[:2]

    # ========================================
    # Step 2: Create masks from provided layer mask
    # ========================================
    print(f"      [Layer {layer_id}] Creating path structure from mask...")

    # Convert mask to the format expected by the pipeline
    # The mask from Step 1 is the full layer mask
    # We use it directly as the primary structural mask
    mask_bool = layer_mask > 127

    # Create a simple layered structure with just this mask
    # layer_segmented_masks returns a 2-level list: [[mask, ...], [mask, ...], ...]
    # Since we already have the segmented layer, we just use a single-layer list.

    # CRITICAL FIX: Force background pixels in target_img to be PURE WHITE (255, 255, 255)
    # This prevents SDXL VAE noise in the background from being vectorized as opacity.
    target_img_bg_white = target_img.copy()
    bg_mask = ~mask_bool
    target_img_bg_white[bg_mask] = [255, 255, 255]
    target_img = target_img_bg_white

    # Update target_img on disk for debugging reference
    Image.fromarray(target_img).save(os.path.join(output_dir, "input_clean_bg.png"))
    print(f"      [Layer {layer_id}] Applied clean white background mask to target image")

    primary_mask = layer_mask.copy()

    # Limit number of paths (match app3 quality settings)
    max_paths = getattr(args, 'max_path_num_limit', 256)

    # ========================================
    # Run SAM on the layer to detect internal structure (like app3)
    # This captures detail like holes, shading regions, etc.
    # Uses sam_internal config (more sensitive) if available, otherwise falls back to sam
    # ========================================
    print(f"      [Layer {layer_id}] Running SAM for internal structure...")
    masks_save_path = os.path.join(output_dir, "masks")
    os.makedirs(masks_save_path, exist_ok=True)

    # Use sam_internal config for finer detail capture (smaller min_mask_region_area)
    # Falls back to sam config if sam_internal not defined
    sam_internal_conf = getattr(args, 'sam_internal', None)
    sam_conf = sam_internal_conf if sam_internal_conf is not None else getattr(args, 'sam', None)
    if sam_internal_conf is not None:
        print(f"      [Layer {layer_id}] Using sam_internal config (min_area={getattr(sam_internal_conf, 'min_mask_region_area', 'default')})")
    masks = sam_img_seq(device, [target_img], masks_save_path, sam_conf)

    # Clear CUDA cache after SAM to free VRAM
    torch.cuda.empty_cache()

    # Filter SAM masks: only keep those that overlap significantly with our layer
    # (SAM might detect the white background as a separate mask)
    primary_mask_bool = primary_mask > 127
    primary_area = primary_mask_bool.sum()
    filtered_masks = []
    for m in masks:
        m_bool = m > 127
        overlap = np.logical_and(m_bool, primary_mask_bool).sum()
        m_area = m_bool.sum()

        # Skip if mask is empty or doesn't overlap enough with layer
        if m_area == 0 or overlap / m_area <= 0.5:
            continue

        # Skip if mask is too large (likely the full boundary or inverted mask)
        # Internal structure masks should be smaller than the primary mask
        if m_area > primary_area * 0.95:
            continue

        # Skip if mask covers most of the image (inverted/background mask)
        total_pixels = m.shape[0] * m.shape[1]
        if m_area > total_pixels * 0.5:
            continue

        filtered_masks.append(m)

    if len(filtered_masks) > 1:
        # Use primary_mask as the base, filtered SAM masks as internal structure
        layerd_struct_masks = layer_segmented_masks([[primary_mask]], filtered_masks[1:])
        print(f"      [Layer {layer_id}] SAM found {len(filtered_masks)} masks inside layer")
    else:
        # Fallback: no internal structure detected
        print(f"      [Layer {layer_id}] SAM found no internal structure, using outer mask")
        layerd_struct_masks = [[primary_mask]]

    layerd_struct_masks = get_struct_masks_by_area(
        layerd_struct_masks,
        int(max_paths * 0.4)
    )

    # ========================================
    # Step 3: Initialize SVG paths from mask
    # ========================================
    print(f"      [Layer {layer_id}] Initializing Bezier paths...")

    shapes, shape_groups = init_svg_by_mask(
        layerd_struct_masks,
        target_img,
        args.approxpolydp_epsilon,
        use_vtracer_enhancement=vtracer_enable,
        staircase_area=staircase_area,
        corner_angle=corner_angle,
        simplify_error=simplify_error,
        smooth_iterations=smooth_iterations
    )

    if len(shapes) == 0:
        print(f"      [Layer {layer_id}] Warning: No paths created from mask")
        # Create a simple rectangle as fallback
        return None

    print(f"      [Layer {layer_id}] Created {len(shapes)} initial paths")

    # ========================================
    # Step 4: Structural Optimization
    # ========================================
    print(f"      [Layer {layer_id}] Structural optimization...")

    shapes, shape_groups = svg_optimize_img_struct_v11(
        device,
        shapes,
        shape_groups,
        target_img,
        layerd_struct_masks,
        struct_svgs_save_path,
        args.train,
        args.base_lr
    )

    # ========================================
    # Step 5: Color Fitting
    # ========================================
    print(f"      [Layer {layer_id}] Color fitting...")

    if args.color_fitting_type == "dominan":
        shape_groups, target_img_cluster = color_fitting(
            shape_groups,
            target_img,
            layerd_struct_masks,
            args.is_cluster_target_img,
            args.kmeas_k
        )
        if args.is_cluster_target_img:
            Image.fromarray(target_img_cluster).save(os.path.join(output_dir, "cluster_img.png"))
    else:
        target_img_cluster = target_img

    # Save color-adjusted SVG
    pydiffvg.save_svg(
        os.path.join(output_dir, "color-adjusted.svg"),
        img_height, img_width, shapes, shape_groups
    )

    # ========================================
    # Step 6: Visual Refinement
    # ========================================
    print(f"      [Layer {layer_id}] Visual refinement...")

    pseudo_struct_masks = [mask for sublist in layerd_struct_masks for mask in sublist]
    is_opt_list = []
    count = 0
    struct_path_num = len(shapes)

    visual_iters = getattr(args, 'add_visual_path_num_iters', 3)

    for i in range(visual_iters):
        iter_save_path = os.path.join(visual_svgs_save_path, f"{i}_add_paths")
        os.makedirs(iter_save_path, exist_ok=True)

        if i == visual_iters - 1:
            remaining_path_num = max_paths - len(shapes)
        else:
            remaining_path_num = int((max_paths - len(shapes)) * 0.6)

        shapes, shape_groups, pseudo_struct_masks, is_opt_list, struct_path_num = add_visual_paths(
            shapes, shape_groups, device,
            struct_path_num,
            target_img_cluster,
            pseudo_struct_masks,
            is_opt_list,
            epsilon=args.approxpolydp_epsilon,
            N=remaining_path_num,
            use_vtracer_enhancement=vtracer_enable,
            staircase_area=staircase_area,
            corner_angle=corner_angle,
            simplify_error=simplify_error,
            smooth_iterations=smooth_iterations
        )

        if struct_path_num == -1:
            print(f"      [Layer {layer_id}] No more paths to add")
            break

        shapes, shape_groups, count = svg_optimize_img_visual_v11(
            device, shapes, shape_groups,
            target_img,
            iter_save_path,
            is_opt_list,
            args.train,
            args.base_lr,
            count,
            struct_path_num
        )

        # Skip merge/cleanup on last iteration (match v3 behavior)
        if i == visual_iters - 1:
            break

        # Remove low quality paths
        shapes, shape_groups = remove_lowquality_paths(
            shapes, shape_groups, device,
            img_width, img_height,
            visual_difference_threshold=args.paths_remove_visual_threshold,
            struct_path_num=struct_path_num
        )

        # Merge paths with similar colors (re-enabled feature)
        # Clear CUDA cache before merge_path to avoid OOM
        torch.cuda.empty_cache()

        print(f"      [Layer {layer_id}] Path merging...")
        merge_save_path = os.path.join(visual_svgs_save_path, f"{i}_merge_paths")
        os.makedirs(merge_save_path, exist_ok=True)

        try:
            shapes, shape_groups, pseudo_struct_masks, is_opt_list, struct_path_num = merge_path(
                shapes, shape_groups, device,
                img_width, img_height,
                struct_path_num,
                pseudo_struct_masks,
                is_opt_list,
                color_threshold=args.paths_merge_color_threshold,
                overlapping_area_threshold=args.paths_merge_distance_threshold
            )

            # Re-optimize after merging
            shapes, shape_groups, count = svg_optimize_img_visual_v11(
                device, shapes, shape_groups,
                target_img,
                merge_save_path,
                is_opt_list,
                args.train,
                args.base_lr,
                count,
                struct_path_num,
                is_path_merging_phase=True
            )
        except torch.cuda.OutOfMemoryError:
            print(f"      [Layer {layer_id}] CUDA OOM during merge - skipping merge phase, keeping current paths")
            torch.cuda.empty_cache()
            # Continue without merging - paths from before merge are still valid

    # ========================================
    # Step 7: Save Final SVG
    # ========================================
    final_svg_path = os.path.join(output_dir, "final.svg")
    pydiffvg.save_svg(final_svg_path, img_height, img_width, shapes, shape_groups)

    print(f"      [Layer {layer_id}] Final SVG saved: {final_svg_path} ({len(shapes)} paths)")

    return final_svg_path
