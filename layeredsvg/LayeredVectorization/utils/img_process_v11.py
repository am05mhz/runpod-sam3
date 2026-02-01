"""
V11 Image Processing - Independent version for app11

Enhanced Image Processing with VTracer-inspired techniques:
- Staircase Removal (collinearity test)
- Corner Detection (preserve sharp angles)
- Error-Penalized Simplification
- 4-Point Subdivision Smoothing
- Improved color fitting with fallback for small mask regions

This is the v11-specific version. Changes here won't affect other app versions.
"""

import cv2
# SAM imports are done lazily in sam_image() to support both standard SAM and HQ-SAM
import numpy as np
from PIL import Image
import os
import glob
from torchvision import transforms
import torch
import pydiffvg
from collections import Counter
from scipy.spatial.distance import cdist
from typing import Union,List
from tqdm import tqdm
import time
import random
from scipy.spatial import cKDTree
import math

# ============================================================================
# VTracer-inspired Path Enhancement Functions
# ============================================================================

def triangle_area(p1, p2, p3):
    """Calculate triangle area using cross product (for collinearity test)."""
    return abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1])) / 2.0

def point_distance(p1, p2):
    """Euclidean distance between two points."""
    return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

def calculate_angle(p1, p2, p3):
    """
    Calculate angle at p2 formed by p1-p2-p3 in degrees.
    Returns angle in range [0, 180].
    """
    v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
    v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])

    len1 = np.linalg.norm(v1)
    len2 = np.linalg.norm(v2)

    if len1 == 0 or len2 == 0:
        return 180.0

    cos_angle = np.dot(v1, v2) / (len1 * len2)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle = math.degrees(math.acos(cos_angle))

    return angle

def remove_staircase_artifacts(points, area_threshold=1.5):
    """
    VTracer-style Staircase Removal.
    Removes intermediate points that form near-collinear sequences
    (staircase patterns from pixel grids).

    Uses triangle area test: if three consecutive points form a very
    small triangle, the middle point can be removed.

    Args:
        points: List of (x, y) tuples or numpy array
        area_threshold: Maximum triangle area to consider collinear (default 1.5 sq pixels)

    Returns:
        Cleaned points list
    """
    if len(points) < 3:
        return points

    points = [tuple(p) if isinstance(p, np.ndarray) else p for p in points]
    result = [points[0]]

    i = 1
    while i < len(points) - 1:
        p_prev = result[-1]
        p_curr = points[i]
        p_next = points[i + 1]

        area = triangle_area(p_prev, p_curr, p_next)

        # If triangle area is large enough, keep the point (not collinear)
        if area > area_threshold:
            result.append(p_curr)
        # Otherwise skip it (collinear/staircase artifact)

        i += 1

    # Always keep the last point
    result.append(points[-1])

    return result

def detect_corners(points, angle_threshold=135.0, min_segment_length=3.0):
    """
    VTracer-style Corner Detection.
    Identifies points that form sharp corners which should be preserved.

    Args:
        points: List of (x, y) tuples
        angle_threshold: Angles below this (sharper turns) are marked as corners (default 135°)
        min_segment_length: Minimum distance between points to consider for angle calculation

    Returns:
        Set of indices that are corners
    """
    if len(points) < 3:
        return set()

    corners = set()
    corners.add(0)  # First point is always a corner
    corners.add(len(points) - 1)  # Last point is always a corner

    for i in range(1, len(points) - 1):
        p_prev = points[i - 1]
        p_curr = points[i]
        p_next = points[i + 1]

        # Check segment lengths
        dist_prev = point_distance(p_prev, p_curr)
        dist_next = point_distance(p_curr, p_next)

        if dist_prev < min_segment_length or dist_next < min_segment_length:
            continue

        angle = calculate_angle(p_prev, p_curr, p_next)

        if angle < angle_threshold:
            corners.add(i)

    return corners

def error_penalized_simplification(points, max_error=2.0, corner_indices=None):
    """
    VTracer-style Error-Penalized Simplification.
    Similar to Douglas-Peucker but preserves marked corners and uses
    perpendicular distance error metric.

    Args:
        points: List of (x, y) tuples
        max_error: Maximum perpendicular distance error allowed
        corner_indices: Set of indices that must be preserved (corners)

    Returns:
        Simplified points list
    """
    if len(points) < 3:
        return points

    if corner_indices is None:
        corner_indices = set()

    def perpendicular_distance(point, line_start, line_end):
        """Calculate perpendicular distance from point to line segment."""
        if line_start == line_end:
            return point_distance(point, line_start)

        line_len = point_distance(line_start, line_end)
        if line_len == 0:
            return point_distance(point, line_start)

        # Calculate perpendicular distance using cross product
        dx = line_end[0] - line_start[0]
        dy = line_end[1] - line_start[1]

        # Vector from line_start to point
        px = point[0] - line_start[0]
        py = point[1] - line_start[1]

        # Cross product magnitude / line length
        dist = abs(dx * py - dy * px) / line_len
        return dist

    def simplify_segment(start_idx, end_idx):
        """Recursively simplify a segment."""
        if end_idx - start_idx < 2:
            return [points[start_idx]]

        # Find the point with maximum distance from the line
        max_dist = 0
        max_idx = start_idx

        for i in range(start_idx + 1, end_idx):
            # If this is a corner, we must keep it
            if i in corner_indices:
                # Treat corners as having infinite distance
                return (simplify_segment(start_idx, i) +
                       simplify_segment(i, end_idx))

            dist = perpendicular_distance(points[i], points[start_idx], points[end_idx])
            if dist > max_dist:
                max_dist = dist
                max_idx = i

        if max_dist > max_error:
            # Recursively simplify both segments
            left = simplify_segment(start_idx, max_idx)
            right = simplify_segment(max_idx, end_idx)
            return left + right
        else:
            # All points are within tolerance, just keep the start
            return [points[start_idx]]

    result = simplify_segment(0, len(points) - 1)
    result.append(points[-1])

    return result

def four_point_subdivision_smooth(points, iterations=1, corner_indices=None):
    """
    VTracer-style 4-Point Subdivision Smoothing.
    Smooths curves by subdividing and averaging, while preserving corners.

    The 4-point scheme uses neighboring points to calculate smooth positions:
    new_point = (-1/16 * p0 + 9/16 * p1 + 9/16 * p2 - 1/16 * p3)

    Args:
        points: List of (x, y) tuples
        iterations: Number of smoothing iterations
        corner_indices: Set of indices that should not be smoothed

    Returns:
        Smoothed points list
    """
    if len(points) < 4 or iterations < 1:
        return points

    if corner_indices is None:
        corner_indices = set()

    result = list(points)

    for _ in range(iterations):
        new_points = []
        new_corners = set()

        for i in range(len(result)):
            # Always keep the original point
            new_points.append(result[i])
            if i in corner_indices:
                new_corners.add(len(new_points) - 1)

            # Add midpoint (subdivide)
            if i < len(result) - 1:
                # Get 4 points for subdivision
                p0 = result[max(0, i - 1)]
                p1 = result[i]
                p2 = result[min(len(result) - 1, i + 1)]
                p3 = result[min(len(result) - 1, i + 2)]

                # If either endpoint is a corner, just use simple midpoint
                if i in corner_indices or (i + 1) in corner_indices:
                    mid_x = (p1[0] + p2[0]) / 2
                    mid_y = (p1[1] + p2[1]) / 2
                else:
                    # 4-point subdivision formula
                    mid_x = (-1/16 * p0[0] + 9/16 * p1[0] + 9/16 * p2[0] - 1/16 * p3[0])
                    mid_y = (-1/16 * p0[1] + 9/16 * p1[1] + 9/16 * p2[1] - 1/16 * p3[1])

                new_points.append((mid_x, mid_y))

        result = new_points
        corner_indices = new_corners

    return result

def vtracer_enhanced_simplification(contour, epsilon=2.0,
                                     staircase_area=1.5,
                                     corner_angle=135.0,
                                     simplify_error=2.0,
                                     smooth_iterations=0):
    """
    Apply VTracer-inspired enhancement pipeline to a contour.

    Pipeline:
    1. Staircase Removal - remove pixel grid artifacts
    2. Corner Detection - identify sharp angles to preserve
    3. Error-Penalized Simplification - reduce points while preserving corners
    4. Optional: 4-Point Subdivision Smoothing

    Args:
        contour: OpenCV contour (Nx1x2 array)
        epsilon: Fallback to approxPolyDP if enhanced pipeline produces too few points
        staircase_area: Threshold for staircase removal
        corner_angle: Angle threshold for corner detection (degrees)
        simplify_error: Max error for simplification
        smooth_iterations: Number of smoothing passes (0 to disable)

    Returns:
        Simplified contour as Nx1x2 numpy array
    """
    # Convert contour to list of tuples
    points = [(int(p[0][0]), int(p[0][1])) for p in contour]

    if len(points) < 4:
        return contour

    # Step 1: Remove staircase artifacts
    points = remove_staircase_artifacts(points, area_threshold=staircase_area)

    if len(points) < 4:
        # Fallback to original approxPolyDP
        return cv2.approxPolyDP(contour, epsilon, closed=True)

    # Step 2: Detect corners
    corners = detect_corners(points, angle_threshold=corner_angle)

    # Step 3: Error-penalized simplification
    points = error_penalized_simplification(points, max_error=simplify_error, corner_indices=corners)

    if len(points) < 3:
        # Fallback to original approxPolyDP
        return cv2.approxPolyDP(contour, epsilon, closed=True)

    # Re-detect corners after simplification
    corners = detect_corners(points, angle_threshold=corner_angle)

    # Step 4: Optional smoothing (disabled by default to preserve detail)
    if smooth_iterations > 0:
        points = four_point_subdivision_smooth(points, iterations=smooth_iterations, corner_indices=corners)

    # Convert back to OpenCV contour format
    result = np.array([[[p[0], p[1]]] for p in points], dtype=np.int32)

    return result


# ============================================================================
# Original functions (unchanged) + enhanced version
# ============================================================================

def svg_to_img(width: int, height: int, shapes, shape_groups, device):
    scene_args = pydiffvg.RenderFunction.serialize_scene(
        width, height, shapes, shape_groups
        )
    _render = pydiffvg.RenderFunction.apply
    img = _render(width,  # width
                height,  # height
                2,  # num_samples_x
                2,  # num_samples_y
                0,  # seed
                None,
                *scene_args)
    img = img.to(device)
    return img

def rgba_to_rgb(img: torch.Tensor, device, para_bg=None):
    if para_bg is None:
        para_bg = torch.tensor([1., 1., 1.], requires_grad=False, device=device)
    img = img[:, :, 3:4] * img[:, :, :3] + para_bg * (1 - img[:, :, 3:4])
    img = img.permute(2, 0, 1)
    return img

def sam_img_seq(device, simp_img_seq: list, masks_save_path: str = "-1", sam_conf: dict = None) -> list:
    def mask_preprocessing(masks):
        prepro_masks = []
        for mask in masks:
            filled_mask = filling_mask_holes(mask)
            num_labels, labels = cv2.connectedComponents(filled_mask)
            for i in range(1, num_labels):
                single_region = np.zeros_like(mask)
                single_region[labels == i] = 255
                prepro_masks.append(single_region)
        return prepro_masks

    def sorted_masks(masks):
        area_list=[]
        for mask in masks:
            area = cv2.countNonZero(mask)
            area_list.append(area)
        sorted_indices = sorted(range(len(area_list)), key=lambda k: area_list[k])
        sorted_indices.reverse()
        sorted_masks = []
        for i in sorted_indices:
            sorted_masks.append(masks[i])
        return sorted_masks

    all_masks = []
    with tqdm(total=len(simp_img_seq), desc="Processing image", unit="value") as pbar:
        for i,simp_img in enumerate(simp_img_seq):
            if masks_save_path != "-1":
                masks_save_path_i = f"{masks_save_path}/{i+1}"
                os.makedirs(masks_save_path_i,exist_ok=True)
                masks = sam_image(device, simp_img, masks_save_path_i, sam_conf)
            else:
                masks = sam_image(device, simp_img, "-1", sam_conf)
            masks = mask_preprocessing(masks)
            all_masks.append(sorted_masks(masks))
            pbar.update(1)
    all_masks.reverse()
    all_masks = [mask for sublist in all_masks for mask in sublist]
    return all_masks

def sam_image(device, image: Union[str, np.ndarray], masks_save_path: str = "-1", sam_conf: dict = None) -> list:
    sam_checkpoint = sam_conf["sam_checkpoint"]
    model_type = sam_conf["model_type"]
    use_samhq = sam_conf.get("use_samhq", False)

    if isinstance(image, str):
        image = cv2.imread(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Use HQ-SAM if configured
    if use_samhq:
        try:
            from segment_anything_hq import sam_model_registry as hq_sam_registry
            from segment_anything_hq import SamAutomaticMaskGenerator as HQSamAutomaticMaskGenerator

            print(f"Loading HQ-SAM for internal structure detection...")
            sam = hq_sam_registry[model_type](checkpoint=sam_checkpoint)
            sam.to(device=device)

            mask_generator_2 = HQSamAutomaticMaskGenerator(
                model=sam,
                points_per_side=sam_conf["points_per_side"],
                pred_iou_thresh=sam_conf["pred_iou_thresh"],
                stability_score_thresh=sam_conf["stability_score_thresh"],
                crop_n_layers=sam_conf["crop_n_layers"],
                crop_n_points_downscale_factor=sam_conf["crop_n_points_downscale_factor"],
                min_mask_region_area=sam_conf["min_mask_region_area"],
                box_nms_thresh=sam_conf["box_nms_thresh"],
            )
        except ImportError as e:
            print(f"HQ-SAM not available ({e}), falling back to standard SAM")
            use_samhq = False

    # Standard SAM fallback
    if not use_samhq:
        # IMPORTANT: Use build functions directly instead of sam_model_registry
        # to avoid potential registry pollution from segment_anything_hq
        from segment_anything import SamAutomaticMaskGenerator
        from segment_anything.build_sam import build_sam_vit_h, build_sam_vit_l, build_sam_vit_b

        # Build SAM model directly (bypasses potentially polluted registry)
        build_functions = {
            "vit_h": build_sam_vit_h,
            "vit_l": build_sam_vit_l,
            "vit_b": build_sam_vit_b,
        }

        if model_type not in build_functions:
            raise ValueError(f"Unknown SAM model type: {model_type}. Expected one of: {list(build_functions.keys())}")

        print(f"  Using SAM: {model_type} with checkpoint {sam_checkpoint}")
        sam = build_functions[model_type](checkpoint=sam_checkpoint)
        sam.to(device=device)

        mask_generator_2 = SamAutomaticMaskGenerator(
            model=sam,
            points_per_side=sam_conf["points_per_side"],
            pred_iou_thresh=sam_conf["pred_iou_thresh"],
            stability_score_thresh=sam_conf["stability_score_thresh"],
            crop_n_layers=sam_conf["crop_n_layers"],
            crop_n_points_downscale_factor=sam_conf["crop_n_points_downscale_factor"],
            min_mask_region_area=sam_conf["min_mask_region_area"],
            box_nms_thresh=sam_conf["box_nms_thresh"],
        )

    # Fix for torchvision batched_nms device mismatch bug with larger images (1024x1024)
    # This is a known issue where some tensors end up on CPU during NMS
    # Workaround: temporarily disable CUDA for the mask generation, then re-enable
    try:
        masks2 = mask_generator_2.generate(image)
    except RuntimeError as e:
        if "indices should be either on cpu" in str(e):
            print("SAM device mismatch detected, retrying with CPU fallback for NMS...")
            # Move SAM to CPU as fallback
            sam.to(device="cpu")
            if use_samhq:
                from segment_anything_hq import SamAutomaticMaskGenerator as HQSamAutomaticMaskGenerator
                mask_generator_2 = HQSamAutomaticMaskGenerator(
                    model=sam,
                    points_per_side=sam_conf["points_per_side"],
                    pred_iou_thresh=sam_conf["pred_iou_thresh"],
                    stability_score_thresh=sam_conf["stability_score_thresh"],
                    crop_n_layers=sam_conf["crop_n_layers"],
                    crop_n_points_downscale_factor=sam_conf["crop_n_points_downscale_factor"],
                    min_mask_region_area=sam_conf["min_mask_region_area"],
                    box_nms_thresh=sam_conf["box_nms_thresh"],
                )
            else:
                from segment_anything import SamAutomaticMaskGenerator
                mask_generator_2 = SamAutomaticMaskGenerator(
                    model=sam,
                    points_per_side=sam_conf["points_per_side"],
                    pred_iou_thresh=sam_conf["pred_iou_thresh"],
                    stability_score_thresh=sam_conf["stability_score_thresh"],
                    crop_n_layers=sam_conf["crop_n_layers"],
                    crop_n_points_downscale_factor=sam_conf["crop_n_points_downscale_factor"],
                    min_mask_region_area=sam_conf["min_mask_region_area"],
                    box_nms_thresh=sam_conf["box_nms_thresh"],
                )
            masks2 = mask_generator_2.generate(image)
        else:
            raise e
    masks = [np.full(image.shape[:2], 255, dtype=np.uint8)]
    for i,mask in enumerate(masks2):
        image = np.where(mask['segmentation'], 255, 0).astype(np.uint8)
        masks.append(image)
        if masks_save_path != "-1":
            image = Image.fromarray(image)
            image.save(f'{masks_save_path}/{i}.png')
    return masks

def filling_mask_holes(image):
    if isinstance(image,str):
        image = cv2.imread(mask, cv2.IMREAD_GRAYSCALE)
    flood_fill_image = image.copy()
    h, w = image.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    cv2.floodFill(flood_fill_image, mask, (0, 0), 255)
    flood_fill_inv = cv2.bitwise_not(flood_fill_image)
    filled_image = cv2.bitwise_or(image, flood_fill_inv)
    return filled_image

def layer_segmented_masks(layered_masks: list, unlayered_masks: list) -> list:
    while len(unlayered_masks)>0:
        found = False
        unlayered_mask = unlayered_masks.pop(0)
        for layer_i,layer in enumerate(layered_masks):
            if len(layer) > 0:
                for mask_i,mask in enumerate(layer):
                    area0 = cv2.countNonZero(mask)
                    area1 = cv2.countNonZero(unlayered_mask)
                    intersection_area = np.sum(mask.astype(np.float32)+unlayered_mask.astype(np.float32) == 510)
                    total_area = cv2.countNonZero(mask+unlayered_mask)
                    if intersection_area/total_area>0.70:  # 0.65
                        found = True
                        break
                    if intersection_area/area0>=0.5 and area1>area0:
                        found = True
                        break
                    if intersection_area/area1>=0.5 and area1<area0:
                        if layer_i == len(layered_masks)-1:
                            new_layer = [unlayered_mask]
                            layered_masks.append(new_layer)
                        break
                    if mask_i == len(layer)-1:
                        layered_masks[layer_i].append(unlayered_mask)
                        found = True
                        break

                if found:
                    break
    return layered_masks

def get_struct_masks_by_area(layerd_struct_masks: list, N: int):
    def get_top_n_indices(matrix,n):
        elements = []
        for i in range(len(matrix)):
            for j in range(len(matrix[i])):
                elements.append((matrix[i][j], i, j))
        sorted_elements = sorted(elements, key=lambda x: (-x[0], x[1], x[2]))
        indices = [(i, j) for val, i, j in sorted_elements[:n]]
        return indices

    if sum([len(x) for x in layerd_struct_masks])<= N:
        return layerd_struct_masks

    layer_area_list = []
    for struct_masks in layerd_struct_masks:
        area_list = []
        for struct_mask in struct_masks:
            area = np.sum(struct_mask.astype(np.float32))
            area_list.append(area)
        layer_area_list.append(area_list)
    indices = get_top_n_indices(layer_area_list,N)

    new_layerd_struct_masks = []
    for i,struct_masks in enumerate(layerd_struct_masks):
        new_struct_masks = []
        for j,struct_mask in enumerate(struct_masks):
            if (i,j) in indices:
                new_struct_masks.append(struct_mask)
        if len(new_struct_masks)>0:
            new_layerd_struct_masks.append(new_struct_masks)
    return new_layerd_struct_masks


def init_struct_target_imgs(layerd_struct_masks: list):
    struct_target_imgs = []
    struct_colors_list = []
    for i,monolayer_masks in enumerate(layerd_struct_masks):
        seg_image = 0
        mask_colors = []
        for mask in monolayer_masks:
            # Get mask dimensions dynamically (supports 512, 1024, etc.)
            mask_h, mask_w = mask.shape[:2]
            tensor0 = torch.zeros((3, mask_h, mask_w))
            color = []
            for channel in range(3):
                channel_value = 0.2 + (1 - 0.2) * torch.rand(1)
                tensor0[channel, :, :] = channel_value
                color.append(channel_value)
            mask_colors.append(color)

            # mask_image1 = Image.open(mask)
            mask_image1 = transforms.ToTensor()(Image.fromarray(mask))
            seg_image += tensor0*mask_image1
            seg_image = torch.clamp(seg_image, max=1.0)

        struct_target_imgs.append(seg_image)
        struct_colors_list.append(mask_colors)
    return struct_target_imgs, struct_colors_list


def find_closest_contours(contours):
    min_distance = float('inf')
    contours = [x[:, 0, :] for x in contours]
    for i, contour1 in enumerate(contours):
        for j, contour2 in enumerate(contours):
            if i >= j:
                continue
            distances = cdist(contour1, contour2)
            distance = np.min(distances)
            if distance < min_distance:
                min_distance = distance
                row_idx, col_idx = np.where(distances == np.min(min_distance))
                closest_point_1 = contour1[row_idx[0]]
                closest_point_2 = contour2[col_idx[0]]
    return closest_point_1, closest_point_2



def save_contours_as_single_png(contour1, contour2, filename):
    # 找出所有轮廓点的最小和最大坐标
    all_points = np.vstack((contour1, contour2))
    min_x = np.min(all_points[:, 0, 0])
    max_x = np.max(all_points[:, 0, 0])
    min_y = np.min(all_points[:, 0, 1])
    max_y = np.max(all_points[:, 0, 1])

    # 创建空白图像
    width = max_x - min_x + 1
    height = max_y - min_y + 1
    image = np.zeros((height, width, 3), dtype=np.uint8)

    # 调整轮廓坐标以适应图像
    contour1_adj = contour1 - [min_x, min_y]
    contour2_adj = contour2 - [min_x, min_y]

    # 绘制轮廓
    cv2.drawContours(image, [contour1_adj], -1, (255, 255, 255), 1)
    cv2.drawContours(image, [contour2_adj], -1, (255, 255, 255), 1)

    # 保存图像
    cv2.imwrite(filename, image)


def init_path_by_mask(mask: Union[np.ndarray,torch.Tensor], epsilon: int = 5,
                      use_vtracer_enhancement: bool = True,
                      staircase_area: float = 1.5,
                      corner_angle: float = 135.0,
                      simplify_error: float = 2.0,
                      smooth_iterations: int = 0):
    """
    Initialize SVG path from mask.

    Enhanced version with VTracer-inspired techniques.

    Args:
        mask: Binary mask
        epsilon: approxPolyDP epsilon (used as fallback)
        use_vtracer_enhancement: Enable VTracer techniques (default True)
        staircase_area: Threshold for staircase removal
        corner_angle: Angle threshold for corner detection
        simplify_error: Max error for simplification
        smooth_iterations: Number of smoothing passes
    """
    def insert_points_in_segments(points, num_interpolations=2):
        def interpolate_points(point1, point2, num_interpolations):
            x_values = np.linspace(point1[0], point2[0], num=num_interpolations + 2)[1:-1]
            y_values = np.linspace(point1[1], point2[1], num=num_interpolations + 2)[1:-1]
            interpolated_points = np.column_stack((x_values, y_values))
            return interpolated_points
        new_points = []
        for i in range(len(points) - 1):
            point1 = points[i]
            point2 = points[i + 1]
            new_points.append(point1)
            interpolated_points = interpolate_points(point1, point2, num_interpolations)
            new_points.extend(interpolated_points)
        new_points.append(points[-1])
        return np.array(new_points)

    if isinstance(mask,torch.Tensor):
        mask = (mask).detach().cpu().numpy()
        mask = (mask*255).astype(np.uint8)

    mask = np.where(mask > 1, 255, 0).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    while len(contours)>1:
        point_1, point_2 = find_closest_contours(contours)
        cv2.line(mask, point_1, point_2, (255), 3)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    contour = contours[0]

    # Apply VTracer enhancement or fallback to original approxPolyDP
    if use_vtracer_enhancement and len(contour) >= 4:
        simplified_contour = vtracer_enhanced_simplification(
            contour,
            epsilon=epsilon,
            staircase_area=staircase_area,
            corner_angle=corner_angle,
            simplify_error=simplify_error,
            smooth_iterations=smooth_iterations
        )
    else:
        simplified_contour = cv2.approxPolyDP(contour, epsilon, closed=True)

    if len(simplified_contour)==2:
        new_point = np.array([simplified_contour[0,0,0]+random.uniform(2, 5),simplified_contour[0,0,1]+random.uniform(2, 5)])
        simplified_contour = np.insert(simplified_contour, 1, new_point, axis=0)
    if len(simplified_contour)==1:
        new_point = np.array([simplified_contour[0,0,0]+3,simplified_contour[0,0,1]+3])
        simplified_contour = np.insert(simplified_contour, 1, new_point, axis=0)
        new_point = np.array([simplified_contour[0,0,0]-3,simplified_contour[0,0,1]+3])
        simplified_contour = np.insert(simplified_contour, 1, new_point, axis=0)
    simplified_contour = simplified_contour[:,0,:]
    points = np.vstack((simplified_contour, simplified_contour[0]))
    # Insert control points for cubic Bezier curves
    points = insert_points_in_segments(points, num_interpolations=2)
    points = torch.FloatTensor(points[:-1])
    num_control_points = [2] * len(simplified_contour)
    path = pydiffvg.Path(
                        num_control_points=torch.LongTensor(num_control_points),
                        points=points,
                        stroke_width=torch.tensor(0.0),
                        is_closed=True
                    )
    return path

def get_mean_color(image: np.ndarray, mask: np.ndarray):
        masked_pixels = image[mask > 0]
        if len(masked_pixels) > 0:
            average_color = tuple(np.mean(masked_pixels, axis=0, dtype=int))
        else:
            average_color = (0, 0, 0)
        return average_color

def init_svg_by_mask(layerd_masks: list, target_img: np.ndarray, epsilon: int = 5,
                     use_vtracer_enhancement: bool = True,
                     staircase_area: float = 1.5,
                     corner_angle: float = 135.0,
                     simplify_error: float = 2.0,
                     smooth_iterations: int = 0):
    """
    Initialize SVG from layered masks.

    Enhanced version with VTracer-inspired techniques.
    """
    shapes =[]
    shape_groups=[]
    for masks in layerd_masks:
        for mask in masks:
            path = init_path_by_mask(mask, epsilon=epsilon,
                                     use_vtracer_enhancement=use_vtracer_enhancement,
                                     staircase_area=staircase_area,
                                     corner_angle=corner_angle,
                                     simplify_error=simplify_error,
                                     smooth_iterations=smooth_iterations)
            mean_color = get_mean_color(target_img,mask)
            path_group = pydiffvg.ShapeGroup(
                                    shape_ids=torch.LongTensor([len(shapes)]),
                                    fill_color=torch.tensor(list(mean_color)+[255])/255,
                                    stroke_color=torch.FloatTensor([0,0,0,1])
                                )
            shapes.append(path)
            shape_groups.append(path_group)
    return shapes,shape_groups


def color_fitting(shape_groups, target_img: np.ndarray, layerd_struct_masks: list, is_cluster: bool, k: int = 30):
    def combine_masks(masks):
        combine_masks = masks[0].copy()
        if len(masks)>1:
            for mask in masks[1:]:
                combine_masks+=mask
        return combine_masks
    if is_cluster:
        data = target_img.reshape((-1, 3))
        data = np.float32(data)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, centers = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        centers = np.uint8(centers)
        segmented_data = centers[labels.flatten()]
        target_img = segmented_data.reshape((target_img.shape))

    most_common_color_list =[]
    for i,masks in enumerate(layerd_struct_masks):
        if i<=len(layerd_struct_masks)-2:
            masks_combine = combine_masks(layerd_struct_masks[i+1])
        else:
            masks_combine = np.zeros_like(masks[0])
        for j,mask_orig in enumerate(masks):
            # Subtract upper layer masks to avoid double-counting
            mask = mask_orig - masks_combine
            mask_indices = np.where(mask == 255)
            colors = target_img[mask_indices]

            # If subtraction left too few pixels (<100), use original mask instead
            # This prevents color sampling from wrong regions when masks overlap heavily
            min_pixels = 100
            if len(colors) < min_pixels:
                mask_indices_orig = np.where(mask_orig == 255)
                colors = target_img[mask_indices_orig]

            color_tuples = [tuple(color) for color in colors]
            color_counts = Counter(color_tuples)
            if color_counts:
                most_common_color = color_counts.most_common(1)[0][0]
            else:
                # Fallback: compute mean color from original mask if still no colors
                mask_indices_orig = np.where(mask_orig == 255)
                fallback_colors = target_img[mask_indices_orig]
                if len(fallback_colors) > 0:
                    most_common_color = tuple(np.mean(fallback_colors, axis=0).astype(int))
                else:
                    most_common_color = (0,0,0)
            most_common_color_list.append(list(most_common_color))
    for i,shape_group in enumerate(shape_groups):
        shape_group.fill_color=torch.FloatTensor(most_common_color_list[i]+[255])/255
    return shape_groups,target_img

def select_mask_by_conn_area(pred: np.ndarray, gt: np.ndarray, N: int = -1):
    map = ((pred - gt)**2).sum(0)

    quantile_interval=139
    # Lowered threshold from 0.025 to 0.008 to catch subtle color differences
    # 0.025 = ~9% color diff per channel ignored (too much for internal detail)
    # 0.008 = ~5% color diff per channel ignored (catches holes, shading better)
    nodiff_thres=0.008
    map[map < nodiff_thres] = 0

    quantile_interval = np.linspace(0., 1., quantile_interval)
    quantized_interval = np.quantile(map, quantile_interval)
    # remove redundant
    quantized_interval = np.unique(quantized_interval)
    quantized_interval = sorted(quantized_interval[1:-1])
    map = np.digitize(map, quantized_interval, right=False)
    map = np.clip(map, 0, 255).astype(np.uint8)

    csize_list = []
    component_list = []
    for idi in np.unique(map):
        if idi==0:
            continue
        _, component, cstats, _ = cv2.connectedComponentsWithStats(
            (map==idi).astype(np.uint8), connectivity=4)
        csize = [c[-1] for c in cstats[1:]]
        csize_list.append(csize)
        component_list.append(component)

    if len(csize_list)==0:
        return []

    max_length = max(len(sublist) for sublist in csize_list)
    csize_arr = np.array([sublist + [0] * (max_length - len(sublist)) for sublist in csize_list])
    mask_ = csize_arr >= 1
    values = csize_arr[mask_]
    indices = np.argwhere(mask_)
    sorted_indices = indices[np.argsort(-values)]
    if N >= 0:
        sorted_indices = sorted_indices[:N]
    # filled_masks = []
    masks = []
    if len(sorted_indices)>0:
        for index in sorted_indices:
            mask = (component_list[index[0]] == index[1]+1).astype(np.uint8)
            # filled_mask = filling_mask_holes(mask)
            # filled_masks.append(filled_mask)
            masks.append(mask*255)
    return masks

def insert_in_struct_layer(mask: Union[np.ndarray,torch.Tensor],struct_masks: List[np.ndarray]):
    if isinstance(mask,torch.Tensor):
        mask = (mask).detach().cpu().numpy()
        mask = (mask*255).astype(np.uint8)
    is_pseudo_struct_mask = False
    for index,pseudo_struct_mask in enumerate(struct_masks):
        mask_area = np.sum(mask==255)
        pseudo_struct_area = np.sum(pseudo_struct_mask==255)
        intersection_area = np.sum((mask.astype(np.uint16)+pseudo_struct_mask.astype(np.uint16))==510)
        if intersection_area/pseudo_struct_area>=0.7 and mask_area > 1.1*pseudo_struct_area:
            struct_masks.insert(index,np.full((mask.shape[0], mask.shape[1]), 255, dtype=np.uint8))
            is_pseudo_struct_mask = True
            break
    return is_pseudo_struct_mask,struct_masks,index


def add_visual_paths(shapes,shape_groups,device,
                     struct_path_num: int,
                     target_img: np.ndarray,
                     pseudo_struct_masks: List[np.ndarray],
                     is_opt_list: List[int] = [],
                     epsilon: int=5,
                     N: int = 50,
                     use_vtracer_enhancement: bool = True,
                     staircase_area: float = 1.5,
                     corner_angle: float = 135.0,
                     simplify_error: float = 2.0,
                     smooth_iterations: int = 0):
    """
    Add visual paths with optional VTracer enhancement.
    """
    if len(is_opt_list)==0:
        is_opt_list = [0 for i in range(len(shapes))]

    img_height, img_width = target_img.shape[:2]
    raster_img = svg_to_img(img_height,img_width,shapes,shape_groups,device)
    raster_img = rgba_to_rgb(raster_img,device=device)
    raster_img = raster_img.detach().cpu().numpy()
    target_img1 = np.transpose((target_img/255).astype(np.float16), (2, 0, 1))

    masks = select_mask_by_conn_area(raster_img,target_img1,N)

    if len(masks) == 0:
        return shapes,shape_groups,pseudo_struct_masks,is_opt_list,-1

    for mask in masks:
        color = get_mean_color(target_img,mask)
        path_group = pydiffvg.ShapeGroup(
                                shape_ids=torch.LongTensor([0]),
                                fill_color=torch.FloatTensor(list(color)+[255])/255,
                                stroke_color=torch.FloatTensor([0,0,0,1])
                            )

        mask = connect_mask_interior_exterior(mask)
        # print(np.max(mask))
        path = init_path_by_mask(mask, epsilon,
                                 use_vtracer_enhancement=use_vtracer_enhancement,
                                 staircase_area=staircase_area,
                                 corner_angle=corner_angle,
                                 simplify_error=simplify_error,
                                 smooth_iterations=smooth_iterations)
        is_pseudo_struct_mask,pseudo_struct_masks,insert_index=insert_in_struct_layer(mask,pseudo_struct_masks)
        if is_pseudo_struct_mask:
            shapes.insert(insert_index,path)
            shape_groups.insert(insert_index,path_group)
            is_opt_list.insert(insert_index,1)
            struct_path_num += 1
        else:
            shapes.append(path)
            shape_groups.append(path_group)
            is_opt_list.append(1)

    for index,shape_group1 in enumerate(shape_groups):
        shape_group1.shape_ids=torch.LongTensor([index])
    return shapes,shape_groups,pseudo_struct_masks,is_opt_list,struct_path_num


def merge_path(shapes,shape_groups,device,
               img_width: int,
               img_height: int,
               struct_path_num: int,
               pseudo_struct_masks: List[np.ndarray],
               is_opt_list: List[int],
               color_threshold: float = 0.05,
               overlapping_area_threshold: int = 3):

    def is_merge(shape_img,shape_img1,color,color1):
        if torch.sum((shape_img+shape_img1) > 1.9).item() >= overlapping_area_threshold and torch.sum(torch.abs(color-color1))<=color_threshold:
            return True
        else:
            return False

    def merge_obj(shape_img,shape_img1,color,color1,record1,record2):
        new_shape_img = shape_img+shape_img1
        new_shape_img = torch.clamp(new_shape_img,max=1)
        new_color = ((color+color1)/2).detach()
        new_record=record1+record2
        return new_shape_img,new_color,new_record

    def merge_objs_based_on_threshold(path_raster_imgs,path_color_list):
        record_list = [[i] for i in range(len(path_raster_imgs))]
        merged = False
        i = 0
        while i < len(path_raster_imgs):
            j = i + 1
            while j < len(path_raster_imgs):
                if is_merge(path_raster_imgs[i],path_raster_imgs[j],path_color_list[i],path_color_list[j]):
                    new_shape_img,new_color,new_record = merge_obj(path_raster_imgs[i],path_raster_imgs[j],path_color_list[i],path_color_list[j],record_list[i],record_list[j])
                    path_raster_imgs[i] = new_shape_img
                    path_raster_imgs.pop(j)
                    path_color_list[i] = new_color
                    path_color_list.pop(j)
                    record_list[i] = new_record
                    record_list.pop(j)
                    merged = True
                else:
                    j += 1
            if not merged:
                i += 1
            else:
                merged = False
        path_raster_imgs = [x for i,x in enumerate(path_raster_imgs) if len(record_list[i])>1]
        path_color_list = [x for i,x in enumerate(path_color_list) if len(record_list[i])>1]
        record_list = [x for i,x in enumerate(record_list) if len(record_list[i])>1]
        return record_list,path_raster_imgs,path_color_list

    struct_shapes,struct_shape_groups = shapes[:struct_path_num],shape_groups[:struct_path_num]
    visual_shapes,visual_shape_groups = shapes[struct_path_num:],shape_groups[struct_path_num:]

    black_pg = torch.tensor([0., 0., 0.], requires_grad=False, device=device)
    white_path_group = pydiffvg.ShapeGroup(
                                shape_ids=torch.LongTensor([0]),
                                fill_color=torch.FloatTensor([1,1,1,1]),
                                stroke_color=torch.FloatTensor([1,1,1,1])
                            )
    path_raster_imgs=[]
    for shape in visual_shapes:
        img = svg_to_img(img_width,img_height,[shape],[white_path_group],device)
        img = rgba_to_rgb(img,device=device,para_bg=black_pg)
        img = img[0]
        path_raster_imgs.append(img)
    path_color_list=[]
    for shape_group in visual_shape_groups:
        path_color_list.append(shape_group.fill_color)

    record_list,merge_imgs,merge_color_list = merge_objs_based_on_threshold(path_raster_imgs,path_color_list)

    flattened_record_list = [item for sublist in record_list for item in sublist]
    visual_shapes = [x for i,x in enumerate(visual_shapes) if i not in flattened_record_list]
    visual_shape_groups = [x for i,x in enumerate(visual_shape_groups) if i not in flattened_record_list]
    is_opt_list = is_opt_list[:len(shapes)-len(flattened_record_list)]

    for i,record in enumerate(record_list):
        if len(record)>1:
            new_mask = connect_mask_interior_exterior(merge_imgs[i])
            merge_path = init_path_by_mask(new_mask)
            merge_path_group = pydiffvg.ShapeGroup(
                            shape_ids=torch.LongTensor([0]),
                            fill_color=merge_color_list[i],
                            stroke_color=torch.FloatTensor([0,0,0,1])
                        )

            visual_shapes.append(merge_path)
            visual_shape_groups.append(merge_path_group)
            is_opt_list.append(1)
    shapes = struct_shapes+visual_shapes
    shape_groups = struct_shape_groups+visual_shape_groups

    for index,shape_group in enumerate(shape_groups):
        shape_group.shape_ids=torch.LongTensor([index])

    return shapes,shape_groups,pseudo_struct_masks,is_opt_list,struct_path_num

def find_closest_pair(contour1,contour2):
        closest_pair = (None, None)
        min_distance = float('inf')
        for outer_point in contour1:
            for inner_point in contour2:
                dist = np.linalg.norm(outer_point[0] - inner_point[0])  # 计算距离
                if dist < min_distance:
                    min_distance = dist
                    closest_pair = (tuple(outer_point[0]), tuple(inner_point[0]))
        return closest_pair

def connect_mask_interior_exterior(mask: Union[np.ndarray,torch.Tensor]):
    if isinstance(mask,torch.Tensor):
        mask = (mask).detach().cpu().numpy()
        mask = (mask*255).astype(np.uint8)
    mask = np.where(mask > 1, 255, 0).astype(np.uint8)
    # contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contour_areas = [cv2.contourArea(contour) for contour in contours]
    sorted_contours = sorted(zip(contour_areas, contours), key=lambda x: x[0], reverse=True)
    if len(contours)>1:
        for contou in sorted_contours[1:]:
            closest_pair=find_closest_pair(sorted_contours[0][1],contou[1])
            cv2.line(mask, closest_pair[0], closest_pair[1], (0), 3)  # 连接线
    return mask

def remove_lowquality_paths(shapes, shape_groups,device,
                img_width: int,
                img_height: int,
                visual_difference_threshold: float = 8.0,
                struct_path_num: int = 0,
                ):
    img = svg_to_img(img_width,img_height,shapes,shape_groups,device)
    img = rgba_to_rgb(img,device)
    remove_index_list = []
    for i in range(len(shapes)):
        shapes_copy=[x for j,x in enumerate(shapes) if j != i]
        shape_groups_copy=[x for j,x in enumerate(shape_groups) if j != i]
        for j in range(len(shape_groups_copy)):
            shape_groups_copy[j].shape_ids=torch.tensor([j])
        img_j = svg_to_img(img_width,img_height,shapes_copy,shape_groups_copy,device)
        img_j = rgba_to_rgb(img_j,device)
        pixel_difference = torch.sum(torch.abs(img_j-img)).item()
        if pixel_difference<=visual_difference_threshold:
            remove_index_list.append(i)

    if struct_path_num>0:
        remove_index_list = [x for x in remove_index_list if x >= struct_path_num]

    shapes = [x for i,x in enumerate(shapes) if i not in remove_index_list]
    shape_groups = [x for i,x in enumerate(shape_groups) if i not in remove_index_list]
    for i in range(len(shape_groups)):
        shape_groups[i].shape_ids=torch.tensor([i])
    return shapes,shape_groups
