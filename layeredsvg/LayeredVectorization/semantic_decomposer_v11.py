"""
Semantic-aware image layer decomposition (CVPR 2025 Section 3.2).

Uses SAM for segmentation + Depth Anything for depth ordering.
Depth Anything V2/V3 provides better layer separation than MoGe for this use case.
Clusters masks into foreground/background using K-means on depth values.

V11: Added SAMRefiner integration for smoother mask edges.
     Switched from MoGe to Depth Anything for better mask/depth alignment.
"""

import numpy as np
import cv2
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass, field
from sklearn.cluster import KMeans
from PIL import Image
import torch
import os
import sys

# Depth Anything for depth estimation (better layer separation than MoGe)
from .depth_anything_wrapper import DepthEstimator, get_depth_estimator

# Add SAMRefiner to path
_sam_refiner_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "SAMRefiner")
if _sam_refiner_path not in sys.path:
    sys.path.insert(0, _sam_refiner_path)


@dataclass
class ImageLayer:
    """Represents a single semantic layer in the image."""
    mask: np.ndarray                    # Binary mask (H, W), uint8
    depth: float                        # Average depth value
    average_color: np.ndarray           # Average RGB color
    is_foreground: bool = False         # True if foreground layer
    layer_id: int = 0                   # Unique layer identifier
    area: int = 0                       # Pixel count

    # These will be filled by later pipeline stages
    bezier_shape: Optional[object] = None
    proxy_geometries: Optional[List] = None
    global_features: Optional[np.ndarray] = None
    local_features: Optional[np.ndarray] = None


class SemanticLayerDecomposer:
    """
    Decomposes an image into semantic layers using SAM + Depth Anything estimation.

    Pipeline:
    1. Run SAM to get segmentation masks
    2. Run Depth Anything to get depth map (better layer separation)
    3. Compute average depth per mask
    4. Cluster into foreground/background using K-means
    5. Sort layers by depth for correct rendering order
    """

    def __init__(
        self,
        depth_version: str = "auto",
        depth_resolution: str = "low",
        sam_checkpoint: Optional[str] = None,
        sam_model_type: str = "vit_h",
        min_mask_area: int = 500,  # Match app9's default for cleaner masks
        device: Optional[str] = None,
        refine_masks: bool = True,
        refine_iterations: int = 3,
        use_samhq: bool = False,  # Standard SAM (HQ-SAM can over-filter masks)
        # Legacy MoGe parameters (ignored, kept for API compatibility)
        moge_version: str = None,
        moge_resolution: str = None
    ):
        """
        Args:
            depth_version: Depth Anything version ("v3", "v2", "v1", "auto")
            depth_resolution: Depth resolution ("high" or "low", V3 only)
            sam_checkpoint: Path to SAM checkpoint (auto-downloads if None)
            sam_model_type: SAM model type ("vit_h", "vit_l", "vit_b")
            min_mask_area: Minimum mask area in pixels
            device: torch device
            refine_masks: Whether to use SAMRefiner for smoother mask edges
            refine_iterations: Number of SAMRefiner iterations (default 3)
            use_samhq: Whether to use HQ-SAM for better mask quality
            moge_version: (IGNORED) Legacy parameter for API compatibility
            moge_resolution: (IGNORED) Legacy parameter for API compatibility
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.min_mask_area = min_mask_area
        self.refine_masks = refine_masks
        self.refine_iterations = refine_iterations
        self.use_samhq = use_samhq

        # Initialize Depth Anything estimator (better layer separation than MoGe)
        self.depth_estimator = get_depth_estimator(
            version=depth_version,
            resolution=depth_resolution,
            device=self.device
        )

        # SAM will be loaded lazily
        self.sam_model = None
        self.sam_checkpoint = sam_checkpoint
        self.sam_model_type = sam_model_type
        self._sam_raw_model = None  # Store raw SAM model for refinement

        # Store last outputs for intermediate saving
        self.last_depth_map = None
        self.last_normal_map = None
        self.last_points_map = None

    def _load_sam(self):
        """Lazy load SAM model (standard SAM or HQ-SAM based on use_samhq setting)."""
        if self.sam_model is not None:
            return

        try:
            import os
            module_dir = os.path.dirname(os.path.abspath(__file__))

            # Try HQ-SAM first if enabled
            if self.use_samhq:
                try:
                    from segment_anything_hq import sam_model_registry as hq_sam_registry
                    from segment_anything_hq import SamAutomaticMaskGenerator as HQSamAutomaticMaskGenerator

                    # HQ-SAM checkpoint paths
                    hq_checkpoint_paths = [
                        self.sam_checkpoint,  # From config (as-is)
                        os.path.join(module_dir, self.sam_checkpoint) if self.sam_checkpoint else None,
                        os.path.join(module_dir, "checkpoints", "sam_hq_vit_h.pth"),
                        "checkpoints/sam_hq_vit_h.pth",
                        "sam_hq_vit_h.pth",
                    ]

                    hq_checkpoint = None
                    for path in hq_checkpoint_paths:
                        if path and os.path.exists(path):
                            hq_checkpoint = path
                            break

                    if hq_checkpoint:
                        print(f"Loading HQ-SAM model from {hq_checkpoint}...")
                        sam = hq_sam_registry[self.sam_model_type](checkpoint=hq_checkpoint)
                        sam.to(device=self.device)
                        self._sam_raw_model = sam

                        self.sam_model = HQSamAutomaticMaskGenerator(
                            model=sam,
                            points_per_side=32,
                            pred_iou_thresh=0.86,
                            stability_score_thresh=0.92,
                            crop_n_layers=1,
                            crop_n_points_downscale_factor=2,
                            min_mask_region_area=self.min_mask_area,
                        )
                        print(f"Loaded HQ-SAM model: {self.sam_model_type} from {hq_checkpoint}")
                        return
                    else:
                        print("HQ-SAM checkpoint not found, falling back to standard SAM")
                except ImportError as e:
                    print(f"HQ-SAM not available ({e}), falling back to standard SAM")

            # Standard SAM fallback
            # IMPORTANT: Use build functions directly instead of sam_model_registry
            # to avoid potential registry pollution from segment_anything_hq
            from segment_anything import SamAutomaticMaskGenerator
            from segment_anything.build_sam import build_sam_vit_h, build_sam_vit_l, build_sam_vit_b

            # Build functions map (bypasses potentially polluted registry)
            build_functions = {
                "vit_h": build_sam_vit_h,
                "vit_l": build_sam_vit_l,
                "vit_b": build_sam_vit_b,
            }

            # Default checkpoint paths - search common locations
            if self.sam_checkpoint is None or not os.path.exists(self.sam_checkpoint):
                checkpoint_paths = [
                    # Project checkpoints folder - prefer vit_h (better quality)
                    os.path.join(module_dir, "checkpoints", "sam_vit_h_4b8939.pth"),
                    os.path.join(module_dir, "checkpoints", "sam_vit_b_01ec64.pth"),
                    # Relative paths
                    "sam_vit_h_4b8939.pth",
                    "sam_vit_b_01ec64.pth",
                    "checkpoints/sam_vit_h_4b8939.pth",
                    "checkpoints/sam_vit_b_01ec64.pth",
                ]

                for path in checkpoint_paths:
                    if os.path.exists(path):
                        self.sam_checkpoint = path
                        # Update model type based on checkpoint
                        if "vit_b" in path:
                            self.sam_model_type = "vit_b"
                        elif "vit_l" in path:
                            self.sam_model_type = "vit_l"
                        elif "vit_h" in path:
                            self.sam_model_type = "vit_h"
                        break

            if self.sam_checkpoint is None:
                print("SAM checkpoint not found, using fallback color-based segmentation")
                self.sam_model = "fallback"
                return

            if self.sam_model_type not in build_functions:
                raise ValueError(f"Unknown SAM model type: {self.sam_model_type}. Expected one of: {list(build_functions.keys())}")

            sam = build_functions[self.sam_model_type](checkpoint=self.sam_checkpoint)
            sam.to(device=self.device)
            self._sam_raw_model = sam  # Store for SAMRefiner

            self.sam_model = SamAutomaticMaskGenerator(
                model=sam,
                points_per_side=32,
                pred_iou_thresh=0.86,
                stability_score_thresh=0.92,
                crop_n_layers=1,
                crop_n_points_downscale_factor=2,
                min_mask_region_area=self.min_mask_area,
            )
            print(f"Loaded SAM model: {self.sam_model_type} from {self.sam_checkpoint}")

        except ImportError as e:
            print(f"SAM not available ({e}), using fallback segmentation")
            self.sam_model = "fallback"
        except Exception as e:
            print(f"Error loading SAM ({e}), using fallback segmentation")
            self.sam_model = "fallback"

    def decompose(self, image: np.ndarray, n_clusters: int = 2) -> List[ImageLayer]:
        """
        Decompose image into semantic layers sorted by depth.

        Args:
            image: RGB numpy array (H, W, 3) with values in [0, 255]
            n_clusters: Number of depth clusters (2 = fg/bg)

        Returns:
            List of ImageLayer objects sorted by depth (background first)
        """
        H, W = image.shape[:2]

        # Step 1: Segment with SAM
        print("Running segmentation...")
        masks = self._segment(image)
        print(f"Found {len(masks)} raw segments")

        # Step 1.5: Deduplicate overlapping masks
        # SAM often creates both "whole object" and "object parts" (e.g., fish + fish head)
        # We keep only the larger/more complete masks
        masks = self._deduplicate_masks(masks)
        print(f"After deduplication: {len(masks)} segments")

        # Step 1.6: Refine masks for smoother edges using SAMRefiner
        if self.refine_masks and len(masks) > 0:
            masks = self._refine_masks(image, masks)

        if len(masks) == 0:
            # Return single layer covering entire image
            return [ImageLayer(
                mask=np.ones((H, W), dtype=np.uint8) * 255,
                depth=0.5,
                average_color=np.mean(image, axis=(0, 1)).astype(np.uint8),
                is_foreground=True,
                layer_id=0,
                area=H * W
            )]

        # Step 2: Estimate depth using Depth Anything
        version_info = self.depth_estimator.get_version_info()
        print(f"Estimating depth with Depth Anything {version_info['actual_version']}...")
        depth_map = self.depth_estimator.estimate(image)
        self.last_depth_map = depth_map  # Store for intermediate saving

        # Note: Depth Anything output convention (after V2-style normalization):
        # Higher values = closer to camera (foreground)
        # Lower values = further from camera (background)

        # Step 3: Compute average depth per mask and create layers
        layers = []
        for i, mask in enumerate(masks):
            mask_bool = mask > 127
            area = np.sum(mask_bool)

            if area < self.min_mask_area:
                continue

            # Average depth for this mask region
            avg_depth = float(np.mean(depth_map[mask_bool]))

            # Average color
            avg_color = np.mean(image[mask_bool], axis=0).astype(np.uint8)

            layers.append(ImageLayer(
                mask=mask,
                depth=avg_depth,
                average_color=avg_color,
                is_foreground=False,  # Will be set in clustering
                layer_id=i,
                area=area
            ))

        if len(layers) == 0:
            # Fallback: single layer
            return [ImageLayer(
                mask=np.ones((H, W), dtype=np.uint8) * 255,
                depth=0.5,
                average_color=np.mean(image, axis=(0, 1)).astype(np.uint8),
                is_foreground=True,
                layer_id=0,
                area=H * W
            )]

        # Step 4: Cluster into foreground/background
        print("Clustering layers...")
        if len(layers) > 1 and n_clusters > 1:
            layers = self._cluster_fg_bg(layers, n_clusters)
        else:
            # Single layer = foreground
            layers[0].is_foreground = True

        # Step 5: Sort by depth (background first, then foreground by depth)
        # Background layers (is_foreground=False) come first, sorted by depth (far to near)
        # Foreground layers (is_foreground=True) come last, sorted by depth (far to near)
        # Within same depth, larger areas go behind smaller ones (secondary sort by -area)
        layers.sort(key=lambda l: (l.is_foreground, l.depth, -l.area))

        # Reassign layer IDs after sorting
        for i, layer in enumerate(layers):
            layer.layer_id = i

        # Debug: print layer order
        n_bg = sum(1 for l in layers if not l.is_foreground)
        n_fg = sum(1 for l in layers if l.is_foreground)
        print(f"Decomposed into {len(layers)} layers ({n_bg} bg, {n_fg} fg)")
        for i, layer in enumerate(layers):
            fg_label = "FG" if layer.is_foreground else "BG"
            print(f"  Layer {i}: {fg_label}, depth={layer.depth:.3f}, area={layer.area}")

        return layers

    def _segment(self, image: np.ndarray) -> List[np.ndarray]:
        """Run segmentation on image."""
        self._load_sam()

        if self.sam_model == "fallback":
            return self._segment_fallback(image)

        # Run SAM
        results = self.sam_model.generate(image)

        # Extract masks with areas for sorting
        masks_with_area = []
        for result in results:
            mask = result['segmentation'].astype(np.uint8) * 255
            area = result.get('area', np.sum(mask > 127))
            masks_with_area.append((mask, area))

        # Sort by area descending (larger masks first for deduplication)
        masks_with_area.sort(key=lambda x: x[1], reverse=True)
        masks = [m[0] for m in masks_with_area]

        return masks

    def _deduplicate_masks(self, masks: List[np.ndarray]) -> List[np.ndarray]:
        """
        Remove redundant overlapping masks.

        SAM often creates both "whole object" and "object parts" masks
        (e.g., "whole fish" + "fish head" + "fish tail"). We keep only
        the larger/more complete masks by removing smaller masks that
        are significantly contained within larger ones.

        Args:
            masks: List of binary masks, sorted by area descending

        Returns:
            Deduplicated list of masks
        """
        if len(masks) <= 1:
            return masks

        # Calculate areas
        areas = [np.sum(m > 127) for m in masks]

        # Track which masks to keep
        keep = [True] * len(masks)

        for i in range(len(masks)):
            if not keep[i]:
                continue

            mask_i = masks[i] > 127

            for j in range(i + 1, len(masks)):
                if not keep[j]:
                    continue

                mask_j = masks[j] > 127

                # Calculate overlap
                intersection = np.logical_and(mask_i, mask_j).sum()

                if intersection == 0:
                    continue

                # IoU check (near-duplicates)
                union = np.logical_or(mask_i, mask_j).sum()
                iou = intersection / union if union > 0 else 0

                if iou > 0.70:
                    # Nearly same mask - keep larger (i comes first, so it's larger)
                    # Threshold 0.70 matches app8 for consistent layer merging
                    keep[j] = False
                    print(f"    Dedup: removing mask {j} (IoU {iou:.2f} with {i})")
                    continue

                # Containment check
                # How much of mask_j is inside mask_i?
                containment = intersection / areas[j] if areas[j] > 0 else 0

                if containment > 0.65:
                    # mask_j is mostly inside mask_i - it's a sub-part, remove it
                    # Threshold 0.65 matches app8 (merges "fish head" into "fish")
                    keep[j] = False
                    print(f"    Dedup: removing mask {j} ({containment:.0%} inside mask {i})")

        return [masks[i] for i in range(len(masks)) if keep[i]]

    def _refine_masks(self, image: np.ndarray, masks: List[np.ndarray]) -> List[np.ndarray]:
        """
        Refine masks using SAMRefiner for smoother edges.

        SAMRefiner (ICLR 2025) iteratively refines coarse masks by generating
        noise-tolerant prompts (points, boxes, soft masks) and re-running SAM.

        Args:
            image: RGB numpy array (H, W, 3)
            masks: List of binary masks to refine

        Returns:
            List of refined masks with smoother edges
        """
        if not self.refine_masks or self._sam_raw_model is None:
            return masks

        if len(masks) == 0:
            return masks

        try:
            # Import SAMRefiner components - need to handle utils conflict
            import importlib.util
            from segment_anything.utils.transforms import ResizeLongestSide

            # Load SAMRefiner modules directly to avoid utils naming conflict
            sam_refiner_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "SAMRefiner")

            # Load utils from SAMRefiner
            utils_spec = importlib.util.spec_from_file_location("sam_refiner_utils", os.path.join(sam_refiner_dir, "utils.py"))
            sam_refiner_utils = importlib.util.module_from_spec(utils_spec)
            sys.modules['sam_refiner_utils'] = sam_refiner_utils
            utils_spec.loader.exec_module(sam_refiner_utils)

            # Temporarily replace utils in sys.modules so sam_refiner can import it
            original_utils = sys.modules.get('utils')
            sys.modules['utils'] = sam_refiner_utils

            try:
                # Now load sam_refiner
                refiner_spec = importlib.util.spec_from_file_location("sam_refiner", os.path.join(sam_refiner_dir, "sam_refiner.py"))
                sam_refiner_module = importlib.util.module_from_spec(refiner_spec)
                refiner_spec.loader.exec_module(sam_refiner_module)
                sam_refiner = sam_refiner_module.sam_refiner
            finally:
                # Restore original utils
                if original_utils is not None:
                    sys.modules['utils'] = original_utils
                elif 'utils' in sys.modules:
                    del sys.modules['utils']

            hq_str = " with HQ-SAM" if self.use_samhq else ""
            print(f"Refining {len(masks)} masks with SAMRefiner{hq_str} ({self.refine_iterations} iterations)...")

            # Prepare resize transform
            resize_transform = ResizeLongestSide(self._sam_raw_model.image_encoder.img_size)

            # SAMRefiner expects image as path, but we can modify the flow
            # Save image temporarily
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
                temp_path = f.name
                # Convert RGB to BGR for cv2.imwrite
                cv2.imwrite(temp_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

            try:
                refined_masks = []
                for i, mask in enumerate(masks):
                    # Convert mask to 0-1 range
                    mask_01 = (mask > 127).astype(np.uint8)

                    # Run SAMRefiner
                    refined, _, _ = sam_refiner(
                        temp_path,
                        [mask_01],
                        self._sam_raw_model,
                        resize_transform=resize_transform,
                        use_point=True,
                        use_box=True,
                        use_mask=True,
                        add_neg=True,
                        iters=self.refine_iterations,
                        gamma=4.0,
                        strength=30,
                        use_samhq=self.use_samhq
                    )

                    # Convert back to 0-255 range
                    refined_mask = (refined[0] * 255).astype(np.uint8)
                    refined_masks.append(refined_mask)

                    if (i + 1) % 5 == 0 or (i + 1) == len(masks):
                        print(f"  Refined {i + 1}/{len(masks)} masks")

                print(f"Mask refinement complete")
                return refined_masks

            finally:
                # Clean up temp file
                if os.path.exists(temp_path):
                    os.remove(temp_path)

        except ImportError as e:
            print(f"SAMRefiner not available ({e}), skipping mask refinement")
            return masks
        except Exception as e:
            print(f"Mask refinement failed ({e}), using original masks")
            import traceback
            traceback.print_exc()
            return masks

    def _segment_fallback(self, image: np.ndarray) -> List[np.ndarray]:
        """
        Fallback segmentation using color clustering and edge detection.
        Used when SAM is not available.
        """
        H, W = image.shape[:2]

        # Convert to LAB color space for better clustering
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)

        # Reshape for clustering
        pixels = lab.reshape(-1, 3).astype(np.float32)

        # K-means clustering
        n_clusters = min(8, max(2, (H * W) // 10000))  # Adaptive number of clusters
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        _, labels, _ = cv2.kmeans(pixels, n_clusters, None, criteria, 10, cv2.KMEANS_PP_CENTERS)

        # Create masks from labels
        labels = labels.reshape(H, W)
        masks = []

        for i in range(n_clusters):
            mask = (labels == i).astype(np.uint8) * 255

            # Clean up mask
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

            # Find connected components
            num_labels, labels_im, stats, centroids = cv2.connectedComponentsWithStats(mask)

            for j in range(1, num_labels):
                area = stats[j, cv2.CC_STAT_AREA]
                if area >= self.min_mask_area:
                    component_mask = (labels_im == j).astype(np.uint8) * 255
                    masks.append(component_mask)

        return masks

    def _cluster_fg_bg(self, layers: List[ImageLayer], n_clusters: int = 2) -> List[ImageLayer]:
        """
        K-means clustering to separate foreground/background (Eq. 2 in paper).

        Higher depth = closer to camera = foreground.
        """
        depths = np.array([l.depth for l in layers]).reshape(-1, 1)

        print(f"Clustering {len(layers)} layers with depths: "
              f"min={depths.min():.3f}, max={depths.max():.3f}")

        # K-means clustering
        n_clusters = min(n_clusters, len(layers))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(depths)

        # Determine which cluster is foreground (higher average depth)
        cluster_means = []
        for i in range(n_clusters):
            cluster_depths = depths[cluster_labels == i]
            cluster_means.append(np.mean(cluster_depths))
            print(f"  Cluster {i}: mean depth={np.mean(cluster_depths):.3f}, "
                  f"n_layers={len(cluster_depths)}")

        # Cluster with highest mean depth is foreground
        fg_cluster = np.argmax(cluster_means)
        print(f"Foreground cluster: {fg_cluster} (highest mean depth)")

        n_fg = 0
        for i, layer in enumerate(layers):
            layer.is_foreground = (cluster_labels[i] == fg_cluster)
            if layer.is_foreground:
                n_fg += 1

        print(f"Assigned {n_fg} foreground, {len(layers) - n_fg} background layers")

        return layers

    def visualize_layers(self, image: np.ndarray, layers: List[ImageLayer]) -> np.ndarray:
        """
        Create a visualization of the layer decomposition.

        Returns:
            Visualization image with colored overlays
        """
        H, W = image.shape[:2]
        vis = image.copy().astype(np.float32)

        # Define colors for visualization
        colors = [
            [255, 0, 0],    # Red
            [0, 255, 0],    # Green
            [0, 0, 255],    # Blue
            [255, 255, 0],  # Yellow
            [255, 0, 255],  # Magenta
            [0, 255, 255],  # Cyan
            [255, 128, 0],  # Orange
            [128, 0, 255],  # Purple
        ]

        for i, layer in enumerate(layers):
            color = colors[i % len(colors)]
            mask_bool = layer.mask > 127

            # Tint the layer with its color
            alpha = 0.3 if layer.is_foreground else 0.2
            for c in range(3):
                vis[:, :, c] = np.where(
                    mask_bool,
                    vis[:, :, c] * (1 - alpha) + color[c] * alpha,
                    vis[:, :, c]
                )

            # Draw contour
            contours, _ = cv2.findContours(
                layer.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(vis.astype(np.uint8), contours, -1, color, 2)

        return vis.astype(np.uint8)


def decompose_image(
    image: np.ndarray,
    depth_version: str = "auto",
    depth_resolution: str = "low",
    n_clusters: int = 2
) -> List[ImageLayer]:
    """
    Convenience function to decompose an image into semantic layers.

    Args:
        image: RGB numpy array (H, W, 3)
        depth_version: Depth model version
        depth_resolution: "high" or "low"
        n_clusters: Number of fg/bg clusters

    Returns:
        List of ImageLayer objects
    """
    decomposer = SemanticLayerDecomposer(
        depth_version=depth_version,
        depth_resolution=depth_resolution
    )
    return decomposer.decompose(image, n_clusters=n_clusters)


if __name__ == "__main__":
    # Test the decomposer
    import sys

    # Create a simple test image with clear fg/bg
    H, W = 256, 256
    test_image = np.zeros((H, W, 3), dtype=np.uint8)

    # Background (blue sky gradient)
    for y in range(H):
        test_image[y, :, 2] = 200 - y // 2  # Blue decreasing
        test_image[y, :, 1] = 150 + y // 4  # Green increasing

    # Foreground object (red circle)
    cv2.circle(test_image, (W // 2, H // 2), 50, (200, 50, 50), -1)

    print("Testing Semantic Layer Decomposer...")
    decomposer = SemanticLayerDecomposer(depth_version="auto")

    layers = decomposer.decompose(test_image)
    print(f"\nFound {len(layers)} layers:")
    for layer in layers:
        print(f"  Layer {layer.layer_id}: depth={layer.depth:.3f}, "
              f"fg={layer.is_foreground}, area={layer.area}")

    print("\nSemantic decomposition test passed!")
