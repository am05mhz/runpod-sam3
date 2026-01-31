"""
Depth Anything wrapper with V3/V2/V1 fallback chain.

Supports:
- Depth Anything V3 (best, requires depth_anything_3 package)
- Depth Anything V2 (good, via transformers)
- Depth Anything V1 (original, via transformers)
- Fallback heuristics (gradient + saliency)

For the CVPR 2025 vectorization method, we recommend V3 with V2-style
normalization for best layer separation results.
"""

import numpy as np
import torch
from PIL import Image
import cv2


class DepthEstimator:
    """
    Unified depth estimation interface with automatic fallback.

    Tries to load models in order of preference:
    1. Depth Anything V3 (best quality, nested architecture)
    2. Depth Anything V2 (good quality, inverse depth)
    3. Depth Anything V1 (original)
    4. Fallback heuristics (gradient + saliency)
    """

    def __init__(self, version="auto", resolution="low", device=None):
        """
        Args:
            version: "v3", "v2", "v1", "fallback", or "auto" (try best available)
            resolution: "high" or "low" (V3 only, others ignore)
            device: torch device (auto-detected if None)
        """
        self.version = version
        self.resolution = resolution
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.actual_version = None

        self._load_model()

    def _load_model(self):
        """Load the best available depth model."""
        if self.version == "auto":
            self._auto_load()
        elif self.version == "v3":
            self._load_v3()
        elif self.version == "v2":
            self._load_v2()
        elif self.version == "v1":
            self._load_v1()
        else:
            self._load_fallback()

    def _auto_load(self):
        """Try to load models in order of preference."""
        # Try V3 first
        try:
            self._load_v3()
            return
        except Exception as e:
            print(f"Depth Anything V3 not available: {e}")

        # Try V2
        try:
            self._load_v2()
            return
        except Exception as e:
            print(f"Depth Anything V2 not available: {e}")

        # Try V1
        try:
            self._load_v1()
            return
        except Exception as e:
            print(f"Depth Anything V1 not available: {e}")

        # Fallback to heuristics
        print("Using fallback depth estimation (gradient + saliency)")
        self._load_fallback()

    def _load_v3(self):
        """Load Depth Anything V3."""
        try:
            from depth_anything_3.app.modules.model_inference import ModelInference

            # Model options:
            # - depth-anything/DA3NESTED-GIANT-LARGE (best, ~12GB VRAM)
            # - depth-anything/DA3NESTED-LARGE (~8GB VRAM)
            # - depth-anything/DA3NESTED-BASE (~4GB VRAM)
            model_id = "depth-anything/DA3NESTED-LARGE"
            self.model = ModelInference(model_dir=model_id)
            self.actual_version = "v3"
            print(f"Loaded Depth Anything V3: {model_id}")
        except ImportError:
            raise ImportError("depth_anything_3 package not installed. Install with: pip install depth-anything-3")

    def _load_v2(self):
        """Load Depth Anything V2 via transformers."""
        from transformers import pipeline

        # V2-Large provides better depth accuracy than V2-Base
        # Available models: Base (~335M), Large (~670M), Giant (~1.3B)
        model_id = "depth-anything/Depth-Anything-V2-Large-hf"
        self.model = pipeline(
            "depth-estimation",
            model=model_id,
            device=0 if self.device == "cuda" else -1
        )
        self.actual_version = "v2"
        print(f"Loaded Depth Anything V2: {model_id}")

    def _load_v1(self):
        """Load Depth Anything V1 via transformers."""
        from transformers import pipeline

        model_id = "LiheYoung/depth-anything-base-hf"
        self.model = pipeline(
            "depth-estimation",
            model=model_id,
            device=0 if self.device == "cuda" else -1
        )
        self.actual_version = "v1"
        print(f"Loaded Depth Anything V1: {model_id}")

    def _load_fallback(self):
        """Initialize fallback heuristic estimator."""
        self.model = None
        self.actual_version = "fallback"
        print("Using fallback depth estimation")

    def estimate(self, image: np.ndarray, apply_v2_style: bool = True) -> np.ndarray:
        """
        Estimate depth map for image.

        Args:
            image: RGB numpy array (H, W, 3) with values in [0, 255]
            apply_v2_style: If True and using V3, apply V2-style normalization
                           for high-contrast output (recommended for layer separation)

        Returns:
            depth_map: Normalized depth (H, W), float32 in [0, 1]
                      Higher values = closer to camera (after V2-style conversion)
        """
        if self.actual_version == "v3":
            depth = self._estimate_v3(image)
            if apply_v2_style:
                depth = self._v3_to_v2_style(depth)
            return depth
        elif self.actual_version in ("v1", "v2"):
            return self._estimate_pipeline(image)
        else:
            return self._estimate_fallback(image)

    def _estimate_v3(self, image: np.ndarray) -> np.ndarray:
        """V3 estimation with resolution mode support."""
        # V3 supports HIGH-RES (better quality, slower) and LOW-RES (faster)
        # Convert numpy to PIL if needed
        if isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image.astype(np.uint8))
        else:
            pil_image = image

        result = self.model.run_inference(
            pil_image,
            mode=self.resolution  # "high" or "low"
        )

        depth = result['depth']
        if isinstance(depth, torch.Tensor):
            depth = depth.cpu().numpy()

        # Normalize to [0, 1]
        depth = depth.astype(np.float32)
        depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)

        return depth

    def _estimate_pipeline(self, image: np.ndarray) -> np.ndarray:
        """Estimate depth using transformers pipeline (V1/V2)."""
        # Convert to PIL
        if isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image.astype(np.uint8))
        else:
            pil_image = image

        # Run inference
        result = self.model(pil_image)
        depth = np.array(result['depth'])

        # Normalize to [0, 1]
        depth = depth.astype(np.float32)
        depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)

        # V1/V2 output inverse depth (closer = brighter), which is what we want
        return depth

    def _estimate_fallback(self, image: np.ndarray) -> np.ndarray:
        """
        Fallback depth estimation using gradient and saliency heuristics.

        This is a simple approximation when no depth model is available.
        Assumes: sharper edges and more salient regions are in foreground.
        """
        H, W = image.shape[:2]

        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        # Compute gradient magnitude (sharper = closer assumption)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient = np.sqrt(sobelx**2 + sobely**2)

        # Normalize gradient
        gradient = (gradient - gradient.min()) / (gradient.max() - gradient.min() + 1e-8)

        # Compute saliency (center-weighted, assuming subject in center)
        y_coords, x_coords = np.mgrid[0:H, 0:W]
        center_y, center_x = H / 2, W / 2
        distance_from_center = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
        max_dist = np.sqrt(center_x**2 + center_y**2)
        center_weight = 1.0 - (distance_from_center / max_dist)

        # Combine gradient and center weight
        # Higher gradient + more central = closer
        depth = 0.6 * gradient + 0.4 * center_weight

        # Smooth the result
        depth = cv2.GaussianBlur(depth.astype(np.float32), (15, 15), 0)

        # Normalize to [0, 1]
        depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)

        return depth.astype(np.float32)

    def _v3_to_v2_style(self, depth_v3: np.ndarray) -> np.ndarray:
        """
        Convert V3 direct depth to V2-style for high contrast layer separation.

        V3 outputs: sky=white (far), closer=darker (linear depth)
        V2 outputs: sky=black, closer=brighter (inverse depth/disparity)

        For layer separation, V2-style gives better contrast between fg/bg.
        """
        # Invert to V2-style (disparity-like)
        depth_inverted = 1.0 - depth_v3

        # Normalize to full range for maximum contrast
        depth_min = depth_inverted.min()
        depth_max = depth_inverted.max()
        depth_normalized = (depth_inverted - depth_min) / (depth_max - depth_min + 1e-8)

        return depth_normalized.astype(np.float32)

    def get_version_info(self) -> dict:
        """Return information about the loaded model."""
        return {
            "requested_version": self.version,
            "actual_version": self.actual_version,
            "resolution_mode": self.resolution,
            "device": self.device
        }


def get_depth_estimator(version="auto", resolution="low", device=None) -> DepthEstimator:
    """
    Factory function to create a depth estimator.

    Args:
        version: "v3", "v2", "v1", "fallback", or "auto"
        resolution: "high" or "low" (V3 only)
        device: torch device

    Returns:
        DepthEstimator instance
    """
    return DepthEstimator(version=version, resolution=resolution, device=device)


if __name__ == "__main__":
    # Test the depth estimator
    import sys

    # Create a simple test image
    test_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)

    print("Testing Depth Estimator...")
    estimator = get_depth_estimator(version="auto")

    print(f"\nVersion info: {estimator.get_version_info()}")

    depth = estimator.estimate(test_image)
    print(f"Depth map shape: {depth.shape}")
    print(f"Depth range: [{depth.min():.4f}, {depth.max():.4f}]")
    print("\nDepth estimation test passed!")
