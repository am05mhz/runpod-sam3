"""
app3.py - Bezier Splatting Flask App with CORRECT SVG Generation

UNDERSTANDING FROM THE PAPER:
- Paper's Algorithm 2 describes SVG conversion for 3k+1 control points (k cubic segments)
- BUT for closed mode with bezier_degree=4, the model stores 10 points = 2*(4+1)
- These 10 points represent TWO degree-4 Bezier curves forming the boundaries of a closed region
- The "paired Bezier curve structure" (Section 3.3) uses two curves B1(t) and BR+1(t)
  that share start/end points, forming a closed region

CORRECT SVG GENERATION FOR CLOSED MODE:
According to the paper's Section 3.3 on "Sampling Gaussians on closed curves":
- Two Bezier curves form the boundaries (bezier1 and bezier2)
- The area BETWEEN them is filled using interpolated curves
- For SVG, we should trace the outline: bezier1 -> bezier2_reversed -> close

Control points structure (closed mode, bezier_degree=4, num_beziers=2):
- total_pts = 2 * (4+1) = 10 points
- M = (total_pts - 2) // 2 = 4 (the bezier degree)
- bezier1 = points[0:M+2] = points[0:6] (6 control points, degree 5)
- bezier2 = points[M+1:] + points[0:1], flipped (6 control points, degree 5)

The SVG path traces the BOUNDARY of the filled region.
"""
import os
import uuid
import math
import time
import json
import threading
from datetime import datetime
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_from_directory, url_for
from flask_cors import CORS
from werkzeug.utils import secure_filename
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from pytorch_msssim import ms_ssim
import torch.nn.functional as F
import svgwrite
from scipy.special import comb

app = Flask(__name__)
CORS(app, origins="*")

# Configuration
UPLOAD_FOLDER = Path('./uploads')
RESULT_FOLDER = Path('./result')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

UPLOAD_FOLDER.mkdir(exist_ok=True)
RESULT_FOLDER.mkdir(exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max

# Store job status
jobs = {}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def image_path_to_tensor(image_path: Path):
    img = Image.open(image_path).convert('RGB')
    transform = transforms.ToTensor()
    img_tensor = transform(img).unsqueeze(0)
    return img_tensor


def sample_bezier_curve(control_points, num_samples=50):
    """
    Sample points along a Bezier curve of any degree using Bernstein polynomials.
    """
    n = len(control_points) - 1  # degree
    t_values = np.linspace(0, 1, num_samples)

    result = np.zeros((num_samples, 2))
    for i, t in enumerate(t_values):
        point = np.zeros(2)
        for j in range(n + 1):
            basis = comb(n, j) * (t ** j) * ((1 - t) ** (n - j))
            point += basis * control_points[j]
        result[i] = point
    return result


def generate_svg_from_model(gaussian_model, svg_path, canvas_width, canvas_height,
                            samples_per_curve=100, use_splats=False):
    """
    Generate SVG from a trained model.

    Args:
        gaussian_model: The trained model
        svg_path: Output path for SVG
        canvas_width, canvas_height: Dimensions
        samples_per_curve: Number of sample points per Bezier curve
        use_splats: If True, use Gaussian splat circles (experimental)
                   If False, use interpolated fill (recommended, matches PNG)

    For CLOSED mode:
    - Uses interpolated fill approach (same as PNG renderer)
    - Creates overlapping bands between boundary curves
    - Eliminates gaps between shapes
    """
    mode = gaussian_model.mode

    # Call forward() to populate self.xyz and self.xyz_area
    # This may fail for some modes, so wrap in try-except
    with torch.no_grad():
        try:
            _ = gaussian_model()
        except Exception as e:
            print(f"Warning: forward() failed: {e}")

    # If using splats for closed mode, use the splat method (experimental)
    if mode == 'closed' and use_splats:
        return generate_svg_splats(gaussian_model, svg_path, canvas_width, canvas_height)

    # Use interpolated fill for closed mode (matches PNG approach)
    if mode == 'closed':
        return generate_svg_interpolated_fill(gaussian_model, svg_path, canvas_width, canvas_height, samples_per_curve)

    # For open curves, use polygon method
    return generate_svg_polygons(gaussian_model, svg_path, canvas_width, canvas_height, samples_per_curve)


def generate_svg_interpolated_fill(gaussian_model, svg_path, canvas_width, canvas_height,
                                    samples_per_curve=100, fill_resolution=10):
    """
    Generate SVG using the same interpolated fill approach as PNG renderer.

    Instead of just tracing the boundary, we create multiple overlapping bands
    between interpolation levels, similar to how the PNG uses xyz_area.
    This eliminates the white gaps between adjacent shapes.
    """
    from scipy.stats import norm

    with torch.no_grad():
        control_points = gaussian_model._control_points.detach().cpu()
        features_dc = torch.sigmoid(gaussian_model._features_dc.detach()).cpu()

        # Get the same xyz that PNG uses for depth ordering
        xyz = gaussian_model.xyz

    # Depth ordering (larger areas drawn first = background)
    num_curves = control_points.shape[0]
    try:
        if xyz is not None and xyz.shape[0] == num_curves:
            boxes = gaussian_model.compute_aabb(xyz.view(xyz.shape[0], -1, 2))
            ratio = gaussian_model.W / gaussian_model.H
            widths = (boxes[:, 2] - boxes[:, 0]) * ratio
            heights = boxes[:, 3] - boxes[:, 1]
            depth = widths * heights
            sorted_indices = torch.argsort(depth, descending=True).cpu()
        else:
            sorted_indices = torch.arange(num_curves)
    except Exception as e:
        print(f"Warning: depth sorting failed in interpolated_fill: {e}")
        sorted_indices = torch.arange(num_curves)

    # Verify sorted_indices are valid
    if sorted_indices.max() >= num_curves:
        sorted_indices = torch.arange(num_curves)

    # Create SVG
    dwg = svgwrite.Drawing(svg_path, size=(canvas_width, canvas_height), profile='full')
    dwg.add(dwg.rect(insert=(0, 0), size=(canvas_width, canvas_height), fill='white'))

    def scale_point(p):
        x = (p[0] + 1) / 2 * canvas_width
        y = (p[1] + 1) / 2 * canvas_height
        return x, y

    for curve_idx in sorted_indices:
        # Get color
        color = features_dc[curve_idx].cpu().numpy()
        r, g, b = (np.clip(color * 255, 0, 255)).astype(int)
        hex_color = svgwrite.rgb(int(r), int(g), int(b), mode='rgb')

        pts = control_points[curve_idx].cpu().numpy()
        total_pts = len(pts)

        if (total_pts - 2) % 2 == 0:
            M = (total_pts - 2) // 2

            # Get the two boundary curves
            bezier1_pts = pts[:M+2]
            bezier2_pts = np.concatenate([pts[M+1:], pts[0:1]], axis=0)[::-1]

            # Sample both boundary curves
            boundary1 = sample_bezier_curve(bezier1_pts, samples_per_curve)
            boundary2 = sample_bezier_curve(bezier2_pts, samples_per_curve)

            # Create the outer boundary polygon
            boundary2_rev = boundary2[::-1]
            all_boundary = np.vstack([boundary1, boundary2_rev[1:]])
            all_boundary_scaled = [scale_point(p) for p in all_boundary]

            # Build main boundary path
            d_parts = [f"M {all_boundary_scaled[0][0]:.2f} {all_boundary_scaled[0][1]:.2f}"]
            for p in all_boundary_scaled[1:]:
                d_parts.append(f"L {p[0]:.2f} {p[1]:.2f}")
            d_parts.append("Z")

            # Add the main filled shape with stroke to cover gaps
            path = dwg.path(
                d=" ".join(d_parts),
                fill=hex_color,
                stroke=hex_color,
                stroke_width=2.0,
                stroke_linejoin='round',
                fill_opacity=1.0
            )
            dwg.add(path)

            # Add interior fill bands (like xyz_area in PNG renderer)
            t_linspace = np.linspace(-2, 2, fill_resolution)
            t_vals = norm(0, 0.85).cdf(t_linspace)

            for level in range(fill_resolution - 1):
                t1 = t_vals[level]
                t2 = t_vals[level + 1]

                # Interpolate control points
                interp1_pts = (1 - t1) * bezier1_pts + t1 * bezier2_pts
                interp2_pts = (1 - t2) * bezier1_pts + t2 * bezier2_pts

                # Sample the interpolated curves
                interp1_samples = sample_bezier_curve(interp1_pts, samples_per_curve // 2)
                interp2_samples = sample_bezier_curve(interp2_pts, samples_per_curve // 2)

                # Create a band between the two interpolation levels
                band_pts = np.vstack([interp1_samples, interp2_samples[::-1]])
                band_scaled = [scale_point(p) for p in band_pts]

                d_band = [f"M {band_scaled[0][0]:.2f} {band_scaled[0][1]:.2f}"]
                for p in band_scaled[1:]:
                    d_band.append(f"L {p[0]:.2f} {p[1]:.2f}")
                d_band.append("Z")

                band_path = dwg.path(
                    d=" ".join(d_band),
                    fill=hex_color,
                    stroke=hex_color,
                    stroke_width=1.0,
                    fill_opacity=0.9
                )
                dwg.add(band_path)
        else:
            # Fallback for non-standard point counts
            pts_scaled = np.array([scale_point(p) for p in pts])
            sampled = sample_bezier_curve(pts_scaled, samples_per_curve)
            d_parts = [f"M {sampled[0][0]:.2f} {sampled[0][1]:.2f}"]
            for p in sampled[1:]:
                d_parts.append(f"L {p[0]:.2f} {p[1]:.2f}")
            d_parts.append("Z")
            path = dwg.path(d=" ".join(d_parts), fill=hex_color, stroke=hex_color,
                           stroke_width=2.0, fill_opacity=1.0)
            dwg.add(path)

    dwg.save()
    return True


def generate_svg_splats(gaussian_model, svg_path, canvas_width, canvas_height):
    """
    Generate SVG using Gaussian splat circles - same approach as PNG renderer.

    This places overlapping circles at the same sample positions used for PNG,
    eliminating gaps between shapes.
    """
    with torch.no_grad():
        # Get the sample positions (same as PNG uses)
        xyz = gaussian_model.xyz  # (N, 2, num_samples, 2) - boundary curves
        xyz_area = gaussian_model.xyz_area  # (N, resolution, num_samples, 2) - fill

        # Get scaling for circle sizes
        scaling = gaussian_model.get_scaling(factor=1)

    # Depth ordering (larger areas = background, drawn first)
    boxes = gaussian_model.compute_aabb(xyz.view(xyz.shape[0], -1, 2))
    ratio = gaussian_model.W / gaussian_model.H
    widths = (boxes[:, 2] - boxes[:, 0]) * ratio
    heights = boxes[:, 3] - boxes[:, 1]
    depth = widths * heights
    sorted_indices = torch.argsort(depth, descending=True)

    # Create SVG
    dwg = svgwrite.Drawing(svg_path, size=(canvas_width, canvas_height), profile='full')
    dwg.add(dwg.rect(insert=(0, 0), size=(canvas_width, canvas_height), fill='white'))

    num_curves = xyz.shape[0]

    for curve_idx in sorted_indices:
        # Get color for this curve
        base_color = torch.sigmoid(gaussian_model._features_dc[curve_idx]).detach().cpu().numpy()
        r, g, b = (np.clip(base_color * 255, 0, 255)).astype(int)
        hex_color = svgwrite.rgb(int(r), int(g), int(b), mode='rgb')

        # Get all sample points for this curve (boundary + interior fill)
        boundary_pts = xyz[curve_idx].reshape(-1, 2).cpu().numpy()
        fill_pts = xyz_area[curve_idx].reshape(-1, 2).cpu().numpy()
        all_pts = np.vstack([boundary_pts, fill_pts])

        # Scale to canvas coordinates
        all_pts_scaled = (all_pts + 1) / 2
        all_pts_scaled[:, 0] *= canvas_width
        all_pts_scaled[:, 1] *= canvas_height

        # Get circle radius from scaling
        try:
            scale_val = abs(scaling[curve_idx * xyz.shape[1] * xyz.shape[2]].item())
            radius = max(2.0, scale_val * 4)
        except:
            radius = 2.5

        # Draw circles at each sample point
        # Subsample to keep SVG file size reasonable (~200-400 circles per curve)
        step = max(1, len(all_pts_scaled) // 300)

        for i in range(0, len(all_pts_scaled), step):
            x, y = all_pts_scaled[i]
            dwg.add(dwg.circle(
                center=(float(x), float(y)),
                r=float(radius),
                fill=hex_color,
                stroke='none'
            ))

    dwg.save()
    return True


def generate_svg_polygons(gaussian_model, svg_path, canvas_width, canvas_height,
                          samples_per_curve=100):
    """
    Generate SVG using filled polygons (for closed mode) or strokes (for unclosed mode).

    For UNCLOSED mode: Uses gaussian_model.xyz directly - these are the actual sampled
    points that the PNG renderer uses. This ensures SVG matches PNG output.
    """
    mode = gaussian_model.mode

    # Get data from model - move to CPU first to avoid CUDA issues
    control_points = gaussian_model._control_points.detach().cpu()
    features_dc = torch.sigmoid(gaussian_model._features_dc.detach()).cpu()
    num_curves = control_points.shape[0]

    # Get opacity
    opacities = torch.sigmoid(gaussian_model._opacity.detach()).cpu()
    if opacities.dim() > 1 and opacities.shape[1] > 1:
        opacities = opacities.mean(dim=1)
    opacities = opacities.squeeze()

    # Get scaling for stroke width
    scaling = gaussian_model._scaling.detach().cpu()

    # For UNCLOSED mode, use the actual xyz points from the model
    # These are the same points the PNG renderer uses
    xyz_per_curve = None
    if mode == 'unclosed' and hasattr(gaussian_model, 'xyz') and gaussian_model.xyz is not None:
        xyz = gaussian_model.xyz.cpu()
        total_samples = xyz.shape[0]
        samples_per_curve_actual = total_samples // num_curves
        if total_samples == num_curves * samples_per_curve_actual:
            xyz_per_curve = xyz.view(num_curves, samples_per_curve_actual, 2)
            print(f"Using actual xyz points: {xyz_per_curve.shape}")

    # Compute depth ordering
    sorted_indices = torch.arange(num_curves)
    try:
        if hasattr(gaussian_model, 'xyz') and gaussian_model.xyz is not None:
            xyz = gaussian_model.xyz
            total_samples = xyz.shape[0]
            samples_per_curve_actual = total_samples // num_curves
            if total_samples == num_curves * samples_per_curve_actual:
                xyz_reshaped = xyz.view(num_curves, samples_per_curve_actual, 2)
                boxes = gaussian_model.compute_aabb(xyz_reshaped)
                ratio = gaussian_model.W / gaussian_model.H
                widths = (boxes[:, 2] - boxes[:, 0]) * ratio
                heights = boxes[:, 3] - boxes[:, 1]
                depth = widths * heights
                sorted_indices = torch.argsort(depth, descending=True).cpu()
    except Exception as e:
        print(f"Warning: depth sorting failed: {e}")
        sorted_indices = torch.arange(num_curves)

    # Verify sorted_indices are valid
    if sorted_indices.max() >= num_curves:
        sorted_indices = torch.arange(num_curves)

    # Create SVG
    dwg = svgwrite.Drawing(svg_path, size=(canvas_width, canvas_height), profile='full')
    dwg.add(dwg.rect(insert=(0, 0), size=(canvas_width, canvas_height), fill='white'))

    def scale_point(p):
        """Scale a single point from [-1, 1] to canvas coordinates."""
        x = (p[0] + 1) / 2 * canvas_width
        y = (p[1] + 1) / 2 * canvas_height
        return x, y

    def scale_points_array(pts):
        """Scale numpy array of points from [-1, 1] to canvas coordinates."""
        pts = pts.copy()
        pts = (pts + 1) / 2
        pts[:, 0] *= canvas_width
        pts[:, 1] *= canvas_height
        return pts

    for idx, curve_idx in enumerate(sorted_indices):
        curve_idx = int(curve_idx)
        pts = control_points[curve_idx].numpy()
        color = features_dc[curve_idx].numpy()
        r, g, b = (np.clip(color * 255, 0, 255)).astype(int)
        hex_color = svgwrite.rgb(int(r), int(g), int(b), mode='rgb')

        if opacities.dim() > 0:
            opacity = float(opacities[curve_idx].item())
        else:
            opacity = float(opacities.item())
        opacity = max(0.0, min(1.0, opacity))

        total_pts = len(pts)

        if mode == 'closed':
            # CLOSED MODE: Use paired Bezier curve structure
            # The control points define TWO boundary curves

            if (total_pts - 2) % 2 == 0:
                # Standard closed curve format: 2*(M+1) points for degree M
                M = (total_pts - 2) // 2  # Bezier degree

                # Split into two boundary curves (same logic as sample_bezier_area)
                # bezier1: first M+2 points
                # bezier2: points[M+1:] + points[0:1], then flipped
                bezier1_pts = pts[:M+2]
                bezier2_pts = np.concatenate([pts[M+1:], pts[0:1]], axis=0)
                bezier2_pts = bezier2_pts[::-1]  # Reverse

                # Sample both curves
                # bezier1 goes p0 -> p5 (forward direction)
                # bezier2 goes p0 -> p5 (after flip), we need p5 -> p0
                samples1 = sample_bezier_curve(bezier1_pts, samples_per_curve)
                samples2 = sample_bezier_curve(bezier2_pts, samples_per_curve)
                samples2 = samples2[::-1]  # Reverse to get p5 -> p0

                # Scale to canvas
                samples1 = scale_points_array(samples1)
                samples2 = scale_points_array(samples2)

                # Combine all points into a single polygon
                all_points = np.vstack([samples1, samples2[1:]])  # Skip first of samples2

                # Ensure consistent winding direction (counter-clockwise for SVG)
                signed_area = 0.0
                n_pts = len(all_points)
                for j in range(n_pts):
                    k = (j + 1) % n_pts
                    signed_area += all_points[j][0] * all_points[k][1]
                    signed_area -= all_points[k][0] * all_points[j][1]

                # If clockwise (negative area), reverse to make counter-clockwise
                if signed_area < 0:
                    all_points = all_points[::-1]

                # Build closed polygon path
                d_parts = [f"M {all_points[0][0]:.2f} {all_points[0][1]:.2f}"]
                for p in all_points[1:]:
                    d_parts.append(f"L {p[0]:.2f} {p[1]:.2f}")
                d_parts.append("Z")

            elif (total_pts - 1) % 3 == 0 and total_pts >= 4:
                # Alternative: 3k+1 format (cubic segments) - use Algorithm 2
                pts_scaled = scale_points_array(pts)
                n_segs = (total_pts - 1) // 3
                d_parts = [f"M {pts_scaled[0][0]:.2f} {pts_scaled[0][1]:.2f}"]
                for seg in range(n_segs):
                    p1 = pts_scaled[1 + 3*seg]
                    p2 = pts_scaled[2 + 3*seg]
                    p3_idx = 3 * seg + 3
                    p3 = pts_scaled[p3_idx] if p3_idx < total_pts else pts_scaled[0]
                    d_parts.append(f"C {p1[0]:.2f} {p1[1]:.2f}, {p2[0]:.2f} {p2[1]:.2f}, {p3[0]:.2f} {p3[1]:.2f}")
                d_parts.append("Z")

            else:
                # Fallback: sample as single curve
                pts_scaled = scale_points_array(pts)
                sampled = sample_bezier_curve(pts_scaled, samples_per_curve)
                d_parts = [f"M {sampled[0][0]:.2f} {sampled[0][1]:.2f}"]
                for p in sampled[1:]:
                    d_parts.append(f"L {p[0]:.2f} {p[1]:.2f}")
                d_parts.append("Z")

            # Add a stroke matching the fill to eliminate white gaps between shapes
            # Using larger stroke width (4.0) to ensure overlap between adjacent shapes
            path = dwg.path(
                d=" ".join(d_parts),
                fill=hex_color,
                stroke=hex_color,
                stroke_width=4.0,
                stroke_linejoin='round',
                fill_opacity=1.0,
                stroke_opacity=1.0
            )

        else:
            # OPEN/UNCLOSED MODE: Use native SVG Bezier strokes
            # For 3k+1 format (k cubic segments): use native 'C' commands
            # Otherwise: sample and draw as polyline stroke

            pts_scaled = scale_points_array(pts)

            # Get stroke width from scaling parameter
            try:
                stroke_width = float(torch.sigmoid(scaling[curve_idx]).mean().item()) * 15
                stroke_width = max(1.0, min(stroke_width, 30.0))
            except Exception:
                stroke_width = 3.0

            # Check for 3k+1 format (native SVG cubic Bezier)
            if (total_pts - 1) % 3 == 0 and total_pts >= 4:
                # Use native SVG 'C' commands for best quality
                n_segs = (total_pts - 1) // 3
                d_parts = [f"M {pts_scaled[0][0]:.2f} {pts_scaled[0][1]:.2f}"]
                for seg in range(n_segs):
                    p1 = pts_scaled[1 + 3*seg]
                    p2 = pts_scaled[2 + 3*seg]
                    p3_idx = 3 * seg + 3
                    p3 = pts_scaled[p3_idx] if p3_idx < total_pts else pts_scaled[-1]
                    d_parts.append(f"C {p1[0]:.2f} {p1[1]:.2f}, {p2[0]:.2f} {p2[1]:.2f}, {p3[0]:.2f} {p3[1]:.2f}")
            else:
                # Sample the curve and use polyline
                sampled = sample_bezier_curve(pts, samples_per_curve)
                sampled_scaled = scale_points_array(sampled)
                d_parts = [f"M {sampled_scaled[0][0]:.2f} {sampled_scaled[0][1]:.2f}"]
                for p in sampled_scaled[1:]:
                    d_parts.append(f"L {p[0]:.2f} {p[1]:.2f}")

            # Create stroke path (no fill for open curves)
            path = dwg.path(
                d=" ".join(d_parts),
                fill='none',
                stroke=hex_color,
                stroke_width=stroke_width,
                stroke_opacity=opacity,
                stroke_linecap='round',
                stroke_linejoin='round'
            )

        dwg.add(path)

    dwg.save()
    return True


class Args:
    """Arguments class to mimic argparse namespace"""
    def __init__(self, **kwargs):
        self.num_curves = kwargs.get('num_curves', 512)
        self.bezier_degree = kwargs.get('bezier_degree', 4)
        self.num_samples = kwargs.get('num_samples', 64)
        self.iterations = kwargs.get('iterations', 10000)
        self.mode = kwargs.get('mode', 'closed')
        self.lr = kwargs.get('lr', 0.01)
        self.save_imgs = kwargs.get('save_imgs', True)
        self.data_name = kwargs.get('data_name', 'upload')
        self.image_name = kwargs.get('image_name', 'image.png')


def run_bezier_splatting(job_id, image_path, args):
    """Run Bezier splatting training in background"""
    try:
        jobs[job_id]['status'] = 'processing'
        jobs[job_id]['progress'] = 0

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        gt_image = image_path_to_tensor(image_path).to(device)

        BLOCK_H, BLOCK_W = 16, 16
        H, W = gt_image.shape[2], gt_image.shape[3]

        from gaussianimage_cholesky_svg import GaussianImage_Cholesky

        gaussian_model = GaussianImage_Cholesky(
            loss_type="L2",
            opt_type="adan",
            num_curves=args.num_curves,
            num_samples=args.num_samples,
            H=H, W=W,
            BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W,
            device=device,
            lr=args.lr,
            mode=args.mode,
            bezier_degree=args.bezier_degree,
            quantize=False
        ).to(device)

        from utils import sparse_coord_init

        gaussian_model.train()
        start_time = time.time()
        remove_iter = 500
        remove_num = 0

        for iter in range(1, args.iterations + 1):
            if args.mode == 'unclosed':
                loss, psnr, pred_image = gaussian_model.train_iter_opencurves(gt_image)
            else:
                loss, psnr, pred_image = gaussian_model.train_iter(gt_image)

            jobs[job_id]['progress'] = int((iter / args.iterations) * 100)
            jobs[job_id]['psnr'] = f"{psnr:.2f}"
            jobs[job_id]['iteration'] = iter

            with torch.no_grad():
                max_iter = 14000 if args.mode == "unclosed" else 9200
                if iter % remove_iter == 0 and 1000 <= iter < max_iter:
                    if (iter // remove_iter) % 2 == 1:
                        prune_mask = gaussian_model.remove_curves_mask()
                        gaussian_model.num_curves = prune_mask.sum()
                        remove_num += (~prune_mask).sum()
                        gaussian_model.prune_beizer_curves(prune_mask)
                    elif (iter // remove_iter) % 2 == 0 and remove_num > 0:
                        pos_init_method = sparse_coord_init(gt_image, pred_image)
                        gaussian_model.densify(remove_num, pos_init_method, gt_image)
                        remove_num = 0

                gaussian_model.optimizer.zero_grad(set_to_none=True)

        job_folder = RESULT_FOLDER / job_id
        job_folder.mkdir(exist_ok=True)

        gaussian_model.eval()
        with torch.no_grad():
            renderpkg = gaussian_model()
            image = renderpkg['render']
            image = image.squeeze(0)
            to_pil = transforms.ToPILImage()
            img = to_pil(image)

            result_filename = "result.png"
            result_path = job_folder / result_filename
            img.save(result_path)

            svg_filename = "result.svg"
            svg_path = job_folder / svg_filename
            generate_svg_from_model(gaussian_model, str(svg_path), W, H)

            mse_loss = F.mse_loss(renderpkg["render"].float(), gt_image.float())
            final_psnr = 10 * math.log10(1.0 / mse_loss.item())
            final_ms_ssim = ms_ssim(renderpkg["render"].float(), gt_image.float(), data_range=1, size_average=True).item()

        training_time = time.time() - start_time

        jobs[job_id]['status'] = 'completed'
        jobs[job_id]['progress'] = 100
        jobs[job_id]['result_file'] = f"{job_id}/{result_filename}"
        jobs[job_id]['svg_file'] = f"{job_id}/{svg_filename}"
        jobs[job_id]['final_psnr'] = f"{final_psnr:.2f}"
        jobs[job_id]['final_ms_ssim'] = f"{final_ms_ssim:.4f}"
        jobs[job_id]['training_time'] = f"{training_time:.2f}s"

        checkpoint_path = job_folder / "model.pth.tar"
        torch.save(gaussian_model.state_dict(), checkpoint_path)
        jobs[job_id]['model_file'] = f"{job_id}/model.pth.tar"

        metadata = {
            'job_id': job_id,
            'status': 'completed',
            'created_at': datetime.now().isoformat(),
            'original_filename': args.image_name,
            'input_file': jobs[job_id]['input_file'],
            'result_file': f"{job_id}/{result_filename}",
            'svg_file': f"{job_id}/{svg_filename}",
            'model_file': f"{job_id}/model.pth.tar",
            'params': {
                'num_curves': args.num_curves,
                'bezier_degree': args.bezier_degree,
                'num_samples': args.num_samples,
                'iterations': args.iterations,
                'mode': args.mode,
                'lr': args.lr
            },
            'total_iterations': args.iterations,
            'metrics': {
                'psnr': final_psnr,
                'ms_ssim': final_ms_ssim,
                'training_time': training_time
            },
            'image_size': {'width': W, 'height': H}
        }
        metadata_path = job_folder / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

    except Exception as e:
        jobs[job_id]['status'] = 'failed'
        jobs[job_id]['error'] = str(e)
        import traceback
        print(f"Error processing job {job_id}: {traceback.format_exc()}")


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        job_id = str(uuid.uuid4())[:8]

        filename = secure_filename(file.filename)
        ext = filename.rsplit('.', 1)[1].lower()
        saved_filename = f"{job_id}_input.{ext}"
        filepath = UPLOAD_FOLDER / saved_filename
        file.save(filepath)

        args = Args(
            num_curves=int(request.form.get('num_curves', 512)),
            bezier_degree=int(request.form.get('bezier_degree', 4)),
            num_samples=int(request.form.get('num_samples', 64)),
            iterations=int(request.form.get('iterations', 10000)),
            mode=request.form.get('mode', 'closed'),
            lr=float(request.form.get('lr', 0.01)),
            image_name=filename
        )

        jobs[job_id] = {
            'status': 'queued',
            'progress': 0,
            'input_file': saved_filename,
            'params': {
                'num_curves': args.num_curves,
                'bezier_degree': args.bezier_degree,
                'iterations': args.iterations,
                'mode': args.mode,
                'lr': args.lr
            }
        }

        thread = threading.Thread(
            target=run_bezier_splatting,
            args=(job_id, filepath, args)
        )
        thread.daemon = True
        thread.start()

        return jsonify({
            'job_id': job_id,
            'message': 'Processing started',
            'status_url': url_for('job_status', job_id=job_id)
        })

    return jsonify({'error': 'Invalid file type'}), 400


@app.route('/status/<job_id>')
def job_status(job_id):
    if job_id not in jobs:
        return jsonify({'error': 'Job not found'}), 404
    return jsonify(jobs[job_id])


@app.route('/result/<path:filename>')
def get_result(filename):
    return send_from_directory(app.config['RESULT_FOLDER'], filename)


@app.route('/uploads/<filename>')
def get_upload(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/jobs')
def list_jobs():
    return jsonify(jobs)


@app.route('/history')
def list_history():
    """List past jobs from result folder metadata files (last 10)"""
    history = []
    metadata_files = list(RESULT_FOLDER.glob('*/metadata.json'))

    for meta_file in metadata_files:
        try:
            with open(meta_file, 'r') as f:
                metadata = json.load(f)
                history.append(metadata)
        except Exception as e:
            print(f"Error reading {meta_file}: {e}")

    history.sort(key=lambda x: x.get('created_at', ''), reverse=True)
    return jsonify(history[:10])


def run_continue_training(job_id, original_job_id, additional_iterations, image_path, args):
    """Continue training from a checkpoint"""
    try:
        jobs[job_id]['status'] = 'processing'
        jobs[job_id]['progress'] = 0

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        gt_image = image_path_to_tensor(image_path).to(device)

        BLOCK_H, BLOCK_W = 16, 16
        H, W = gt_image.shape[2], gt_image.shape[3]

        from gaussianimage_cholesky_svg import GaussianImage_Cholesky

        gaussian_model = GaussianImage_Cholesky(
            loss_type="L2",
            opt_type="adan",
            num_curves=args.num_curves,
            num_samples=args.num_samples,
            H=H, W=W,
            BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W,
            device=device,
            lr=args.lr,
            mode=args.mode,
            bezier_degree=args.bezier_degree,
            quantize=False
        ).to(device)

        checkpoint_path = RESULT_FOLDER / original_job_id / "model.pth.tar"
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model_dict = gaussian_model.state_dict()
        pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict and v.shape == model_dict[k].shape}
        model_dict.update(pretrained_dict)
        gaussian_model.load_state_dict(model_dict, strict=False)

        gaussian_model.train()
        start_time = time.time()

        for iter in range(1, additional_iterations + 1):
            if args.mode == 'unclosed':
                loss, psnr, pred_image = gaussian_model.train_iter_opencurves(gt_image)
            else:
                loss, psnr, pred_image = gaussian_model.train_iter(gt_image)

            jobs[job_id]['progress'] = int((iter / additional_iterations) * 100)
            jobs[job_id]['psnr'] = f"{psnr:.2f}"
            jobs[job_id]['iteration'] = iter

            with torch.no_grad():
                gaussian_model.optimizer.zero_grad(set_to_none=True)

        job_folder = RESULT_FOLDER / job_id
        job_folder.mkdir(exist_ok=True)

        gaussian_model.eval()
        with torch.no_grad():
            renderpkg = gaussian_model()
            image = renderpkg['render']
            image = image.squeeze(0)
            to_pil = transforms.ToPILImage()
            img = to_pil(image)

            result_filename = "result.png"
            result_path = job_folder / result_filename
            img.save(result_path)

            svg_filename = "result.svg"
            svg_path = job_folder / svg_filename
            generate_svg_from_model(gaussian_model, str(svg_path), W, H)

            mse_loss = F.mse_loss(renderpkg["render"].float(), gt_image.float())
            final_psnr = 10 * math.log10(1.0 / mse_loss.item())
            final_ms_ssim = ms_ssim(renderpkg["render"].float(), gt_image.float(), data_range=1, size_average=True).item()

        training_time = time.time() - start_time

        jobs[job_id]['status'] = 'completed'
        jobs[job_id]['progress'] = 100
        jobs[job_id]['result_file'] = f"{job_id}/{result_filename}"
        jobs[job_id]['svg_file'] = f"{job_id}/{svg_filename}"
        jobs[job_id]['final_psnr'] = f"{final_psnr:.2f}"
        jobs[job_id]['final_ms_ssim'] = f"{final_ms_ssim:.4f}"
        jobs[job_id]['training_time'] = f"{training_time:.2f}s"

        checkpoint_path = job_folder / "model.pth.tar"
        torch.save(gaussian_model.state_dict(), checkpoint_path)
        jobs[job_id]['model_file'] = f"{job_id}/model.pth.tar"

        original_meta_path = RESULT_FOLDER / original_job_id / "metadata.json"
        original_total = 0
        if original_meta_path.exists():
            with open(original_meta_path, 'r') as f:
                original_meta = json.load(f)
                original_total = original_meta.get('total_iterations', 0)

        metadata = {
            'job_id': job_id,
            'status': 'completed',
            'created_at': datetime.now().isoformat(),
            'continued_from': original_job_id,
            'original_filename': args.image_name,
            'input_file': jobs[job_id]['input_file'],
            'result_file': f"{job_id}/{result_filename}",
            'svg_file': f"{job_id}/{svg_filename}",
            'model_file': f"{job_id}/model.pth.tar",
            'params': {
                'num_curves': args.num_curves,
                'bezier_degree': args.bezier_degree,
                'num_samples': args.num_samples,
                'iterations': additional_iterations,
                'mode': args.mode,
                'lr': args.lr
            },
            'total_iterations': original_total + additional_iterations,
            'metrics': {
                'psnr': final_psnr,
                'ms_ssim': final_ms_ssim,
                'training_time': training_time
            },
            'image_size': {'width': W, 'height': H}
        }
        metadata_path = job_folder / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

    except Exception as e:
        jobs[job_id]['status'] = 'failed'
        jobs[job_id]['error'] = str(e)
        import traceback
        print(f"Error continuing job {job_id}: {traceback.format_exc()}")


@app.route('/continue/<original_job_id>', methods=['POST'])
def continue_training(original_job_id):
    """Continue training from a previous job"""
    metadata_path = RESULT_FOLDER / original_job_id / "metadata.json"
    if not metadata_path.exists():
        return jsonify({'error': 'Original job not found'}), 404

    with open(metadata_path, 'r') as f:
        original_meta = json.load(f)

    additional_iterations = int(request.form.get('iterations', 5000))
    job_id = str(uuid.uuid4())[:8]

    input_file = original_meta['input_file']
    image_path = UPLOAD_FOLDER / input_file

    if not image_path.exists():
        return jsonify({'error': 'Original image not found'}), 404

    original_params = original_meta['params']
    args = Args(
        num_curves=original_params['num_curves'],
        bezier_degree=original_params['bezier_degree'],
        num_samples=original_params.get('num_samples', 64),
        iterations=additional_iterations,
        mode=original_params['mode'],
        lr=original_params['lr'],
        image_name=original_meta.get('original_filename', 'image.png')
    )

    jobs[job_id] = {
        'status': 'queued',
        'progress': 0,
        'input_file': input_file,
        'continued_from': original_job_id,
        'params': original_params
    }

    thread = threading.Thread(
        target=run_continue_training,
        args=(job_id, original_job_id, additional_iterations, image_path, args)
    )
    thread.daemon = True
    thread.start()

    return jsonify({
        'job_id': job_id,
        'message': f'Continuing training from {original_job_id}',
        'additional_iterations': additional_iterations,
        'status_url': url_for('job_status', job_id=job_id)
    })


@app.route('/convert/<job_id>', methods=['POST'])
def regenerate_svg(job_id):
    """Regenerate SVG from an existing model checkpoint"""
    metadata_path = RESULT_FOLDER / job_id / "metadata.json"
    if not metadata_path.exists():
        return jsonify({'error': 'Job not found'}), 404

    model_path = RESULT_FOLDER / job_id / "model.pth.tar"
    if not model_path.exists():
        return jsonify({'error': 'Model checkpoint not found'}), 404

    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    try:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        input_file = metadata['input_file']
        image_path = UPLOAD_FOLDER / input_file
        gt_image = image_path_to_tensor(image_path).to(device)
        H, W = gt_image.shape[2], gt_image.shape[3]

        params = metadata['params']

        from gaussianimage_cholesky_svg import GaussianImage_Cholesky

        gaussian_model = GaussianImage_Cholesky(
            loss_type="L2",
            opt_type="adan",
            num_curves=params['num_curves'],
            num_samples=params.get('num_samples', 64),
            H=H, W=W,
            BLOCK_H=16, BLOCK_W=16,
            device=device,
            lr=params['lr'],
            mode=params['mode'],
            bezier_degree=params['bezier_degree'],
            quantize=False
        ).to(device)

        checkpoint = torch.load(model_path, map_location=device)
        model_dict = gaussian_model.state_dict()
        pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict and v.shape == model_dict[k].shape}
        model_dict.update(pretrained_dict)
        gaussian_model.load_state_dict(model_dict, strict=False)

        gaussian_model.eval()

        svg_path = RESULT_FOLDER / job_id / "result.svg"
        generate_svg_from_model(gaussian_model, str(svg_path), W, H)

        return jsonify({
            'success': True,
            'message': 'SVG regenerated successfully',
            'svg_file': f"{job_id}/result.svg"
        })

    except Exception as e:
        import traceback
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500


if __name__ == '__main__':
    templates_dir = Path('./templates')
    templates_dir.mkdir(exist_ok=True)

    app.run(host='0.0.0.0', port=5000, debug=True)
