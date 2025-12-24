from typing import List, Dict, Any, Optional
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from pydantic import BaseModel
import uvicorn
import numpy as np
import io
import json
import base64
import uuid
import traceback
import asyncio
import aiohttp
import cv2
from PIL import Image
import torch
from transformers import Sam3Model, Sam3Processor, Sam3TrackerModel, Sam3TrackerProcessor
import os
import argparse
from dotenv import load_dotenv
from torchvision import transforms
from skimage.segmentation import slic
import pydiffvg
import cairosvg
from models.supersvg_coarse import SuperSVG_coarse

load_dotenv()

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
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp', 'bmp'}
CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), 'ckpts', 'coarse-model.pt')

device = "cuda" if torch.cuda.is_available() else "cpu"
sam3_model = None
sam3_processor = None
sam3_trackmodel = None
sam3_trackprocessor = None
supersvg_model = None

app = FastAPI(title="SAM3 and SuperSVG", version="1.0.0")

job_queue = asyncio.Queue()
job_results = {}

class JobStatus:
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    ERROR = "error"

# -----------------------
# Utility functions
# -----------------------
def load_models():
    load_sam3_model()
    load_supersvg()

def load_sam3_model(ptype: string, decive: string):
    global sam3_model
    global sam3_trackmodel
    global sam3_processor
    global sam3_trackprocessor  
    print("Loading SAM3 models...")
    if sam3_model is None:
        sam3_model = Sam3Model.from_pretrained("facebook/sam3").to(device)
        sam3_processor = Sam3Processor.from_pretrained("facebook/sam3")
    
    if sam3_trackmodel is None:
        sam3_trackmodel = Sam3TrackerModel.from_pretrained("facebook/sam3").to(device)
        sam3_trackprocessor = Sam3TrackerProcessor.from_pretrained("facebook/sam3")
    print("SAM3 model loaded.")

def load_supersvg(ptype: string, decive: string):
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

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def resize_mask_to_original(
    mask: np.ndarray,
    original_size: tuple[int, int],
) -> np.ndarray:
    """
    Safely resize a binary mask to original image size.
    Handles empty masks and invalid sizes.
    """
    orig_h, orig_w = original_size

    # ---- Validate original image size ----
    if orig_h <= 0 or orig_w <= 0:
        raise ValueError(f"Invalid original image size: {original_size}")

    # ---- Ensure mask is 2D ----
    mask = np.asarray(mask)
    # Remove singleton dimensions
    if mask.ndim == 3:
        mask = np.squeeze(mask)
    if mask.ndim != 2:
        raise ValueError(f"Mask must be 2D before resize, got {mask.shape}")

    h, w = mask.shape

    # ---- Empty mask: return empty mask at original size ----
    if h == 0 or w == 0 or mask.sum() == 0:
        return np.zeros((orig_h, orig_w), dtype=bool)

    # ---- No resize needed ----
    if h == orig_h and w == orig_w:
        return mask.astype(bool)

    # ---- Resize safely ----
    resized = cv2.resize(
        mask.astype(np.uint8),
        (orig_w, orig_h),   # (W, H)
        interpolation=cv2.INTER_NEAREST
    )

    return resized.astype(bool)

async def load_image_from_url(url: str) -> Image.Image:
    """
    Asynchronously download an image from URL and return PIL Image.
    """
    timeout = aiohttp.ClientTimeout(total=20)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.get(url) as resp:
            if resp.status != 200:
                raise HTTPException(
                    status_code=400,
                    detail=f"Failed to fetch image URL (status={resp.status})"
                )
            data = await resp.read()

    try:
        return Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as e:
        raise HTTPException(400, f"Invalid image from URL: {e}")

def get_server_port():
    # --- CLI ---
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=None, help="Port for the server")
    args, _ = parser.parse_known_args()

    if args.port:
        return args.port

    # --- System environment variables ---
    if "PORT" in os.environ:
        return int(os.environ["PORT"])

    # At this point, .env is already merged into os.environ by load_dotenv()

    # --- Default ---
    return 8000

def mask_to_base64_png(mask: np.ndarray) -> str:
    """
    Convert a SAM3 mask to base64 PNG.
    Handles masks of shape:
      (H, W)
      (1, H, W)
      (H, W, 1)
    """
    # Convert to numpy
    mask = np.asarray(mask)

    # --- Handle SAM3 multi-mask output (K, H, W) ---
    if mask.ndim == 3 and mask.shape[0] > 1:
        # Choose the mask with the largest foreground area
        areas = mask.sum(axis=(1, 2))
        mask = mask[areas.argmax()]

    # --- Handle singleton dimensions ---
    if mask.ndim == 3 and mask.shape[0] == 1:
        mask = mask[0]
    elif mask.ndim == 3 and mask.shape[-1] == 1:
        mask = mask[:, :, 0]

    if mask.ndim != 2:
        raise ValueError(f"Invalid mask shape after squeeze: {mask.shape}")

    # Convert to uint8 (0 or 255)
    mask_uint8 = (mask.astype(np.uint8)) * 255

    pil = Image.fromarray(mask_uint8, mode="L")
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")

def bbox_from_mask(mask: np.ndarray) -> List[int]:
    # mask is boolean 2D
    ys, xs = np.where(mask)
    if ys.size == 0:
        return [0, 0, 0, 0]
    y_min, y_max = int(ys.min()), int(ys.max())
    x_min, x_max = int(xs.min()), int(xs.max())
    w = x_max - x_min + 1
    h = y_max - y_min + 1
    return [int(x_min), int(y_min), int(w), int(h)]

def x1y1x2y2_to_xywh(box):
    x1, y1, x2, y2 = box
    return [x1, y1, x2 - x1, y2 - y1]

def clear_gpu_memory():
    """Clear GPU memory cache."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    print("GPU memory cleared")

# -----------------------
# SAM inference wrapper
# -----------------------
async def run_sam_inference(
    image: np.ndarray,
    prompt: str | None = None,
    points: list[list[int]] | None = None,
    point_labels: list[int] | None = None,
    box: list[int] | None = None,
    min_area: int = 0,
) -> list[dict]:
    """
    Async SAM3 inference using Hugging Face transformers SAM3 model.
    Supports:
      - text_prompt (string)
      - box prompt ([x0,y0,x1,y1]) optionally
    Returns list of dicts: {mask: np.ndarray, bbox: [x,y,w,h], score: float}
    """

    def _worker():
        try:
            # Convert numpy -> PIL
            pil_img = Image.fromarray(image)

            if prompt is not None:
                # Build inputs for processor
                # boxes and labels must be lists for HF API
                inputs = sam3_processor(
                    images=pil_img,
                    text=prompt,
                    input_boxes=[[box]] if box else None,
                    input_boxes_labels=[[1]] if box else None,
                    return_tensors="pt"
                ).to(device)

                with torch.no_grad():
                    outputs = sam3_model(**inputs)

                # Post-process to get masks & boxes
                results = sam3_processor.post_process_instance_segmentation(
                    outputs,
                    threshold=0.8,                          # score threshold
                    mask_threshold=0.8,                     # binarize mask
                    target_sizes=inputs.get("original_sizes").tolist()
                )[0]  # first image only

            else:
                labels = (point_labels or [1] * len(points)) if points else None
                inputs = sam3_trackprocessor(
                    images=pil_img,
                    input_points=[[points]] if points else None,
                    input_labels=[[labels]] if points else None,
                    input_boxes=[[box]] if box else None,
                    return_tensors="pt"
                ).to(sam3_trackmodel.device)

                with torch.no_grad():
                    outputs = sam3_trackmodel(**inputs)

                results = {
                    "masks": hf_sam3_tprocessor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"])[0],
                }

            masks = results["masks"]   # NxHxW boolean or [0,1] float

            out_list = []
            orig_h, orig_w = image.shape[:2]

            for i in range(len(masks)):
                mask = masks[i].cpu().numpy().astype(bool)

                # Clean/truncate tiny masks
                area = int(mask.sum())
                if area < min_area:
                    continue

                out_list.append({
                    "mask": mask,
                })

            return out_list

        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            raise RuntimeError(f"SAM3 HF inference error:\n{e}\n{tb}")

    # Run heavy work in thread
    return await asyncio.to_thread(_worker)

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

async def gpu_worker():
    while True:
        job = await job_queue.get()
        job_id = job["job_id"]

        job_results[job_id]["status"] = JobStatus.RUNNING

        try:
            match job.job_type:
                case 'super-svg':
                    process_image_sam3(**job["payload"])
                case 'sam3-segment':
                    run_sam_inference(**job["payload"])
            job_results[job_id]["status"] = JobStatus.DONE
        except Exception as e:
            traceback.print_exc()
            job_results[job_id]["status"] = JobStatus.ERROR
            job_results[job_id]["error"] = str(e)
        finally:
            job_queue.task_done()


# -----------------------
# FastAPI endpoints
# -----------------------
@app.on_event("startup")
async def startup_event():
    load_models()
    asyncio.create_task(gpu_worker())

@app.get("/status")
async def status():
    return {
        "device": device,
        "supersvg_loaded": supersvg_model is not None,
        "sam3_loaded": sam3_model is not None,
        "sam3_track_loaded": sam3_trackmodel is not None,
        "queue_size": job_queue.qsize()
    }
    
@app.post("/upload")
async def upload(
    file: UploadFile = File(...),
    use_ollama: bool = Form(True),
    conf_thresh: float = Form(0.3),
    num_rounds: int = Form(1),
    quality: str = Form("default"),
):
    if not allowed_file(file.filename):
        return JSONResponse({"error": "Invalid file type"}, status_code=400)

    if quality not in QUALITY_SETTINGS:
        quality = "default"

    job_id = str(uuid.uuid4())[:8]

    input_path = os.path.join(UPLOAD_FOLDER, f"{job_id}_input.png")
    output_svg = os.path.join(OUTPUT_FOLDER, f"{job_id}_output.svg")
    output_png = os.path.join(OUTPUT_FOLDER, f"{job_id}_output.png")

    with open(input_path, "wb") as f:
        f.write(await file.read())

    job_results[job_id] = {
        "status": JobStatus.PENDING,
        "svg_url": f"/output/{os.path.basename(output_svg)}",
        "png_url": f"/output/{os.path.basename(output_png)}",
    }

    await job_queue.put({
        "job_id": job_id,
        "job_type": "super-svg",
        "payload": dict(
            image_path=input_path,
            output_svg_path=output_svg,
            output_png_path=output_png,
            use_ollama=use_ollama,
            conf_thresh=conf_thresh,
            num_rounds=num_rounds,
            quality=quality,
        )
    })

    return {
        "job_id": job_id,
        "status_url": f"/jobs/{job_id}"
    }

@app.get("/jobs/{job_id}")
async def job_status(job_id: str):
    if job_id not in job_results:
        return JSONResponse({"error": "Job not found"}, status_code=404)
    return job_results[job_id]

@app.get("/download/{filename}")
async def download(filename: str):
    path = os.path.join(OUTPUT_FOLDER, filename)
    if not os.path.exists(path):
        return JSONResponse({"error": "File not found"}, status_code=404)

    return FileResponse(path, filename=filename)

@app.post("/segment")
async def segment_endpoint(
    # One of these must be provided
    image: UploadFile = File(None),
    image_url: Optional[str] = Form(None),

    # Prompts
    prompt: Optional[str] = Form(None),

    # Optional point prompts (JSON strings)
    points: Optional[str] = Form(None),
    point_labels: Optional[str] = Form(None),

    # Optional box prompt (JSON string)
    box: Optional[str] = Form(None),

    min_area: Optional[int] = Form(0),
):
    # -----------------------------
    # Load image (file OR URL)
    # -----------------------------
    if image is None and image_url is None:
        raise HTTPException(
            status_code=400,
            detail="Either 'image' file or 'image_url' must be provided"
        )

    if image is not None and image_url is not None:
        raise HTTPException(
            status_code=400,
            detail="Provide only one of 'image' or 'image_url'"
        )

    if image_url:
        pil_img = await load_image_from_url(image_url)
    else:
        contents = await image.read()
        try:
            pil_img = Image.open(io.BytesIO(contents)).convert("RGB")
        except Exception as e:
            raise HTTPException(400, f"Could not decode uploaded image: {e}")

    img_np = np.array(pil_img)

    original_h, original_w = img_np.shape[:2]

    pts = json.loads(points) if points else None
    lbls = json.loads(point_labels) if point_labels else None
    bx = json.loads(box) if box else None

    # async SAM3 inference
    sam_outputs = await run_sam_inference(
        image=img_np,
        prompt=prompt,
        points=pts,
        point_labels=lbls,
        box=bx,
        min_area=min_area,
    )

    segments_out = []
    for idx, seg in enumerate(sam_outputs):
        mask = seg["mask"]
        mask_b64 = mask_to_base64_png(mask)

        segments_out.append({
            "id": idx,
            # "bbox": seg["bbox"],
            "mask_base64": mask_b64,
            "area": int(mask.sum()),
            # "score": seg["score"]
        })

    return {"segments": segments_out, "num_segments": len(segments_out)}


# -----------------------
# Run server
# -----------------------
if __name__ == "__main__":
    port = get_server_port()
    print(f"Starting server on port {port}")
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=True)
