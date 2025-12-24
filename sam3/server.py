from typing import List, Dict, Any, Optional
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn
import numpy as np
import io
import json
import base64
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

load_dotenv()

# -------------------------
# Global SAM3 model (loaded once)
# -------------------------
print("Loading SAM3 model...")

device = "cuda" if torch.cuda.is_available() else "cpu"
hf_sam3_tmodel = Sam3TrackerModel.from_pretrained("facebook/sam3").to(device)
hf_sam3_tprocessor = Sam3TrackerProcessor.from_pretrained("facebook/sam3")
hf_sam3_model = Sam3Model.from_pretrained("facebook/sam3").to(device)
hf_sam3_processor = Sam3Processor.from_pretrained("facebook/sam3")

print("SAM3 model loaded.")

app = FastAPI(title="SAM segmentation server")

# -----------------------
# Utility functions
# -----------------------
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

def image_from_base64(b64_string):
    """Convert base64 string to PIL Image."""
    image_data = base64.b64decode(b64_string)
    return Image.open(BytesIO(image_data)).convert('RGB')

def masks_to_base64(masks):
    """Convert numpy masks to base64 encoded compressed format."""
    buffer = BytesIO()
    np.savez_compressed(buffer, masks=masks)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode('utf-8')

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
    for_svg: bool = False,
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
                hf_inputs = hf_sam3_processor(
                    images=pil_img,
                    text=prompt,
                    input_boxes=[[box]] if box else None,
                    input_boxes_labels=[[1]] if box else None,
                    return_tensors="pt"
                ).to(device)

                with torch.no_grad():
                    outputs = hf_sam3_model(**hf_inputs)

                # Post-process to get masks & boxes
                results = hf_sam3_processor.post_process_instance_segmentation(
                    outputs,
                    threshold=0.8,                          # score threshold
                    mask_threshold=0.8,                     # binarize mask
                    target_sizes=hf_inputs.get("original_sizes").tolist()
                )[0]  # first image only

                if for_svg:
                    raw_masks = results['masks'].cpu().numpy()
                    raw_scores = results['scores'].cpu().numpy().tolist()

                    if len(raw_masks) > 0:
                        masks_array = raw_masks.astype(np.uint8)
                    else:
                        masks_array = np.array([])

                    masks_b64 = masks_to_base64(masks_array)

                    return {
                        'masks': masks_b64,
                        'scores': raw_scores
                    }

            else:
                labels = (point_labels or [1] * len(points)) if points else None
                hf_inputs = hf_sam3_tprocessor(
                    images=pil_img,
                    input_points=[[points]] if points else None,
                    input_labels=[[labels]] if points else None,
                    input_boxes=[[box]] if box else None,
                    return_tensors="pt"
                ).to(hf_sam3_tmodel.device)

                with torch.no_grad():
                    outputs = hf_sam3_tmodel(**hf_inputs)

                results = {
                    "masks": hf_sam3_tprocessor.post_process_masks(outputs.pred_masks.cpu(), hf_inputs["original_sizes"])[0],
                    # "boxes": [],
                    # "scores": [],
                }

            masks = results["masks"]   # NxHxW boolean or [0,1] float
            # boxes_out = results["boxes"]  # Nx4 xyxy
            # scores = results["scores"]    # N

            out_list = []
            orig_h, orig_w = image.shape[:2]

            for i in range(len(masks)):
                mask = masks[i].cpu().numpy().astype(bool)

                # Clean/truncate tiny masks
                area = int(mask.sum())
                if area < min_area:
                    continue

                # Convert box xyxy -> [x,y,w,h]
                # x0, y0, x1, y1 = map(int, boxes_out[i].tolist())
                # bbox = [x0, y0, x1 - x0, y1 - y0]

                out_list.append({
                    "mask": mask,
                    # "bbox": bbox,
                    # "score": float(scores[i].item()) if scores is not None else None
                })

            return out_list

        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            raise RuntimeError(f"SAM3 HF inference error:\n{e}\n{tb}")

    # Run heavy work in thread
    return await asyncio.to_thread(_worker)

# -----------------------
# FastAPI endpoint
# -----------------------
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

@app.post("/segment_text")
async def segment_text_endpoint(
    # One of these must be provided
    image: str = Form(None),
    image_url: Optional[str] = Form(None),

    # Prompts
    prompt: Optional[str] = Form(None),

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
        try:
            pil_img = image_from_base64(image)
        except Exception as e:
            raise HTTPException(400, f"Could not decode uploaded image: {e}")

    img_np = np.array(pil_img)

    original_h, original_w = img_np.shape[:2]

    # async SAM3 inference
    sam_outputs = await run_sam_inference(
        image=img_np,
        prompt=prompt,
        min_area=min_area,
        for_svg=True,
    )

    return {"masks": sam_outputs["masks"], "num_masks": len(sam_outputs["masks"]), "scores": sam_outputs["scores"]}


# -----------------------
# Run server
# -----------------------
if __name__ == "__main__":
    port = get_server_port()
    print(f"Starting server on port {port}")
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=True)
