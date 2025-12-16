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
import torch
import cv2
from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
import os
import argparse
from dotenv import load_dotenv

load_dotenv()

# -------------------------
# Global SAM3 model (loaded once)
# -------------------------
print("Loading SAM3 model...")

_model = build_sam3_image_model()
_processor = Sam3Processor(_model)

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
    Resize a binary mask to original image size.

    Args:
        mask: (H, W) boolean or uint8 mask
        original_size: (orig_h, orig_w)

    Returns:
        (orig_h, orig_w) boolean mask
    """
    orig_h, orig_w = original_size

    if mask.shape[0] == orig_h and mask.shape[1] == orig_w:
        return mask.astype(bool)

    mask_uint8 = mask.astype(np.uint8)

    resized = cv2.resize(
        mask_uint8,
        (orig_w, orig_h),   # cv2 uses (W, H)
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

def read_imagefile_to_numpy(file: UploadFile) -> np.ndarray:
    contents = file.file.read()
    try:
        img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read image: {e}")
    arr = np.array(img)
    return arr

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

    # Remove singleton dimensions
    if mask.ndim == 3:
        mask = np.squeeze(mask)

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

# -----------------------
# SAM inference wrapper
# -----------------------
async def run_sam_inference(
    image: np.ndarray,
    prompt: Optional[str] = None,
    points: Optional[List[List[int]]] = None,        # [[x,y], ...]
    point_labels: Optional[List[int]] = None,        # [1, 0, ...]
    box: Optional[List[int]] = None,                 # [x1,y1,x2,y2]
    min_area: int = 0,
) -> List[Dict[str, Any]]:
    """
    Async SAM3 inference with:
      - text (concept) prompt
      - visual prompts: points + labels, and box
    """
    def _worker():
        try:
            pil_img = Image.fromarray(image)

            # 1) Set image
            inference_state = _processor.set_image(pil_img)

            # 2) Handle prompts
            # First handle text prompt (concept prompt)
            if prompt:
                out = _processor.set_text_prompt(
                    state=inference_state,
                    prompt=prompt,
                )
            else:
                out = None

            # Then handle visual prompts
            if points is not None or box is not None:
                # Convert points
                pt_coords = None
                pt_lbls = None
                if points is not None:
                    pt_coords = torch.tensor(points, dtype=torch.float32)
                    if point_labels is None:
                        pt_lbls = torch.ones(len(points), dtype=torch.int64)
                    else:
                        pt_lbls = torch.tensor(point_labels, dtype=torch.int64)

                # Convert box
                box_coords = None
                if box is not None:
                    box_coords = torch.tensor([box], dtype=torch.float32)

                vis_out = _processor.set_visual_prompt(
                    state=inference_state,
                    point_coords=pt_coords,
                    point_labels=pt_lbls,
                    box_coords=box_coords,
                )
                # If text was also given, concatenate outputs
                if out is None:
                    out = vis_out
                else:
                    # Merge text + visual results
                    # We simply append masks/boxes/scores
                    out["masks"] += vis_out.get("masks", [])
                    out["boxes"] += vis_out.get("boxes", [])
                    out["scores"] += vis_out.get("scores", [])

            if out is None:
                raise RuntimeError("No prompt provided")

            masks = out.get("masks", [])
            boxes = out.get("boxes", [])
            scores = out.get("scores", [])

            results = []
            for i in range(len(masks)):
                mask = masks[i]
                if isinstance(mask, torch.Tensor):
                    mask = mask.cpu().numpy()
                mask = resize_mask_to_original(
                    mask=mask,
                    original_size=(image.shape[0], image.shape[1]),
                )

                area = int(mask.sum())
                if area < min_area:
                    continue

                box_i = boxes[i]
                if isinstance(box_i, torch.Tensor):
                    box_i = box_i.cpu().numpy().tolist()

                # Convert xyxy to [x, y, w, h]
                x1, y1, x2, y2 = map(int, box_i)
                bbox = [x1, y1, x2 - x1, y2 - y1]

                score_val = float(scores[i]) if scores is not None else None

                results.append({
                    "mask": mask,
                    "bbox": bbox,
                    "score": score_val,
                })

            return results

        except Exception as e:
            import traceback
            raise RuntimeError(f"SAM3 inference error:\n{traceback.format_exc()}")

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
            "bbox": seg["bbox"],
            "mask_base64": mask_b64,
            "area": int(mask.sum()),
            "score": seg["score"]
        })

    return {"segments": segments_out, "num_segments": len(segments_out)}


# -----------------------
# Run server
# -----------------------
if __name__ == "__main__":
    port = get_server_port()
    print(f"Starting server on port {port}")
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=True)
