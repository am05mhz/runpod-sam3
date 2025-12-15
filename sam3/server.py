from typing import List, Dict, Any, Optional
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn
import numpy as np
from PIL import Image
import io
import base64
import traceback
import asyncio
import torch
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
    Convert binary mask (H x W) to PNG encoded bytes, then to base64 string.
    Mask expected dtype=bool or {0,1}
    """
    # Ensure boolean
    mask_uint8 = (mask.astype(np.uint8) * 255)
    pil = Image.fromarray(mask_uint8, mode="L")
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    b = buf.getvalue()
    return base64.b64encode(b).decode("ascii")

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
async def run_sam_inference(image: np.ndarray, prompt: Optional[str] = None, min_area: int = 0) -> List[Dict[str, Any]]:
    """
    ASYNC version of SAM3 inference.
    Uses asyncio.to_thread() so GPU/CPU heavy work does not block main event loop.
    """

    async def _worker():
        """
        Heavy computation executed in a separate thread.
        All SAM3 operations stay inside this thread.
        """
        try:
            # Convert numpy array → PIL
            pil_img = Image.fromarray(image)

            # Preprocessing: bind image to state
            inference_state = _processor.set_image(pil_img)

            # Text prompt (SAM3 expects a string)
            if prompt is None:
                t_prompt = ""
            else:
                t_prompt = prompt

            output = _processor.set_text_prompt(state=inference_state, prompt=t_prompt)

            masks = output["masks"]    # NxHxW
            boxes = output["boxes"]    # Nx4
            scores = output["scores"]  # N

            results = []
            N = len(masks)

            for i in range(N):
                mask = masks[i]
                if isinstance(mask, torch.Tensor):
                    mask = mask.cpu().numpy()
                mask = mask.astype(bool)

                area = int(mask.sum())
                if area < min_area:
                    continue

                # boxes: [x_min, y_min, x_max, y_max]
                box = boxes[i]
                if isinstance(box, torch.Tensor):
                    box = box.cpu().numpy().tolist()
                else:
                    box = list(box)

                x_min, y_min, x_max, y_max = map(int, box)
                w = x_max - x_min
                h = y_max - y_min
                bbox = [x_min, y_min, w, h]

                score = float(scores[i]) if scores is not None else None

                results.append({
                    "mask": mask,
                    "bbox": bbox,
                    "score": score
                })

            return results

        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            raise RuntimeError(f"SAM3 inference error: {e}\n{tb}")

    # Run SAM3 inside worker thread
    return await asyncio.to_thread(_worker)

# -----------------------
# FastAPI endpoint
# -----------------------
@app.post("/segment")
async def segment_endpoint(
    image: UploadFile = File(...),
    prompt: Optional[str] = Form(None),
    min_area: Optional[int] = Form(0)
):
    # read image file asynchronously
    contents = await image.read()

    try:
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        img_np = np.array(img)
    except Exception as e:
        raise HTTPException(400, f"Could not decode image: {e}")

    # async SAM3 inference
    sam_outputs = await run_sam_inference(img_np, prompt=prompt, min_area=min_area)

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
