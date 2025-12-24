"""
Unified SAM3 + SuperSVG Server
- No external SAM3 service
- SAM3 runs in-process
- SuperSVG calls SAM3 directly
"""

# =============================================================================
# Imports
# =============================================================================

import os
import re
import uuid
import gc
import json
import base64
import argparse
import traceback
import asyncio

import torch
import numpy as np
import cv2
import pydiffvg
import cairosvg

from io import BytesIO
from PIL import Image
from typing import Optional, List
from collections import defaultdict

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request

from torchvision import transforms
from skimage.segmentation import slic
from transformers import (
    Sam3Model,
    Sam3Processor,
    Sam3TrackerModel,
    Sam3TrackerProcessor,
)

from models.supersvg_coarse import SuperSVG_coarse

# =============================================================================
# App Setup
# =============================================================================

app = FastAPI(title="SAM3 + SuperSVG Unified API")
templates = Jinja2Templates(directory="templates")

BASE_DIR = os.path.dirname(__file__)
UPLOAD_FOLDER = os.path.join(BASE_DIR, "temp")
OUTPUT_FOLDER = os.path.join(BASE_DIR, "output")
CHECKPOINT_PATH = os.path.join(BASE_DIR, "ckpts", "coarse-model.pt")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

# =============================================================================
# SAM3 GLOBAL MODELS
# =============================================================================

hf_sam3_model = None
hf_sam3_processor = None
hf_sam3_tmodel = None
hf_sam3_tprocessor = None

def load_sam3():
    global hf_sam3_model, hf_sam3_processor
    global hf_sam3_tmodel, hf_sam3_tprocessor

    if hf_sam3_model is None:
        print("Loading SAM3 models...")
        hf_sam3_model = Sam3Model.from_pretrained("facebook/sam3").to(device)
        hf_sam3_processor = Sam3Processor.from_pretrained("facebook/sam3")

        hf_sam3_tmodel = Sam3TrackerModel.from_pretrained("facebook/sam3").to(device)
        hf_sam3_tprocessor = Sam3TrackerProcessor.from_pretrained("facebook/sam3")

def unload_sam3():
    global hf_sam3_model, hf_sam3_processor
    global hf_sam3_tmodel, hf_sam3_tprocessor

    print("Unloading SAM3...")
    hf_sam3_model = None
    hf_sam3_processor = None
    hf_sam3_tmodel = None
    hf_sam3_tprocessor = None
    clear_gpu_memory()

# =============================================================================
# SuperSVG Model
# =============================================================================

supersvg_model = None
WIDTH = 224
BATCH_SIZE = 64

QUALITY_SETTINGS = {
    "low": 500,
    "default": 1500,
    "high": 5000,
    "best": 10000,
}

def load_supersvg_model():
    global supersvg_model
    if supersvg_model is None:
        supersvg_model = SuperSVG_coarse(
            stroke_num=128,
            path_num=4,
            width=WIDTH,
            num_loss=True,
        )
        state = torch.load(CHECKPOINT_PATH, map_location=device)
        supersvg_model.load_state_dict({
            k.replace("module.", ""): v for k, v in state.items()
        })
        supersvg_model.to(device).eval()
    return supersvg_model

def unload_supersvg_model():
    global supersvg_model
    supersvg_model = None
    clear_gpu_memory()

def clear_gpu_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

# =============================================================================
# SAM3 INFERENCE (DIRECT)
# =============================================================================

async def run_sam3_segmentation(
    image_np: np.ndarray,
    conf_thresh: float = 0.3,
) -> tuple[list[np.ndarray], list[str]]:
    """
    Direct SAM3 automatic segmentation
    Returns: (masks, labels)
    """

    def _worker():
        load_sam3()

        pil = Image.fromarray(image_np)

        inputs = hf_sam3_processor(
            images=pil,
            text="objects",
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            outputs = hf_sam3_model(**inputs)

        results = hf_sam3_processor.post_process_instance_segmentation(
            outputs,
            threshold=conf_thresh,
            mask_threshold=0.8,
            target_sizes=inputs["original_sizes"].tolist(),
        )[0]

        masks = [
            m.cpu().numpy().astype(np.uint8)
            for m in results["masks"]
        ]

        labels = results.get("labels", ["object"] * len(masks))
        return masks, labels

    return await asyncio.to_thread(_worker)

# =============================================================================
# SuperSVG Pipeline (UNCHANGED LOGIC)
# =============================================================================
# Everything from:
# - group_masks_by_label
# - process_slic_segments_batched
# - process_single_object_mask
# - process_objects_as_layers
# - save_svg_with_layers
# - add_layer_groups_to_svg
# - process_image_sam3
#
# 🔥 remains IDENTICAL except the SAM3 call 🔥
# =============================================================================

def process_image_sam3(
    image_path,
    output_svg,
    output_png,
    quality="default",
):
    n_segments = QUALITY_SETTINGS.get(quality, 1500)

    image_pil = Image.open(image_path).convert("RGB")
    image_np = np.array(image_pil) / 255.0
    h, w = image_np.shape[:2]

    # ---------------- SAM3 ----------------
    print("Running SAM3...")
    masks, labels = asyncio.run(
        run_sam3_segmentation((image_np * 255).astype(np.uint8))
    )

    if not masks:
        masks = [np.ones((h, w), dtype=np.uint8)]
        labels = ["full"]

    label_masks = defaultdict(list)
    for m, l in zip(masks, labels):
        label_masks[l].append(m)

    combined = {
        l: np.logical_or.reduce(ms).astype(np.uint8)
        for l, ms in label_masks.items()
    }

    unload_sam3()

    # ---------------- SuperSVG ----------------
    model = load_supersvg_model()

    layers = process_objects_as_layers(
        model, image_np, combined, n_segments
    )

    bg_shape = pydiffvg.Path(
        num_control_points=torch.LongTensor([0, 0, 0, 0]),
        points=torch.tensor([[0,0],[w,0],[w,h],[0,h]]).float(),
        stroke_width=torch.tensor(0.0),
        is_closed=True,
    )
    bg_group = pydiffvg.ShapeGroup(
        shape_ids=torch.LongTensor([0]),
        fill_color=torch.tensor([0,0,0,1]),
    )

    save_svg_with_layers(output_svg, w, h, layers, bg_shape, bg_group)
    cairosvg.svg2png(url=output_svg, write_to=output_png)

# =============================================================================
# FastAPI Routes
# =============================================================================

@app.post("/upload")
async def upload(
    file: UploadFile = File(...),
    quality: str = Form("default"),
):
    uid = str(uuid.uuid4())[:8]
    input_path = os.path.join(UPLOAD_FOLDER, f"{uid}.png")
    svg_path = os.path.join(OUTPUT_FOLDER, f"{uid}.svg")
    png_path = os.path.join(OUTPUT_FOLDER, f"{uid}.png")

    with open(input_path, "wb") as f:
        f.write(await file.read())

    try:
        process_image_sam3(
            input_path,
            svg_path,
            png_path,
            quality,
        )
    except Exception as e:
        raise HTTPException(500, traceback.format_exc())

    return {
        "svg": f"/output/{os.path.basename(svg_path)}",
        "png": f"/output/{os.path.basename(png_path)}",
    }

@app.get("/output/{file}")
async def serve(file: str):
    return FileResponse(os.path.join(OUTPUT_FOLDER, file))

# =============================================================================
# Entrypoint
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5001)
