"""
Combined FastAPI entrypoint that aggregates routes from each component folder,
prefixes routes with the original filename (e.g. /supersvg/upload, /sam3/segment),
and uses a GPU-safe worker + queue system.

Place this file at:
  /home/amos/docs/repo/container/sam3-qwen-llm-dock/combined_api.py
"""
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from starlette.requests import Request
import os
import asyncio
import shutil
import traceback
from pathlib import Path
import uvicorn
import time
import threading
import queue
import argparse
import inspect
from contextlib import asynccontextmanager

# local common utils
from common_utils import (
    BASE_DIR,
    JOB_QUEUE,
    JOBS,
    GPU_SEMAPHORE,
    WORKER_TASKS,
    import_module_from_path,
    safe_mkdir,
    make_id,
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage app startup and shutdown with lifespan context manager"""
    # Startup
    def run_worker_thread():
        asyncio.run(worker_loop(0))
    
    worker_thread = threading.Thread(target=run_worker_thread, daemon=True)
    worker_thread.start()
    print("Combined API: worker started in separate thread")
    
    yield
    
    # Shutdown
    for t in WORKER_TASKS:
        t.cancel()
    await asyncio.gather(*WORKER_TASKS, return_exceptions=True)
    print("Combined API: workers shut down")


app = FastAPI(
    title="Combined API - supersvg | sam3 | bezier | layeredsvg",
    lifespan=lifespan,
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))

COMPONENT_PATHS = {
    "supersvg": str(Path(__file__).parent / "supersvg" / "server.py"),
    "sam3": str(Path(__file__).parent / "sam3" / "server.py"),
    "bezier": str(Path(__file__).parent / "bezier" / "server.py"),
    "layeredsvg": str(Path(__file__).parent / "layeredsvg" / "app11.py"),
}

UPLOAD_DIR = str(Path(__file__).parent / "combined_temp")
OUTPUT_DIR = str(Path(__file__).parent / "combined_output")
safe_mkdir(UPLOAD_DIR)
safe_mkdir(OUTPUT_DIR)


async def worker_loop(worker_idx: int = 0):
    print(f"[worker-{worker_idx}] started")
    loop = asyncio.get_event_loop()
    while True:
        # Use asyncio to wrap blocking queue.get() with timeout
        try:
            job_id = await asyncio.wait_for(
                loop.run_in_executor(None, lambda: JOB_QUEUE.get(timeout=1.0)),
                timeout=2.0
            )
        except (asyncio.TimeoutError, queue.Empty):
            # No job available, keep waiting
            continue
        
        job = JOBS.get(job_id)
        if job is None:
            continue

        job["status"] = "started"
        try:
            async with GPU_SEMAPHORE:
                job["status"] = "processing"
                try:
                    module_key = job.get("module")
                    func_name = job.get("callable")
                    module_path = COMPONENT_PATHS.get(module_key)
                    if not module_path or not os.path.exists(module_path):
                        raise RuntimeError(f"Module path not found for '{module_key}'")

                    module = import_module_from_path(f"{module_key}_mod", module_path)

                    fn = getattr(module, func_name, None)
                    if fn is None:
                        for alt in ("process_image_sam3", "process_image", "run_bezier_splatting", "run_vectorization"):
                            if hasattr(module, alt):
                                fn = getattr(module, alt)
                                break
                    if fn is None:
                        raise RuntimeError(f"No callable found for job {job_id} in {module_key}")

                    module_kwargs = job.get("kwargs", {})
                    module_args = job.get("args", [])

                    def _call_target():
                        try:
                            return fn(*module_args, **module_kwargs)
                        except TypeError as te:
                            # Try filtering kwargs to only accepted parameters
                            try:
                                sig = inspect.signature(fn)
                                params = sig.parameters
                                allowed = {k: v for k, v in module_kwargs.items() if k in params}
                                if allowed:
                                    return fn(*module_args, **allowed)
                                else:
                                    # No kwargs match the signature, try without any
                                    return fn(*module_args)
                            except Exception as sig_error:
                                # Log the signature inspection failure and original error
                                print(f"Warning: inspect.signature failed: {sig_error}")
                                print(f"Original TypeError: {te}")
                                print(f"Attempting call with all original kwargs: {module_kwargs.keys()}")
                                # Try one more time with the original call
                                try:
                                    return fn(*module_args, **module_kwargs)
                                except TypeError:
                                    # If it still fails, try without kwargs as last resort
                                    print(f"Fallback: calling without kwargs")
                                    return fn(*module_args)

                    result = await asyncio.to_thread(_call_target)

                    job["status"] = "completed"
                    job["result"] = result
                    svg = job.get("svg_out")
                    png = job.get("png_out")
                    if svg and os.path.exists(svg):
                        job["svg_url"] = f"/combined_output/{os.path.basename(svg)}"
                    if png and os.path.exists(png):
                        job["png_url"] = f"/combined_output/{os.path.basename(png)}"

                except Exception as e:
                    job["status"] = "error"
                    job["error"] = str(e)
                    traceback.print_exc()
                finally:
                    if 'module' in locals() and hasattr(module, "unload_supersvg_model"):
                        try:
                            getattr(module, "unload_supersvg_model")()
                        except Exception:
                            pass
        except asyncio.CancelledError:
            job["status"] = "cancelled"
            raise
        finally:
            pass


@app.get("/combined_output/{path:path}")
def serve_combined_output(path: str):
    full = os.path.join(OUTPUT_DIR, path)
    if not os.path.exists(full):
        raise HTTPException(status_code=404)
    return FileResponse(full)

@app.get("/supersvg")
async def supersvg_home(request: Request):
    return templates.TemplateResponse("supersvg.html", {"request": request})

@app.get("/bezier")
async def bezier_home(request: Request):
    return templates.TemplateResponse("bezier.html", {"request": request})

@app.get("/layeredsvg")
async def layeredsvg_home(request: Request):
    return templates.TemplateResponse("layeredsvg.html", {"request": request})

@app.get("/layeredsvg/view/{run_id}")
async def layeredsvg_view(request: Request, run_id: str):
    return templates.TemplateResponse("layeredsvg_view.html", {"request": request, "run_id": run_id})

@app.post("/supersvg/upload")
async def supersvg_upload(
    file: UploadFile = File(...),
    conf_thresh: float = Form(0.3),
    num_rounds: int = Form(1),
    quality: str = Form("default"),
    mode: str | None = Form("layered"),
    labels: str | None = Form(None),
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file")
    uid = make_id("sup")
    ext = file.filename.rsplit(".", 1)[-1]
    inp = os.path.join(UPLOAD_DIR, f"{uid}_input.{ext}")
    svg_out = os.path.join(OUTPUT_DIR, f"{uid}_output.svg")
    png_out = os.path.join(OUTPUT_DIR, f"{uid}_output.png")
    with open(inp, "wb") as f:
        f.write(await file.read())

    if mode == "layered":
        callable = "process_image_sam3"
        module_kwargs = {
            "use_ollama": False,
            "use_labels": labels,
            "conf_thresh": conf_thresh,
            "num_rounds": num_rounds,
            "quality": quality,
        }
    else:
        callable = "process_image"
        module_kwargs = {
            "n_segments": supersvg.QUALITY_SETTINGS.get(quality, supersvg.N_SEGMENTS),
        }

    job_id = uid
    JOBS[job_id] = {
        "module": "supersvg",
        "callable": callable,
        "args": [inp, svg_out, png_out],
        "kwargs": module_kwargs,
        "svg_out": None,
        "png_out": None,
        "status": "queued",
        "created_at": time.time(),
    }
    JOB_QUEUE.put(job_id)
    return {"job_id": job_id, "poll_url": f"/job/{job_id}/status"}


@app.post("/sam3/segment")
async def sam3_segment(image: UploadFile = File(None), image_url: str | None = Form(None), prompt: str | None = Form(None), min_area: int = Form(0)):
    module_path = COMPONENT_PATHS.get("sam3")
    if not module_path or not os.path.exists(module_path):
        raise HTTPException(status_code=500, detail="sam3 module missing")
    sam3_mod = import_module_from_path("sam3_mod", module_path)
    if image is None and image_url is None:
        raise HTTPException(status_code=400, detail="Provide image or image_url")
    if image_url:
        if hasattr(sam3_mod, "load_image_from_url") and hasattr(sam3_mod, "run_sam_inference"):
            pil = await sam3_mod.load_image_from_url(image_url)
            img_np = sam3_mod.numpy.array(pil) if hasattr(sam3_mod, "numpy") else None
        else:
            raise HTTPException(status_code=500, detail="sam3 helpers missing")
    else:
        contents = await image.read()
        try:
            pil_img = sam3_mod.Image.open(sam3_mod.io.BytesIO(contents)).convert("RGB")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image: {e}")
        img_np = sam3_mod.np.array(pil_img)
        out = await sam3_mod.run_sam_inference(img_np, prompt, None, None, None, 0, 0.8, True)
        return JSONResponse(content={"segments": out})


@app.post("/bezier/upload")
async def bezier_upload(
    file: UploadFile = File(...),
    num_curves: int = Form(512),
    iterations: int = Form(10000),
    mode: str = Form("closed"),
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file")
    uid = make_id("bez")
    ext = file.filename.rsplit(".", 1)[-1]
    inp = os.path.join(UPLOAD_DIR, f"{uid}_input.{ext}")
    with open(inp, "wb") as f:
        f.write(await file.read())

    job_id = uid
    JOBS[job_id] = {
        "module": "bezier",
        "callable": "run_bezier_splatting",
        "args": [job_id, inp, {"num_curves": num_curves, "iterations": iterations, "mode": mode}],
        "kwargs": {},
        "svg_out": None,
        "png_out": None,
        "status": "queued",
        "created_at": time.time(),
    }
    JOB_QUEUE.put(job_id)
    return {"job_id": job_id, "poll_url": f"/job/{job_id}/status"}


@app.post("/layeredsvg/upload")
async def layeredsvg_upload(file: UploadFile = File(...), quality: str = Form("fast")):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file")
    uid = make_id("lv")
    ext = file.filename.rsplit(".", 1)[-1]
    run_folder = os.path.join(OUTPUT_DIR, uid)
    os.makedirs(run_folder, exist_ok=True)
    inp = os.path.join(run_folder, f"input_{file.filename}")
    with open(inp, "wb") as f:
        f.write(await file.read())

    job_id = uid
    JOBS[job_id] = {
        "module": "layeredsvg",
        "callable": "run_vectorization",
        "args": [job_id],
        "kwargs": {},
        "status": "queued",
        "created_at": time.time(),
    }
    JOB_QUEUE.put(job_id)
    return {"job_id": job_id, "poll_url": f"/job/{job_id}/status"}


@app.get("/job/{job_id}/status")
async def job_status(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return {
        "job_id": job_id,
        "module": job.get("module"),
        "status": job.get("status"),
        "svg_url": job.get("svg_url"),
        "png_url": job.get("png_url"),
        "error": job.get("error"),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combined API server")
    parser.add_argument("--port", type=int, default=8000, help="Port to run the server on (default: 8000)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to (default: 0.0.0.0)")
    args = parser.parse_args()
    
    print(f"Starting combined_api on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)