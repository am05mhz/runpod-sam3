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

app = FastAPI(title="Combined API - supersvg | sam3 | bezier | layeredsvg")
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
    while True:
        job_id = await JOB_QUEUE.get()
        job = JOBS.get(job_id)
        if job is None:
            JOB_QUEUE.task_done()
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

                    result = await asyncio.to_thread(lambda: fn(*job.get("args", []), **module_kwargs))

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
            JOB_QUEUE.task_done()


@app.on_event("startup")
async def startup_workers():
    loop = asyncio.get_event_loop()
    t = loop.create_task(worker_loop(0))
    WORKER_TASKS.append(t)
    print("Combined API: worker started")


@app.on_event("shutdown")
async def shutdown_workers():
    for t in WORKER_TASKS:
        t.cancel()
    await asyncio.gather(*WORKER_TASKS, return_exceptions=True)
    print("Combined API: workers shut down")


@app.get("/combined_output/{path:path}")
def serve_combined_output(path: str):
    full = os.path.join(OUTPUT_DIR, path)
    if not os.path.exists(full):
        raise HTTPException(status_code=404)
    return FileResponse(full)


@app.post("/supersvg/upload")
async def supersvg_upload(
    file: UploadFile = File(...),
    use_ollama: bool = Form(True),
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

    job_id = uid
    JOBS[job_id] = {
        "module": "supersvg",
        "callable": "process_image_sam3",
        "args": [inp, svg_out, png_out],
        "kwargs": {
            "use_ollama": use_ollama,
            "use_labels": labels,
            "conf_thresh": conf_thresh,
            "num_rounds": num_rounds,
            "quality": quality,
        },
        "svg_out": svg_out,
        "png_out": png_out,
        "status": "queued",
        "created_at": time.time(),
    }
    JOB_QUEUE.put_nowait(job_id)
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
        "args": [job_id, inp, None],
        "kwargs": {"num_curves": num_curves, "iterations": iterations, "mode": mode},
        "status": "queued",
        "created_at": time.time(),
    }
    JOB_QUEUE.put_nowait(job_id)
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
    JOB_QUEUE.put_nowait(job_id)
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
    print("Starting combined_api on port 8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)