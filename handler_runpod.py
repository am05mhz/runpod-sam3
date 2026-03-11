"""
RunPod Serverless handler for combined API
Adapted from combined_api.py for queue-based serverless execution.

This handler processes requests for: supersvg, sam3, bezier, layeredsvg

Input format:
{
    "module": "supersvg|sam3|bezier|layeredsvg",
    "image_url": "https://...",  # or "image_base64": "...",
    "params": { ... }  # module-specific parameters
}

Output format:
{
    "status": "success|error",
    "module": "...",
    "result_svg": "...",  # if applicable
    "result_png": "...",  # if applicable
    "result": { ... },   # module-specific results
    "error": "..."       # if status is error
}
"""

import os
import sys
import traceback
import inspect
import tempfile
import base64
import io
import asyncio
from pathlib import Path
from typing import Dict, Any

# utility serialization helper
import numpy as np

def sanitize_for_json(obj):
    """Recursively convert numpy arrays (and other non-serializable types) to JSON-friendly forms."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_for_json(v) for v in obj]
    elif isinstance(obj, tuple):
        return [sanitize_for_json(v) for v in obj]
    # add more conversions if needed
    else:
        return obj

# Import runpod SDK
try:
    import runpod
except ImportError:
    print("Warning: runpod not installed. Running in test mode.")
    runpod = None

# Local imports
from common_utils import (
    BASE_DIR,
    QUALITY_SETTINGS,
    N_SEGMENTS,
    import_module_from_path,
    load_image_from_url,
    make_id,
)

# Module path definitions
COMPONENT_PATHS = {
    "supersvg": str(Path(__file__).parent / "supersvg" / "server.py"),
    "sam3": str(Path(__file__).parent / "sam3" / "server.py"),
    "bezier": str(Path(__file__).parent / "bezier" / "server.py"),
    "layeredsvg": str(Path(__file__).parent / "layeredsvg" / "server.py"),
}


async def load_image_from_input(job_input: Dict[str, Any]):
    """
    Load image from either URL or base64 encoded data.
    Returns PIL Image object.
    """
    from PIL import Image
    
    if "image_url" in job_input:
        image_url = job_input["image_url"]
        try:
            # Use the load_image_from_url utility if available
            if asyncio.iscoroutinefunction(load_image_from_url):
                return await load_image_from_url(image_url)
            else:
                return load_image_from_url(image_url)
        except Exception as e:
            raise ValueError(f"Failed to load image from URL: {e}")
    
    elif "image_base64" in job_input:
        try:
            image_data = base64.b64decode(job_input["image_base64"])
            return Image.open(io.BytesIO(image_data)).convert("RGB")
        except Exception as e:
            raise ValueError(f"Failed to decode base64 image: {e}")
    
    else:
        raise ValueError("Provide either 'image_url' or 'image_base64' in input")


async def call_module_function(module, fn, module_args, module_kwargs):
    """
    Call a module function with intelligent argument handling.
    Falls back to different argument combinations if initial call fails.
    Handles both sync and async functions.
    """
    try:
        if asyncio.iscoroutinefunction(fn):
            result = await fn(*module_args, **module_kwargs)
        else:
            result = fn(*module_args, **module_kwargs)
        return result
    except TypeError as te:
        msg = str(te).lower()
        try:
            sig = inspect.signature(fn)
            params = sig.parameters
            accepts_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())

            if accepts_var_kw:
                try:
                    if asyncio.iscoroutinefunction(fn):
                        result = await fn(*module_args, **module_kwargs)
                    else:
                        result = fn(*module_args, **module_kwargs)
                    return result
                except TypeError:
                    pass

            if "unexpected keyword" in msg or "got an unexpected keyword" in msg or "unexpected keyword argument" in msg:
                allowed = {k: v for k, v in module_kwargs.items() if k in params}
                if allowed:
                    if asyncio.iscoroutinefunction(fn):
                        result = await fn(*module_args, **allowed)
                    else:
                        result = fn(*module_args, **allowed)
                    return result
                else:
                    if asyncio.iscoroutinefunction(fn):
                        result = await fn(*module_args)
                    else:
                        result = fn(*module_args)
                    return result

            if asyncio.iscoroutinefunction(fn):
                result = await fn(*module_args)
            else:
                result = fn(*module_args)
            return result
        except Exception as sig_error:
            print(f"Warning: inspect.signature failed: {sig_error}")
            print(f"Original TypeError: {te}")
            try:
                if asyncio.iscoroutinefunction(fn):
                    result = await fn(*module_args, **module_kwargs)
                else:
                    result = fn(*module_args, **module_kwargs)
                return result
            except TypeError:
                print("Fallback: calling without kwargs")
                if asyncio.iscoroutinefunction(fn):
                    result = await fn(*module_args)
                else:
                    result = fn(*module_args)
                return result


async def process_supersvg(job_input: Dict[str, Any]) -> Dict[str, Any]:
    """Process supersvg module request."""
    module_path = COMPONENT_PATHS.get("supersvg")
    if not module_path or not os.path.exists(module_path):
        raise RuntimeError("SuperSVG module path not found")
    
    module = import_module_from_path("supersvg_mod", module_path)
    
    # Load image
    image = await load_image_from_input(job_input)
    
    # Get parameters
    params = job_input.get("params", {})
    conf_thresh = params.get("conf_thresh", 0.3)
    num_rounds = params.get("num_rounds", 1)
    quality = params.get("quality", "default")
    mode = params.get("mode", "layered")
    labels = params.get("labels", None)
    
    # Create temporary output directory
    with tempfile.TemporaryDirectory() as tmpdir:
        # Save input image to temp file
        input_path = os.path.join(tmpdir, "input.png")
        image.save(input_path)
        
        output_svg = os.path.join(tmpdir, "output.svg")
        output_png = os.path.join(tmpdir, "output.png")
        
        # Determine which function to call
        if mode == "layered":
            callable_name = "process_image_sam3"
            module_kwargs = {
                "use_ollama": False,
                "use_labels": labels,
                "conf_thresh": conf_thresh,
                "num_rounds": num_rounds,
                "quality": quality,
            }
        else:
            callable_name = "process_image"
            module_kwargs = {
                "n_segments": QUALITY_SETTINGS.get(quality, N_SEGMENTS),
            }
        
        fn = getattr(module, callable_name, None)
        if fn is None:
            raise RuntimeError(f"Function {callable_name} not found in supersvg module")
        
        # Call the module function
        result = await call_module_function(
            module, fn,
            [input_path, output_svg, output_png],
            module_kwargs
        )
        
        # Prepare output
        svg_data = None
        png_data = None
        
        if os.path.exists(output_svg):
            with open(output_svg, 'rb') as f:
                svg_data = f.read()
        
        if os.path.exists(output_png):
            with open(output_png, 'rb') as f:
                png_data = f.read()
        
        return {
            "status": "success" if result and result.get("status") == "completed" else "error",
            "module": "supersvg",
            "result_svg": base64.b64encode(svg_data).decode('utf-8') if svg_data else None,
            "result_png": base64.b64encode(png_data).decode('utf-8') if png_data else None,
            "result": result,
        }


async def process_sam3(job_input: Dict[str, Any]) -> Dict[str, Any]:
    """Process sam3 module request."""
    module_path = COMPONENT_PATHS.get("sam3")
    if not module_path or not os.path.exists(module_path):
        raise RuntimeError("SAM3 module path not found")
    
    module = import_module_from_path("sam3_mod", module_path)
    
    # Load image
    image = await load_image_from_input(job_input)
    img_np = module.np.array(image)
    
    # Get parameters
    params = job_input.get("params", {})
    prompt = params.get("prompt", None)
    min_area = params.get("min_area", 0)
    
    # Call SAM inference
    if hasattr(module, "run_sam_inference"):
        segments = await call_module_function(
            module, module.run_sam_inference,
            [img_np, prompt, None, None, None, 0, 0.8, True],
            {}
        )
        return {
            "status": "success",
            "module": "sam3",
            "segments": segments,
        }
    else:
        raise RuntimeError("run_sam_inference function not found in sam3 module")


async def process_bezier(job_input: Dict[str, Any]) -> Dict[str, Any]:
    """Process bezier module request."""
    module_path = COMPONENT_PATHS.get("bezier")
    if not module_path or not os.path.exists(module_path):
        raise RuntimeError("Bezier module path not found")
    
    module = import_module_from_path("bezier_mod", module_path)
    
    # Load image
    image = await load_image_from_input(job_input)
    
    # Get parameters
    params = job_input.get("params", {})
    num_curves = params.get("num_curves", 512)
    iterations = params.get("iterations", 10000)
    mode = params.get("mode", "closed")
    
    # Create temporary output directory
    with tempfile.TemporaryDirectory() as tmpdir:
        # Save input image to temp file
        input_path = os.path.join(tmpdir, "input.png")
        image.save(input_path)
        
        job_id = make_id("bez")
        output_svg = os.path.join(tmpdir, "output.svg")
        output_png = os.path.join(tmpdir, "output.png")
        
        fn = getattr(module, "run_bezier_splatting", None)
        if fn is None:
            raise RuntimeError("run_bezier_splatting function not found in bezier module")
        
        # Call the module function
        result = await call_module_function(
            module, fn,
            [job_id, input_path, {"num_curves": num_curves, "iterations": iterations, "mode": mode}],
            {}
        )
        
        # Prepare output
        svg_data = None
        png_data = None
        
        if result and result.get("result_svg") and os.path.exists(result.get("result_svg")):
            with open(result.get("result_svg"), 'rb') as f:
                svg_data = f.read()
        
        if result and result.get("result_png") and os.path.exists(result.get("result_png")):
            with open(result.get("result_png"), 'rb') as f:
                png_data = f.read()
        
        return {
            "status": "success" if result and result.get("status") == "completed" else "error",
            "module": "bezier",
            "result_svg": base64.b64encode(svg_data).decode('utf-8') if svg_data else None,
            "result_png": base64.b64encode(png_data).decode('utf-8') if png_data else None,
            "result": result,
        }


async def process_layeredsvg(job_input: Dict[str, Any]) -> Dict[str, Any]:
    """Process layeredsvg module request."""
    module_path = COMPONENT_PATHS.get("layeredsvg")
    if not module_path or not os.path.exists(module_path):
        raise RuntimeError("LayeredSVG module path not found")
    
    module = import_module_from_path("layeredsvg_mod", module_path)
    
    # Get the action (segmentation or vectorization)
    action = job_input.get("action", "segment")
    
    if action == "segment":
        # Load image
        image = await load_image_from_input(job_input)
        
        # Get parameters
        params = job_input.get("params", {})
        keywords = params.get("keywords", [])
        
        # Create job directory
        job_id = make_id("lyr")
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "input.png")
            image.save(input_path)
            
            fn = getattr(module, "run_segmentation", None)
            if fn is None:
                raise RuntimeError("run_segmentation function not found in layeredsvg module")
            
            result = await call_module_function(
                module, fn,
                [job_id],
                {"keywords_with_conf": keywords}
            )
            
            return {
                "status": "success" if result and result.get("status") in ("layers_ready", "completed") else "error",
                "module": "layeredsvg",
                "action": "segment",
                "job_id": job_id,
                "layers_info": result.get("layers_info", []) if result else [],
                "result": result,
            }
    
    elif action == "vectorize":
        # Vectorization requires job_id
        job_id = job_input.get("job_id")
        if not job_id:
            raise ValueError("job_id required for vectorization action")
        
        params = job_input.get("params", {})
        selected_layers = params.get("selected_layers", [])
        quality = params.get("quality", "fast")
        
        if not selected_layers:
            raise ValueError("selected_layers required for vectorization")
        
        fn = getattr(module, "run_vectorization", None)
        if fn is None:
            raise RuntimeError("run_vectorization function not found in layeredsvg module")
        
        result = await call_module_function(
            module, fn,
            [job_id],
            {"selected_layers": selected_layers, "quality": quality}
        )
        
        # Prepare output
        svg_data = None
        png_data = None
        
        if result and result.get("result_svg"):
            if isinstance(result.get("result_svg"), str):
                with open(result.get("result_svg"), 'rb') as f:
                    svg_data = f.read()
            else:
                svg_data = result.get("result_svg")
        
        if result and result.get("result_png"):
            if isinstance(result.get("result_png"), str):
                with open(result.get("result_png"), 'rb') as f:
                    png_data = f.read()
            else:
                png_data = result.get("result_png")
        
        return {
            "status": "success" if result and result.get("status") == "completed" else "error",
            "module": "layeredsvg",
            "action": "vectorize",
            "job_id": job_id,
            "result_svg": base64.b64encode(svg_data).decode('utf-8') if svg_data else None,
            "result_png": base64.b64encode(png_data).decode('utf-8') if png_data else None,
            "result": result,
        }
    
    else:
        raise ValueError(f"Unknown action for layeredsvg: {action}")


async def handler(job):
    """
    Main RunPod handler function.
    
    Expected input format:
    {
        "module": "supersvg|sam3|bezier|layeredsvg",
        "image_url": "https://...",
        "params": {...}
    }
    """
    try:
        job_input = job.get("input", {})
        
        # Validate input
        module = job_input.get("module")
        if not module:
            return {
                "status": "error",
                "error": "Missing 'module' in input"
            }
        
        if module not in COMPONENT_PATHS:
            return {
                "status": "error",
                "error": f"Unknown module: {module}. Must be one of: {list(COMPONENT_PATHS.keys())}"
            }
        
        # Send progress update
        if runpod:
            runpod.serverless.progress_update(job, f"Processing {module} request")
        
        # Route to appropriate processor
        if module == "supersvg":
            result = await process_supersvg(job_input)
        elif module == "sam3":
            result = await process_sam3(job_input)
        elif module == "bezier":
            result = await process_bezier(job_input)
        elif module == "layeredsvg":
            result = await process_layeredsvg(job_input)
        else:
            return {
                "status": "error",
                "error": f"Unhandled module: {module}"
            }
        
        # Send final progress update
        if runpod:
            runpod.serverless.progress_update(job, "Completed")
        
        # sanitize result to make JSON serializable
        result = sanitize_for_json(result)
        return result
    
    except Exception as e:
        error_msg = str(e)
        print(f"Error in handler: {error_msg}")
        traceback.print_exc()
        
        return {
            "status": "error",
            "error": error_msg,
            "traceback": traceback.format_exc()
        }


# Start the RunPod serverless worker
if __name__ == "__main__":
    if runpod:
        runpod.serverless.start({"handler": handler})
    else:
        # For local testing without runpod installed
        print("RunPod SDK not available. Running test handler...")
        test_job = {
            "id": "test-job-001",
            "input": {
                "module": "supersvg",
                "image_url": "https://example.com/image.jpg",
                "params": {
                    "quality": "default",
                    "mode": "layered"
                }
            }
        }
        print("Test job:", test_job)
        # Uncomment to test:
        # result = asyncio.run(handler(test_job))
        # print("Result:", result)
