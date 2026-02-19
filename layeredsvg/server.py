"""
FastAPI App (Version 13) - Ollama + SAM3 Text-Prompted Layered Vectorization

Three-phase interactive pipeline:
  Phase 1: Ollama Qwen2.5 VL keyword detection
  Phase 2: SAM3 text-prompted segmentation (iterative)
  Phase 3: User confirms layers -> DiffVG vectorization

Run with: python app13.py
Access at: http://localhost:8000
Requires: conda activate lv13
"""

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import os
import argparse
import sys
from datetime import datetime
import re
import shutil
import threading
import importlib.util
import uvicorn
from starlette.requests import Request

# moved local imports to global
import time
import importlib.util
import torch
import traceback

# Add LayeredVectorization to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'LayeredVectorization'))


if __name__ == "__main__":
    app = FastAPI(title="V13 Layered Vectorization")
    app.config = {
        'UPLOAD_FOLDER': 'uploads',
        'RESULTS_FOLDER': 'results_v13',
        'MAX_CONTENT_LENGTH': 50 * 1024 * 1024
    }
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)

    # Mount static files
    app.mount("/static", StaticFiles(directory="static"), name="static")
    app.mount("/results", StaticFiles(directory="results_v11"), name="serve_result")

    # Setup Jinja2 templates
    templates = Jinja2Templates(directory="templates")

    processing_status = {}
    processing_threads = {}

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}

def get_main_v13():
    """Dynamically load main_v13 module."""
    module_path = os.path.join(os.path.dirname(__file__), 'LayeredVectorization', 'main_v13.py')
    spec = importlib.util.spec_from_file_location("main_v13", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def run_keyword_detection(run_id, **kwargs):
    """Background thread: run Ollama keyword detection."""
    try:
        if __name__ == "__main__":
            status = processing_status[run_id]
            status['status'] = 'detecting_keywords'
            status['progress'] = 5
            status['message'] = 'Starting keyword detection...'

        else:
            status = kwargs

        main_v13 = get_main_v13()

        def progress_cb(pct, msg):
            if __name__ == "__main__":
                status['progress'] = pct
                status['message'] = msg

        keywords = main_v13.detect_keywords_v13(
            status['filepath'],
            progress_cb=progress_cb
        )

        if __name__ == "__main__":
            status['keywords'] = keywords
            status['status'] = 'keywords_ready'
            status['progress'] = 100
            status['message'] = f'Detected {len(keywords)} objects'

    except Exception as e:
        traceback.print_exc()
        if __name__ == "__main__":
            status['status'] = 'error'
            status['message'] = str(e)
    finally:
        if __name__ == "__main__" and run_id in processing_threads:
            processing_threads.pop(run_id, None)

def run_segmentation(run_id, **kwargs):
    """Background thread: run SAM3 segmentation."""
    try:
        if __name__ == "__main__":
            status = processing_status[run_id]
            status['status'] = 'segmenting'
            status['progress'] = 5
            status['message'] = 'Starting segmentation...'

        else:
            status = kwargs

        main_v13 = get_main_v13()

        def progress_cb(pct, msg):
            if __name__ == "__main__":
                status['progress'] = pct
                status['message'] = msg

        keywords_with_conf = status.get('keywords_with_conf', [])

        layers_info = main_v13.segment_keywords_v13(
            status['filepath'],
            keywords_with_conf,
            status['run_folder'],
            progress_cb=progress_cb
        )

        if __name__ == "__main__":
            status['layers_info'] = layers_info
            status['status'] = 'layers_ready'
            status['progress'] = 100
            status['message'] = f'Segmented {len(layers_info)} layers'
            status['n_layers'] = len(layers_info)

    except Exception as e:
        traceback.print_exc()
        if __name__ == "__main__":
            status['status'] = 'error'
            status['message'] = str(e)
    finally:
        if __name__ == "__main__" and run_id in processing_threads:
            processing_threads.pop(run_id, None)

def run_vectorization(run_id, **kwargs):
    """Run V11 vectorization: SAM + Depth Anything decomposition + per-layer DiffVG"""
    original_dir = os.getcwd()

    try:
        if __name__ == "__main__":
            status = processing_status[run_id]
            status['status'] = 'processing'
            status['phase'] = 'vectorize'
            status['progress'] = 5
            status['message'] = 'Starting vectorization...'

        else:
            status = kwargs

        selected_layers = status.get('selected_layers', [])
        quality = status.get('quality', 'fast')

        input_path = os.path.join(status['run_folder'], status['filename'])
        run_folder_abs = os.path.abspath(status['run_folder'])
        workdir_path = os.path.join(run_folder_abs, 'workdir')
        os.makedirs(workdir_path, exist_ok=True)

        # Change to LayeredVectorization dir (required for imports)
        layered_vec_dir = os.path.join(os.path.dirname(__file__), 'LayeredVectorization')
        os.chdir(layered_vec_dir)

        # Setup workdir symlink
        workdir_existing = os.path.join(layered_vec_dir, 'workdir')
        if os.path.islink(workdir_existing):
            os.unlink(workdir_existing)
        elif os.path.exists(workdir_existing):
            shutil.rmtree(workdir_existing)

        try:
            os.symlink(workdir_path, workdir_existing)
        except OSError:
            shutil.copytree(workdir_path, workdir_existing)

        if __name__ == "__main__":
            status['progress'] = 10
            status['message'] = 'Loading models...'

        # Load config
        main_v13 = get_main_v13()

        config_map = {
            'fast': 'base_config_v11.yaml',
            'balanced': 'high_quality_config_v11.yaml',
            'balanced_plus': 'balanced_plus_config_v11.yaml',
            'medium': 'medium_quality_config_v11.yaml',
            'high': 'ultra_quality_config_v11.yaml',
            'best': 'best_quality_config_v11.yaml',
        }
        config_file = config_map.get(quality, 'base_config_v11.yaml')
        config_path = os.path.join(os.path.dirname(__file__),
                                    'LayeredVectorization', 'config', config_file)

        # Setup args
        # parser = argparse.ArgumentParser()
        # parser.add_argument("-c", "--config", type=str)
        # parser.add_argument("-timg", "--target_image", type=str)
        # parser.add_argument("-fsn", "--file_save_name", type=str, default="output")
        # parser.add_argument("--moge_version", type=str, default="v2")
        # parser.add_argument("--moge_resolution", type=str, default="High")
        # parser.add_argument("--max_layers", type=int, default=10)
        # parser.add_argument("--n_depth_clusters", type=int, default=3)
        # parser.add_argument("--min_mask_area", type=int, default=500)
        # parser.add_argument("--mask_dilation_px", type=int, default=3)
        # parser.add_argument("--background_method", type=str, default="depth")
        # parser.add_argument("--vtracer_enable", type=bool, default=True)
        # parser.add_argument("--staircase_area", type=float, default=1.5)
        # parser.add_argument("--corner_angle", type=float, default=135.0)
        # parser.add_argument("--simplify_error", type=float, default=2.0)
        # parser.add_argument("--smooth_iterations", type=int, default=0)
        # parser.add_argument("--skip_sds", action="store_true")

        # args = parser.parse_args([])
        # args.config = os.path.join(layered_vec_dir, "config", config_file)
        # args.target_image = os.path.abspath(os.path.join(original_dir, input_path))
        # args.file_save_name = "output"

        # Get max_layers from form data
        # max_layers_str = status.get('max_layers', '10')
        # args.max_layers = int(max_layers_str)

        # n_depth_clusters_str = status.get('n_depth_clusters', '3')
        # args.n_depth_clusters = int(n_depth_clusters_str)

        # args.moge_version = status.get('moge_version', args.moge_version)
        # args.moge_resolution = status.get('moge_resolution', args.moge_resolution)

        # # EDGE GAP FIX: Get mask dilation parameter
        # mask_dilation_str = status.get('mask_dilation_px', '3')
        # args.mask_dilation_px = int(mask_dilation_str)

        # # Background detection method: "depth" or "area"
        # args.background_method = status.get('background_method', 'depth')

        # args = load_config(args.config, args)

        args = argparse.Namespace()
        args.target_image = status['filepath']
        args.file_save_name = run_id
        args.mask_dilation_px = 3
        args.skip_sds = False
        args.vtracer_enable = True
        args.staircase_area = 1.5
        args.corner_angle = 135.0
        args.simplify_error = 2.0
        args.smooth_iterations = 0

        args = main_v13.load_config(config_path, args)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        main_v13.init_diffvg(device)

        # Make run_folder absolute before chdir
        run_folder_abs = os.path.abspath(status['run_folder'])

        # Setup workdir symlink
        lv_dir = os.path.join(os.path.dirname(__file__), 'LayeredVectorization')
        workdir_link = os.path.join(lv_dir, 'workdir')
        run_workdir = os.path.join(run_folder_abs, 'workdir')
        os.makedirs(run_workdir, exist_ok=True)

        if os.path.islink(workdir_link):
            os.unlink(workdir_link)
        elif os.path.exists(workdir_link):
            import shutil
            shutil.rmtree(workdir_link)
        os.symlink(os.path.abspath(run_workdir), workdir_link)

        os.chdir(lv_dir)
        
        if __name__ == "__main__":
            status['progress'] = 15
            status['message'] = 'Decomposing image into layers...'

        # Progress callback
        def progress_callback(progress, message):
            if __name__ == "__main__":
                status['progress'] = progress
                status['message'] = message

        result = main_v13.vectorize_confirmed_v13(
            device, args,
            run_folder_abs,
            selected_layers,
            progress_cb=progress_callback
        )

        if __name__ == "__main__":
            status['status'] = 'completed'
            status['progress'] = 100
            status['message'] = f'Complete! {result["n_layers"]} layers vectorized'
            status['result_svg'] = result.get('svg_path', '')
            status['result_layers'] = result.get('layers', [])
            status['n_vectorized'] = result.get('n_layers', 0)

        else:
            return {
                'status': 'completed',
                'n_layers': result['n_layers'],
                'result_svg': final_svg if os.path.exists(final_svg) else None,
                'result_png': fullsize_png if os.path.exists(fullsize_png) else None,
            }

    except Exception as e:
        traceback.print_exc()
        if __name__ == "__main__":
            status['status'] = 'error'
            status['message'] = f'Error: {str(e)}'

    finally:
        try:
            # Cleanup symlink
            layered_vec_dir = os.path.join(os.path.dirname(__file__), 'LayeredVectorization')
            workdir_existing = os.path.join(layered_vec_dir, 'workdir')
            if os.path.islink(workdir_existing):
                os.unlink(workdir_existing)

            os.chdir(original_dir)
        except:
            pass

        if __name__ == "__main__" and run_id in processing_threads:
            del processing_threads[run_id]


if __name__ == "__main__":
    @app.get("/")
    async def index(request: Request):
        return templates.TemplateResponse("index_v13.html", {"request": request})

    @app.post("/upload")
    async def upload_file(file: UploadFile = File(...)):
    if file is None or file.filename == '':
        return JSONResponse(status_code=400, content={"error": "No selected file"})

    if file and allowed_file(file.filename):
        run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        run_folder = os.path.join(app.config['RESULTS_FOLDER'], run_id)
        os.makedirs(run_folder, exist_ok=True)

        filename = f"input_{file.filename}"
        filepath = os.path.join(run_folder, filename)
        file.save(filepath)

        processing_status[run_id] = {
            'status': 'uploaded',
            'progress': 0,
            'message': 'File uploaded',
            'filename': filename,
            'filepath': filepath,
            'run_folder': run_folder,
        }

        return {'success': True, 'run_id': run_id, 'filename': filename}

    return JSONResponse(status_code=400, content={"error": "Invalid file type"})

    @app.post("/detect_keywords/{run_id}")
    async def detect_keywords(run_id: str):
        if run_id not in processing_status:
            return JSONResponse(status_code=400, content={"error": "Invalid run ID"})
        if run_id in processing_threads:
            return JSONResponse(status_code=400, content={"error": "Already processing"})

        thread = threading.Thread(target=run_keyword_detection, args=(run_id,))
        thread.daemon = True
        thread.start()
        processing_threads[run_id] = thread

        return {'success': True, 'message': 'Keyword detection started'}

    class JsonKeyword(BaseModel):
        keywords: str

    @app.post("/segment/{run_id}")
    async def segment_keywords(run_id: str, params: JsonKeyword):
        if run_id not in processing_status:
            return JSONResponse(status_code=400, content={"error": "Invalid run ID"})
        if run_id in processing_threads:
            return JSONResponse(status_code=400, content={"error": "Already processing"})

        if not keywords:
            return JSONResponse(status_code=400, content={"error": "No keywords provided"})

        # Store keywords with confidence for the background thread
        processing_status[run_id]['keywords_with_conf'] = params.keywords

        thread = threading.Thread(target=run_segmentation, args=(run_id,))
        thread.daemon = True
        thread.start()
        processing_threads[run_id] = thread

        return {'success': True, 'message': f'Segmenting {len(keywords)} keywords'}

    @app.get("/layers/{run_id}")
    async def get_layers(run_id: str):
        if run_id not in processing_status:
            return JSONResponse(status_code=400, content={"error": "Invalid run ID"})

        status = processing_status[run_id]
        if status['status'] not in ('layers_ready', 'vectorizing', 'completed'):
            return JSONResponse(status_code=400, content={"error": "Layers not ready yet"})

        return {
            'layers': status.get('layers_info', []),
            'n_layers': status.get('n_layers', 0),
            'merge_preview_url': 'merge_preview.png',
        }

    @app.get("/layer_asset/{run_id}/{filename:path}")
    async def serve_layer_asset(run_id: str, filename: str):
        if run_id in processing_status:
            layers_dir = os.path.join(processing_status[run_id]['run_folder'], 'layers')
        else:
            # Fallback: serve from results folder (for historical results)
            layers_dir = os.path.join(app.config['RESULTS_FOLDER'], run_id, 'layers')

        full_path = os.path.join(layers_dir, filename)
        if not os.path.exists(full_path):
            return JSONResponse(status_code=404, content={"error": "Not found"})
        return FileResponse(full_path)

    class JsonLayerConfig(BaseModel):
        selected_layers: list[int] = []
        quality: str = "fast"

    @app.post("/vectorize/{run_id}")
    async def vectorize_confirmed(run_id: str, params: JsonLayerConfig):
        if run_id not in processing_status:
            return JSONResponse(status_code=400, content={"error": "Invalid run ID"})
        if run_id in processing_threads:
            return JSONResponse(status_code=400, content={"error": "Already processing"})

        if not selected_layers:
            return JSONResponse(status_code=400, content={"error": "No layers selected"})

        processing_status[run_id]['selected_layers'] = params.selected_layers
        processing_status[run_id]['quality'] = params.quality

        thread = threading.Thread(target=run_vectorization, args=(run_id,))
        thread.daemon = True
        thread.start()
        processing_threads[run_id] = thread

        return {'success': True, 'message': f'Vectorizing {len(selected_layers)} layers with {quality} quality'}

    @app.get("/status/{run_id}")
    async def get_status(run_id: str):
        if run_id not in processing_status:
            return JSONResponse(status_code=404, content={"error": "Invalid run ID"})
        status = processing_status[run_id]
        resp = {
            'status': status['status'],
            'progress': status['progress'],
            'message': status['message'],
            'keywords': status.get('keywords', []),
            'n_layers': status.get('n_layers', 0),
        }
        if status['status'] == 'completed':
            resp['n_layers'] = status.get('n_vectorized', 0)
            svg_path = status.get('result_svg', '')
            if svg_path and os.path.isabs(svg_path):
                # Convert to relative path from results folder
                svg_path = os.path.relpath(svg_path, app.config['RESULTS_FOLDER'])
            resp['result_svg'] = svg_path
            # Check for PNG preview
            run_folder = status.get('run_folder', '')
            workdir = os.path.join(run_folder, 'workdir', run_id)
            for png_name in ['final_fullsize.png', 'final.png']:
                png_path = os.path.join(workdir, png_name)
                if os.path.exists(png_path):
                    resp['result_png'] = os.path.relpath(png_path, app.config['RESULTS_FOLDER'])
                    break
        return resp

    @app.get("/results")
    async def list_results():
        results = []
        results_dir = app.config['RESULTS_FOLDER']
        if os.path.exists(results_dir):
            for run_id in sorted(os.listdir(results_dir), reverse=True):
                run_folder = os.path.join(results_dir, run_id)
                if os.path.isdir(run_folder):
                    meta_path = os.path.join(run_folder, 'layers_meta.json')
                has_svg = os.path.exists(os.path.join(run_folder, 'workdir', run_id, 'final.svg'))
                has_meta = os.path.exists(meta_path)
                # Find input image
                input_files = [f for f in os.listdir(run_folder) if f.startswith('input_')]
                results.append({
                    'run_id': run_id,
                    'has_svg': has_svg,
                    'has_meta': has_meta,
                    'input_file': input_files[0] if input_files else None,
                })
        return results

    @app.get("/results/{file_path:path}")
    async def serve_result(file_path: str):
        full_path = os.path.join(app.config['RESULTS_FOLDER'], file_path)
        if not os.path.exists(full_path):
            return JSONResponse(status_code=404, content={"error": "File not found"})
        return FileResponse(full_path)

    @app.get("/view/{run_id}")
    async def view_result(request: Request, run_id: str):
        run_folder = os.path.join(app.config['RESULTS_FOLDER'], run_id)
        if not os.path.exists(run_folder):
            return JSONResponse(status_code=404, content={"error": "Run not found"})

        # Gather result file info
        workdir = os.path.join(run_folder, 'workdir', run_id)
        layers_dir = os.path.join(run_folder, 'layers')

        files = {
            'run_id': run_id,
            'has_svg': os.path.exists(os.path.join(workdir, 'final.svg')),
            'has_fullsize_svg': os.path.exists(os.path.join(workdir, 'final_fullsize.svg')),
            'has_png': os.path.exists(os.path.join(workdir, 'final.png')),
            'has_fullsize_png': os.path.exists(os.path.join(workdir, 'final_fullsize.png')),
        }

        # Find input image
        input_files = [f for f in os.listdir(run_folder) if f.startswith('input_')]
        files['input_file'] = input_files[0] if input_files else None

        # Load layer metadata
        meta_path = os.path.join(run_folder, 'layers_meta.json')
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            files['layers'] = meta.get('layers', [])
        else:
            files['layers'] = []

        return templates.TemplateResponse("view_result_v13.html", {"request": request, "files": files, "run_id": run_id})

    parser = argparse.ArgumentParser(description='V13: Ollama + SAM3 Layered Vectorization')
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', type=int, default=8000)
    cli_args = parser.parse_args()

    print("=" * 60)
    print("V13 Layered Vectorization - Ollama + SAM3")
    print("=" * 60)
    print(f"Server: http://localhost:{cli_args.port}")
    print(f"Prerequisites:")
    print(f"  - Ollama running: ollama serve")
    print(f"  - Model pulled: ollama pull qwen2.5vl:7b")
    print("=" * 60)

    uvicorn.run(app, host=cli_args.host, port=cli_args.port)