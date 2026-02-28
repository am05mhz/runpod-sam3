"""
Flask App (Version 13) - Ollama + SAM3 Text-Prompted Layered Vectorization

Three-phase interactive pipeline:
  Phase 1: Ollama Qwen2.5 VL keyword detection
  Phase 2: SAM3 text-prompted segmentation (iterative)
  Phase 3: User confirms layers -> DiffVG vectorization

Run with: python app13.py
Access at: http://localhost:5013
Requires: conda activate lv13
"""

from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import sys
import json
from datetime import datetime
import threading
import importlib.util

# Add LayeredVectorization to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'LayeredVectorization'))

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULTS_FOLDER'] = 'results_v13'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)

processing_status = {}
processing_threads = {}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg', 'webp', 'bmp'}


def get_main_v13():
    """Dynamically load main_v13 module."""
    module_path = os.path.join(os.path.dirname(__file__), 'LayeredVectorization', 'main_v13.py')
    spec = importlib.util.spec_from_file_location("main_v13", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# =========================================================================
# Routes
# =========================================================================

@app.route('/')
def index():
    return render_template('index_v13.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

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

        return jsonify({'success': True, 'run_id': run_id, 'filename': filename})

    return jsonify({'error': 'Invalid file type'}), 400


# =========================================================================
# Phase 1: Detect Keywords
# =========================================================================

def run_keyword_detection(run_id):
    """Background thread: run Ollama keyword detection."""
    try:
        status = processing_status[run_id]
        status['status'] = 'detecting_keywords'
        status['progress'] = 5
        status['message'] = 'Starting keyword detection...'

        main_v13 = get_main_v13()

        def progress_cb(pct, msg):
            status['progress'] = pct
            status['message'] = msg

        keywords = main_v13.detect_keywords_v13(
            status['filepath'],
            progress_cb=progress_cb
        )

        status['keywords'] = keywords
        status['status'] = 'keywords_ready'
        status['progress'] = 100
        status['message'] = f'Detected {len(keywords)} objects'

    except Exception as e:
        import traceback
        traceback.print_exc()
        status['status'] = 'error'
        status['message'] = str(e)
    finally:
        processing_threads.pop(run_id, None)


@app.route('/detect_keywords/<run_id>', methods=['POST'])
def detect_keywords(run_id):
    if run_id not in processing_status:
        return jsonify({'error': 'Invalid run ID'}), 400
    if run_id in processing_threads:
        return jsonify({'error': 'Already processing'}), 400

    thread = threading.Thread(target=run_keyword_detection, args=(run_id,))
    thread.daemon = True
    thread.start()
    processing_threads[run_id] = thread

    return jsonify({'success': True, 'message': 'Keyword detection started'})


# =========================================================================
# Phase 2: Segment Keywords
# =========================================================================

def run_segmentation(run_id):
    """Background thread: run SAM3 segmentation."""
    try:
        status = processing_status[run_id]
        status['status'] = 'segmenting'
        status['progress'] = 5
        status['message'] = 'Starting segmentation...'

        main_v13 = get_main_v13()

        def progress_cb(pct, msg):
            status['progress'] = pct
            status['message'] = msg

        keywords_with_conf = status.get('keywords_with_conf', [])

        layers_info = main_v13.segment_keywords_v13(
            status['filepath'],
            keywords_with_conf,
            status['run_folder'],
            progress_cb=progress_cb
        )

        status['layers_info'] = layers_info
        status['status'] = 'layers_ready'
        status['progress'] = 100
        status['message'] = f'Segmented {len(layers_info)} layers'
        status['n_layers'] = len(layers_info)

    except Exception as e:
        import traceback
        traceback.print_exc()
        status['status'] = 'error'
        status['message'] = str(e)
    finally:
        processing_threads.pop(run_id, None)


@app.route('/segment/<run_id>', methods=['POST'])
def segment_keywords(run_id):
    if run_id not in processing_status:
        return jsonify({'error': 'Invalid run ID'}), 400
    if run_id in processing_threads:
        return jsonify({'error': 'Already processing'}), 400

    data = request.get_json()
    keywords = data.get('keywords', [])

    if not keywords:
        return jsonify({'error': 'No keywords provided'}), 400

    # Store keywords with confidence for the background thread
    processing_status[run_id]['keywords_with_conf'] = keywords

    thread = threading.Thread(target=run_segmentation, args=(run_id,))
    thread.daemon = True
    thread.start()
    processing_threads[run_id] = thread

    return jsonify({'success': True, 'message': f'Segmenting {len(keywords)} keywords'})


# =========================================================================
# Layer Review
# =========================================================================

@app.route('/layers/<run_id>', methods=['GET'])
def get_layers(run_id):
    if run_id not in processing_status:
        return jsonify({'error': 'Invalid run ID'}), 400

    status = processing_status[run_id]
    if status['status'] not in ('layers_ready', 'vectorizing', 'completed'):
        return jsonify({'error': 'Layers not ready yet'}), 400

    return jsonify({
        'layers': status.get('layers_info', []),
        'n_layers': status.get('n_layers', 0),
        'merge_preview_url': 'merge_preview.png',
    })


@app.route('/layer_asset/<run_id>/<path:filename>')
def serve_layer_asset(run_id, filename):
    """Serve layer preview/mask images."""
    if run_id in processing_status:
        layers_dir = os.path.join(processing_status[run_id]['run_folder'], 'layers')
    else:
        # Fallback: serve from results folder (for historical results)
        layers_dir = os.path.join(app.config['RESULTS_FOLDER'], run_id, 'layers')
    if not os.path.exists(layers_dir):
        return 'Not found', 404
    return send_from_directory(layers_dir, filename)


# =========================================================================
# Phase 3: Vectorize
# =========================================================================

def run_vectorization(run_id):
    """Background thread: run DiffVG vectorization."""
    import torch
    original_dir = os.getcwd()

    try:
        status = processing_status[run_id]
        status['status'] = 'vectorizing'
        status['phase'] = 'vectorize'
        status['progress'] = 5
        status['message'] = 'Starting vectorization...'

        selected_layers = status.get('selected_layers', [])
        quality = status.get('quality', 'fast')

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

        import argparse
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

        def progress_cb(pct, msg):
            status['progress'] = pct
            status['message'] = msg

        result = main_v13.vectorize_confirmed_v13(
            device, args,
            run_folder_abs,
            selected_layers,
            progress_cb=progress_cb
        )

        # Copy results
        status['result_svg'] = result.get('svg_path', '')
        status['result_layers'] = result.get('layers', [])
        status['n_vectorized'] = result.get('n_layers', 0)
        status['status'] = 'completed'
        status['progress'] = 100
        status['message'] = f'Complete! {result["n_layers"]} layers vectorized'

    except Exception as e:
        import traceback
        traceback.print_exc()
        status['status'] = 'error'
        status['message'] = str(e)
    finally:
        os.chdir(original_dir)
        # Clean up symlink
        try:
            lv_dir = os.path.join(os.path.dirname(__file__), 'LayeredVectorization')
            workdir_link = os.path.join(lv_dir, 'workdir')
            if os.path.islink(workdir_link):
                os.unlink(workdir_link)
        except Exception:
            pass
        processing_threads.pop(run_id, None)


@app.route('/vectorize/<run_id>', methods=['POST'])
def vectorize_confirmed(run_id):
    if run_id not in processing_status:
        return jsonify({'error': 'Invalid run ID'}), 400
    if run_id in processing_threads:
        return jsonify({'error': 'Already processing'}), 400

    data = request.get_json()
    selected_layers = data.get('selected_layers', [])
    quality = data.get('quality', 'fast')

    if not selected_layers:
        return jsonify({'error': 'No layers selected'}), 400

    processing_status[run_id]['selected_layers'] = selected_layers
    processing_status[run_id]['quality'] = quality

    thread = threading.Thread(target=run_vectorization, args=(run_id,))
    thread.daemon = True
    thread.start()
    processing_threads[run_id] = thread

    return jsonify({
        'success': True,
        'message': f'Vectorizing {len(selected_layers)} layers with {quality} quality'
    })


# =========================================================================
# Status & Results
# =========================================================================

@app.route('/status/<run_id>', methods=['GET'])
def get_status(run_id):
    if run_id not in processing_status:
        return jsonify({'error': 'Invalid run ID'}), 400
    status = processing_status[run_id]
    resp = {
        'status': status['status'],
        'progress': status['progress'],
        'message': status['message'],
        'keywords': status.get('keywords', []),
        'n_layers': status.get('n_layers', 0),
    }
    # Include result paths when completed
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
    return jsonify(resp)


@app.route('/results', methods=['GET'])
def list_results():
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
    return jsonify({'results': results})


@app.route('/results/<path:filepath>')
def serve_result(filepath):
    return send_from_directory(app.config['RESULTS_FOLDER'], filepath)


@app.route('/view/<run_id>')
def view_result(run_id):
    run_folder = os.path.join(app.config['RESULTS_FOLDER'], run_id)
    if not os.path.exists(run_folder):
        return 'Not found', 404

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

    return render_template('view_result_v13.html', **files)


# =========================================================================
# Main
# =========================================================================

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='V13: Ollama + SAM3 Layered Vectorization')
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', type=int, default=5013)
    parser.add_argument('--debug', action='store_true')
    cli_args = parser.parse_args()

    print("=" * 60)
    print("V13 Layered Vectorization - Ollama + SAM3")
    print("=" * 60)
    print(f"Server: http://localhost:{cli_args.port}")
    print(f"Prerequisites:")
    print(f"  - Ollama running: ollama serve")
    print(f"  - Model pulled: ollama pull qwen2.5vl:7b")
    print("=" * 60)

    app.run(host=cli_args.host, port=cli_args.port, debug=cli_args.debug)
