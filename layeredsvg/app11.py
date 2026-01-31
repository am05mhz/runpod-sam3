"""
Flask App (Version 11) - App3 SVG Quality + App8 Layer Editability

Combines:
- App3's DiffVG optimization for perfect SVG output
- App8's SAM + Depth Anything semantic layer decomposition

Result: Each object can be moved independently without leaving holes.

Run with: python app11.py
Access at: http://localhost:5011
Requires: conda activate lv8
"""

from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import argparse
import sys
from datetime import datetime
import re
import shutil
import threading

# Add LayeredVectorization to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'LayeredVectorization'))

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULTS_FOLDER'] = 'results_v11'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)

processing_status = {}
processing_threads = {}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}


@app.route('/')
def index():
    return render_template('index_v11.html')


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

        quality = request.form.get('quality', 'fast')
        max_layers = request.form.get('max_layers', '10')
        n_depth_clusters = request.form.get('n_depth_clusters', '3')
        moge_version = request.form.get('moge_version', 'v2')
        moge_resolution = request.form.get('moge_resolution', 'High')
        # EDGE GAP FIX: Dilate background mask by this many pixels to fill gaps
        # Only background layer gets dilation (foreground objects stay clean)
        mask_dilation_px = request.form.get('mask_dilation_px', '3')
        # Background detection method: "depth" (Depth Anything) or "area" (largest mask)
        background_method = request.form.get('background_method', 'depth')

        processing_status[run_id] = {
            'status': 'uploaded',
            'progress': 0,
            'message': 'File uploaded',
            'filename': filename,
            'run_folder': run_folder,
            'quality': quality,
            'max_layers': max_layers,
            'n_depth_clusters': n_depth_clusters,
            'moge_version': moge_version,
            'moge_resolution': moge_resolution,
            'mask_dilation_px': mask_dilation_px,
            'background_method': background_method
        }

        return jsonify({'success': True, 'run_id': run_id, 'filename': filename})

    return jsonify({'error': 'Invalid file type'}), 400


def run_vectorization(run_id):
    """Run V11 vectorization: SAM + Depth Anything decomposition + per-layer DiffVG"""
    import time
    original_dir = os.getcwd()

    try:
        status = processing_status[run_id]
        status['status'] = 'processing'
        status['progress'] = 5
        status['message'] = 'Starting V11 layered vectorization...'

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

        status['progress'] = 10
        status['message'] = 'Loading models...'

        # Now import (after chdir - use local imports, not package imports)
        import torch
        import importlib.util

        # Load main_v3 from file
        spec = importlib.util.spec_from_file_location("main_v3", os.path.join(layered_vec_dir, "main_v3.py"))
        main_v3 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(main_v3)
        init_diffvg = main_v3.init_diffvg
        load_config = main_v3.load_config

        # Load main_v11 from file
        spec10 = importlib.util.spec_from_file_location("main_v11", os.path.join(layered_vec_dir, "main_v11.py"))
        main_v11 = importlib.util.module_from_spec(spec10)
        spec10.loader.exec_module(main_v11)
        layered_vectorization_v11 = main_v11.layered_vectorization_v11

        quality = status.get('quality', 'fast')
        # V11-specific configs with HQ-SAM support
        config_map = {
            'fast': 'base_config_v11.yaml',
            'balanced': 'high_quality_config_v11.yaml',
            'high': 'ultra_quality_config_v11.yaml',
            'best': 'best_quality_config_v11.yaml',
        }
        config_file = config_map.get(quality, 'base_config_v11.yaml')

        # Setup args
        parser = argparse.ArgumentParser()
        parser.add_argument("-c", "--config", type=str)
        parser.add_argument("-timg", "--target_image", type=str)
        parser.add_argument("-fsn", "--file_save_name", type=str, default="output")
        # Legacy params kept for backward compat (now uses Depth Anything internally)
        parser.add_argument("--moge_version", type=str, default="v2")
        parser.add_argument("--moge_resolution", type=str, default="High")
        parser.add_argument("--max_layers", type=int, default=10)
        parser.add_argument("--n_depth_clusters", type=int, default=3)
        parser.add_argument("--min_mask_area", type=int, default=500)
        # EDGE GAP FIX: Dilate masks to create overlap between layers
        parser.add_argument("--mask_dilation_px", type=int, default=3)
        # Background detection: "depth" (Depth Anything) or "area" (largest mask)
        parser.add_argument("--background_method", type=str, default="depth")
        parser.add_argument("--vtracer_enable", type=bool, default=True)
        parser.add_argument("--staircase_area", type=float, default=1.5)
        parser.add_argument("--corner_angle", type=float, default=135.0)
        parser.add_argument("--simplify_error", type=float, default=2.0)
        parser.add_argument("--smooth_iterations", type=int, default=0)
        parser.add_argument("--skip_sds", action="store_true")

        args = parser.parse_args([])
        args.config = os.path.join(layered_vec_dir, "config", config_file)
        args.target_image = os.path.abspath(os.path.join(original_dir, input_path))
        args.file_save_name = "output"

        # Get max_layers from form data
        max_layers_str = status.get('max_layers', '10')
        args.max_layers = int(max_layers_str)

        n_depth_clusters_str = status.get('n_depth_clusters', '3')
        args.n_depth_clusters = int(n_depth_clusters_str)

        args.moge_version = status.get('moge_version', args.moge_version)
        args.moge_resolution = status.get('moge_resolution', args.moge_resolution)

        # EDGE GAP FIX: Get mask dilation parameter
        mask_dilation_str = status.get('mask_dilation_px', '3')
        args.mask_dilation_px = int(mask_dilation_str)

        # Background detection method: "depth" or "area"
        args.background_method = status.get('background_method', 'depth')

        args = load_config(args.config, args)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        init_diffvg(device=device)

        status['progress'] = 15
        status['message'] = 'Decomposing image into layers...'

        # Progress callback
        def progress_callback(progress, message):
            status['progress'] = progress
            status['message'] = message

        # Run V11 pipeline
        result = layered_vectorization_v11(device, args, progress_callback=progress_callback)

        # Cleanup symlink
        if os.path.islink(workdir_existing):
            os.unlink(workdir_existing)

        # Set results
        output_dir = os.path.join(workdir_path, 'output')
        final_svg = os.path.join(output_dir, 'final.svg')

        if os.path.exists(final_svg):
            status['status'] = 'completed'
            status['progress'] = 100
            status['message'] = f'Completed! {result["n_layers"]} editable layers.'
            status['result_svg'] = os.path.relpath(final_svg, app.config['RESULTS_FOLDER'])
            status['n_layers'] = result['n_layers']

            fullsize_svg = os.path.join(output_dir, 'final_fullsize.svg')
            if os.path.exists(fullsize_svg):
                status['result_svg_fullsize'] = os.path.relpath(fullsize_svg, app.config['RESULTS_FOLDER'])

            masked_svg = os.path.join(output_dir, 'final_masked.svg')
            if os.path.exists(masked_svg):
                status['result_svg_masked'] = os.path.relpath(masked_svg, app.config['RESULTS_FOLDER'])

            masked_fullsize_svg = os.path.join(output_dir, 'final_fullsize_masked.svg')
            if os.path.exists(masked_fullsize_svg):
                status['result_svg_fullsize_masked'] = os.path.relpath(masked_fullsize_svg, app.config['RESULTS_FOLDER'])

            fullsize_png = os.path.join(output_dir, 'final_fullsize.png')
            if os.path.exists(fullsize_png):
                status['result_png'] = os.path.relpath(fullsize_png, app.config['RESULTS_FOLDER'])
        else:
            status['status'] = 'error'
            status['message'] = 'SVG not found after processing'

    except Exception as e:
        import traceback
        traceback.print_exc()
        processing_status[run_id]['status'] = 'error'
        processing_status[run_id]['message'] = f'Error: {str(e)}'

        # Cleanup on error
        try:
            layered_vec_dir = os.path.join(os.path.dirname(__file__), 'LayeredVectorization')
            workdir_existing = os.path.join(layered_vec_dir, 'workdir')
            if os.path.islink(workdir_existing):
                os.unlink(workdir_existing)
        except:
            pass

    finally:
        try:
            os.chdir(original_dir)
        except:
            pass
        if run_id in processing_threads:
            del processing_threads[run_id]


@app.route('/process/<run_id>', methods=['POST'])
def process_image(run_id):
    if run_id not in processing_status:
        return jsonify({'error': 'Invalid run ID'}), 400
    if run_id in processing_threads:
        return jsonify({'error': 'Already processing'}), 400

    thread = threading.Thread(target=run_vectorization, args=(run_id,))
    thread.daemon = True
    thread.start()
    processing_threads[run_id] = thread

    return jsonify({'success': True, 'message': 'V11 processing started'})


@app.route('/status/<run_id>')
def get_status(run_id):
    if run_id not in processing_status:
        return jsonify({'error': 'Invalid run ID'}), 404
    return jsonify(processing_status[run_id])


@app.route('/results')
def list_results():
    results = []
    if os.path.exists(app.config['RESULTS_FOLDER']):
        for run_id in sorted(os.listdir(app.config['RESULTS_FOLDER']), reverse=True):
            run_path = os.path.join(app.config['RESULTS_FOLDER'], run_id)
            if os.path.isdir(run_path):
                final_svg = os.path.join(run_path, 'workdir', 'output', 'final.svg')
                layers_dir = os.path.join(run_path, 'workdir', 'output', 'layers')
                n_layers = None
                if os.path.exists(layers_dir):
                    layer_svgs = [f for f in os.listdir(layers_dir) if f.endswith('.svg')]
                    n_layers = len(layer_svgs)
                input_files = [f for f in os.listdir(run_path) if f.startswith('input_')]
                results.append({
                    'run_id': run_id,
                    'has_result': os.path.exists(final_svg),
                    'input_file': input_files[0] if input_files else None,
                    'n_layers': n_layers
                })
    return jsonify(results)


@app.route('/results/<path:filename>')
def serve_result(filename):
    return send_from_directory(app.config['RESULTS_FOLDER'], filename)


@app.route('/view/<run_id>')
def view_result(run_id):
    run_folder = os.path.join(app.config['RESULTS_FOLDER'], run_id)
    if not os.path.exists(run_folder):
        return "Run not found", 404

    files = {}

    input_files = [f for f in os.listdir(run_folder) if f.startswith('input_')]
    if input_files:
        files['input'] = os.path.join(run_id, input_files[0])

    output_dir = os.path.join(run_folder, 'workdir', 'output')

    final_svg_fullsize = os.path.join(output_dir, 'final_fullsize.svg')
    if os.path.exists(final_svg_fullsize):
        files['final_svg_fullsize'] = os.path.relpath(final_svg_fullsize, app.config['RESULTS_FOLDER'])

    final_svg_fullsize_masked = os.path.join(output_dir, 'final_fullsize_masked.svg')
    if os.path.exists(final_svg_fullsize_masked):
        files['final_svg_fullsize_masked'] = os.path.relpath(final_svg_fullsize_masked, app.config['RESULTS_FOLDER'])

    final_svg = os.path.join(output_dir, 'final.svg')
    if os.path.exists(final_svg):
        files['final_svg'] = os.path.relpath(final_svg, app.config['RESULTS_FOLDER'])

    final_svg_masked = os.path.join(output_dir, 'final_masked.svg')
    if os.path.exists(final_svg_masked):
        files['final_svg_masked'] = os.path.relpath(final_svg_masked, app.config['RESULTS_FOLDER'])

    final_png = os.path.join(output_dir, 'final_fullsize.png')
    if os.path.exists(final_png):
        files['final_png'] = os.path.relpath(final_png, app.config['RESULTS_FOLDER'])

    layers_dir = os.path.join(output_dir, 'layers')
    if os.path.exists(layers_dir):
        layer_svgs = []
        layer_previews = []
        layer_masks = []
        for filename in os.listdir(layers_dir):
            if filename.endswith('.svg'):
                layer_svgs.append(filename)
            elif filename.endswith('_preview.png'):
                layer_previews.append(filename)
            elif filename.endswith('_mask.png'):
                layer_masks.append(filename)

        def layer_sort_key(name):
            match = re.search(r'layer_(\d+)', name)
            return int(match.group(1)) if match else name

        for filenames, key in [
            (layer_svgs, 'layer_svgs'),
            (layer_previews, 'layer_pngs'),
            (layer_masks, 'layer_masks'),
        ]:
            if filenames:
                files[key] = [
                    os.path.relpath(os.path.join(layers_dir, f), app.config['RESULTS_FOLDER'])
                    for f in sorted(filenames, key=layer_sort_key)
                ]

    return render_template('view_result_v11.html', run_id=run_id, files=files)


if __name__ == '__main__':
    print("=" * 60)
    print("V11 Layered Vectorization")
    print("App3 SVG Quality + App8 Layer Editability")
    print("=" * 60)
    print("")
    print("Features:")
    print("  - SAM + Depth Anything layer decomposition")
    print("  - Per-layer DiffVG vectorization")
    print("  - Each object can be moved independently")
    print("")
    print("Port: 5011")
    print("=" * 60)
    app.run(debug=True, host='0.0.0.0', port=5011)
