"""
Experimental Flask App (Version 3) - SDXL + VTracer

Uses SDXL (1024x1024) instead of SD 1.5 (512x512) for better detail preservation.

Key features:
- SDXL at 1024x1024 native resolution
- Aspect ratio preservation (padding instead of stretching)
- VTracer-inspired path simplification (from app2)

Files used:
- main_v3.py - Main pipeline with SDXL
- img_process_v3.py - Dynamic resolution image processing
- sds_image_simplicity_sdxl.py - SDXL SDS simplification

Run with: python app3.py
Access at: http://localhost:5002
"""

from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import argparse
import yaml
import sys
from datetime import datetime
import shutil
from pathlib import Path
import json
import threading

# Add LayeredVectorization to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'LayeredVectorization'))

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULTS_FOLDER'] = 'results_v3'  # Experimental results folder
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure folders exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)

# Store processing status
processing_status = {}

# Store running threads to prevent duplicate processing
processing_threads = {}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}

@app.route('/')
def index():
    return render_template('index_v3.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        # Create unique run ID based on timestamp
        run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        run_folder = os.path.join(app.config['RESULTS_FOLDER'], run_id)
        os.makedirs(run_folder, exist_ok=True)

        # Save uploaded file
        filename = f"input_{file.filename}"
        filepath = os.path.join(run_folder, filename)
        file.save(filepath)

        # Get quality setting from form (default to standard)
        quality = request.form.get('quality', 'standard')

        # Get VTracer parameters from form (with defaults)
        vtracer_enable = request.form.get('vtracer_enable', 'true').lower() == 'true'
        staircase_area = float(request.form.get('staircase_area', '1.5'))
        corner_angle = float(request.form.get('corner_angle', '135.0'))
        simplify_error = float(request.form.get('simplify_error', '2.0'))
        smooth_iterations = int(request.form.get('smooth_iterations', '0'))

        # Initialize status
        processing_status[run_id] = {
            'status': 'uploaded',
            'progress': 0,
            'message': 'File uploaded successfully',
            'filename': filename,
            'run_folder': run_folder,
            'quality': quality,
            # VTracer parameters
            'vtracer_enable': vtracer_enable,
            'staircase_area': staircase_area,
            'corner_angle': corner_angle,
            'simplify_error': simplify_error,
            'smooth_iterations': smooth_iterations
        }

        return jsonify({
            'success': True,
            'run_id': run_id,
            'filename': filename,
            'vtracer_enabled': vtracer_enable
        })

    return jsonify({'error': 'Invalid file type. Please upload PNG or JPG'}), 400

def write_log(log_file, message):
    """Append timestamped message to log file"""
    try:
        with open(log_file, 'a', encoding='utf-8') as f:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            f.write(f"[{timestamp}] {message}\n")
            f.flush()
    except Exception as e:
        print(f"Warning: Could not write to log: {e}")

def run_vectorization(run_id):
    """Run vectorization in a separate thread using VTracer-enhanced pipeline"""
    import time
    from datetime import datetime

    original_dir = os.getcwd()
    start_time = time.time()
    log_file = None

    # Estimated times per quality preset (in seconds)
    quality_time_estimates = {
        'fast': 240,           # ~4 minutes
        'balanced': 480,       # ~8 minutes
        'high': 960,           # ~16 minutes
        'best': 1500,          # ~25 minutes
        'ultra': 4500,         # ~75 minutes (2048 paths, 250 colors)
        'extreme': 6000,       # ~100 minutes (2560 paths, requires 20GB+ GPU)
        'max': 9000            # ~150 minutes (4096 paths, requires 24GB+ GPU)
    }

    try:
        # Import torch and vectorization modules
        import torch
        # Use the experimental main_v3
        from LayeredVectorization.main_v3 import layered_vectorization, init_diffvg, load_config

        status = processing_status[run_id]
        status['status'] = 'processing'
        status['progress'] = 5
        status['message'] = 'Initializing experimental v3 vectorization...'
        status['start_time'] = start_time

        # Get VTracer parameters from status
        vtracer_enable = status.get('vtracer_enable', True)
        staircase_area = status.get('staircase_area', 1.5)
        corner_angle = status.get('corner_angle', 135.0)
        simplify_error = status.get('simplify_error', 2.0)
        smooth_iterations = status.get('smooth_iterations', 0)

        # Get quality preset and estimated time
        quality = status.get('quality', 'fast')
        estimated_total_time = quality_time_estimates.get(quality, 240)
        status['estimated_time'] = estimated_total_time

        # Get the uploaded file path
        input_path = os.path.join(status['run_folder'], status['filename'])

        # Get absolute path for the run folder
        run_folder_abs = os.path.abspath(status['run_folder'])

        # Create workdir inside the run folder
        workdir_path = os.path.join(run_folder_abs, 'workdir')
        os.makedirs(workdir_path, exist_ok=True)

        # Log file path
        log_file = os.path.join(run_folder_abs, 'workdir', 'processing_log.txt')

        # Write initial log entry
        write_log(log_file, "=" * 80)
        write_log(log_file, "EXPERIMENTAL V3 LAYERED VECTORIZATION LOG")
        write_log(log_file, "=" * 80)
        write_log(log_file, f"Run ID: {run_id}")
        write_log(log_file, f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        write_log(log_file, f"Input Image: {status['filename']}")
        write_log(log_file, f"Quality Preset: {quality.upper()}")
        write_log(log_file, f"Estimated Duration: {estimated_total_time // 60} minutes")
        write_log(log_file, "")
        write_log(log_file, "VTracer Enhancement Settings:")
        write_log(log_file, f"  Enabled: {vtracer_enable}")
        write_log(log_file, f"  Staircase area threshold: {staircase_area}")
        write_log(log_file, f"  Corner angle threshold: {corner_angle}°")
        write_log(log_file, f"  Simplification error: {simplify_error}")
        write_log(log_file, f"  Smooth iterations: {smooth_iterations}")
        write_log(log_file, "")

        write_log(log_file, "Stage 1: Setting up environment")

        # Change to LayeredVectorization directory
        layered_vec_dir = os.path.join(os.path.dirname(__file__), 'LayeredVectorization')
        os.chdir(layered_vec_dir)

        # Backup existing workdir if it exists
        workdir_backup = os.path.join(layered_vec_dir, 'workdir_backup')
        workdir_existing = os.path.join(layered_vec_dir, 'workdir')

        # Clean up any existing workdir (might be leftover from a crash)
        if os.path.islink(workdir_existing):
            os.unlink(workdir_existing)
        elif os.path.exists(workdir_existing):
            shutil.rmtree(workdir_existing)

        # Also clean up old backup if exists
        if os.path.exists(workdir_backup):
            shutil.rmtree(workdir_backup)

        # Create symlink or copy to point to run's workdir
        try:
            os.symlink(workdir_path, workdir_existing)
        except OSError:
            # If symlink fails, create a copy
            shutil.copytree(workdir_path, workdir_existing)

        status['progress'] = 10
        status['message'] = 'Loading models and configurations...'
        write_log(log_file, "Environment setup complete")

        # Select SDXL-specific config based on quality setting
        # These configs have SAM parameters optimized for 1024x1024 resolution
        config_map = {
            'fast': 'base_config_sdxl.yaml',
            'balanced': 'high_quality_config_sdxl.yaml',
            'high': 'ultra_quality_config_sdxl.yaml',
            'best': 'best_quality_config_sdxl.yaml',
            'ultra': 'ultra_detail_config_sdxl.yaml',
            'extreme': 'extreme_detail_config_sdxl.yaml',
            'max': 'max_detail_config_sdxl.yaml'
        }
        config_file = config_map.get(quality, 'base_config_sdxl.yaml')

        write_log(log_file, "")
        write_log(log_file, f"Stage 2: Loading configuration - {config_file}")

        # Create argument parser and load config
        parser = argparse.ArgumentParser()
        parser.add_argument("-c", "--config", type=str, default=f"./config/{config_file}")
        parser.add_argument("-timg", "--target_image", type=str, default=input_path)
        parser.add_argument("-fsn", "--file_save_name", type=str, default="output")
        # VTracer arguments
        parser.add_argument("--vtracer_enable", type=bool, default=True)
        parser.add_argument("--staircase_area", type=float, default=1.5)
        parser.add_argument("--corner_angle", type=float, default=135.0)
        parser.add_argument("--simplify_error", type=float, default=2.0)
        parser.add_argument("--smooth_iterations", type=int, default=0)

        args = parser.parse_args([])
        args.config = os.path.join(layered_vec_dir, "config", config_file)
        args.target_image = os.path.abspath(os.path.join(original_dir, input_path))
        args.file_save_name = "output"

        # Set VTracer parameters
        args.vtracer_enable = vtracer_enable
        args.staircase_area = staircase_area
        args.corner_angle = corner_angle
        args.simplify_error = simplify_error
        args.smooth_iterations = smooth_iterations

        # Load config
        args = load_config(args.config, args)

        # Log key configuration parameters
        write_log(log_file, f"  Max paths: {getattr(args, 'max_path_num_limit', 'N/A')}")
        write_log(log_file, f"  Color clusters (kmeans_k): {getattr(args, 'kmeas_k', 'N/A')}")
        write_log(log_file, f"  Visual refinement iterations: {getattr(args, 'add_visual_path_num_iters', 'N/A')}")
        write_log(log_file, f"  Structural optimization iterations: {getattr(args, 'struct_opt_num_iters', 'N/A')}")
        write_log(log_file, f"  Visual optimization iterations: {getattr(args, 'visual_opt_num_iters', 'N/A')}")

        # Initialize device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        write_log(log_file, f"  Device: {device}")
        init_diffvg(device=device)

        status['progress'] = 20
        status['message'] = 'Configuration loaded. Starting VTracer-enhanced vectorization...'
        write_log(log_file, "Configuration loaded successfully")

        # Calculate time remaining
        elapsed = time.time() - start_time
        remaining = max(0, estimated_total_time - elapsed)
        status['time_remaining'] = remaining

        write_log(log_file, "")
        write_log(log_file, "Stage 3: Running VTracer-enhanced vectorization pipeline")
        write_log(log_file, "  This includes: SDS simplification, SAM segmentation, VTracer path optimization")

        status['progress'] = 25
        status['message'] = 'Running SDS-based image simplification...'

        # Run the vectorization
        pipeline_start = time.time()
        layered_vectorization(args, device)
        pipeline_duration = time.time() - pipeline_start

        write_log(log_file, f"Vectorization pipeline completed in {pipeline_duration:.1f} seconds ({pipeline_duration/60:.1f} minutes)")

        status['progress'] = 90
        status['message'] = 'Finalizing results...'

        write_log(log_file, "")
        write_log(log_file, "Stage 4: Post-processing")

        # Clean up workdir symlink
        if os.path.islink(workdir_existing):
            os.unlink(workdir_existing)
        elif os.path.exists(workdir_existing):
            shutil.rmtree(workdir_existing)

        # Restore backup if it existed
        if os.path.exists(workdir_backup):
            shutil.move(workdir_backup, workdir_existing)

        # Find the final SVG
        final_svg = os.path.join(run_folder_abs, 'workdir', 'output', 'final.svg')

        if os.path.exists(final_svg):
            write_log(log_file, f"  Final SVG generated: final.svg")

            status['progress'] = 92
            status['message'] = 'Creating full-size outputs...'

            input_file = os.path.join(run_folder_abs, status['filename'])
            output_dir = os.path.join(run_folder_abs, 'workdir', 'output')
            fullsize_svg = os.path.join(output_dir, 'final_fullsize.svg')
            fullsize_png = os.path.join(output_dir, 'final_fullsize.png')

            # Create full-size SVG (at original image dimensions)
            try:
                from PIL import Image
                import re

                # Get original image dimensions
                with Image.open(input_file) as img:
                    orig_width, orig_height = img.size

                # Read SVG and modify dimensions
                with open(final_svg, 'r', encoding='utf-8') as f:
                    svg_content = f.read()

                # Update to full original dimensions
                svg_content = re.sub(r'width="\d+"', f'width="{orig_width}"', svg_content)
                svg_content = re.sub(r'height="\d+"', f'height="{orig_height}"', svg_content)

                # Ensure viewBox is set (SDXL uses 1024x1024)
                if 'viewBox=' in svg_content:
                    svg_content = re.sub(r'viewBox="[^"]*"', 'viewBox="0 0 1024 1024"', svg_content)
                else:
                    svg_content = re.sub(r'(height="\d+")', r'\1 viewBox="0 0 1024 1024"', svg_content)

                # Add preserveAspectRatio
                if 'preserveAspectRatio=' in svg_content:
                    svg_content = re.sub(r'preserveAspectRatio="[^"]*"', 'preserveAspectRatio="none"', svg_content)
                else:
                    svg_content = re.sub(r'(viewBox="[^"]*")', r'\1 preserveAspectRatio="none"', svg_content)

                with open(fullsize_svg, 'w', encoding='utf-8') as f:
                    f.write(svg_content)

                status['result_svg_fullsize'] = os.path.relpath(fullsize_svg, app.config['RESULTS_FOLDER'])
                write_log(log_file, f"  Full-size SVG created: final_fullsize.svg ({orig_width}x{orig_height})")
                print(f"Created full-size SVG: {fullsize_svg}")

                # Create PNG from full-size SVG
                status['progress'] = 96
                status['message'] = 'Generating PNG preview...'

                try:
                    import cairosvg
                    cairosvg.svg2png(url=fullsize_svg, write_to=fullsize_png,
                                     output_width=orig_width, output_height=orig_height)
                    status['result_png'] = os.path.relpath(fullsize_png, app.config['RESULTS_FOLDER'])
                    write_log(log_file, f"  Full-size PNG created: final_fullsize.png ({orig_width}x{orig_height})")
                    print(f"Created full-size PNG: {fullsize_png}")
                except ImportError:
                    write_log(log_file, "  Warning: cairosvg not installed, skipping PNG generation")
                    print("Warning: cairosvg not installed, skipping PNG generation")
                except Exception as e:
                    write_log(log_file, f"  Warning: Could not create PNG: {e}")
                    print(f"Warning: Could not create PNG: {e}")

            except Exception as e:
                write_log(log_file, f"  Warning: Could not create full-size outputs: {e}")
                print(f"Warning: Could not create full-size outputs: {e}")

            # Calculate total time
            total_time = time.time() - start_time
            total_minutes = total_time / 60

            status['status'] = 'completed'
            status['progress'] = 100
            status['message'] = 'Experimental v3 vectorization completed!'
            status['result_svg'] = os.path.relpath(final_svg, app.config['RESULTS_FOLDER'])
            status['time_remaining'] = 0

            # Write final summary
            write_log(log_file, "")
            write_log(log_file, "=" * 80)
            write_log(log_file, "EXPERIMENTAL V3 PROCESSING COMPLETED")
            write_log(log_file, "=" * 80)
            write_log(log_file, f"Total Duration: {total_time:.1f} seconds ({total_minutes:.1f} minutes)")
            write_log(log_file, f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            write_log(log_file, "")
            write_log(log_file, "VTracer Enhancements Applied:")
            write_log(log_file, f"  - Staircase Removal (area threshold: {staircase_area})")
            write_log(log_file, f"  - Corner Detection (angle threshold: {corner_angle}°)")
            write_log(log_file, f"  - Error-Penalized Simplification (max error: {simplify_error})")
            if smooth_iterations > 0:
                write_log(log_file, f"  - 4-Point Subdivision Smoothing ({smooth_iterations} iterations)")
            write_log(log_file, "")
            write_log(log_file, "Output Files:")
            write_log(log_file, f"  - final.svg (512x512 square)")
            write_log(log_file, f"  - final_fullsize.svg (original dimensions)")
            write_log(log_file, f"  - final_fullsize.png (PNG at original dimensions)")
            write_log(log_file, "=" * 80)
        else:
            status['status'] = 'error'
            status['message'] = 'Processing completed but final SVG not found'
            if log_file:
                write_log(log_file, "")
                write_log(log_file, "ERROR: Final SVG not found after processing")

    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print("Error during processing:")
        print(error_trace)

        processing_status[run_id]['status'] = 'error'
        processing_status[run_id]['message'] = f'Error: {str(e)}'
        processing_status[run_id]['progress'] = 0
        processing_status[run_id]['time_remaining'] = 0

        # Log the error
        if log_file:
            write_log(log_file, "")
            write_log(log_file, "=" * 80)
            write_log(log_file, "ERROR OCCURRED")
            write_log(log_file, "=" * 80)
            write_log(log_file, f"Error: {str(e)}")
            write_log(log_file, "")
            write_log(log_file, "Full traceback:")
            for line in error_trace.split('\n'):
                write_log(log_file, f"  {line}")
            write_log(log_file, "=" * 80)

        # Clean up workdir symlink on error
        try:
            workdir_existing = os.path.join(layered_vec_dir, 'workdir')
            if os.path.islink(workdir_existing):
                os.unlink(workdir_existing)
            if os.path.exists(workdir_backup):
                shutil.move(workdir_backup, workdir_existing)
        except:
            pass

    finally:
        # Always restore directory
        try:
            os.chdir(original_dir)
        except:
            pass

        # Mark thread as complete
        if run_id in processing_threads:
            del processing_threads[run_id]

@app.route('/process/<run_id>', methods=['POST'])
def process_image(run_id):
    if run_id not in processing_status:
        return jsonify({'error': 'Invalid run ID'}), 400

    if run_id in processing_threads:
        return jsonify({'error': 'Processing already in progress for this run'}), 400

    # Start processing in a separate thread
    try:
        thread = threading.Thread(target=run_vectorization, args=(run_id,))
        thread.daemon = True
        thread.start()
        processing_threads[run_id] = thread

        return jsonify({
            'success': True,
            'message': 'Experimental v3 processing started'
        })
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print("Error starting processing thread:")
        print(error_trace)
        return jsonify({'error': f'Failed to start processing: {str(e)}'}), 500

@app.route('/status/<run_id>')
def get_status(run_id):
    if run_id not in processing_status:
        return jsonify({'error': 'Invalid run ID'}), 404

    return jsonify(processing_status[run_id])

@app.route('/results')
def list_results():
    results = []
    results_path = app.config['RESULTS_FOLDER']

    if os.path.exists(results_path):
        for run_id in sorted(os.listdir(results_path), reverse=True):
            run_path = os.path.join(results_path, run_id)
            if os.path.isdir(run_path):
                final_svg = os.path.join(run_path, 'workdir', 'output', 'final.svg')
                input_files = [f for f in os.listdir(run_path) if f.startswith('input_')]

                result_info = {
                    'run_id': run_id,
                    'has_result': os.path.exists(final_svg),
                    'input_file': input_files[0] if input_files else None,
                    'result_file': final_svg,
                }
                results.append(result_info)

    return jsonify(results)

@app.route('/results/<path:filename>')
def serve_result(filename):
    return send_from_directory(app.config['RESULTS_FOLDER'], filename)

@app.route('/view/<run_id>')
def view_result(run_id):
    run_folder = os.path.join(app.config['RESULTS_FOLDER'], run_id)

    if not os.path.exists(run_folder):
        return "Run not found", 404

    # Get all available files
    files = {}

    # Input image
    input_files = [f for f in os.listdir(run_folder) if f.startswith('input_')]
    if input_files:
        files['input'] = os.path.join(run_id, input_files[0])

    # Final SVG (full-size takes priority)
    final_svg_fullsize = os.path.join(run_folder, 'workdir', 'output', 'final_fullsize.svg')
    if os.path.exists(final_svg_fullsize):
        files['final_svg_fullsize'] = os.path.relpath(final_svg_fullsize, app.config['RESULTS_FOLDER'])

    final_svg = os.path.join(run_folder, 'workdir', 'output', 'final.svg')
    if os.path.exists(final_svg):
        files['final_svg'] = os.path.relpath(final_svg, app.config['RESULTS_FOLDER'])

    # Final PNG (full-size)
    final_png = os.path.join(run_folder, 'workdir', 'output', 'final_fullsize.png')
    if os.path.exists(final_png):
        files['final_png'] = os.path.relpath(final_png, app.config['RESULTS_FOLDER'])

    # Simplified images
    simp_img_dir = os.path.join(run_folder, 'workdir', 'output', 'simplified_image_sequence')
    if os.path.exists(simp_img_dir):
        files['simplified_images'] = [
            os.path.relpath(os.path.join(simp_img_dir, f), app.config['RESULTS_FOLDER'])
            for f in sorted(os.listdir(simp_img_dir))
            if f.endswith('.png')
        ]

    # Intermediate SVGs
    struct_svg_dir = os.path.join(run_folder, 'workdir', 'output', 'struct_svgs')
    if os.path.exists(struct_svg_dir):
        svg_files = [f for f in os.listdir(struct_svg_dir) if f.endswith('.svg')]
        if svg_files:
            files['struct_svgs'] = os.path.relpath(
                os.path.join(struct_svg_dir, sorted(svg_files)[-1]),
                app.config['RESULTS_FOLDER']
            )

    return render_template('view_result.html', run_id=run_id, files=files)

if __name__ == '__main__':
    print("=" * 60)
    print("V3 Layered Vectorization - SDXL + VTracer")
    print("=" * 60)
    print("SDXL Improvements:")
    print("  - 1024x1024 resolution (vs 512x512)")
    print("  - Better detail preservation")
    print("  - Aspect ratio preservation (padding)")
    print("")
    print("VTracer features (from app2):")
    print("  - Staircase Removal")
    print("  - Corner Detection")
    print("  - Error-Penalized Simplification")
    print("")
    print("Other versions:")
    print("  - app.py (port 5000) - Original SD 1.5")
    print("  - app2.py (port 5001) - SD 1.5 + VTracer")
    print("This version runs on port 5002")
    print("=" * 60)
    app.run(debug=True, host='0.0.0.0', port=5002)
