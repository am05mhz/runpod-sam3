import os
import sys
import uuid
import asyncio
import importlib.util
import traceback
import queue
import threading
import json
from pathlib import Path

BASE_DIR = Path(__file__).parent

# Job system shared data (importable)
JOB_QUEUE = queue.Queue()  # Thread-safe queue for cross-thread communication
JOBS = {}
GPU_SEMAPHORE = asyncio.Semaphore(1)  # serialize GPU-heavy jobs
WORKER_TASKS = []

WIDTH = 224
N_SEGMENTS = 1500  # Default segment count
N_SEGMENTS_HIGH = 5000  # High quality
N_SEGMENTS_BEST = 10000  # Best quality - finest detail
BATCH_SIZE = 64  # Batch size for GPU inference

QUALITY_SETTINGS = {
    "low": 500,
    "default": 1500,
    "high": 5000,
    "best": 10000,
}

def make_id(prefix: str = "") -> str:
    return (prefix + "_" if prefix else "") + uuid.uuid4().hex[:8]


def import_module_from_path(name: str, path: str):
    """Dynamically import a module from a Python file path.
    
    Supports modules with relative imports by properly setting up the package
    structure and setting the __package__ attribute.
    """
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot create spec for {path}")
    
    module = importlib.util.module_from_spec(spec)
    
    # Add the module's directory and parent directory to sys.path
    module_dir = str(Path(path).parent)
    parent_dir = str(Path(path).parent.parent)
    
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    
    # Set __package__ to enable relative imports
    # For a module at /path/to/supersvg/server.py, __package__ should be "supersvg"
    # so that "from . import something" correctly resolves to the supersvg package
    package_name = Path(path).parent.name
    module.__package__ = package_name
    
    # Add to sys.modules before executing to support relative imports
    sys.modules[name] = module
    
    try:
        spec.loader.exec_module(module)  # type: ignore
    except Exception:
        # Clean up on failure
        sys.modules.pop(name, None)
        traceback.print_exc()
        raise
    
    return module


def safe_mkdir(p: str):
    os.makedirs(p, exist_ok=True)


def join(*parts):
    return os.path.join(*parts)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}

def loadFromFile(filepath):
    if not os.path.exist(filepath):
        return None

    with open(filepath, 'r') as f:
        return json.load(f)

def saveToFile(filepath, data):
    if not os.path.exist(filepath):
        return None

    with open(filepath, 'w') as f:
        return json.dump(data, f, indent=2)
