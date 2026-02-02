import os
import sys
import uuid
import asyncio
import importlib.util
import traceback
from pathlib import Path

BASE_DIR = Path(__file__).parent

# Job system shared data (importable)
JOB_QUEUE = asyncio.Queue()
JOBS = {}
GPU_SEMAPHORE = asyncio.Semaphore(1)  # serialize GPU-heavy jobs
WORKER_TASKS = []


def make_id(prefix: str = "") -> str:
    return (prefix + "_" if prefix else "") + uuid.uuid4().hex[:8]


def import_module_from_path(name: str, path: str):
    """Dynamically import a module from a Python file path."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)  # type: ignore
    except Exception:
        traceback.print_exc()
        raise
    return module


def safe_mkdir(p: str):
    os.makedirs(p, exist_ok=True)


def join(*parts):
    return os.path.join(*parts)