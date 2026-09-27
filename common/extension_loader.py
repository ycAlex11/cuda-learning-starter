"""Import locally built CUDA extensions from build/python/."""

import importlib
from pathlib import Path
import sys


def load_cuda_extension(module_name: str):
    # Initialize PyTorch's native libraries before importing a linked extension.
    import torch

    extension_dir = Path(__file__).resolve().parents[1] / "build" / "python"
    if str(extension_dir) not in sys.path:
        sys.path.insert(0, str(extension_dir))
    return importlib.import_module(module_name)
