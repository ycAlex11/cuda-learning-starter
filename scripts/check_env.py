"""Print the CUDA/PyTorch environment needed by the examples."""
from __future__ import annotations
import platform
import struct
import sys

def main() -> None:
    print(f"Python: {sys.version.split()[0]} ({struct.calcsize('P') * 8}-bit)")
    print(f"Platform: {platform.platform()}")
    if struct.calcsize("P") * 8 != 64:
        raise SystemExit("ERROR: install and use 64-bit Python before continuing.")
    try:
        import torch
    except ImportError as exc:
        raise SystemExit("ERROR: PyTorch is not installed in this interpreter.") from exc
    print(f"PyTorch: {torch.__version__}")
    print(f"PyTorch CUDA runtime: {torch.version.cuda}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        raise SystemExit("ERROR: PyTorch cannot access CUDA. Check the NVIDIA driver and PyTorch CUDA wheel.")
    index = torch.cuda.current_device()
    print(f"GPU {index}: {torch.cuda.get_device_name(index)}")
    print(f"Compute capability: {torch.cuda.get_device_capability(index)}")
    print(f"VRAM: {torch.cuda.get_device_properties(index).total_memory / 2**30:.2f} GiB")

if __name__ == "__main__":
    main()
