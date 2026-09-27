"""Correctness check and timing for the locally built CUDA extension."""
from __future__ import annotations
import statistics
import torch
try:
    from common.extension_loader import load_cuda_extension
    vector_add_cuda = load_cuda_extension("vector_add_cuda")
except ImportError as exc:
    raise SystemExit("Extension not built. Run .\\scripts\\build_extension.ps1 first.") from exc

def peak_extra_memory_mb(operation):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    before = torch.cuda.memory_allocated()

    result = operation()
    torch.cuda.synchronize()

    peak = torch.cuda.max_memory_allocated()

    del result
    return(peak-before)/(1024*1024)

def benchmark(name,operation,warmup=10,runs =100):
    for _ in range(warmup):
        result = operation()
    
    torch.cuda.synchronize()
    times_ms = []

    for _ in range(runs):
        start = torch.cuda.Event(enable_timing= True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        result = operation()
        end.record()

        end.synchronize()
        times_ms.append(start.elapsed_time(end))
    print(f"{name} median: {statistics.median(times_ms):.4f} ms")
    print(f"{name} min / max: {min(times_ms):.4f} / {max(times_ms):.4f} ms")

def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable.")
    x = torch.randn(4 * 1024 * 1024, device="cuda", dtype=torch.float32)
    scale = 1.5
    bias = 2.0

    torch_scale_add = x * scale + bias
    extension_scale_add = vector_add_cuda.scale_add(x, scale, bias)

    torch.testing.assert_close(extension_scale_add, torch_scale_add)
    print("scale_add correctness: PASS")
    for _ in range(100):
        vector_add_cuda.scale_add(x,scale,bias)
        x*scale+bias

    torch.cuda.synchronize()
    for round_index in range(3):
        print(f"\n===== Round {round_index + 1} =====")

        if round_index % 2 == 0:
            benchmark("CUDA Extension add_one", lambda: vector_add_cuda.scale_add(x,scale,bias),)
            benchmark("PyTorch x + 1", lambda:x*scale+bias, )
        else:
            benchmark("PyTorch x + 1", lambda:x*scale+bias, )
            benchmark("CUDA Extension add_one", lambda: vector_add_cuda.scale_add(x,scale,bias),)
    pytorch_peak_mb= peak_extra_memory_mb(lambda:x*scale+bias)
    extension_peak_mb = peak_extra_memory_mb(
    lambda: vector_add_cuda.scale_add(x, scale, bias)
)
    print(f"PyTorch separate x * scale + bias: {pytorch_peak_mb:.2f} MiB")
    print(f"CUDA Extension fused scale_add: {extension_peak_mb:.2f} MiB")

if __name__ == "__main__":
    main()
