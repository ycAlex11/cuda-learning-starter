from __future__ import annotations
import statistics
import torch

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
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        result = operation()
        end.record()

        end.synchronize()
        times_ms.append(start.elapsed_time(end))
    print(f"{name} median: {statistics.median(times_ms):.4f} ms")
    print(f"{name} min / max: {min(times_ms):.4f} / {max(times_ms):.4f} ms")
