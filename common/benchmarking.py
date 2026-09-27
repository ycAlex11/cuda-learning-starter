
import torch

def benchmark_cuda(operation,warmup: int = 10,runs:int=10)->list[float]:

    for _ in range(warmup):
        operation()
    
    torch.cuda.synchronize()

    latencies_ms = []

    for _ in range(runs):
        start = torch .cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        operation()
        end.record()

        end.synchronize()

        latencies_ms.append(start.elapsed_time(end))
    return latencies_ms



