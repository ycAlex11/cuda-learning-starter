import torch
import time
import statistics

a = torch.tensor([1.0,2.0,3.0],device="cuda")
b = torch.tensor([4.0,5.0,6.0],device="cuda")

c = a+b
print(a.device)
print(b.device)
print(c.device)
print(c)
print("**********************")
n = 1_000_000
x = torch.ones(n,device="cuda")
y = torch.ones(n,device="cuda")
times_ms = []
for _ in range(10):
    z = x+y
torch.cuda.synchronize()
for _ in range(100):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    z = x+y
    end.record()
    end.synchronize()
    times_ms.append(start.elapsed_time(end))
print(f"median: {statistics.median(times_ms):.4f} ms")
print(f"min: {min(times_ms):.4f} ms")
print(f"max: {max(times_ms):.4f} ms")

bytes_moved = 3 * n * x.element_size()
bandwidth_gbs = bytes_moved / (statistics.median(times_ms) * 1e-3) / 1e9

print(f"estimated effective bandwidth: {bandwidth_gbs:.2f} GB/s")
