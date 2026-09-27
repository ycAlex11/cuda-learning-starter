import torch

n = 1_000_000
warmup = 10
samples = 100

x = torch.ones(n,device="cuda",dtype=torch.float32)
gup_sum = torch.sum(x).item()

print(f"GPU sum: {gup_sum:.1f}")

if gup_sum!=float(n):
    raise RuntimeError("Correctness: FAIL")

print("Correctness: PASS")

for _ in range(warmup):
    result = torch.sum(x)

torch.cuda.synchronize()

times = []

start = torch.cuda.Event(enable_timing=True)

end = torch.cuda.Event(enable_timing=True)

for _ in range(samples):
    start.record() 
    result = torch.sum(x)
    end.record()

    end.synchronize()
    times.append(start.elapsed_time(end))

times.sort()
median_ms = 0.5 * (times[samples // 2 - 1] + times[samples // 2])
print(f"PyTorch sum median: {median_ms:.4f} ms")
print(f"PyTorch sum min / max: {times[0]:.4f} / {times[-1]:.4f} ms")
