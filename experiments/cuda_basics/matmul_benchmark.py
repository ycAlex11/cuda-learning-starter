"""A small, repeatable PyTorch CUDA matrix-multiplication benchmark."""
from __future__ import annotations
import argparse
import statistics
import torch

def timed_matmul(a: torch.Tensor, b: torch.Tensor, warmup: int, iterations: int) -> list[float]:
    for _ in range(warmup):
        torch.mm(a, b)
    torch.cuda.synchronize()
    samples_ms: list[float] = []
    for _ in range(iterations):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        torch.mm(a, b)
        end.record()
        end.synchronize()
        samples_ms.append(start.elapsed_time(end))
    return samples_ms

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=2048, help="square matrix dimension")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--dtype", choices=("float32", "float16"), default="float32")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable; run scripts/check_env.py first.")
    dtype = getattr(torch, args.dtype)
    torch.manual_seed(0)
    a = torch.randn(args.size, args.size, device="cuda", dtype=dtype)
    b = torch.randn_like(a)
    samples_ms = timed_matmul(a, b, args.warmup, args.iterations)
    median_ms = statistics.median(samples_ms)
    tflops = (2 * args.size**3) / (median_ms * 1e-3) / 1e12
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"matmul: {args.size}x{args.size}, {args.dtype}, {args.iterations} timed iterations")
    print(f"median: {median_ms:.3f} ms | mean: {statistics.mean(samples_ms):.3f} ms | {tflops:.2f} TFLOP/s")

if __name__ == "__main__":
    main()
