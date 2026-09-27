import torch
import torch.nn.functional as F
from common.benchmark_reporting import benchmark
from common.extension_loader import load_cuda_extension
rmsnorm_cuda = load_cuda_extension("rmsnorm_cuda")

def run_rmsnorm_test(label, dtype, atol, rtol):
    x = torch.randn(1, 256, 896, device="cuda", dtype=dtype)
    weight = torch.randn(896, device="cuda", dtype=dtype)
    eps = 1e-6

    pytorch_output = F.rms_norm(
        x,
        normalized_shape=(896,),
        weight=weight,
        eps=eps,
    )

    extension_output = rmsnorm_cuda.rmsnorm_forward(x, weight, eps)

    correct = torch.allclose(
        extension_output,
        pytorch_output,
        atol=atol,
        rtol=rtol,
    )

    max_abs_error = (
        extension_output.float() - pytorch_output.float()
    ).abs().max().item()

    print(f"\n===== {label} =====")
    print(f"Correctness: {'PASS' if correct else 'FAIL'}")
    print(f"Maximum absolute error: {max_abs_error:.8f}")

    benchmark(
        f"{label} PyTorch RMSNorm",
        lambda: F.rms_norm(x, (896,), weight, eps),
    )

    benchmark(
        f"{label} CUDA extension RMSNorm",
        lambda: rmsnorm_cuda.rmsnorm_forward(x, weight, eps),
    )


def run_residual_rmsnorm_test(label, dtype, atol, rtol):
    x = torch.randn(1, 256, 896, device="cuda", dtype=dtype)
    residual = torch.randn_like(x)
    weight = torch.randn(896, device="cuda", dtype=dtype)
    eps = 1e-6
    if dtype == torch.float16:
        pytorch_output = F.rms_norm(
            x.float() + residual.float(),
            normalized_shape=(896,),
            weight=weight.float(),
            eps=eps,
        ).to(torch.float16)
    else:
        pytorch_output = F.rms_norm(
            x + residual,
            normalized_shape=(896,),
            weight=weight,
            eps=eps,
        )

    extension_output = rmsnorm_cuda.residual_rmsnorm_forward(
        x,
        residual,
        weight,
        eps,
    )
    correct = torch.allclose(
        extension_output,
        pytorch_output,
        atol=atol,
        rtol=rtol,
    )
    max_abs_error = (
        extension_output - pytorch_output
    ).abs().max().item()

    print(f"\n===== {label} =====")
    print(f"Correctness: {'PASS' if correct else 'FAIL'}")
    print(f"Maximum absolute error: {max_abs_error:.8f}")

    benchmark(
    "PyTorch separate residual add + RMSNorm",
    lambda: F.rms_norm(x + residual, (896,), weight, eps),
    )

    benchmark(
        "CUDA fused residual add + RMSNorm",
        lambda: rmsnorm_cuda.residual_rmsnorm_forward(
            x,
            residual,
            weight,
            eps,
        ),
    )

torch.manual_seed(0)

run_rmsnorm_test("float32", torch.float32, 1e-5, 1e-5)
run_rmsnorm_test("float16", torch.float16, 1e-3, 1e-3)

run_residual_rmsnorm_test("float32", torch.float32, 1e-5, 1e-5)
run_residual_rmsnorm_test("float16", torch.float16, 1e-3, 1e-3)
