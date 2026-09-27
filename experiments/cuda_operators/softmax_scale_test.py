import torch 
from common.extension_loader import load_cuda_extension
softmax_cuda = load_cuda_extension("softmax_cuda")
from common.benchmark_reporting import benchmark


SCALE = 1.0 / 8.0

def checkscheck_fused_scaled_softmax(x: torch.Tensor,atol: float,rtol: float,)->None:
    pytorch_output = torch.softmax(x*SCALE,dim=-1)
    extension_output = softmax_cuda.softmax_forward_warp_scale(x, SCALE)
    correct = torch.allclose(
        extension_output,
        pytorch_output,
        atol,
        rtol,
    )
    max_abs_error = (
        extension_output - pytorch_output
    ).abs().max().item()

    print(f"Fused scaled Softmax correctness: {correct}")
    print(f"Maximum absolute error: {max_abs_error:.8f}")

def benchmark_fused_scaled_softmax(x:torch.Tensor,label:str)->None:
    print()
    print(f"===== {label} =====")
    benchmark(
        "PyTorch separate scale + Softmax:",
        lambda: torch.softmax(x * SCALE, dim=1),
    )

    benchmark(
        "CUDA fused scale + Softmax:",
        lambda: softmax_cuda.softmax_forward_warp_scale(x, SCALE),
    )



def main()->None:
    torch.manual_seed(0)
    x = torch.randn(4096, 256, device="cuda", dtype=torch.float32)
    x2 = torch.randn(4096, 256, device="cuda", dtype=torch.float16)

    checkscheck_fused_scaled_softmax(x, 1e-5, 1e-5)
    checkscheck_fused_scaled_softmax(x2, 1e-3, 1e-3)

    benchmark_fused_scaled_softmax(x,"float32")
    
    benchmark_fused_scaled_softmax(x2,"float16")
    print("**********************")
    benchmark(
        "CUDA separate scale + Softmax:",
        lambda: softmax_cuda.softmax_forward_warp(x * SCALE),
    )

    print("**********************")
    benchmark(
        "CUDA separate scale + Softmax:",
        lambda: softmax_cuda.softmax_forward_warp(x2 * SCALE),
    )
    
    

if __name__ == "__main__":
    main()
