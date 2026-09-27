import torch 
from common.extension_loader import load_cuda_extension
softmax_cuda = load_cuda_extension("softmax_cuda")
from common.benchmark_reporting import benchmark

def main()->None:
    torch.manual_seed(0)
    x = torch.randn(4096,256,device="cuda",dtype=torch.float16)
    pytorch_output = torch.softmax(x,dim =1)
    extension_output = softmax_cuda.softmax_forward_warp(x)
    correct = torch.allclose(extension_output,pytorch_output,atol=1e-3,rtol= 1e-3)

    max_abs_error = (extension_output.float() - pytorch_output.float()).abs().max().item()
    row_sums = extension_output.float().sum(dim=1)
    print(f"Correctness: {'PASS' if correct else 'FAIL'}")
    print(f"Maximum absolute error: {max_abs_error:.8f}")
    print(f"Row-sum minimum: {row_sums.min().item():.8f}")
    print(f"Row-sum maximum: {row_sums.max().item():.8f}")
    benchmark(
        "FP16 PyTorch Softmax:",
        lambda: torch.softmax(x, dim=1),
    )
    benchmark(
        "FP16 CUDA extension Softmax:",
        lambda: softmax_cuda.softmax_forward_warp(x),
    )
if __name__ == "__main__":
    main()
