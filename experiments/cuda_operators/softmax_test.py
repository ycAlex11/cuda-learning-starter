import torch 
from common.extension_loader import load_cuda_extension
softmax_cuda = load_cuda_extension("softmax_cuda")
from common.benchmark_reporting import benchmark,peak_extra_memory_mb

def main() ->None:
    torch.manual_seed(0)
    x = torch.randn(4096,256,device="cuda",dtype=torch.float32,)
    pytorch_output = torch.softmax(x,dim =1)
    extenstion_output = softmax_cuda.softmax_forward(x)
    extenstion_output2 = softmax_cuda.softmax_forward_warp(x)
    correct = torch.allclose(extenstion_output,pytorch_output,atol=1e-5,rtol=1e-5)
    warp_correct = torch.allclose(extenstion_output2, pytorch_output, atol=1e-5, rtol=1e-5)
    if not correct or not warp_correct: 
        raise RuntimeError("softmax extension output does not match PyTorch")
    
    else:
        max_abs_error = (extenstion_output - pytorch_output
        ).abs().max().item()
        row_sums = extenstion_output.sum(dim=1)
        print(f"Correctness: {'PASS' if correct else 'FAIL'}")
        print(f"Maximum absolute error: {max_abs_error:.8f}")
        print(f"Row-sum minimum: {row_sums.min().item():.8f}")
        print(f"Row-sum maximum: {row_sums.max().item():.8f}")
    
    benchmark("pytorch-softmax: ",lambda:torch.softmax(x,dim =1))
    benchmark("cuda-extension-softmax: ",lambda:softmax_cuda.softmax_forward(x))
    benchmark("cuda-extension-softmax with warp: ",lambda:softmax_cuda.softmax_forward_warp(x))
    a1 = peak_extra_memory_mb(lambda:torch.softmax(x,dim =1))
    a2 = peak_extra_memory_mb(lambda:softmax_cuda.softmax_forward(x))
    a3 = peak_extra_memory_mb(lambda:softmax_cuda.softmax_forward_warp(x))
    print(a1)
    print("****************")
    print(a2)
    print("****************")
    print(a3)


if __name__ == "__main__":
    main()
