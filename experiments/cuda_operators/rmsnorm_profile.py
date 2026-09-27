import torch 
import torch.nn.functional as F 
from torch.profiler import profile, ProfilerActivity,record_function

from common.extension_loader import load_cuda_extension
rmsnorm_cuda = load_cuda_extension("rmsnorm_cuda")

def profile_once(lable,opreation):
    for _ in range(10):
        opreation
    
    torch.cuda.synchronize()

    with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA]) as prof:
        with record_function(lable):
            opreation()

        torch.cuda.synchronize()
    print(f"\n===== {lable} =====")
    print(prof.key_averages().table(sort_by= "self_cuda_time_total",row_limit=20))


def main():
    torch.manual_seed(0)

    x = torch.randn(1,256,896,device="cuda",dtype = torch.float16)
    weight = torch.randn(896,device="cuda",dtype = torch.float16)

    eps = 1e-6

    profile_once("PyTorch FP16 RMSNorm",lambda: F.rms_norm(x, (896,), weight, eps),)
    profile_once("CUDA extension FP16 RMSNorm",lambda: rmsnorm_cuda.rmsnorm_forward(x, weight, eps))


if __name__ == "__main__":
    main()

