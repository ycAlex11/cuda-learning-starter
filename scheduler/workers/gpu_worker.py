import torch 

from scheduler.models import Task


from common.extension_loader import load_cuda_extension
vector_add_cuda = load_cuda_extension("vector_add_cuda")

def execute_gpu_task(task:Task)-> str:
    if task.operation!="scale_add":
        raise ValueError(f"unsupported operation: {task.operation}")
    
    num_elements = int(task.payload["num_elements"])
    scale = float(task.payload["scale"])
    bias = float(task.payload["bias"])


    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    before_bytes = torch.cuda.memory_allocated()
    x = torch.ones(num_elements,device="cuda",dtype=torch.float32,)
    output = vector_add_cuda.scale_add(x, scale, bias)
    torch.cuda.synchronize()
    peak_extra_bytes = (
        torch.cuda.max_memory_allocated() - before_bytes
    )
    peak_extra_mib = peak_extra_bytes / (1024 * 1024)

    first_value = float(output[0].item())
    expected_value = 1.0*scale+bias
    if abs(first_value - expected_value) > 1e-5:
        raise RuntimeError(
            f"unexpected result: got {first_value}, "
            f"expected {expected_value}"
        )
    return (
        f"scale_add completed: "
        f"num_elements={num_elements}, "
        f"first_value={first_value},"
        f"peak_extra_mib={peak_extra_mib:.2f}"
    )
