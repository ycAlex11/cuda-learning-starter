import sys

import torch 


BATCH_SIZE = 1
SEQUENCE_LENGTH = 256

Q_HEADS = 14
KV_HEADS = 2
HEAD_DIM = 64

import statistics

def benchmark(operation,warmup: int =10,runs :int =100)->tuple[float, float, float]:
    for _ in range(warmup):
        operation()

    torch.cuda.synchronize()
    times_ms = []

    for _ in range(runs):
        start = torch.cuda.Event(enable_timing= True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        operation()
        end.record()

        end.synchronize()
        times_ms.append(start.elapsed_time(end))

    return (statistics.median(times_ms),min(times_ms),max(times_ms))    

def pytorch_attention(q:torch.Tensor,k:torch.Tensor,v:torch.Tensor)->torch.Tensor:
    scores = torch.matmul(q,k.transpose(-2,-1))
    scores = scores/(HEAD_DIM**0.5)
    weights = torch.softmax(scores, dim=-1)
    return torch.matmul(weights,v)
'''
def extension_attention(q:torch.Tensor,k:torch.Tensor,v:torch.Tensor)->torch.Tensor:
    scores = torch.matmul(q,k.transpose(-2,-1))
    scores = scores/(HEAD_DIM**0.5)
    flat_scores = scores.reshape(-1,SEQUENCE_LENGTH)
    flat_weights = softmax_cuda.softmax_forward_warp(flat_scores)
    weights = flat_weights.reshape_as(scores)
    return torch.matmul(weights,v)
'''
def main() -> None:
    torch.manual_seed(0)

    q = torch.randn(BATCH_SIZE,Q_HEADS,SEQUENCE_LENGTH,HEAD_DIM,device="cuda",dtype=torch.float16)

    k = torch.randn(BATCH_SIZE,KV_HEADS,SEQUENCE_LENGTH,HEAD_DIM,device="cuda",dtype=torch.float16)

    v = torch.randn(BATCH_SIZE,KV_HEADS,SEQUENCE_LENGTH,HEAD_DIM,device="cuda",dtype=torch.float16)
    KV_GROUP_SIZE = Q_HEADS // KV_HEADS
    k_for_q = k.repeat_interleave(KV_GROUP_SIZE, dim=1)
    #print(k_for_q.shape)
    v_for_q = v.repeat_interleave(KV_GROUP_SIZE, dim=1)

    torch.cuda.synchronize()
    baseline_active_bytes = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    attention_probe_output = pytorch_attention(q, k_for_q, v_for_q)
    torch.cuda.synchronize()
    peak_active_bytes = torch.cuda.max_memory_allocated()
    peak_extra_mib = (peak_active_bytes - baseline_active_bytes) / (1024 ** 2)
    print(f"FP16 attention extra peak active memory: {peak_extra_mib:.2f} MiB")
    del attention_probe_output

    scores = torch.matmul(q, k_for_q.transpose(-2, -1))
    scores = scores / (HEAD_DIM ** 0.5)
    weights = torch.softmax(scores,dim = -1)
    attention_output = torch.matmul(weights, v_for_q)


    flat_scores = scores.reshape(-1,SEQUENCE_LENGTH)
    #extension_flat_weights = softmax_cuda.softmax_forward_warp(flat_scores)
    #extension_weights = extension_flat_weights.reshape_as(scores)
    #print(torch.allclose(extension_weights, weights, atol=1e-5, rtol=1e-5))
    merged_output = attention_output.transpose(1, 2).reshape(
        BATCH_SIZE,
        SEQUENCE_LENGTH,
        Q_HEADS * HEAD_DIM,
    )

    print("*****************************")
    print(merged_output.shape)
    print(attention_output.shape)
    print(weights.shape)
    print(weights[0][0][0].sum())
    print(scores.shape)
    print("***************")
    torch_median, torch_min, torch_max = benchmark(
        lambda: torch.softmax(flat_scores, dim=1)
    )
    #extension_median, extension_min, extension_max = benchmark(lambda: softmax_cuda.softmax_forward_warp(flat_scores))

    
    print()
    print(f"PyTorch Softmax median: {torch_median:.4f} ms")
    print(f"PyTorch Softmax min / max: {torch_min:.4f} / {torch_max:.4f} ms")
    #print(f"CUDA extension Softmax median: {extension_median:.4f} ms")
    #print(f"CUDA extension Softmax min / max: "f"{extension_min:.4f} / {extension_max:.4f} ms")

    pytorch_attention_output = pytorch_attention(q, k_for_q, v_for_q)
    #extension_attention_output = extension_attention(q, k_for_q, v_for_q)

   
    pytorch_attention_median, _, _ = benchmark(
        lambda: pytorch_attention(q, k_for_q, v_for_q)
    )
    #extension_attention_median, _, _ = benchmark(lambda: extension_attention(q, k_for_q, v_for_q))

    print(f"PyTorch full attention median: {pytorch_attention_median:.4f} ms")


if __name__ == "__main__":
    main()
