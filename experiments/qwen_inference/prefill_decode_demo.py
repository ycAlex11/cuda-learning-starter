import statistics
import time 

import torch 
from  common.model_loader import load_causal_lm
from common.benchmarking import benchmark_cuda
from common.causal_lm import(decode_step,extend_attention_mask,prefill,select_greedy_token)


MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

def decode_without_cache(model,prompt_input_ids,prompt_attention_mask,prefill_outputs,step:int):
    first_generated_token = select_greedy_token(prefill_outputs)
    current_input_ids = torch.cat([prompt_input_ids,first_generated_token],dim=1)
    curren_attention_mask = extend_attention_mask(prompt_attention_mask)

    latencies_ms = []

    for _ in range(step):
        torch.cuda.synchronize()
        start = time.perf_counter()

        decode_outputs = decode_step(model,input_ids=current_input_ids,attention_mask=curren_attention_mask,past_key_values=None,use_cache=False)
        torch.cuda.synchronize()
        latencies_ms.append((time.perf_counter() - start) * 1000)
        next_token = select_greedy_token(decode_outputs)

        current_input_ids=torch.cat(
            [
                current_input_ids,
                next_token,
            ],
            dim=1,
        )

        curren_attention_mask = extend_attention_mask(
            curren_attention_mask
        )
    return latencies_ms




def decode_with_cache(model,prefill_outputs,prompt_attention_mask,step:int):
    current_inputs_ids = select_greedy_token(prefill_outputs)

    current_attention_mask = extend_attention_mask(prompt_attention_mask)
    current_cache = prefill_outputs.past_key_values
    decode_latencies_ms = []

    for _ in range(step):
        torch.cuda.synchronize()
        start = time.perf_counter()

        decode_outputs = decode_step(model,input_ids = current_inputs_ids,attention_mask=current_attention_mask,past_key_values =current_cache,use_cache = True)
        torch.cuda.synchronize()
        decode_latencies_ms.append((time.perf_counter() - start) * 1000)
        current_inputs_ids = select_greedy_token(decode_outputs)
        current_cache = decode_outputs.past_key_values
        current_attention_mask = extend_attention_mask(current_attention_mask)
    return decode_latencies_ms,current_cache


def benchmark_test(model,inputs,steps):
    print("\n===== Full generation benchmark =====")
    
    cached_latencies_ms = benchmark_cuda(
        operation=lambda: decode_with_cache(
            model,
            prefill(
                model,
                inputs,
                use_cache=True,
            ),
            inputs["attention_mask"],
            step=steps,
        ),
        warmup=3,
        runs=10,
    )
    uncached_latencies_ms = benchmark_cuda(
        operation=lambda: decode_without_cache(
            model,
            prompt_input_ids=inputs["input_ids"],
            prompt_attention_mask=inputs["attention_mask"],
            prefill_outputs=prefill(
                model,
                inputs,
                use_cache=False,
            ),
            step=steps,
        ),
        warmup=3,
        runs=10,
    )
    cached_median_ms = statistics.median(cached_latencies_ms)
    uncached_median_ms = statistics.median(
        uncached_latencies_ms
    )
    print("\n===== Full generation benchmark =====")

    print(
        f"Cached median: {cached_median_ms:.2f} ms"
    )
    print(
        f"Cached min / max: "
        f"{min(cached_latencies_ms):.2f} / "
        f"{max(cached_latencies_ms):.2f} ms"
    )

    print(
        f"Uncached median: {uncached_median_ms:.2f} ms"
    )
    print(
        f"Uncached min / max: "
        f"{min(uncached_latencies_ms):.2f} / "
        f"{max(uncached_latencies_ms):.2f} ms"
    )

    speedup = uncached_median_ms / cached_median_ms

    print(f"KV Cache speedup: {speedup:.2f}x")


def main() ->None:
    tokenizer,model = load_causal_lm(model_name = MODEL_NAME) 

    prompt = "Explain why KV Cache helps autoregressive LLM inference."

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
    ).to("cuda")
    prompt_tokens = inputs["input_ids"].shape[1]
    torch.cuda.synchronize()
    start = time.perf_counter()
    
    prefill_outputs = prefill(model,inputs,use_cache=True)
    torch.cuda.synchronize()
    prefill_seconds = time.perf_counter() - start
    cache = prefill_outputs.past_key_values
    print(f"Prompt tokens: {prompt_tokens}")
    print(f"Prefill latency: {prefill_seconds * 1000:.2f} ms")
    print(f"Cache type: {type(cache).__name__}")
    steps = 10

    decode_latencies_ms,final_cache = decode_with_cache(model,prefill_outputs,inputs["attention_mask"],step=steps)
    print(f"Decode steps: {steps}")
    print(
        f"Decode median latency: "
        f"{statistics.median(decode_latencies_ms):.2f} ms"
    )
    print(
        f"Decode min / max: "
        f"{min(decode_latencies_ms):.2f} / "
        f"{max(decode_latencies_ms):.2f} ms"
    )
    print(
        "Layer 0 KV length after decode: "
        f"{final_cache.layers[0].keys.shape[2]}"
    )

    
    uncached_prefill_outputs = prefill(model,inputs,use_cache = False)
    uncached_decode_latencies_ms = decode_without_cache(model,prompt_input_ids=inputs["input_ids"],prompt_attention_mask=inputs["attention_mask"],prefill_outputs=uncached_prefill_outputs,step=steps)
    print("\nWithout KV Cache")
    print(
        f"Decode median latency: "
        f"{statistics.median(uncached_decode_latencies_ms):.2f} ms"
    )
    print(
        f"Decode min / max: "
        f"{min(uncached_decode_latencies_ms):.2f} / "
        f"{max(uncached_decode_latencies_ms):.2f} ms"
    )

    print("*****************\n")
    benchmark_test(model,inputs,steps)

if __name__ == "__main__":
    main()

