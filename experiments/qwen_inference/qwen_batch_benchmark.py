import statistics
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
MAX_NEW_TOKENS = 16

def benchmark(name:str,operation,warmup:int =2,runs:int =5)->float:
    for _ in range(warmup):
        operation()
    
    torch.cuda.synchronize()
    esconds = []

    for _ in range(runs):
        torch.cuda.synchronize()

        start = time.perf_counter()
        operation()

        torch.cuda.synchronize()
        esconds.append(time.perf_counter()-start)
    msecond = statistics.median(esconds)
    print(f"{name} median: {msecond:.3f} s")
    print(
        f"{name} min / max: "
        f"{min(esconds):.3f} / "
        f"{max(esconds):.3f} s"
    )

    return msecond


def generate_sequential(model,tokenizer,prompts:list[str]):
    output_ids_list = []

    with torch.inference_mode():
        for prompt in prompts:
            inputs = tokenizer(prompt,return_tensors = "pt").to("cuda")

            output_ids = model.generate(**inputs,max_new_tokens =MAX_NEW_TOKENS,do_sample=False,use_cache = True)

            output_ids_list.append(output_ids)
    return output_ids_list

def generate_batch(model,tokenizer,prompts:list[str]):
    inputs = tokenizer(prompts,return_tensors = "pt",padding = True).to("cuda")
    with torch.inference_mode():
        output_ids = model.generate(**inputs,max_new_tokens=MAX_NEW_TOKENS,do_sample = False,use_cache = True)
    return output_ids

def main()->None:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME,dtype = torch.float16).to("cuda")

    model.eval()
    prompts = [
        "Explain KV Cache in one short sentence.",
        "What is a GPU worker? Answer briefly.",
    ]
    sequential_seconds = benchmark(
        "Sequential Qwen generate",
        lambda: generate_sequential(
            model,
            tokenizer,
            prompts,
        ),
    )
    batch_seconds = benchmark(
        "Batched Qwen generate",
        lambda: generate_batch(
            model,
            tokenizer,
            prompts,
        ),
    )
    total_generated_tokens = len(prompts) * MAX_NEW_TOKENS

    print(
        f"\nSequential throughput: "
        f"{total_generated_tokens / sequential_seconds:.2f} tokens/s"
    )
    print(
        f"Batched throughput: "
        f"{total_generated_tokens / batch_seconds:.2f} tokens/s"
    )
    print(f"Batch speedup: {sequential_seconds / batch_seconds:.2f}x")
if __name__ == "__main__":
    main()
