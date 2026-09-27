import torch
from transformers import AutoModelForCausalLM, AutoTokenizer



from common.benchmark_reporting import benchmark

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"


def decode_with_cache(
    model,
    input_ids,
    attention_mask,
    num_new_tokens,
):
    generated_ids = input_ids
    current_input_ids = input_ids
    current_mask = attention_mask
    cache = None

    with torch.inference_mode():
        for _ in range(num_new_tokens):
            outputs = model(
                input_ids=current_input_ids,
                attention_mask=current_mask,
                past_key_values=cache,
                use_cache=True,
            )

            next_token = outputs.logits[:, -1, :].argmax(
                dim=-1,
                keepdim=True,
            )

            cache = outputs.past_key_values
            generated_ids = torch.cat(
                [generated_ids, next_token],
                dim=1,
            )

            current_input_ids = next_token
            current_mask = torch.cat(
                [
                    current_mask,
                    torch.ones(
                        (generated_ids.size(0), 1),
                        device=generated_ids.device,
                        dtype=current_mask.dtype,
                    ),
                ],
                dim=1,
            )

    return generated_ids

def decode_without_cache(
    model,
    input_ids,
    attention_mask,
    num_new_tokens,
):
    generated_ids = input_ids
    current_mask = attention_mask

    with torch.inference_mode():
        for _ in range(num_new_tokens):
            outputs = model(
                input_ids=generated_ids,
                attention_mask=current_mask,
                use_cache=False,
            )

            next_token = outputs.logits[:, -1, :].argmax(
                dim=-1,
                keepdim=True,
            )

            generated_ids = torch.cat(
                [generated_ids, next_token],
                dim=1,
            )

            current_mask = torch.cat(
                [
                    current_mask,
                    torch.ones(
                        (generated_ids.size(0), 1),
                        device=generated_ids.device,
                        dtype=current_mask.dtype,
                    ),
                ],
                dim=1,
            )

    return generated_ids

def cache_size_mib(cache):
    total_bytes = 0

    for layer in cache.layers:
        total_bytes += layer.keys.numel() * layer.keys.element_size()
        total_bytes += layer.values.numel() * layer.values.element_size()

    return total_bytes / (1024 * 1024)

def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME,dtype = torch.float16).to("cuda")
    model.eval()

    prompt = (
    "KV cache avoids recomputing historical keys and values "
    "during autoregressive language model decoding. "
    * 16
    )
    inputs = tokenizer(prompt,return_tensors = "pt").to("cuda")
    with torch.inference_mode():
        outputs = model(**inputs, use_cache=True)
    cache = outputs.past_key_values
    prefill_cache_mib = cache_size_mib(cache)

    print(f"Cache type: {type(cache).__name__}")
    print(f"Cache layers: {len(cache)}")

    first_layer = cache.layers[0]
    first_key = first_layer.keys
    first_value = first_layer.values

    print(f"Prompt tokens: {inputs['input_ids'].shape[1]}")
    print(f"Layer 0 key shape: {first_key.shape}")
    print(f"Layer 0 value shape: {first_value.shape}")
    next_token = outputs.logits[:,-1,:].argmax(
        dim = -1,
        keepdim = True 
    )

    next_attention_mask = torch.cat(
        [
            inputs["attention_mask"],
            torch.ones((1, 1), device="cuda", dtype=torch.long),
        ],
        dim=1,
    )

    with torch.inference_mode():
        next_outputs = model(
            input_ids = next_token,
            attention_mask = next_attention_mask,
            past_key_values = cache,
            use_cache = True
        )
    
    next_cache = next_outputs.past_key_values
    next_key = next_cache.layers[0].keys
    decode_cache_mib = cache_size_mib(next_cache)
    print(f"KV Cache after prefill: {prefill_cache_mib:.4f} MiB")
    print(f"KV Cache after one decode step: {decode_cache_mib:.4f} MiB")
    print(
        "KV Cache growth per token: "
        f"{(decode_cache_mib - prefill_cache_mib) * 1024:.2f} KiB"
    )
    return

    print(f"Generated token id: {next_token.item()}")
    print(f"Layer 0 key shape after one decode step: {next_key.shape}")

    num_new_tokens = 16

    output_without_cache = decode_without_cache(
        model,
        inputs["input_ids"],
        inputs["attention_mask"],
        num_new_tokens,
    )

    output_with_cache = decode_with_cache(
        model,
        inputs["input_ids"],
        inputs["attention_mask"],
        num_new_tokens,
    )

    correct = torch.equal(
        output_without_cache,
        output_with_cache,
    )

    print(f"KV Cache correctness: {correct}")
    print(f"Final token shape: {output_with_cache.shape}")

    if not correct:
        raise RuntimeError(
            "Cached decode output does not match uncached decode output"
        )


    benchmark(
        "Uncached decode",
        lambda: decode_without_cache(
            model,
            inputs["input_ids"],
            inputs["attention_mask"],
            num_new_tokens,
        ),
        warmup=2,
        runs=5,
    )

    benchmark(
        "KV Cache decode",
        lambda: decode_with_cache(
            model,
            inputs["input_ids"],
            inputs["attention_mask"],
            num_new_tokens,
        ),
        warmup=2,
        runs=5,
    )

if __name__ == "__main__":
    main()
