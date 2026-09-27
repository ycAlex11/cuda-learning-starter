import copy 
import statistics
import torch 

from common.benchmarking import benchmark_cuda

from common.causal_lm import decode_step,prefill,append_text_suffix

from common.model_loader import load_causal_lm

from common.validation import compare_tensors,compare_next_token_outputs
from common.prefix_cache import PrefixCache,run_request_from_prefix_cache 


MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

COMMON_PREFIX = (
    "You are a helpful CUDA and AI infrastructure assistant. "
    "Explain technical concepts accurately and concisely. "
) * 50

QUESTIONS = [
    "What is a KV cache?",
    "What is continuous batching?",
]


def prefill_full_requests(model,full_inputs):
    outputs = []
    for inputs in full_inputs:
        request_outputs = prefill(model,inputs,use_cache=True)
        outputs.append(request_outputs)
    return outputs

def prefill_with_prefix_cache(model,prefix_inputs,full_inputs,prefix_tokens:int):
    prefix_outputs = prefill(model,prefix_inputs,use_cache=True)
    prefix_cache = prefix_outputs.past_key_values
    suffix_outputs = []

    for inputs in full_inputs:
        suffix_inputs_idx = inputs["input_ids"][:,prefix_tokens:]
        request_outputs = decode_step(model,input_ids=suffix_inputs_idx,attention_mask=inputs["attention_mask"],past_key_values=copy.deepcopy(prefix_cache),use_cache=True)
        suffix_outputs.append(request_outputs)
    return suffix_outputs


def benchmark_prefix_cache(
    model,
    prefix_inputs,
    prefix_ids,
    full_inputs,
):
    full_latencies_ms = benchmark_cuda(
        operation=lambda: prefill_full_requests(
            model,
            full_inputs,
        ),
        warmup=2,
        runs=5,
    )

    cached_latencies_ms = benchmark_cuda(
        operation=lambda: run_requests_with_prefix_cache(
            model,
            prefix_inputs,
            prefix_ids,
            full_inputs,
            max_entries=2,
        ),
        warmup=2,
        runs=5,
    )

    full_median_ms = statistics.median(full_latencies_ms)
    cached_median_ms = statistics.median(cached_latencies_ms)

    print("\n===== Prefix Cache Prefill Benchmark =====")
    print(
    f"Full prefill min / max: "
    f"{min(full_latencies_ms):.2f} / "
    f"{max(full_latencies_ms):.2f} ms"
    )

    print(
        f"Prefix-cache prefill min / max: "
        f"{min(cached_latencies_ms):.2f} / "
        f"{max(cached_latencies_ms):.2f} ms"
    )
    print(f"Full prefill median: {full_median_ms:.2f} ms")
    print(f"Prefix-cache prefill median: {cached_median_ms:.2f} ms")
    print(f"Prefix-cache speedup: {full_median_ms / cached_median_ms:.2f}x")


def run_requests_with_prefix_cache(model,prefix_inputs,prefix_ids,full_inputs,max_entries:int|None=None):

    cache = PrefixCache(max_entries=max_entries)
    prefix_token_ids = tuple(prefix_ids[0].cpu().tolist())
    prefix_outputs = prefill(model,prefix_inputs,use_cache=True)

    cache.put(token_ids=prefix_token_ids,past_key_values=prefix_outputs.past_key_values)

    cached_outputs= []
    matched_prefix_token_counts = []
    for inputs in full_inputs:
        outputs, matched_tokens = run_request_from_prefix_cache(
            model,
            cache,
            inputs,
        )

        cached_outputs.append(outputs)
        matched_prefix_token_counts.append(matched_tokens)
    return cached_outputs, matched_prefix_token_counts, cache


def demonstrate_miss_then_followup(model,tokenizer,first_request_inputs):
    cold_cache = PrefixCache(max_entries=2)
    _,first_matched_tokens = run_request_from_prefix_cache(model,cold_cache,first_request_inputs)

    followup_inputs = append_text_suffix(
        tokenizer,
        first_request_inputs,
        " Please answer briefly.",
    )
    followup_output,followup_matched_tokens = (run_request_from_prefix_cache(model,cold_cache,followup_inputs))
    followup_baseline = prefill(model,followup_inputs,use_cache=True)
    correct, same_next_token, max_abs_error = (
        compare_next_token_outputs(
            followup_baseline,
            followup_output,
            atol=1e-1,
            rtol=1e-3,
        )
    )
    print("\n===== Cache miss then follow-up hit =====")
    print(
        f"First request matched prefix tokens: "
        f"{first_matched_tokens}"
    )
    print(
        f"Follow-up matched prefix tokens: "
        f"{followup_matched_tokens}"
    )
    print(f"Cache hits: {cold_cache.hits}")
    print(f"Cache misses: {cold_cache.misses}")
    print(f"Cache entries: {cold_cache.entry_count}")
    print(
        f"Follow-up correctness={correct}, "
        f"next_token_match={same_next_token}, "
        f"max_abs_error={max_abs_error:.8f}"
    )

    if not correct or not same_next_token:
        raise RuntimeError(
            "Follow-up cache output does not match full prefill"
        )
    third_inputs = append_text_suffix(tokenizer,followup_inputs,"give an cuda example")
    third_output,third_matched_tokens =(run_request_from_prefix_cache(model,cold_cache,third_inputs))
    third_baseline = prefill(model,third_inputs,use_cache=True)
    third_correct, third_next_token_match, third_max_abs_error = (
        compare_next_token_outputs(
            third_baseline,
            third_output,
            atol=1e-1,
            rtol=1e-3,
        )
    )
    print("\n===== Third follow-up cache hit =====")
    print(
        f"Third request matched prefix tokens: "
        f"{third_matched_tokens}"
    )
    print(f"Cache hits: {cold_cache.hits}")
    print(f"Cache misses: {cold_cache.misses}")
    print(f"Cache entries: {cold_cache.entry_count}")
    print(
        f"Third follow-up correctness={third_correct}, "
        f"next_token_match={third_next_token_match}, "
        f"max_abs_error={third_max_abs_error:.8f}"
    )

    if not third_correct or not third_next_token_match:
        raise RuntimeError(
            "Third follow-up cache output does not match full prefill"
        )
    
    repeated_output,repeated_matched_tokens = run_request_from_prefix_cache(model,cold_cache,third_inputs)
    correct,same_next_token,max_abs_error = compare_next_token_outputs(third_output,repeated_output,atol=0.0,rtol=0.0)
    print("\n===== Exact cache hit =====")
    print(f"Matched tokens: {repeated_matched_tokens}")
    print(f"Correctness: {correct}")
    print(f"Next token match: {same_next_token}")
    print(f"Maximum absolute error: {max_abs_error:.8f}")

    if (
        repeated_matched_tokens != third_inputs["input_ids"].shape[1]
        or not correct
        or not same_next_token
    ):
        raise RuntimeError("Exact cache hit validation failed")


def demonstrate_lru_eviciton():
    cache = PrefixCache(max_entries=2)

    cache.put((1,), "cache A")
    cache.put((2,), "cache B")

    cache.get((1,))
    cache.put((3,), "cache C")
    entry_a = cache.get((1,))
    entry_b = cache.get((2,))
    entry_c = cache.get((3,))
    print("\n===== LRU eviction test =====")
    print(f"Entry A exists: {entry_a is not None}")
    print(f"Entry B exists: {entry_b is not None}")
    print(f"Entry C exists: {entry_c is not None}")
    print(f"Evictions: {cache.evictions}")


def demonstrate_kv_only_cache(model,tokenizer):
    cache = PrefixCache(max_entries=2)
    inputs = tokenizer("Explain KV cache",return_tensors = "pt").to("cuda")
    token_ids = tuple(inputs["input_ids"][0].cpu().tolist())

    original_output = prefill(model,inputs,use_cache=True)
    cache.put(token_ids=token_ids,past_key_values=original_output.past_key_values)
    repaired_output, first_matched = run_request_from_prefix_cache(
        model, cache, inputs
    )

    cached_output, second_matched = run_request_from_prefix_cache(
        model, cache, inputs
    )

    correct, same_next_token, max_abs_error = compare_next_token_outputs(
        repaired_output,
        cached_output,
        atol=0.0,
        rtol=0.0,
    )

    print("\n===== KV-only cache repair =====")
    print(f"First call reused tokens: {first_matched}")
    print(f"Second call reused tokens: {second_matched}")
    print(f"Prompt tokens: {len(token_ids)}")
    print(f"Correctness: {correct}")
    print(f"Maximum absolute error: {max_abs_error:.8f}")

    if (
        first_matched != 0
        or second_matched != len(token_ids)
        or not correct
        or not same_next_token
    ):
        raise RuntimeError("KV-only cache repair validation failed")



def main():
    tokenizer,model = load_causal_lm(model_name = MODEL_NAME)
    demonstrate_kv_only_cache(model, tokenizer)

    prefix_inputs = tokenizer(COMMON_PREFIX,return_tensors = "pt").to("cuda")
    prefix_ids = prefix_inputs["input_ids"]
    prefix_tokens = prefix_ids.shape[1]

    full_inputs = []

    for question in QUESTIONS:
        suffix_inputs = tokenizer(
            question,
            return_tensors="pt",
            add_special_tokens=False,
        ).to("cuda")

        full_input_ids = torch.cat(
            [prefix_ids, suffix_inputs["input_ids"]],
            dim=1,
        )

        full_attention_mask = torch.ones_like(full_input_ids)

        full_inputs.append(
            {
                "input_ids": full_input_ids,
                "attention_mask": full_attention_mask,
            }
        )
        
    
    print(f"Shared prefix tokens: {prefix_tokens}")
    for index, inputs in enumerate(full_inputs):
        
        if not torch.equal(
            inputs["input_ids"][:, :prefix_tokens],
            prefix_ids,
        ):
            raise RuntimeError("Full prompt does not preserve the shared prefix tokens")
        total_tokens = inputs["input_ids"].shape[1]
        print(
            f"Request {index}: "
            f"total tokens={total_tokens}, "
            f"suffix tokens={total_tokens - prefix_tokens}"
        )

    
    full_outputs = prefill_full_requests(model,full_inputs) 

    cached_outputs, matched_prefix_token_counts, cache = (
        run_requests_with_prefix_cache(
            model,
            prefix_inputs,
            prefix_ids,
            full_inputs,
            max_entries=2,
        )
    )


    for index, matched_tokens in enumerate(matched_prefix_token_counts):
        print(
                f"Request {index}: "
                f"matched_prefix_tokens={matched_tokens}"
            )

    print(f"Prefix-cache hits: {cache.hits}")
    print(f"Prefix-cache misses: {cache.misses}")

    for index, (full_output, cached_output) in enumerate(
        zip(full_outputs, cached_outputs)
    ):
        correct, same_next_token, max_abs_error = (
            compare_next_token_outputs(
                full_output,
                cached_output,
                atol=5e-2,
                rtol=1e-3,
            )
        )
        
        print(
        f"Request {index}: "
        f"correctness={correct}, "
        f"next_token_match={same_next_token}, "
        f"max_abs_error={max_abs_error:.8f}"
    )

        if not correct or not same_next_token:
            raise RuntimeError(
                "Prefix-cache output does not match full prefill"
            )
        

    demonstrate_miss_then_followup(model,tokenizer,full_inputs[0])
    demonstrate_lru_eviciton()
    #benchmark_prefix_cache(model,prefix_inputs,prefix_ids,full_inputs)


if __name__ == "__main__":
    main()