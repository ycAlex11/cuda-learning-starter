from math import ceil
QWEN_NUM_LAYERS = 24
QWEN_NUM_KV_HEADS = 2
QWEN_HEAD_DIM = 64
FP16_BYTES = 2



def estimate_qwen_kv_cache_mib(prompt_tokens:int, max_new_tokens:int,batch_size:int =1) -> int:
    total_tokens = prompt_tokens+max_new_tokens
    total_bytes = (
        batch_size
        * total_tokens
        * QWEN_NUM_LAYERS
        * 2
        * QWEN_NUM_KV_HEADS
        * QWEN_HEAD_DIM
        * FP16_BYTES
    )

    return max(1, ceil(total_bytes / (1024 * 1024)))