import torch
import time 
import statistics
from transformers import AutoModelForCausalLM, AutoTokenizer
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"


def generate_once(modle,inputs):
    with torch.inference_mode():
        return modle.generate(
            ** inputs,
            max_new_tokens = 64,
            do_sample = False,
        )

def main()->None: 
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME,torch_dtype = torch.float16,).to("cuda")

    model.eval()
    print(model.model.layers[0])
    print(f"Layers: {model.config.num_hidden_layers}")
    print(f"Hidden size: {model.config.hidden_size}")
    print(f"Attention heads: {model.config.num_attention_heads}")
    print(f"KV heads: {model.config.num_key_value_heads}")
    messages = [
        {"role": "user", "content": "用一句话解释什么是 CUDA。"}
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    for _ in range(2):
        generate_once(model,inputs)
    torch.cuda.reset_peak_memory_stats()
    latencies_seconds = []
    throughputs = []

    for _ in range(5):
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        output_ids = generate_once(model,inputs)
        torch.cuda.synchronize()
        elapsed_seconds = time.perf_counter() - start_time
        generated_tokens = (output_ids.shape[1]- inputs["input_ids"].shape[1])
        latencies_seconds.append(elapsed_seconds)
        throughputs.append(generated_tokens / elapsed_seconds)
        peak_memory_mib = torch.cuda.max_memory_allocated() / (1024 ** 2)
    generated_ids = output_ids[:,inputs["input_ids"].shape[1]:]
    answer = tokenizer.batch_decode(generated_ids,skip_special_tokens = True)[0]

    print(f"Response: {answer}")
    print(f"Generated tokens: {generated_tokens}")
    print(
        f"Generation median: "
        f"{statistics.median(latencies_seconds):.3f} s"
    )
    print(
        f"Throughput median: "
        f"{statistics.median(throughputs):.2f} tokens/s"
    )
    print(
        f"Throughput min / max: "
        f"{min(throughputs):.2f} / {max(throughputs):.2f} tokens/s"
    )
    print(f"Peak active GPU memory: {peak_memory_mib:.2f} MiB")
    print(f"Loaded: {MODEL_NAME}")
    print(f"Model device: {next(model.parameters()).device}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

if __name__ == "__main__":
    main()