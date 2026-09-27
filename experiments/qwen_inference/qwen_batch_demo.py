import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

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
    inputs = tokenizer(prompts,return_tensors = "pt",padding = True).to("cuda")
    print(f"Batch input_ids shape: {inputs['input_ids'].shape}")
    print(f"Batch attention_mask shape: {inputs['attention_mask'].shape}")

    with torch.inference_mode():
        output_ids = model.generate(**inputs,max_new_tokens = 16,do_sample = False,use_cache = True)

    prompt_length = inputs["input_ids"].shape[1]
    new_token_ids = output_ids[:, prompt_length:]
    responses = tokenizer.batch_decode(
        new_token_ids,
        skip_special_tokens=True,
    )
    for index, response in enumerate(responses):
        print(f"\nRequest {index}")
        print(f"Prompt: {prompts[index]}")
        print(f"Response: {response.strip()}")

if __name__ == "__main__":
    main()
