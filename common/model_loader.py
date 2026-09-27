import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_causal_lm(model_name:str,dtype:torch.dtype = torch.float16,device:str="cuda",padding_side:str = "left"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = padding_side
    model = AutoModelForCausalLM.from_pretrained(model_name,dtype=dtype).to(device)

    model.eval()
    return tokenizer,model