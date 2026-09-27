import torch 

def prefill(model,inputs,use_cache=True):
    with torch.inference_mode():
        return model(**inputs,use_cache =use_cache)


def decode_step(model,input_ids,attention_mask,past_key_values =None,use_cache= True):
    if use_cache and past_key_values is None:
        raise ValueError(
            "past_key_values is required when use_cache=True"
        )
    
    if not use_cache and past_key_values is not None:
        raise ValueError(
            "past_key_values must be None when use_cache=False"
        )
    
    model_inputs = {
        "input_ids":input_ids,
        "attention_mask":attention_mask,
        "use_cache":use_cache
    }

    if past_key_values is not None:
        model_inputs["past_key_values"] = past_key_values
    
    with torch.inference_mode():
        return model(**model_inputs)

def extend_attention_mask(attention_mask):
    batch_size = attention_mask.size(0)

    new_token_mask = torch.ones((batch_size,1),device=attention_mask.device,dtype=attention_mask.dtype)

    return torch.cat([attention_mask,new_token_mask],dim=1)

def select_greedy_token(outputs):
    return outputs.logits[:,-1,:].argmax(dim =-1,keepdim=True)


def append_text_suffix(tokenizer,inputs,suffix_text:str):

    suffix_inputs = tokenizer(suffix_text,return_tensors = "pt",add_special_tokens=False).to(inputs["input_ids"].device)
    input_ids = torch.cat([inputs["input_ids"],suffix_inputs["input_ids"]],dim =1)
    suffix_attention_mask = torch.ones_like(suffix_inputs["input_ids"],dtype= inputs["attention_mask"].dtype)

    attention_mask = torch.cat([inputs["attention_mask"],suffix_attention_mask],dim=1)

    return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }