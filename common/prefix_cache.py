from dataclasses import  dataclass 
from collections import OrderedDict
import copy
from common.causal_lm import decode_step, prefill
import torch 
from transformers.modeling_outputs import CausalLMOutputWithPast

@dataclass
class PrefixCacheEntry:
    token_ids:tuple[int,...]
    past_key_values:object
    next_token_logits:torch.Tensor | None = None


def tensor_bytes(tensor:torch.Tensor|None) ->int:
    if tensor is None:
        return 0
    return tensor.numel() * tensor.element_size()

def past_key_values_bytes(past_key_values:object) ->int:
    total = 0
    if hasattr(past_key_values,"layers"):
        for layer in past_key_values.layers:
            total+= tensor_bytes(layer.keys)
            total+= tensor_bytes(layer.values)
        return total
    
    if isinstance(past_key_values,(list,tuple)):
        for item in past_key_values:
            if isinstance(item,(list,tuple)):
                for tensor in item:
                    total += tensor_bytes(tensor)
            elif torch.is_tensor(item):
                total += tensor_bytes(item)
    return total 
    

class PrefixCache:
    def __init__(self,max_entries:int|None=None,max_bytes: int | None = None,):
        if max_entries is not None and max_entries < 1:
            raise ValueError(
                "max_entries must be at least 1"
            )
        self.max_entries = max_entries
        self.max_bytes = max_bytes
        self._entries:OrderedDict[tuple[int,...],PrefixCacheEntry] = OrderedDict()

        self.hits = 0
        self.misses = 0
        self.evictions = 0

    def get(self,token_ids:tuple[int,...])->PrefixCacheEntry|None:
        entry = self._entries.get(token_ids)

        if entry is None:
            self.misses +=1
        else:
            self.hits +=1
            self._entries.move_to_end(token_ids)
        
        return entry

    def get_longest_prefix(self,request_token_ids:tuple[int,...])->PrefixCacheEntry|None:
        for end in range(len(request_token_ids),0,-1):
            candidate_token_ids = request_token_ids[:end]
            entry = self._entries.get(candidate_token_ids)
            if entry is not None:
                self.hits+=1
                self._entries.move_to_end(
                    candidate_token_ids
                )
                return entry
        self.misses+=1
        return None

    def put(self,token_ids: tuple[int,...],past_key_values:object,next_token_logits:torch.Tensor|None=None)->None:
        self._entries[token_ids] = PrefixCacheEntry(token_ids=token_ids,past_key_values=past_key_values,next_token_logits=next_token_logits)
        self._entries.move_to_end(token_ids)

        if(self.max_entries is not None and len(self._entries)>self.max_entries):
            self._entries.popitem(last=False)
            self.evictions += 1
        
        while(self.max_bytes is not None and self._entries and self.total_bytes>self.max_bytes):
            self._entries.popitem(last = False)
            self.evictions+=1
    
    
    @property
    def entry_count(self)->int:
        return len(self._entries)
    
    @property
    def total_bytes(self) ->int:
        total = 0 
        for entry in self._entries.values():
            total += past_key_values_bytes(entry.past_key_values)
            total += tensor_bytes(entry.next_token_logits)
        
        return total

def run_request_from_prefix_cache(model,cache:PrefixCache,request_inputs):
    request_token_ids = tuple(request_inputs["input_ids"][0].cpu().tolist())
    entry = cache.get_longest_prefix(request_token_ids)
    if entry is None:
        outputs = prefill(model,request_inputs,use_cache = True)
        cache.put(token_ids=request_token_ids,past_key_values=outputs.past_key_values, next_token_logits=outputs.logits[:, -1:, :].detach().clone(),)
        return outputs,0

    matched_tokens = len(entry.token_ids)
    if matched_tokens == len(request_token_ids):
        if entry.next_token_logits is not None:
            outputs = CausalLMOutputWithPast(logits=entry.next_token_logits.clone(),past_key_values=copy.deepcopy(entry.past_key_values))
            return outputs,matched_tokens
        outputs = prefill(model,request_inputs,use_cache=True)
        cache.put(token_ids=request_token_ids,past_key_values=outputs.past_key_values,next_token_logits=outputs.logits[:, -1:, :].detach().clone(),)
        return outputs,0
    suffix_input_ieds = request_inputs["input_ids"][:,matched_tokens:,]
    outputs = decode_step(model,input_ids=suffix_input_ieds,attention_mask=request_inputs["attention_mask"],past_key_values=copy.deepcopy(entry.past_key_values),use_cache=True)
    cache.put(token_ids=request_token_ids,past_key_values=outputs.past_key_values, next_token_logits=outputs.logits[:, -1:, :].detach().clone(),)
    return outputs,matched_tokens

