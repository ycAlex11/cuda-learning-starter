import time 
from multiprocessing import Queue
from threading import Event, Thread

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from scheduler.models import Task 

from scheduler.batching import TaskBatch
from common.prefix_cache import (PrefixCache, run_request_from_prefix_cache)



class WorkerResourceSnapshot:
    def __init__(self):
        self.prefix_cache_bytes: int = 0


MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

def hearbeat_loop(heartbeat_queue:Queue,stop_event:Event,resource_snapshot: WorkerResourceSnapshot,)->None:
    while not stop_event.is_set():
        #heartbeat_queue.put({"type":"heartbeat","worker_id":"gpu-0"})
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        heartbeat_queue.put(
            {
                "type": "heartbeat",
                "worker_id": "gpu-0",
                "reported_free_vram_mb": free_bytes // (1024 * 1024),
                "reported_total_vram_mb": total_bytes // (1024 * 1024),
                "reported_prefix_cache_bytes": resource_snapshot.prefix_cache_bytes,

            }
        )
        stop_event.wait(1.0)

def exectue_qwen_task(model,tokneizer,task:Task)->str:
    if task.operation != "qwen_generate":
        raise ValueError(f"unsupported operation: {task.operation}")
    
    prompt = str(task.payload["prompt"])
    max_new_tokens = int(task.payload["max_new_tokens"])

    inputs = tokneizer(prompt,return_tensors = "pt").to("cuda")

    prompt_tokens = inputs["input_ids"].shape[1]
    torch.cuda.synchronize()

    start = time.perf_counter()
    with torch.inference_mode():
        output_ids = model.generate(**inputs,max_new_tokens=max_new_tokens,do_sample = False,use_cache = True)
    torch.cuda.synchronize()
    elapsed_seconds = time.perf_counter() - start

    generated_ids = output_ids[:,prompt_tokens:]
    generated_tokens = generated_ids.shape[1]

    response = tokneizer.decode(generated_ids[0],skip_special_tokens = True)

    return (
        f"qwen_generate completed: "
        f"prompt_tokens={prompt_tokens}, "
        f"generated_tokens={generated_tokens}, "
        f"elapsed_seconds={elapsed_seconds:.3f}, "
        f"response={response!r}"
    )

def execute_qwen_batch(model,tokenizer,batch:TaskBatch)->dict[str,str]:
    if not batch.tasks:
        raise ValueError("cannot execute an empty batch")
    for task in batch.tasks:
        if task.operation != "qwen_generate":
            raise ValueError(
                f"unsupported operation in batch: {task.operation}"
            )
    max_new_tokens = int(batch.tasks[0].payload["max_new_tokens"])
    for task in batch.tasks:
        if int(task.payload["max_new_tokens"]) != max_new_tokens:
            raise ValueError(
                "all tasks in a batch must use the same max_new_tokens"
            )
    prompts = [
        str(task.payload["prompt"])
        for task in batch.tasks
    ]

    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
    ).to("cuda")

    prompt_token_counts = (
        inputs["attention_mask"].sum(dim=1).tolist()
    )

    torch.cuda.synchronize()
    start = time.perf_counter() 

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
        )

    torch.cuda.synchronize()
    elapsed_seconds = time.perf_counter() - start
    padded_prompt_length = inputs["input_ids"].shape[1]
    generated_ids = output_ids[:, padded_prompt_length:]
    responses = tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )
    
    results = {}
    for task, prompt_tokens, response in zip(
        batch.tasks,
        prompt_token_counts,
        responses,
    ):
        results[task.task_id] = (
            f"qwen_batch_generate completed: "
            f"batch_id={batch.batch_id}, "
            f"batch_size={len(batch.tasks)}, "
            f"prompt_tokens={int(prompt_tokens)}, "
            f"generated_tokens={generated_ids.shape[1]}, "
            f"batch_elapsed_seconds={elapsed_seconds:.3f}, "
            f"response={response.strip()!r}"
        )
    
    return results


def exectue_qwen_prefill_task(model,tokenizer,task:Task,prefix_cache:PrefixCache):
    prompt = str(task.payload["prompt"])
    inputs = tokenizer(prompt,return_tensors= "pt").to("cuda")

    outputs,matched_tokens = run_request_from_prefix_cache(model,prefix_cache,inputs,)
    torch.cuda.synchronize()
    cache_bytes = prefix_cache.total_bytes
    cache_mib = cache_bytes / (1024 ** 2)

    return (
        f"qwen_prefill completed: "
        f"prompt_tokens={inputs['input_ids'].shape[1]}, "
        f"matched_prefix_tokens={matched_tokens}, "
        f"cache_hits={prefix_cache.hits}, "
        f"cache_misses={prefix_cache.misses}, "
        f"cache_entries={prefix_cache.entry_count} "
        f"cache_bytes={cache_bytes}, "
        f"cache_mib={cache_mib:.4f}, "
        f"cache_evictions={prefix_cache.evictions}, "
    )

def qwen_worker_loop(task_queue:Queue,result_queue:Queue,heartbeat_queue:Queue,prefix_cache_max_bytes: int | None = None,)->None:
    print("Qwen worker loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token =  tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float16,
    ).to("cuda")

    model.eval()
    prefix_cache = PrefixCache(max_entries=2,max_bytes=prefix_cache_max_bytes,)
    print("Qwen worker ready.")
    resource_snapshot = WorkerResourceSnapshot()
    stop_heartbeat = Event()
    heartbeat_thread = Thread(target = hearbeat_loop,args=(heartbeat_queue,stop_heartbeat,resource_snapshot),daemon = True)
    heartbeat_thread.start()
    while True:
        task = task_queue.get()

        if task is None:
            stop_heartbeat.set()
            heartbeat_thread.join(timeout=1)
            print("Qwen worker received shutdown signal.")
            return
        
        if isinstance(task,TaskBatch):
            print(
                f"Qwen worker received batch: "
                f"{task.batch_id}, size={len(task.tasks)}"
            )
            try:
                if any(
                    task.payload.get("force_batch_failure", False)
                    for task in task.tasks
                ):
                    raise RuntimeError("forced batch failure for testing")
                results = execute_qwen_batch(
                    model,
                    tokenizer,
                    task,
                )

                result_queue.put(
                    {
                        "type": "batch_result",
                        "batch_id": task.batch_id,
                        "worker_id": "gpu-0",
                        "ok": True,
                        "results": results,
                    }
                )
            except Exception as error:
                result_queue.put(
                    {
                        "type": "batch_result",
                        "batch_id": task.batch_id,
                        "worker_id": "gpu-0",
                        "ok": False,
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
            continue
        if not isinstance(task, Task):
            result_queue.put(
                {
                    "worker_id": "gpu-0",
                    "ok": False,
                    "error": "worker received an invalid task",
                }
            )
            continue
        
        print(f"Qwen worker received task: {task.task_id}")

        try:
            if task.operation == "qwen_prefill":
                result = exectue_qwen_prefill_task(model,tokenizer,task, prefix_cache)
            else:
                result = exectue_qwen_task(model,tokenizer,task)
            resource_snapshot.prefix_cache_bytes = prefix_cache.total_bytes
            if task.payload.get("crash_before_report", False) and task.attempts == 1:
                print("Worker finished attempt 1, then crashes before reporting.")
                raise SystemExit(1)
            result_queue.put({
                "task_id":task.task_id,
                "worker_id":"gpu-0",
                "attempt":task.attempts,
                "ok":True,
                "result":result
            })
        except Exception as error:
            result_queue.put(
                {
                    "task_id": task.task_id,
                    "worker_id": "gpu-0",
                    "attempt": task.attempts,
                    "ok": False,
                    "error": f"{type(error).__name__}: {error}",
                }
            )
