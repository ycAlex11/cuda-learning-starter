from multiprocessing import Process,Queue

from queue import Empty
from transformers import AutoTokenizer

from scheduler.core import Scheduler
from scheduler.models import Task,Worker
from scheduler.workers.qwen_process_worker import (MODEL_NAME,qwen_worker_loop)

from scheduler.qwen_resources import estimate_qwen_kv_cache_mib
from time import sleep



def refresh_heartbeats(scheduler,heartbeat_queue):
    count = 0
    while True:
        try:
            msg = heartbeat_queue.get_nowait()
        except Empty:
            return count 
        
        if msg.get("type") == "heartbeat":
            scheduler.heartbeat(msg["worker_id"],reported_free_vram_mb=msg["reported_free_vram_mb"],reported_total_vram_mb=msg["reported_total_vram_mb"],reported_prefix_cache_bytes=msg.get("reported_prefix_cache_bytes"))
            count+=1

def run_qwen_task(
    scheduler,
    task_queue,
    result_queue,
    heartbeat_queue,
    scheduler_tokenizer,
    task_id,
    prompt,
    max_new_tokens,
    operation= "qwen_generate"
):
    prompt_tokens = scheduler_tokenizer(
        prompt,
        return_tensors="pt",
    )["input_ids"].shape[1]

    required_vram_mb = estimate_qwen_kv_cache_mib(
        prompt_tokens,
        max_new_tokens,
    )

    task = Task(
        task_id=task_id,
        operation=operation,
        required_vram_mb=required_vram_mb,
        payload={
            "prompt": prompt,
            "max_new_tokens": max_new_tokens,
        },
    )
    scheduler.submit_task(task)

    refresh_heartbeats(
        scheduler,
        heartbeat_queue,
    )
    scheduled_task = scheduler.schedule_once()

    if scheduled_task is None:
        raise RuntimeError(f"{task_id} was not scheduled")

    print(
        f"Scheduler assigned {scheduled_task.task_id} "
        f"to {scheduled_task.assigned_worker_id}"
    )
    print(
        f"Reserved KV Cache: "
        f"{scheduled_task.required_vram_mb} MiB"
    )

    task_queue.put(scheduled_task)
    worker_result = result_queue.get(timeout=60)

    print(f"Scheduler received: {worker_result}")

    if worker_result["ok"]:
        accepted = scheduler.complete_task(
            task_id=scheduled_task.task_id,
            worker_id=worker_result["worker_id"],
            attempt=worker_result["attempt"],
            result=worker_result["result"],
        )
    else:
        accepted = scheduler.fail_task(
            task_id=worker_result["task_id"],
            worker_id=worker_result["worker_id"],
            attempt=worker_result["attempt"],
            error=worker_result["error"],
        )

    print(f"Completion accepted: {accepted}")
    print(f"Task status: {task.status.value}")

    return task


def main():
    task_queue = Queue()
    result_queue = Queue()
    heartbeat_queue = Queue()

    worker_process =Process(target=qwen_worker_loop,args=(task_queue,result_queue,heartbeat_queue,600_0000))
    worker_process.start()

    scheduler_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    scheduler = Scheduler()
    worker = Worker(
            worker_id="gpu-0",
            total_vram_mb=4096,
            used_vram_mb=1024,
        )
    scheduler.register_worker(worker)
    first_heartbeat = heartbeat_queue.get(timeout=60,)
    if first_heartbeat.get("type") != "heartbeat":
        raise RuntimeError("worker did not send an initial heartbeat")
    scheduler.heartbeat(first_heartbeat["worker_id"],reported_free_vram_mb=first_heartbeat["reported_free_vram_mb"],reported_total_vram_mb=first_heartbeat["reported_total_vram_mb"],reported_prefix_cache_bytes=first_heartbeat.get("reported_prefix_cache_bytes"),)
    print(
        f"Scheduler received initial heartbeat from "
        f"{first_heartbeat['worker_id']}"
         f"Worker reported prefix cache: "
        f"{worker.reported_prefix_cache_bytes} bytes"
    )

    first_task = run_qwen_task(
        scheduler,
        task_queue,
        result_queue,
        heartbeat_queue,
        scheduler_tokenizer,
        "qwen-process-001",
        "Explain KV Cache in one short sentence.",
        0,
        operation = "qwen_prefill"
    )

    second_task = run_qwen_task(
        scheduler,
        task_queue,
        result_queue,
        heartbeat_queue,
        scheduler_tokenizer,
        "qwen-process-002",
        "Explain KV Cache in one short sentence. Give one example.",
        0,
        operation = "qwen_prefill"
    )

    
    

    third_task = run_qwen_task(
        scheduler,
        task_queue,
        result_queue,
        heartbeat_queue,
        scheduler_tokenizer,
        "qwen-process-003",
        "Explain KV Cache in one short sentence.",
        0,
        operation="qwen_prefill",
    )
    worker = scheduler.workers["gpu-0"]
    print(f"\nFirst task status: {first_task.status.value}")
    print(f"Second task status: {second_task.status.value}")
    print(f"third task status: {third_task.status.value}")
    print(f"Worker free VRAM: {worker.free_vram_mb} MiB")


    sleep(1.2)
    refresh_heartbeats(scheduler, heartbeat_queue)
    print(
        f"Scheduler recorded prefix cache: "
        f"{worker.reported_prefix_cache_bytes} bytes"
    )
    print(f"Worker logical free VRAM: {worker.free_vram_mb} MiB")

    task_queue.put(None)
    worker_process.join(timeout=10)

if __name__ == "__main__":
    main()
