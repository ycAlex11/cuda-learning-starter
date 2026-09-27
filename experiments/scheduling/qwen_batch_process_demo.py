from multiprocessing import Process,Queue
from transformers import AutoTokenizer

from scheduler.batching import MicroBatcher
from scheduler.core import Scheduler
from scheduler.models import Task,Worker
from scheduler.workers.qwen_process_worker import MODEL_NAME,qwen_worker_loop
from scheduler.qwen_resources import estimate_qwen_kv_cache_mib

from time import monotonic, sleep

def run_pending_batches(scheduler:Scheduler,batcher:MicroBatcher,task_queue:Queue,result_queue:Queue)->list[float]:

    batch_number = 1
    completed_latencies_ms = []
    worker_available_at = monotonic()

    while True:
        pending_tasks = []
        for task in scheduler.tasks.values():
            if task.status.value == "pending":
                pending_tasks.append(task)
        if not pending_tasks:
            return completed_latencies_ms
        if not batcher.should_form_batch(pending_tasks):
            sleep(0.01)
            continue 
        
        batch = batcher.from_batch(batch_id=f"inference-batch-{batch_number:03d}",pending_tasks=pending_tasks,)
        if batch is None:
            raise RuntimeError("pending tasks exist, but no batch was formed")
        s_times = []
        for task in batch.tasks:
            if task.submitted_at is not None:
                s_times.append(task.submitted_at)
        
        dispatch_time = monotonic()
        oldest_submitted_at = min(s_times)
        queue_wait_ms = (
            monotonic() - oldest_submitted_at
        ) * 1000

        batching_wait_started_at = max(
            oldest_submitted_at,
            worker_available_at,
        )
        batching_wait_ms = (
            dispatch_time - batching_wait_started_at
        ) * 1000

        assigned_worker = scheduler.schedule_batch(batch)
        if assigned_worker is None:
            raise RuntimeError("batch was not scheduled")
        print(f"\nBatch: {batch.batch_id}")
        print(f"Tasks: {[task.task_id for task in batch.tasks]}")
        print(f"Reserved VRAM: {batch.total_required_vram_mb} MiB")
        print(f"Oldest task queue wait: {queue_wait_ms:.1f} ms")
        print(f"Intentional batching wait: {batching_wait_ms:.1f} ms")
        print(f"Assigned worker: {assigned_worker.worker_id}")

        task_queue.put(batch)
        batch_result = result_queue.get(timeout=60)

        if batch_result.get("type") != "batch_result":
            raise RuntimeError(f"unexpected worker result: {batch_result}")
        if not batch_result["ok"]:
            accepted = scheduler.fail_batch(batch,batch_result["worker_id"],batch_result["error"])
            print(f"Batch failure accepted: {accepted}")

            for task in batch.tasks:
                print(
                    f"{task.task_id}: "
                    f"status={task.status.value}, "
                    f"error={task.error}"
                )
            if not accepted:
                raise RuntimeError(
                    f"failed to record batch failure: {batch_result}"
                )
            batch_number+=1
            continue

        accepted = scheduler.complete_batch(
            batch,
            batch_result["worker_id"],
            batch_result["results"],
        )
        print(f"Completion accepted: {accepted}")
        worker_available_at = monotonic()
        for task in batch.tasks:
            print(f"{task.task_id}: {task.status.value}")
            if(task.submitted_at is not None and task.completed_at is not None ):
                end_to_end_latency_ms =(task.completed_at-task.submitted_at)*1000
                print(
                    f"End-to-end latency: "
                    f"{end_to_end_latency_ms:.1f} ms"
                )
                completed_latencies_ms.append(
                    end_to_end_latency_ms
                )


        batch_number += 1

def run_qwen_batch_policy(max_batch_size:int,max_batch_wait_ms:int)->list[float]:
    task_queue = Queue()
    result_queue = Queue()
    heartbeat_queue = Queue()

    scheduler = Scheduler()
    scheduler.register_worker(Worker(worker_id="gpu-0",total_vram_mb=4096,used_vram_mb=1024))
    worker_process = Process(
        target=qwen_worker_loop,
        args=(task_queue, result_queue, heartbeat_queue),
    )
    worker_process.start()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    first_heartbeat = heartbeat_queue.get(timeout = 60)

    if first_heartbeat.get("type") != "heartbeat":
        raise RuntimeError("worker did not send an initial heartbeat")
    scheduler.heartbeat(
        first_heartbeat["worker_id"],
        reported_free_vram_mb=first_heartbeat["reported_free_vram_mb"],
        reported_total_vram_mb=first_heartbeat["reported_total_vram_mb"],
    )
    worker = scheduler.workers["gpu-0"]
    print(f"Worker reported free VRAM: {worker.reported_free_vram_mb} MiB")
    print(f"Worker logical free VRAM: {worker.free_vram_mb} MiB")

    print("************************")

    prompts = [
        "Explain KV Cache in one short sentence.",
        "What is a GPU worker? Answer briefly.",
        "What is continuous batching? Answer briefly.",
    ]
    max_new_tokens = 16
    tasks: list[Task] = []

    for index,prompt in enumerate(prompts,start =1):
        encoded = tokenizer(prompt, return_tensors="pt")
        prompt_tokens = int(encoded["attention_mask"].sum().item())

        task = Task(
            task_id=f"qwen-batch-{index:03d}",
            operation="qwen_generate",
            required_vram_mb=estimate_qwen_kv_cache_mib(
                prompt_tokens,
                max_new_tokens,
            ),
             payload={
                "prompt": prompt,
                "max_new_tokens": max_new_tokens,
                "force_batch_failure": False,
            },
        )
        scheduler.submit_task(task)
        tasks.append(task)
        print(
            f"{task.task_id}: "
            f"prompt_tokens={prompt_tokens}, "
            f"reserved_kv_cache={task.required_vram_mb} MiB"
        )

    batcher = MicroBatcher(max_batch_size=max_batch_size,max_batch_vram_mb=64,max_batch_wait_ms=max_batch_wait_ms)
    lms = run_pending_batches(
        scheduler,
        batcher,
        task_queue,
        result_queue,
    )

    metrics = scheduler.metrics

    print("\nScheduler metrics")
    print(f"Submitted tasks: {metrics.submitted_tasks}")
    print(f"Scheduled attempts: {metrics.scheduled_attempts}")
    print(f"Completed tasks: {metrics.completed_tasks}")
    print(f"Requeued tasks: {metrics.requeued_tasks}")
    print(f"Rejected completions: {metrics.rejected_completions}")
    print(f"Duplicate completions: {metrics.duplicate_completions}")
    print(f"Failed tasks: {metrics.failed_tasks}")

    task_queue.put(None)
    worker_process.join(timeout=10)
    print("Worker process stopped.")
    return lms

def main():
    run_qwen_batch_policy(max_batch_size=2,max_batch_wait_ms=50)

if __name__ == "__main__":
    main()
