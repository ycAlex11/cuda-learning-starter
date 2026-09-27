import time 

from multiprocessing import Process,Queue
from queue import Empty

from transformers import AutoTokenizer

from scheduler.core import Scheduler
from scheduler.models import Task,Worker
from scheduler.workers.qwen_process_worker import MODEL_NAME,qwen_worker_loop
from scheduler.qwen_resources import estimate_qwen_kv_cache_mib

def start_worker_and_wait_for_heartbeat(
    task_queue,
    result_queue,
    heartbeat_queue,
    scheduler,
):
    worker_process = Process(
        target=qwen_worker_loop,
        args=(task_queue, result_queue, heartbeat_queue),
    )
    worker_process.start()

    msg = heartbeat_queue.get(timeout=60)

    if msg.get("type") != "heartbeat":
        raise RuntimeError("Expected a heartbeat message")

    scheduler.heartbeat(
        msg["worker_id"],
        reported_free_vram_mb=msg["reported_free_vram_mb"],
        reported_total_vram_mb=msg["reported_total_vram_mb"],
        reported_prefix_cache_bytes=msg["reported_prefix_cache_bytes"],
    )

    worker = scheduler.workers[msg["worker_id"]]

    print(f"Scheduler received heartbeat from {worker.worker_id}")
    print(f"Recorded cache: {worker.reported_prefix_cache_bytes} bytes")
    print(f"Logical free budget: {worker.free_vram_mb} MiB")

    return worker_process

def main():
    task_queue = Queue()
    result_queue = Queue()
    heartbeat_queue = Queue()

    scheduler = Scheduler(heartbeat_timeout_seconds=1.0)

    scheduler.register_worker(Worker(worker_id="gpu-0",total_vram_mb=4096,used_vram_mb=1024))


    scheduler_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    first_worker = start_worker_and_wait_for_heartbeat(
        task_queue,
        result_queue,
        heartbeat_queue,
        scheduler,
    )
    print(f"Worker free VRAM: {scheduler.workers['gpu-0'].free_vram_mb} MiB")

    prompt = "Explain KV Cache in one short sentence."
    max_new_tokens = 0

    prompt_tokens = scheduler_tokenizer(
        prompt,
        return_tensors="pt",
    )["input_ids"].shape[1]

    required_vram_mb = estimate_qwen_kv_cache_mib(
        prompt_tokens,
        max_new_tokens,
    )

    crash_task = Task(
        task_id="qwen-retry-001",
        operation="qwen_prefill",
        required_vram_mb=required_vram_mb,
        payload={
            "prompt": prompt,
            "max_new_tokens": max_new_tokens,
            "crash_before_report": True,
            },)
    
    scheduler.submit_task(crash_task)
    scheduled_task = scheduler.schedule_once()
    if scheduled_task is None:
        raise RuntimeError("task was not scheduled")
    print(
    f"Attempt {scheduled_task.attempts}: assigned "
    f"{scheduled_task.task_id} to {scheduled_task.assigned_worker_id}"
    )
    print(
        f"Worker free VRAM after scheduling: "
        f"{scheduler.workers['gpu-0'].free_vram_mb} MiB"
        )
    task_queue.put(scheduled_task)
    try:
        result_queue.get(timeout=5)
        raise RuntimeError("worker unexpectedly reported a result")
    except Empty:
        print("No completion result: worker crashed before reporting.")
    first_worker.join(timeout=10)
    if first_worker.is_alive():
        raise RuntimeError("旧进程尚未退出，不能开始重试")
    print(f"First worker exit code: {first_worker.exitcode}")

    time.sleep(1.2)

    requeued_tasks = scheduler.requeue_tasks_from_lost_workers()
    print(f"Requeued tasks: {[task.task_id for task in requeued_tasks]}")
    print(f"Task status after requeue: {crash_task.status.value}")
    print(
        f"Worker free VRAM after requeue: "
        f"{scheduler.workers['gpu-0'].free_vram_mb} MiB"
    )


    task_queue = Queue()
    result_queue = Queue()
    heartbeat_queue = Queue()

    second_worker = start_worker_and_wait_for_heartbeat(
        task_queue,
        result_queue,
        heartbeat_queue,
        scheduler,
    )

    retry_task = scheduler.schedule_once()

    if retry_task is None:
        raise RuntimeError("requeued task was not scheduled")
    print(
    f"Attempt {retry_task.attempts}: assigned "
    f"{retry_task.task_id} to {retry_task.assigned_worker_id}"
    )

    task_queue.put(retry_task)
    retry_result = result_queue.get(timeout=60)
    print(f"Scheduler received retry result: {retry_result}")
    accepted = scheduler.complete_task(
        task_id=retry_result["task_id"],
        worker_id=retry_result["worker_id"],
        attempt=retry_result["attempt"],
        result=retry_result["result"],
    )
    print(f"Retry completion accepted: {accepted}")
    worker = scheduler.workers["gpu-0"]

    # 等待新进程上报已经建立的缓存
    for _ in range(5):
        msg = heartbeat_queue.get(timeout=5)

        scheduler.heartbeat(
            msg["worker_id"],
            reported_free_vram_mb=msg["reported_free_vram_mb"],
            reported_total_vram_mb=msg["reported_total_vram_mb"],
            reported_prefix_cache_bytes=msg["reported_prefix_cache_bytes"],
        )

        if worker.reported_prefix_cache_bytes > 0:
            break
    else:
        raise RuntimeError("没有收到重试后的缓存占用")

    print(f"Recorded cache after retry: {worker.reported_prefix_cache_bytes} bytes")
    print(f"Final task status: {crash_task.status.value}")
    print(f"Final task attempts: {crash_task.attempts}")
    print(
        f"Final worker free VRAM: "
        f"{scheduler.workers['gpu-0'].free_vram_mb} MiB"
    )

    stale_accepted = scheduler.complete_task(
        task_id=crash_task.task_id,
        worker_id="gpu-0",
        attempt=1,
        result="late result from crashed worker",
        )

    print(f"Stale attempt 1 completion accepted: {stale_accepted}")
    print(f"Task status after stale completion: {crash_task.status.value}")
    print(
        f"Worker free VRAM after stale completion: "
        f"{scheduler.workers['gpu-0'].free_vram_mb} MiB"
    )

    duplicate_accepted = scheduler.complete_task(
        task_id=crash_task.task_id,
        worker_id="gpu-0",
        attempt=2,
        result="duplicate result from retry worker",
        )

    print(f"Duplicate attempt 2 completion accepted: {duplicate_accepted}")
    print(
        f"Worker free VRAM after duplicate completion: "
        f"{scheduler.workers['gpu-0'].free_vram_mb} MiB"
    )

    task_queue.put(None)
    second_worker.join(timeout=10)

if __name__ == "__main__":
    main()
