from multiprocessing import Queue

from transformers import AutoTokenizer

from scheduler.core import Scheduler
from scheduler.models import Task, Worker
from scheduler.workers.qwen_process_worker import MODEL_NAME
from experiments.scheduling.qwen_retry_demo import start_worker_and_wait_for_heartbeat


def main():
    task_queue = Queue()
    result_queue = Queue()
    heartbeat_queue = Queue()

    scheduler = Scheduler(heartbeat_timeout_seconds=5.0)

    scheduler.register_worker(
        Worker(
            worker_id="gpu-0",
            total_vram_mb=4096,
            used_vram_mb=1024,
        )
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    worker_process = start_worker_and_wait_for_heartbeat(
        task_queue,
        result_queue,
        heartbeat_queue,
        scheduler,
    )
    worker = scheduler.workers["gpu-0"]

    too_large_task = Task(
        task_id="qwen-too-large-001",
        operation="qwen_generate",
        required_vram_mb=2500,
        payload={
            "prompt": "This task should not be scheduled.",
            "max_new_tokens": 16,
        },
    )

    scheduler.submit_task(too_large_task)

    print(f"Scheduler logical free VRAM: {worker.free_vram_mb} MiB")
    print(f"Worker reported CUDA free VRAM: {worker.reported_free_vram_mb} MiB")
    print(f"Task required VRAM: {too_large_task.required_vram_mb} MiB")

    scheduled_task = scheduler.schedule_once()

    print(f"Scheduled task: {scheduled_task}")
    print(f"Task status: {too_large_task.status.value}")

    task_queue.put(None)
    worker_process.join(timeout=10)
if __name__ == "__main__":
    main()
