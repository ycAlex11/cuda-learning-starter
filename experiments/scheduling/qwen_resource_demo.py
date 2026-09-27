from scheduler.qwen_resources import estimate_qwen_kv_cache_mib
from scheduler.core import Scheduler
from scheduler.models import Task, Worker

def main():
    prompt_tokens = 289
    max_new_tokens = 16
    required_vram_mb = estimate_qwen_kv_cache_mib(
        prompt_tokens,
        max_new_tokens,
    )

    print(f"Prompt tokens: {prompt_tokens}")
    print(f"Max new tokens: {max_new_tokens}")
    print(f"Estimated KV Cache reservation: {required_vram_mb} MiB")

    scheduler = Scheduler()

    worker = Worker(worker_id="gpu-0",total_vram_mb=4096,used_vram_mb=1024)

    scheduler.register_worker(worker)
    task = Task(task_id="qwen-001",operation="qwen_genrate",required_vram_mb= required_vram_mb,payload = {"prompt_tokens": prompt_tokens,"max_new_tokens": max_new_tokens,})
    scheduler.submit_task(task)
    scheduled_task = scheduler.schedule_once()
    if scheduled_task is None:
        raise RuntimeError("Qwen task was not scheduled")

    print(f"Scheduled task: {scheduled_task.task_id}")
    print(f"Task required KV Cache: {scheduled_task.required_vram_mb} MiB")
    print(f"Worker used VRAM: {worker.used_vram_mb} MiB")
    print(f"Worker free VRAM: {worker.free_vram_mb} MiB")


if __name__ == "__main__":
    main()
