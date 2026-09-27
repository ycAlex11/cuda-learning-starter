from scheduler.core import Scheduler
from scheduler.models import Task,Worker
from scheduler.qwen_resources import estimate_qwen_kv_cache_mib

def main():
    scheduler = Scheduler()

    worker = Worker(worker_id="gpu-0",total_vram_mb=4096,used_vram_mb=1024)

    scheduler.register_worker(worker)

    prompt_tokens = 24_000
    max_new_tokens = 4096

    required_vram_mb= estimate_qwen_kv_cache_mib(prompt_tokens,max_new_tokens)

    print(f"KV Cache per long request: {required_vram_mb} MiB")

    for index in range(1,11):
        task = Task(task_id=f"qwen-{index:03d}",operation="qwen_generate",required_vram_mb=required_vram_mb,payload={"prompt_tokens": prompt_tokens,"max_new_tokens": max_new_tokens,})
        scheduler.submit_task(task)

    scount = 0
    while True:
        stask = scheduler.schedule_once()

        if stask is None:
            break
        
        scount+=1
        print()
        print(
            f"Scheduled {stask.task_id}: "
            f"free VRAM={worker.free_vram_mb} MiB"
        )
    print(f"\nScheduled tasks: {scount}")
    print(f"Worker free VRAM: {worker.free_vram_mb} MiB")
    for task in scheduler.tasks.values():
        print(f"{task.task_id}: {task.status.value}")

    ftask = scheduler.tasks["qwen-001"]
    scheduler.complete_task(task_id=ftask.task_id,worker_id=worker.worker_id,attempt=ftask.attempts,result="simulated Qwen generation completed",)
    scheduled_task = scheduler.schedule_once()

    print("\nAfter qwen-001 completes")

    if scheduled_task is not None:
        print(f"Scheduled pending task: {scheduled_task.task_id}")

    print(f"Worker free VRAM: {worker.free_vram_mb} MiB")
    print(f"qwen-001: {scheduler.tasks['qwen-001'].status.value}")
    print(f"qwen-010: {scheduler.tasks['qwen-010'].status.value}")


if __name__ == "__main__":
    main()
