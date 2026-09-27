from scheduler.batching import MicroBatcher

from scheduler.core import Scheduler
from scheduler.models import Task,Worker

def main()->None:
    scheduler =Scheduler()
    scheduler.register_worker(Worker(worker_id="gpu-0",total_vram_mb=512,used_vram_mb=0,))
    pending_tasks = [Task(task_id = "request-001",operation="qwen",required_vram_mb=64),
                     Task(task_id = "request-002",operation="qwen",required_vram_mb=128),
                     Task(task_id = "request-003",operation="qwen",required_vram_mb=96),
                     ]
    

    for task in pending_tasks:
        scheduler.submit_task(task)

    batcher = MicroBatcher(max_batch_size=3,max_batch_vram_mb=256)

    batch = batcher.from_batch(
        batch_id="batch-001",
        pending_tasks=list(scheduler.tasks.values()),
    )

    if batch is None:
        print("No batch formed")
        return 
    assigned_worker = scheduler.schedule_batch(batch)

    print(f"Batch: {batch.batch_id}")
    print(f"Tasks: {[task.task_id for task in batch.tasks]}")
    print(f"Reserved VRAM: {batch.total_required_vram_mb} MiB")

    if assigned_worker is None:
        print("Batch was not scheduled.")
    else:
        print(f"Batch assigned to: {assigned_worker.worker_id}")
        print(f"Worker used VRAM: {assigned_worker.used_vram_mb} MiB")
        print(f"Worker free VRAM: {assigned_worker.free_vram_mb} MiB")
        results = {
            "request-001": "batch result for request-001",
            "request-002": "batch result for request-002",
        }

        completed = scheduler.complete_batch(batch,assigned_worker.worker_id,results)
        print(f"Batch completion accepted: {completed}")
        print(f"Worker used VRAM after completion: {assigned_worker.used_vram_mb} MiB")
        print(f"Worker free VRAM after completion: {assigned_worker.free_vram_mb} MiB")

    print("\nTask states")
    for task in scheduler.tasks.values():
        print(
            f"{task.task_id}: status={task.status.value}, "
            f"worker={task.assigned_worker_id}, "
            f"attempts={task.attempts}"
        )
        print()
    next_batch = batcher.from_batch(batch_id="batch-002",pending_tasks=list(scheduler.tasks.values()))

    if next_batch is None:
        print("\nNo second batch formed")
        return 

    next_worker = scheduler.schedule_batch(next_batch)
    if next_worker is None:
        print("\nSecond batch was not scheduled.")
        return
    print(f"\nSecond batch: {next_batch.batch_id}")
    print(f"Second batch tasks: {[task.task_id for task in next_batch.tasks]}")

    second_completed = scheduler.complete_batch(
        next_batch,
        next_worker.worker_id,
        {
            "request-003": "batch result for request-003",
        },
    )
    print(f"Second batch completion accepted: {second_completed}")
    print(f"Worker free VRAM after second batch: {next_worker.free_vram_mb} MiB")
    
    print("\nFinal task states")
    for task in scheduler.tasks.values():
        print(f"{task.task_id}: {task.status.value}")


if __name__ == "__main__":
    main()
