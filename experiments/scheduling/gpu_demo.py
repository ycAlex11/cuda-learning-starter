from scheduler.core import Scheduler
from scheduler.workers.gpu_worker import execute_gpu_task
from scheduler.models import Task,Worker

def main():
    sr = Scheduler()

    worker = Worker(worker_id="gpu-0",total_vram_mb= 4096)
    task = Task(task_id="gpu-001",operation= "scale_add",required_vram_mb=64,payload={"num_elements":4*1024*1024,"scale":1.5,"bias":2.0})
    sr.register_worker(worker)
    sr.submit_task(task)

    scheduled_task = sr.schedule_once()

    if scheduled_task is None:
        raise RuntimeError("task was not scheduled")

    print(
        f"Scheduled {scheduled_task.task_id} "
        f"to {scheduled_task.assigned_worker_id}"
    )
    result = execute_gpu_task(scheduled_task)
    accepted = sr.complete_task(
        task_id=scheduled_task.task_id,
        worker_id=scheduled_task.assigned_worker_id,
        attempt=scheduled_task.attempts,
        result=result,
    )
    print(f"Worker result: {result}")
    print(f"Completion accepted: {accepted}")
    print(f"Task status: {scheduled_task.status.value}")
    print(f"Worker free VRAM: {worker.free_vram_mb} MiB")
    metrics = sr.metrics

    print("\nScheduler metrics")
    print(f"Submitted tasks: {metrics.submitted_tasks}")
    print(f"Scheduled attempts: {metrics.scheduled_attempts}")
    print(f"Completed tasks: {metrics.completed_tasks}")
    print(f"Requeued tasks: {metrics.requeued_tasks}")
    print(f"Rejected completions: {metrics.rejected_completions}")
    print(f"Duplicate completions: {metrics.duplicate_completions}")
if __name__ == "__main__":
    main()
