from scheduler.core import Scheduler
from scheduler.models import Task, Worker


def main():
    scheduler = Scheduler()

    workers = [
        Worker(worker_id="small", total_vram_mb=1024),
        Worker(worker_id="large", total_vram_mb=2048),
    ]

    tasks = [
        Task(task_id="001", operation="scale_add", required_vram_mb=1024),
        Task(task_id="002", operation="scale_add", required_vram_mb=1024),
        Task(task_id="003", operation="scale_add", required_vram_mb=1024),
        Task(task_id="004", operation="scale_add", required_vram_mb=1024),
    ]

    for worker in workers:
        scheduler.register_worker(worker)
    for task in tasks:
        scheduler.submit_task(task)

    while True:
        scheduled_task = scheduler.schedule_once()

        if scheduled_task is None:
            break
        print(
            f"Scheduled task {scheduled_task.task_id} "
            f"to worker {scheduled_task.assigned_worker_id}"
        )
    print("\nFinal worker state")
    for worker in scheduler.workers.values():
        print(
            f"{worker.worker_id}: "
            f"used={worker.used_vram_mb} MiB, "
            f"free={worker.free_vram_mb} MiB"
        )

    print("\nFinal task state")

    for task in scheduler.tasks.values():
        print(
            f"{task.task_id}: "
            f"status={task.status.value}, "
            f"worker={task.assigned_worker_id}"
        )
    
        print("\nComplete task 001 and schedule again")

    scheduler.complete_task(
        task_id="001",
        worker_id="small",
        attempt=1,
        result="completed",
    )

    scheduled_task = scheduler.schedule_once()

    if scheduled_task is not None:
        print(
            f"Scheduled task {scheduled_task.task_id} "
            f"to worker {scheduled_task.assigned_worker_id}"
        )

    print(
        f"small: used={scheduler.workers['small'].used_vram_mb} MiB, "
        f"free={scheduler.workers['small'].free_vram_mb} MiB"
    )
    print(
        f"001: status={scheduler.tasks['001'].status.value}, "
        f"result={scheduler.tasks['001'].result}"
    )
    print(
        f"004: status={scheduler.tasks['004'].status.value}, "
        f"worker={scheduler.tasks['004'].assigned_worker_id}"
    )

    print("\nHeartbeat check")

    scheduler.heartbeat("small")

    print(
        "small alive after heartbeat:",
        scheduler.is_worker_alive(
            worker_id="small",
            timeout_seconds=5.0,
        ),
    )

    scheduler.workers["small"].last_heartbeat -= 10.0

    print(
        "small alive after simulated 10-second silence:",
        scheduler.is_worker_alive(
            worker_id="small",
            timeout_seconds=5.0,
        ),
    )
    print("\nRequeue tasks from lost workers")

    requeued_tasks = scheduler.requeue_tasks_from_lost_workers()

    for task in requeued_tasks:
        print(f"Requeued task: {task.task_id}")

    print(
        f"004: status={scheduler.tasks['004'].status.value}, "
        f"worker={scheduler.tasks['004'].assigned_worker_id}, "
        f"error={scheduler.tasks['004'].error}"
    )
    print(
        f"small: used={scheduler.workers['small'].used_vram_mb} MiB, "
        f"free={scheduler.workers['small'].free_vram_mb} MiB"
    )

    retry_now = scheduler.schedule_once()
    print(f"Can 004 retry immediately? {retry_now is not None}")

    scheduler.complete_task(
        task_id="002",
        worker_id="large",
        attempt=1,
        result="completed",
    )

    retried_task = scheduler.schedule_once()

    print(
        f"Retried task: {retried_task.task_id}, "
        f"worker={retried_task.assigned_worker_id}, "
        f"attempts={retried_task.attempts}"
    )
    print("\nStale completion and idempotency check")

    stale_accepted = scheduler.complete_task(
        task_id="004",
        worker_id="small",
        attempt=1,
        result="late result from small",
    )

    print(f"Stale completion accepted: {stale_accepted}")
    print(f"004 status after stale report: {scheduler.tasks['004'].status.value}")
    print(f"large used VRAM after stale report: {scheduler.workers['large'].used_vram_mb} MiB")

    valid_accepted = scheduler.complete_task(
        task_id="004",
        worker_id="large",
        attempt=2,
        result="completed by large",
    )

    print(f"Valid completion accepted: {valid_accepted}")
    print(f"large used VRAM after valid completion: {scheduler.workers['large'].used_vram_mb} MiB")

    duplicate_accepted = scheduler.complete_task(
        task_id="004",
        worker_id="large",
        attempt=2,
        result="duplicate completion",
    )

    print(f"Duplicate completion accepted: {duplicate_accepted}")
    print(f"large used VRAM after duplicate completion: {scheduler.workers['large'].used_vram_mb} MiB")
    
    metrics = scheduler.metrics

    print("\nScheduler metrics")
    print(f"Submitted tasks: {metrics.submitted_tasks}")
    print(f"Scheduled attempts: {metrics.scheduled_attempts}")
    print(f"Completed tasks: {metrics.completed_tasks}")
    print(f"Requeued tasks: {metrics.requeued_tasks}")
    print(f"Rejected completions: {metrics.rejected_completions}")
    print(f"Duplicate completions: {metrics.duplicate_completions}")
if __name__ == "__main__":
    main()
