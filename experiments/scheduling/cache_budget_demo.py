from scheduler.core import Scheduler
from scheduler.models import Task,Worker,TaskStatus

def main():
    scheduler = Scheduler()
    worker = Worker(worker_id="gpu-0",total_vram_mb=4,used_vram_mb=1)
    scheduler.register_worker(worker)

    task = Task(task_id = "sth-1",operation="test",required_vram_mb=3)
    scheduler.submit_task(task)
    scheduler.heartbeat(worker.worker_id,reported_prefix_cache_bytes=1024 * 1024)
    print("With cache")
    print(f"Free budget: {worker.free_vram_mb} MiB")

    scheduled = scheduler.schedule_once()

    print(f"Task assigned: {scheduled is not None}")
    print(f"Task status: {task.status.value}")
    print(f"Free budget after scheduling: {worker.free_vram_mb} MiB")

    assert scheduled is None
    assert task.status == TaskStatus.PENDING
    assert worker.free_vram_mb == 2

    

    scheduler.heartbeat(worker.worker_id,reported_prefix_cache_bytes= 0)
    print("\nAfter cache cleared")
    print(f"Free budget before scheduling: {worker.free_vram_mb} MiB")

    scheduled= scheduler.schedule_once()

    print(f"Task assigned: {scheduled is not None}")
    print(f"Task status: {task.status.value}")
    print(f"Free budget after scheduling: {worker.free_vram_mb} MiB")

    assert scheduled is task
    assert task.status == TaskStatus.RUNNING
    assert worker.free_vram_mb == 0

    

    accepted = scheduler.complete_task(task_id=task.task_id,worker_id=worker.worker_id,attempt=task.attempts,result ="test completed")

    print("\nAfter task completed")
    print(f"Completion accepted: {accepted}")
    print(f"Task status: {task.status.value}")
    print(f"Free budget: {worker.free_vram_mb} MiB")

    assert accepted is True
    assert task.status == TaskStatus.SUCCEEDED
    assert worker.free_vram_mb == 3

    print("\nCache budget test: PASS")

if __name__ == "__main__":
    main()
