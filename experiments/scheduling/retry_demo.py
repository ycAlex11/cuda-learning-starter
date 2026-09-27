from multiprocessing import Process,Queue
from queue import Empty

from scheduler.core import Scheduler
from scheduler.models import Task,Worker
from scheduler.workers.process_worker import worker_loop

def work_crashes(task_queue:Queue) ->None:
    task = task_queue.get()
    print(
        f"First worker received {task.task_id}, "
        "then exits without reporting a result."
    )

    raise SystemExit(1)

def main()->None:
    scheduler = Scheduler(heartbeat_timeout_seconds=0.2)

    worker = Worker(worker_id="gpu-0",total_vram_mb=4096)

    scheduler.register_worker(worker)

    task = Task(task_id="retry-001",operation="scale_add",required_vram_mb=64,payload={"num_elements":4*1024*1024,"scale":1.5,"bias":2.0})

    scheduler.submit_task(task)
    crash_task_queue = Queue()
    crash_result_queue = Queue()

    crash_process =Process(target=work_crashes,args = (crash_task_queue,))

    crash_process.start()
    first_attempt = scheduler.schedule_once()
    if first_attempt is None:
        raise RuntimeError("first attempt was not scheduled")
    
    print(f"Attempt {first_attempt.attempts}: assigned to gpu-0")
    first_attempt_number = first_attempt.attempts

    crash_task_queue.put(first_attempt)

    try:
        crash_result_queue.get(timeout=0.5)
    except Empty:
        print("Attempt 1 timed out without a completion result.")
    
    crash_process.join(timeout=5)

    if crash_process.exitcode is None:
        raise RuntimeError("first worker did not exit within 5 seconds")
    print(f"First worker exit code: {crash_process.exitcode}")
    requeued_tasks = scheduler.requeue_tasks_from_lost_workers()
    print(f"Requeued tasks: {[task.task_id for task in requeued_tasks]}")
    print(f"Task status after requeue: {task.status.value}")
    print(f"Worker free VRAM after requeue: {worker.free_vram_mb} MiB")

    scheduler.heartbeat("gpu-0")

    retry_task_queue = Queue()

    retry_result_queue =Queue()
    retry_process = Process(
        target=worker_loop,
        args=(retry_task_queue, retry_result_queue),
    )
    retry_process.start()
    second_attempt = scheduler.schedule_once()
    if second_attempt is None:
        raise RuntimeError("retry was not scheduled")
    print(f"Attempt {second_attempt.attempts}: assigned to gpu-0")
    retry_task_queue.put(second_attempt)

    worker_result = retry_result_queue.get(timeout=30)
    print(f"Retry worker result: {worker_result}")
    if worker_result["ok"]:
        accepted = scheduler.complete_task(
            task_id=worker_result["task_id"],
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

    print(f"Retry completion accepted: {accepted}")
    
    stale_accepted = scheduler.complete_task(
    task_id=task.task_id,
    worker_id="gpu-0",
    attempt=first_attempt_number,
    result="late result from attempt 1",
)

    print(f"Stale attempt 1 completion accepted: {stale_accepted}")
    print(f"Task status after stale completion: {task.status.value}")
    print(f"Worker free VRAM after stale completion: {worker.free_vram_mb} MiB")


    print(f"Final task status: {task.status.value}")
    print(f"Final task attempts: {task.attempts}")
    print(f"Final worker free VRAM: {worker.free_vram_mb} MiB")

    retry_task_queue.put(None)
    retry_process.join(timeout=5)
    print("\nScheduler metrics")
    print(f"Rejected completions: {scheduler.metrics.rejected_completions}")
    print(f"Submitted tasks: {scheduler.metrics.submitted_tasks}")
    print(f"Scheduled attempts: {scheduler.metrics.scheduled_attempts}")
    print(f"Completed tasks: {scheduler.metrics.completed_tasks}")
    print(f"Requeued tasks: {scheduler.metrics.requeued_tasks}")

if __name__ == "__main__":
    main()





    
