from multiprocessing import Process,Queue
from queue import Empty

from scheduler.core import Scheduler
from scheduler.models import Task,Worker 

def worke_exists_without_result(task_queue:Queue,result_queue:Queue,)->None:
    task = task_queue.get()

    print(
        f"Worker received task: {task.task_id}, "
        "then exits without sending a result."
    )
    raise SystemExit(1)

def main()->None:
    scheduler = Scheduler(heartbeat_timeout_seconds=0.2)

    worker = Worker(worker_id="gpu-0",total_vram_mb=4096)
    scheduler.register_worker(worker)

    task = Task(task_id="crash-001",operation="scale_add",required_vram_mb=64,)
    scheduler.submit_task(task)
    task_queue= Queue()
    result_queue = Queue()

    worker_process = Process(target= worke_exists_without_result,args=(task_queue,result_queue))

    worker_process.start()
    scheduled_task = scheduler.schedule_once()
    if scheduled_task is None:
        raise RuntimeError("task was not scheduled")
    print(f"Scheduler assigned {scheduled_task.task_id} to gpu-0")
    task_queue.put(scheduled_task)

    try:
        worker_result = result_queue.get(timeout=0.5)
        print(f"Unexpected worker result: {worker_result}")
    except Empty:
        print("No completion result before timeout.")
    
    worker_process.join(timeout=1)
    print(f"Worker process exit code: {worker_process.exitcode}")

    requeued_task = scheduler.requeue_tasks_from_lost_workers()
    for sth in requeued_task:
        print(f"Requeued task: {sth.task_id}")
    
    print(f"Final task status: {task.status.value}")
    print(f"Task error: {task.error}")
    print(f"Worker free VRAM: {worker.free_vram_mb} MiB")
    

if __name__ == "__main__":
    main()
