from multiprocessing import Process,Queue

from scheduler.core import Scheduler
from scheduler.models import Task,Worker
from scheduler.workers.process_worker import worker_loop

def main()->None:
    task_queue = Queue()
    result_queue = Queue()
    
    woreker_process = Process(target= worker_loop,args=(task_queue,result_queue))
    woreker_process.start()

    scheduler = Scheduler()
    scheduler.register_worker(Worker(worker_id="gpu-0",total_vram_mb=4096))

    task = Task(task_id="process-001",operation= "sth",required_vram_mb=64, payload={
        "num_elements": 4 * 1024 * 1024,
        "scale": 1.5,
        "bias": 2.0,
    },)

    scheduler.submit_task(task)

    scheduled_task = scheduler.schedule_once()

    if scheduled_task is None:
        raise RuntimeError("task was not scheduled")
    print(
        f"Scheduler assigned {scheduled_task.task_id} "
        f"to {scheduled_task.assigned_worker_id}")
    
    task_queue.put(scheduled_task)
    worker_result = result_queue.get(timeout=5)
    print(f"Scheduler received: {worker_result}")

    if worker_result["ok"]:

        accepted = scheduler.complete_task(
            task_id=scheduled_task.task_id,
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


    print(f"Completion accepted: {accepted}")
    print(f"Final task status: {task.status.value}")
    print(f"Task error: {task.error}")
    print(f"Worker free VRAM: {scheduler.workers['gpu-0'].free_vram_mb} MiB")

    task_queue.put(None)
    woreker_process.join(timeout=5)


if __name__ == "__main__":
    main()   
