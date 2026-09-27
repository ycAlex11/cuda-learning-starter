from multiprocessing import Queue
from scheduler.models import Task
from scheduler.workers.gpu_worker import execute_gpu_task

def worker_loop(task_queue:Queue,result_queue:Queue)->None:

    while True:
        task = task_queue.get()

        if task is None:
            print("Worker received shutdown signal.")
            return
        if not isinstance(task, Task):
            result_queue.put(
                {
                    "worker_id": "gpu-0",
                    "ok": False,
                    "error": "worker received an invalid task",
                }
            )
            continue
        print(f"Worker received task: {task.task_id}")
        try:
            result = execute_gpu_task(task)
            result_queue.put(
                {
                    "task_id": task.task_id,
                    "worker_id": "gpu-0",
                    "attempt":task.attempts,
                    "ok":True,
                    "result": result,
                })
        except Exception as error:
            result_queue.put(
                {
                    "task_id": task.task_id,
                    "worker_id": "gpu-0",
                    "attempt": task.attempts,
                    "ok": False,
                    "error": f"{type(error).__name__}: {error}",
                }
            )
