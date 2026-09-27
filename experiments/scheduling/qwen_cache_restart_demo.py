from multiprocessing import Process,Queue
from scheduler.core import Scheduler
from scheduler.models import Worker,Task
from scheduler.workers.qwen_process_worker import qwen_worker_loop

def main():
    scheduler = Scheduler(heartbeat_timeout_seconds=1.0)

    worker = Worker(worker_id="gpu-0",total_vram_mb=4096,used_vram_mb=1024)

    scheduler.register_worker(worker)

    task_queue = Queue()
    result_queue = Queue()
    heartbeat_queue = Queue()

    first_process = Process(target= qwen_worker_loop,args =(task_queue, result_queue, heartbeat_queue), )
    first_process.start()
    try:
        # 等待模型加载完成后的第一条心跳
        msg = heartbeat_queue.get(timeout=60)

        if msg.get("type") != "heartbeat":
            raise RuntimeError("Expected a heartbeat message")

        scheduler.heartbeat(
            msg["worker_id"],
            reported_free_vram_mb=msg["reported_free_vram_mb"],
            reported_total_vram_mb=msg["reported_total_vram_mb"],
            reported_prefix_cache_bytes=msg["reported_prefix_cache_bytes"],
        )

        print(
            f"Initial prefix cache: "
            f"{worker.reported_prefix_cache_bytes} bytes"
        )

        task = Task(task_id = "sth1",operation="qwen_prefill",required_vram_mb=1,payload={"prompt": "Explain KV Cache in one short sentence.","max_new_tokens": 0,})
        scheduler.submit_task(task)
        scheduled = scheduler.schedule_once()
        if scheduled is None:
            raise RuntimeError("Task was not scheduled")
        task_queue.put(scheduled)
        result = result_queue.get(timeout=60)
        if not result["ok"]:
            raise RuntimeError(result["error"])
        accepted = scheduler.complete_task(task_id= result["task_id"],worker_id=result["worker_id"],attempt=result["attempt"],result = result["result"])
        
        print(f"Completion accepted: {accepted}")
        print(f"Worker result: {result['result']}")

        for _ in range(5):
            msg = heartbeat_queue.get(timeout=5)

            scheduler.heartbeat(msg["worker_id"],reported_free_vram_mb=msg["reported_free_vram_mb"],
                reported_total_vram_mb=msg["reported_total_vram_mb"],
                reported_prefix_cache_bytes=msg["reported_prefix_cache_bytes"],)
            
            if worker.reported_prefix_cache_bytes > 0:
                break
        else:
            raise RuntimeError("没有收到非零缓存占用的心跳")

        print(
            f"Scheduler recorded cache: "
            f"{worker.reported_prefix_cache_bytes} bytes"
        )
        print(f"Logical free budget: {worker.free_vram_mb} MiB")


    finally:
        # 本阶段测试结束，关闭进程
        task_queue.put(None)
        first_process.join(timeout=10)


    if first_process.is_alive():
        raise RuntimeError("旧进程尚未退出，不能继续测试")
    print("second")
    new_task_queue = Queue()
    new_result_queue = Queue()
    new_heartbeat_queue = Queue()

    second_process = Process(target=qwen_worker_loop, args=(new_task_queue, new_result_queue, new_heartbeat_queue),)
    second_process.start()

    try:
        msg = new_heartbeat_queue.get(timeout=60)
        if msg.get("type") != "heartbeat":
            raise RuntimeError("Expected a heartbeat msg")
        scheduler.heartbeat(
            msg["worker_id"],
            reported_free_vram_mb=msg["reported_free_vram_mb"],
            reported_total_vram_mb=msg["reported_total_vram_mb"],
            reported_prefix_cache_bytes=msg["reported_prefix_cache_bytes"],
        )
        print("\nAfter worker restart")
        print(f"Recorded cache: {worker.reported_prefix_cache_bytes} bytes")
        print(f"Logical free budget: {worker.free_vram_mb} MiB")

        assert worker.reported_prefix_cache_bytes == 0
        assert worker.free_vram_mb == 3072
        second_task = Task(
            task_id="cache-after-restart",
            operation="qwen_prefill",
            required_vram_mb=1,
            payload=task.payload.copy(),
        )

        scheduler.submit_task(second_task)
        scheduled = scheduler.schedule_once()

        if scheduled is None:
            raise RuntimeError("新进程的任务未能分配")

        new_task_queue.put(scheduled)

        result = new_result_queue.get(timeout=60)
        if not result["ok"]:
            raise RuntimeError(result["error"])

        accepted = scheduler.complete_task(
            task_id=result["task_id"],
            worker_id=result["worker_id"],
            attempt=result["attempt"],
            result=result["result"],
        )

        assert accepted
        print(f"New worker result: {result['result']}")
    finally:
        new_task_queue.put(None)
        second_process.join(timeout=10)


if __name__== "__main__":
    main()
