from scheduler.models import Task, Worker,TaskStatus

from time import monotonic
from scheduler.metrics import SchedulerMetrics


from scheduler.batching import TaskBatch

class Scheduler:
    def __init__(self, heartbeat_timeout_seconds: float = 5.0):
        self.workers: dict[str, Worker] = {}
        self.tasks: dict[str, Task] = {}
        self.heartbeat_timeout_seconds = heartbeat_timeout_seconds
        self.metrics = SchedulerMetrics()

    def register_worker(self, worker: Worker) -> None:
        if worker.worker_id in self.workers:
            raise ValueError(f"worker already exists: {worker.worker_id}")

        self.workers[worker.worker_id] = worker
        


    def submit_task(self, task: Task) -> None:
        if task.task_id in self.tasks:
            raise ValueError(f"task already exists: {task.task_id}")

        self.tasks[task.task_id] = task
        task.submitted_at = monotonic()
        self.metrics.submitted_tasks += 1

    def schedule_once(self)->Task|None:
        for task in self.tasks.values():
            if task.status!= TaskStatus.PENDING:
                continue
            for worker in self.workers.values():
                if not self.is_worker_alive(
                    worker.worker_id,
                    self.heartbeat_timeout_seconds,
                ):
                    continue
                if worker.free_vram_mb<task.required_vram_mb:
                    continue
                if (worker.reported_free_vram_mb is not None and worker.reported_free_vram_mb < task.required_vram_mb):
                    continue
                
                task.status = TaskStatus.RUNNING
                task.assigned_worker_id = worker.worker_id
                task.attempts+=1

                worker.used_vram_mb+=task.required_vram_mb
                self.metrics.scheduled_attempts+=1
                return task
        return None


    def schedule_batch(self, batch: TaskBatch) -> Worker | None:
        if not batch.tasks:
            return None 
        
        for task in batch.tasks:
            if self.tasks.get(task.task_id) is not task:
                raise ValueError(f"unknown task in batch: {task.task_id}")

            if task.status != TaskStatus.PENDING:
                return None
        required_vram_mb = batch.total_required_vram_mb
        for worker in self.workers.values():
            if not self.is_worker_alive(worker.worker_id,self.heartbeat_timeout_seconds):
                continue

            if worker.free_vram_mb<required_vram_mb:
                continue

            if(worker.reported_free_vram_mb is not None and worker.reported_free_vram_mb<required_vram_mb):
                continue
            
            for task in batch.tasks:
                task.status = TaskStatus.RUNNING
                task.assigned_worker_id =worker.worker_id
                task.attempts+=1
            
            worker.used_vram_mb+= required_vram_mb
            self.metrics.scheduled_attempts+=len(batch.tasks)

            return worker
        
        return None 



    def complete_task(self, task_id:str,worker_id:str,attempt:int,result:str)->bool:
        task = self.tasks[task_id]
        if task.status == TaskStatus.SUCCEEDED:
            is_duplicate = (
                task.assigned_worker_id == worker_id
                and task.attempts == attempt
            )
            if is_duplicate:
                    self.metrics.duplicate_completions += 1
            else:
                    self.metrics.rejected_completions += 1

            return is_duplicate
        if task.status != TaskStatus.RUNNING:
            self.metrics.rejected_completions += 1
            return False
        if task.assigned_worker_id != worker_id:
            self.metrics.rejected_completions += 1
            return False

        if task.attempts != attempt:
            self.metrics.rejected_completions += 1
            return False
        worker = self.workers[worker_id]

        worker.used_vram_mb -= task.required_vram_mb

        task.status = TaskStatus.SUCCEEDED
        task.result = result
        task.error = None
        self.metrics.completed_tasks+=1
        return True
    

    def complete_batch(self,batch:TaskBatch,worker_id:str,result:dict[str,str])->bool:
        worker = self.workers.get(worker_id)
        if worker is None:
            return False 
        
        for task in batch.tasks:
            if task.status!= TaskStatus.RUNNING:
                return False 
            if  task.assigned_worker_id != worker_id:
                return False 
            if task.task_id not in result:
                return False 
        completed_at = monotonic()
        worker.used_vram_mb -= batch.total_required_vram_mb

        for task in batch.tasks:
            task.status = TaskStatus.SUCCEEDED
            task.completed_at = completed_at
            task.result = result [task.task_id]
            task.error = None 
        self.metrics.completed_tasks+=len(batch.tasks)
        return True
     
    def fail_batch(self,batch:TaskBatch,work_id:str,error:str)->bool:
        worker = self.workers.get(work_id)

        if worker is None:
            return False 
        for task in batch.tasks:
            if task.status!= TaskStatus.RUNNING:
                return False 
            if task.assigned_worker_id != work_id:
                return False 
        
        worker.used_vram_mb -= batch.total_required_vram_mb

        for task in batch.tasks:
            task.status = TaskStatus.FAILED
            task.result = None 
            task.error = error 
        self.metrics.failed_tasks += len(batch.tasks)
        return True 


    def heartbeat(self, worker_id: str,reported_free_vram_mb: int | None = None,reported_total_vram_mb: int | None = None,reported_prefix_cache_bytes: int | None = None,) -> None:
        if worker_id not in self.workers:
            raise ValueError(f"unknown worker: {worker_id}")
        worker = self.workers[worker_id]
        worker.last_heartbeat = monotonic()
        if reported_free_vram_mb is not None:
            worker.reported_free_vram_mb = reported_free_vram_mb
        if reported_total_vram_mb is not None:
            worker.reported_total_vram_mb = reported_total_vram_mb
            
        if reported_prefix_cache_bytes is not None:
            worker.reported_prefix_cache_bytes = (
                reported_prefix_cache_bytes
            )   

    def is_worker_alive(
        self,
        worker_id: str,
        timeout_seconds: float,
    ) -> bool:
        if worker_id not in self.workers:
            raise ValueError(f"unknown worker: {worker_id}")

        worker = self.workers[worker_id]

        return monotonic() - worker.last_heartbeat <= timeout_seconds

    def requeue_tasks_from_lost_workers(self) -> list[Task]:
        requeued_tasks: list[Task] = []

        for worker in self.workers.values():
            if self.is_worker_alive(worker.worker_id,self.heartbeat_timeout_seconds):
                continue
            for task in self.tasks.values():
                if task.status != TaskStatus.RUNNING:
                    continue
                
                if task.assigned_worker_id != worker.worker_id:
                    continue

                worker.used_vram_mb -= task.required_vram_mb

                task.status = TaskStatus.PENDING
                task.assigned_worker_id = None
                task.error = "worker lost; task requeued"

                requeued_tasks.append(task)
                self.metrics.requeued_tasks+=1
        
        return requeued_tasks
    
    def fail_task(self, task_id:str,worker_id:str,attempt:int,error:str)->bool:
        
        task = self.tasks.get(task_id)
        if task is None:
            return False
        
        if task.status != TaskStatus.RUNNING:
            return False
        if task.assigned_worker_id != worker_id:
            return False 
        if task.attempts!=attempt:
            return False
        worker = self.workers[worker_id]
        worker.used_vram_mb -= task.required_vram_mb

        task.status = TaskStatus.FAILED
        task.result = None 
        task.error = error

        return True

        