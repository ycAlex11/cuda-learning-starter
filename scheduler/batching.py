from dataclasses import dataclass
from scheduler.models import Task
from time import monotonic

@dataclass
class TaskBatch:
    batch_id : str 
    tasks : list[Task]

    @property
    def total_required_vram_mb(self) -> int:
        return sum(task.required_vram_mb for task in self.tasks)
    

class MicroBatcher:
    def __init__(
        self,
        max_batch_size: int,
        max_batch_vram_mb: int,
        max_batch_wait_ms:int =50
    ) -> None:
        self.max_batch_size = max_batch_size
        self.max_batch_vram_mb = max_batch_vram_mb
        self.max_batch_wait_ms = max_batch_wait_ms

    def from_batch(self,batch_id:str,pending_tasks:list[Task])->TaskBatch|None:
        selected_tasks:list[Task] = []
        reserved_vram_mb = 0

        for task in pending_tasks:
             if task.status.value != "pending":
                 continue 
             if len(selected_tasks)>= self.max_batch_size:
                 break 
             next_reserved_vram_mb = (
                 reserved_vram_mb+task.required_vram_mb
             )

             if next_reserved_vram_mb>self.max_batch_vram_mb:
                 break
             
             selected_tasks.append(task)
             reserved_vram_mb = next_reserved_vram_mb

        if not selected_tasks:
            return None 
        return TaskBatch(batch_id=batch_id,tasks=selected_tasks)
    
    def should_form_batch(self,pending_tasks:list[Task])->bool:
        pending: list[Task] = []

        for task in pending_tasks:
            if task.status.value == "pending":
                pending.append(task)

        if not pending:
            return False 
        if len(pending) >= self.max_batch_size:
            return True 
        submitted_times = []
        for task in pending:
            if task.submitted_at is not None:
                submitted_times.append(task.submitted_at)
        if not submitted_times:
            return False
        
        oldest_submitted_at = min(submitted_times)
        waited_ms = (monotonic() - oldest_submitted_at) * 1000
        return waited_ms >= self.max_batch_wait_ms