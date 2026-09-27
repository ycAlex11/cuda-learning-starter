from dataclasses import dataclass

@dataclass

class SchedulerMetrics:
    submitted_tasks:int = 0
    scheduled_attempts: int = 0
    completed_tasks: int = 0
    requeued_tasks: int = 0
    rejected_completions: int = 0
    duplicate_completions: int = 0 
    failed_tasks:int = 0 