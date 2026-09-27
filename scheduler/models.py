from dataclasses import dataclass, field
from enum import Enum
from time import monotonic


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


@dataclass
class Task:
    task_id: str
    operation: str
    required_vram_mb: int
    payload: dict[str, object] = field(default_factory=dict)

    status: TaskStatus = TaskStatus.PENDING
    attempts: int = 0
    assigned_worker_id: str | None = None
    result: str | None = None
    error: str | None = None
    submitted_at: float|None = None 
    completed_at: float|None = None



@dataclass
class Worker:
    worker_id: str
    total_vram_mb: int

    used_vram_mb: int = 0
    last_heartbeat: float = field(default_factory=monotonic)

    reported_free_vram_mb: int | None = None
    reported_total_vram_mb: int | None = None
    reported_prefix_cache_bytes: int | None = None

    @property
    def free_vram_mb(self) -> int:
        #return self.total_vram_mb - self.used_vram_mb
        cache_bytes = self.reported_prefix_cache_bytes
        if cache_bytes is None:
            cache_bytes = 0

        # 字节转换成 MiB，向上取整
        mib = 1024 * 1024
        cache_mib = (cache_bytes + mib - 1) // mib

        return self.total_vram_mb - self.used_vram_mb - cache_mib