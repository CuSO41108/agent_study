from dataclasses import dataclass


@dataclass
class TaskState:
    id: str
    status: str
    step: int
    stop_reason: str | None = None
