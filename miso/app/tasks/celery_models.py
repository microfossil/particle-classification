from dataclasses import dataclass
from typing import List, Any, Dict, Optional


@dataclass
class CeleryTaskProgress:
    message: str
    count: float
    total: float
    id: Optional[str] = None
    state: Optional[str] = None


@dataclass
class CeleryTask:
    name: str
    description: str
    task_id: str
    result_id: str


@dataclass
class CeleryDeliveryInfo:
    exchange: Optional[str]
    routing_key: str
    priority: int
    redelivered: bool


@dataclass
class CeleryTaskStatus:
    id: str
    name: str
    args: Optional[List[Any]]
    kwargs: Optional[Dict[Any, Any]]
    type: str
    hostname: str
    timestart: Optional[float]
    acknowledged: bool
    delivery_info: CeleryDeliveryInfo
    worker_pid: Optional[int]



@dataclass
class CeleryQueueStatus:
    scheduled: List[CeleryTaskStatus]
    active: List[CeleryTaskStatus]
    reserved: List[CeleryTaskStatus]