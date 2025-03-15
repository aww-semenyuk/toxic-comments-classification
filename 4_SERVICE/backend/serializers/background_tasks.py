from datetime import datetime
from enum import Enum
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class BGTaskStatus(str, Enum):
    running = "running"
    success = "success"
    failure = "failure"


class BGTask(BaseModel):
    uuid: UUID = Field(default_factory=uuid4)
    name: str
    status: BGTaskStatus = BGTaskStatus.running
    result_msg: str | None = None
    updated_at: datetime | None = datetime.now()
