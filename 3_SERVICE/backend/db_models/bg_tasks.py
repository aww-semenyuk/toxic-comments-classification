import datetime as dt
from typing import Optional
from uuid import UUID

from sqlalchemy import DateTime, Enum, Uuid, func
from sqlalchemy.orm import Mapped, mapped_column

from database import Base
from serializers import BGTaskStatus


class BgTask(Base):
    __tablename__ = 'bg_tasks'

    uuid: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        primary_key=True,
        server_default=func.gen_random_uuid()
    )
    name: Mapped[str]
    status: Mapped[str] = mapped_column(
        Enum(BGTaskStatus),
        default=BGTaskStatus.running
    )
    result_msg: Mapped[Optional[str]] = mapped_column(default=None)
    updated_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: dt.datetime.now(dt.UTC)
    )
