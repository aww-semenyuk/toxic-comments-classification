import datetime as dt
from uuid import UUID

from sqlalchemy import select, update, delete
from sqlalchemy.ext.asyncio import AsyncSession

from db_models import BgTask
from serializers import BGTaskSchema


class BgTasksRepository:
    def __init__(self, db_session: AsyncSession) -> None:
        self.db_session = db_session

    async def get_tasks(self) -> list[BgTask]:
        async with self.db_session as session:
            tasks: list[BgTask] = list((
                await session.execute(
                    select(
                        BgTask
                    ).order_by(BgTask.updated_at.desc())
                )
            ).scalars().all())
        return tasks

    async def create_task(self, task: BGTaskSchema) -> UUID:
        db_bg_task = BgTask(name=task.name)
        async with self.db_session as session:
            session.add(db_bg_task)
            await session.commit()
            await session.flush()
            return db_bg_task.uuid

    async def update_task(
        self,
        task_uuid: UUID,
        status: str,
        result_msg: str,
        updated_at: dt.datetime
    ) -> None:
        async with self.db_session as session:
            await session.execute(
                update(
                    BgTask
                ).where(
                    BgTask.uuid == task_uuid
                ).values(
                    status=status,
                    result_msg=result_msg,
                    updated_at=updated_at
                )
            )
            await session.commit()
            await session.flush()

    async def delete_tasks_by_uuid(self, task_uuids: list[UUID]) -> None:
        async with self.db_session as session:
            await session.execute(
                delete(
                    BgTask
                ).where(
                    BgTask.uuid.in_(task_uuids)
                )
            )
            await session.commit()
            await session.flush()
