from fastapi import BackgroundTasks, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from database import get_db_session
from repository import BgTasksRepository, ModelsRepository
from services import BGTasksService, TrainerService
from store import loaded_models


async def get_models_repository(
    db_session: AsyncSession = Depends(get_db_session)
) -> ModelsRepository:
    return ModelsRepository(db_session)


async def get_bg_tasks_repository(
    db_session: AsyncSession = Depends(get_db_session)
) -> BgTasksRepository:
    return BgTasksRepository(db_session)


async def get_bg_tasks_service(
    bg_tasks_repo: BgTasksRepository = Depends(get_bg_tasks_repository)
):
    """Get the background tasks service."""
    return BGTasksService(bg_tasks_repo=bg_tasks_repo)


async def get_trainer_service(
    background_tasks: BackgroundTasks,
    bg_tasks_service: BGTasksService = Depends(get_bg_tasks_service),
    models_repo: ModelsRepository = Depends(get_models_repository),
    bg_tasks_repo: BgTasksRepository = Depends(get_bg_tasks_repository)
):
    """Get the trainer service."""
    return TrainerService(
        models_repo=models_repo,
        loaded_models=loaded_models,
        bg_tasks_repo=bg_tasks_repo,
        background_tasks=background_tasks,
        bg_tasks_service=bg_tasks_service
    )
