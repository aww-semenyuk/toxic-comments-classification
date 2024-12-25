from fastapi import BackgroundTasks, Depends

from services.background_tasks import BGTasksService
from services.trainer import TrainerService
from settings.app_config import AppConfig
from store import models, loaded_models, bg_tasks


async def get_bg_tasks_service():
    return BGTasksService(
        app_config=AppConfig(),
        bg_tasks_store=bg_tasks
    )


async def get_trainer_service(
    background_tasks: BackgroundTasks,
    bg_tasks_service: BGTasksService = Depends(get_bg_tasks_service)
):
    return TrainerService(
        app_config=AppConfig(),
        models_store=models,
        loaded_models=loaded_models,
        bg_tasks_store=bg_tasks,
        background_tasks=background_tasks,
        bg_tasks_service=bg_tasks_service
    )
