from services.trainer import TrainerService
from settings.app_config import AppConfig
from store import models, loaded_models

async def get_trainer_service():
    return TrainerService(
        app_config=AppConfig(),
        models_store=models,
        loaded_models=loaded_models
    )
