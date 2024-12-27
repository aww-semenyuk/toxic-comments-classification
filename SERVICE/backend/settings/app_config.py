import logging
import multiprocessing
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path

from pydantic import field_validator, conint
from pydantic_settings import BaseSettings, SettingsConfigDict

active_processes = multiprocessing.Value('i', 1)

BASE_DIR = Path(__file__).parent.parent.resolve()
MODELS_DIR = BASE_DIR / "models"


class AppConfig(BaseSettings):
    """Configuration settings for the application."""

    cores_cnt: conint(gt=1) = 2
    models_max_cnt: int = 2
    max_saved_bg_tasks: conint(gt=2) = 10
    log_file_path: Path = Path('./app.log')

    model_config = SettingsConfigDict(env_file='.env')

    @field_validator('cores_cnt', mode='before')
    def set_cores_cnt(cls, v):
        """Set the number of CPU cores to use."""
        available_cores = multiprocessing.cpu_count()
        return min(int(v), available_cores)


app_config = AppConfig()

formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

timed_handler = TimedRotatingFileHandler(
    app_config.log_file_path,
    when="midnight",
    interval=1,
    backupCount=7
)
timed_handler.setLevel(logging.INFO)
timed_handler.setFormatter(formatter)

logger = logging.getLogger("toxic_comments_app")
logger.setLevel(logging.INFO)
logger.addHandler(timed_handler)
