import multiprocessing
from pathlib import Path

from pydantic import field_validator, conint
from pydantic_settings import BaseSettings, SettingsConfigDict

active_processes = multiprocessing.Value('i', 1)

DEFAULT_MODEL_NAMES = ("default_logistic", "default_svm")


class AppConfig(BaseSettings):
    models_dir_path: Path = Path('./models')
    cores_cnt: conint(gt=1) = 2
    models_max_cnt: int = 2
    max_saved_bg_tasks: conint(gt=2) = 10
    model_config = SettingsConfigDict(env_file='.env')

    @field_validator('cores_cnt', mode='before')
    def set_cores_cnt(cls, v):
        available_cores = multiprocessing.cpu_count()
        return min(int(v), available_cores)
