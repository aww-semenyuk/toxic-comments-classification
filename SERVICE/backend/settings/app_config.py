import multiprocessing
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

active_processes = multiprocessing.Value('i', 1)


class AppConfig(BaseSettings):
    models_dir_path: Path = Path('./models')
    cores_cnt: int = 2
    models_max_cnt: int = 2

    model_config = SettingsConfigDict(env_file='.env')

    @field_validator('cores_cnt', mode='before')
    def set_cores_cnt(cls, v):
        available_cores = multiprocessing.cpu_count()
        return min(v, available_cores)
