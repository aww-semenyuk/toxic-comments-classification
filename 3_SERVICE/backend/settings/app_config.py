import multiprocessing

from pydantic import field_validator, conint
from pydantic_settings import BaseSettings, SettingsConfigDict

active_processes = multiprocessing.Value('i', 1)


class AppConfig(BaseSettings):
    """Configuration settings for the application."""

    cores_cnt: conint(gt=1) = 2
    models_max_cnt: int = 2
    max_saved_bg_tasks: conint(gt=2) = 10

    postgres_host: str = 'localhost'
    postgres_port: int = 5432
    postgres_user: str = 'postgres'
    postgres_db: str = 'toxic_comments'
    postgres_password: str = 'postgres'
    postgres_driver: str = 'postgresql+asyncpg'

    model_config = SettingsConfigDict(env_file='.env', extra='ignore')

    @property
    def db_url(self):
        return (
            f'{self.postgres_driver}://{self.postgres_user}:'
            f'{self.postgres_password}@{self.postgres_host}:'
            f'{self.postgres_port}/{self.postgres_db}'
        )

    @field_validator('cores_cnt', mode='before')
    def set_cores_cnt(cls, v):
        """Set the number of CPU cores to use."""
        available_cores = multiprocessing.cpu_count()
        return min(int(v), available_cores)


app_config = AppConfig()
