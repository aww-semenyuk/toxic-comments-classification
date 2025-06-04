import os
from concurrent.futures import ProcessPoolExecutor
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, Request, BackgroundTasks
from pydantic import BaseModel, ConfigDict

from api.v1.background_tasks import router as background_tasks_router
from api.v1.trainer import router as trainer_router
from database.database import AsyncSessionFactory
from repository import ModelsRepository, BgTasksRepository
from services import TrainerService, BGTasksService
from settings import logger, app_config, LOG_FILE_PATH
from store import loaded_models


@asynccontextmanager
async def lifespan(application: FastAPI):
    """Context manager for the lifespan of the FastAPI application."""
    os.makedirs(LOG_FILE_PATH.parent, exist_ok=True)

    application.state.process_executor = ProcessPoolExecutor(
        max_workers=app_config.cores_cnt - 1
    )
    logger.info("Пул процессов запущен")

    async with AsyncSessionFactory() as session:
        await TrainerService(
            models_repo=ModelsRepository(session),
            loaded_models=loaded_models,
            bg_tasks_repo=BgTasksRepository(session),
            background_tasks=BackgroundTasks(),
            bg_tasks_service=BGTasksService(BgTasksRepository(session))
        ).create_and_load_models()
    logger.info("Предобученные модели загружены")

    logger.info("Приложение запущено")
    yield
    application.state.process_executor.shutdown(wait=True)
    logger.info("Пул процессов остановлен")

    logger.info("Приложение остановлено")


app = FastAPI(
    title="toxic-comments-classification-model-trainer",
    docs_url="/api/openapi",
    openapi_url="/api/openapi.json",
    lifespan=lifespan
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Middleware function to log HTTP requests."""
    method = request.method
    url = str(request.url)
    http_version = request.scope.get("http_version", "1.1")
    response = await call_next(request)
    logger.info(
        '%s:%d - "%s %s HTTP/%s" %d',
        request.client.host,
        request.client.port,
        method, url,
        http_version,
        response.status_code
    )
    return response


class StatusResponse(BaseModel):
    """Pydantic model for the status response."""
    status: str

    model_config = ConfigDict(
        json_schema_extra={"examples": [{"status": "App healthy"}]}
    )


@app.get(
    "/",
    response_model=list[StatusResponse],
    description="Проверка статуса приложения"
)
async def root():
    """Root endpoint to check the status of the application."""
    return [StatusResponse(status="App healthy")]


app.include_router(
    trainer_router,
    prefix="/api/v1/models",
    tags=["trainer"],
)

app.include_router(
    background_tasks_router,
    prefix="/api/v1/tasks",
    tags=["background_tasks"],
)

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        access_log=False
    )
