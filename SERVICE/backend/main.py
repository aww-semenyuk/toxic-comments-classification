import os
from concurrent.futures import ProcessPoolExecutor
from contextlib import asynccontextmanager

import cloudpickle
import uvicorn
from fastapi import FastAPI, Request
from pydantic import BaseModel, ConfigDict

from api.v1.background_tasks import router as background_tasks_router
from api.v1.trainer import router as trainer_router
from serializers.trainer import MLModel
from serializers.utils.trainer import serialize_params
from settings.app_config import logger, app_config, MODELS_DIR, LOG_FILE_PATH
from store import loaded_models, models, DEFAULT_MODELS_INFO


@asynccontextmanager
async def lifespan(application: FastAPI):
    """Context manager for the lifespan of the FastAPI application."""
    os.makedirs(LOG_FILE_PATH.parent, exist_ok=True)

    for model_id, model_info in DEFAULT_MODELS_INFO.items():
        saved_model_path = MODELS_DIR / "default" / model_info["filename"]
        with open(saved_model_path, "rb") as f:
            pipe = cloudpickle.load(f)

        loaded_models[model_id] = pipe
        models[model_id] = MLModel(
            id=model_id,
            type=model_info["type"],
            is_trained=True,
            is_loaded=True,
            model_params=serialize_params(
                pipe.named_steps['classifier'].get_params()
            ),
            vectorizer_params=serialize_params(
                pipe.named_steps['vectorizer'].get_params()
            ),
            saved_model_file_path=saved_model_path
        )

    logger.info("Предобученные модели загружены")

    application.state.process_executor = ProcessPoolExecutor(
        max_workers=app_config.cores_cnt - 1
    )
    logger.info("Пул процессов запущен")
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
