import shutil
from concurrent.futures import ProcessPoolExecutor
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, Request
from pydantic import BaseModel, ConfigDict

from api.v1.background_tasks import router as background_tasks_router
from api.v1.trainer import router as trainer_router
from settings.app_config import AppConfig
from settings.logger_config import logger


@asynccontextmanager
async def lifespan(app: FastAPI):
    app_config = AppConfig()

    if not app_config.models_dir_path.exists():
        app_config.models_dir_path.mkdir(parents=True)

    app.state.process_executor = ProcessPoolExecutor(
        max_workers=app_config.cores_cnt - 1
    )

    logger.info("Application started")
    yield
    app.state.process_executor.shutdown(wait=True)

    if app_config.models_dir_path.exists():
        shutil.rmtree(app_config.models_dir_path)


app = FastAPI(
    title="toxic-comments-classification-model-trainer",
    docs_url="/api/openapi",
    openapi_url="/api/openapi.json",
    lifespan=lifespan
)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    method = request.method
    url = str(request.url)
    http_version = request.scope.get("http_version", "1.1")
    response = await call_next(request)
    logger.info(f'"{method} {url} HTTP/{http_version}" {response.status_code}')
    return response


class StatusResponse(BaseModel):
    status: str

    model_config = ConfigDict(
        json_schema_extra={"examples": [{"status": "App healthy"}]}
    )


@app.get("/", response_model=list[StatusResponse])
async def root():
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
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
