import shutil
from concurrent.futures import ProcessPoolExecutor
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict

from api.v1.background_tasks import router as background_tasks_router
from api.v1.trainer import router as trainer_router
from settings.app_config import AppConfig


@asynccontextmanager
async def lifespan(app: FastAPI):
    app_config = AppConfig()

    if not app_config.models_dir_path.exists():
        app_config.models_dir_path.mkdir(parents=True)

    app.state.process_executor = ProcessPoolExecutor(
        max_workers=app_config.cores_cnt - 1
    )
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
