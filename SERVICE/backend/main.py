import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict

from api.v1.api_route import router

app = FastAPI(
    title="model_trainer",
    docs_url="/api/openapi",
    openapi_url="/api/openapi.json",
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
    router,
    prefix="/api/v1/models",
    tags=["trainer"],
)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
