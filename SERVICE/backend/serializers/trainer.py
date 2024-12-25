from enum import Enum
from pathlib import Path

from pydantic import BaseModel


class MessageResponse(BaseModel):
    message: str


class ModelListResponse(BaseModel):
    models: list[dict]


class MLModelType(str, Enum):
    linear_regression = "linear"
    logistic_regression = "logistic"


class MLModelConfig(BaseModel):
    id: str
    ml_model_type: MLModelType
    hyperparameters: dict


class LoadRequest(BaseModel):
    id: str


class GetStatusResponse(BaseModel):
    status: str


class UnloadRequest(LoadRequest):
    pass


class PredictRequest(BaseModel):
    id: str
    X: list[list[float]]


class PredictResponse(BaseModel):
    predictions: list


class MLModelInListResponse(BaseModel):
    id: str
    type: MLModelType
    is_trained: bool = False


class MLModel(MLModelInListResponse):
    saved_model_file_path: Path | None = None
