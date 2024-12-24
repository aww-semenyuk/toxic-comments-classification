from enum import Enum

from pydantic import BaseModel


class MessageResponse(BaseModel):
    message: str


class ModelListResponse(BaseModel):
    models: list[dict]


class MLModelType(str, Enum):
    linear_regression = "linear"
    logistic_regression = "logistic"


class FitConfig(BaseModel):
    id: str
    ml_model_type: MLModelType
    hyperparameters: dict


class FitRequest(BaseModel):
    X: list[list[float]]
    y: list[float]
    config: FitConfig


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
    predictions: list[float]
