from enum import Enum
from pathlib import Path

from pydantic import BaseModel

class MessageResponse(BaseModel):
    message: str

class ModelListResponse(BaseModel):
    models: list[dict]

class MLModelType(str, Enum):
    LogisticRegression = "logistic_regression"
    MultinomialNB = "multinomial_naive_bayes"

class VectorizerType(str, Enum):
    CountVectorizer = "bag_of_words"
    TfidfVectorizer = "tf_idf"

class MLModelConfig(BaseModel):
    id: str
    ml_model_type: MLModelType = MLModelType.LogisticRegression
    ml_model_params: dict = {}
    vectorizer_type: VectorizerType = VectorizerType.CountVectorizer
    vectorizer_params: dict = {}

class LoadRequest(BaseModel):
    id: str

    model_config = {"json_schema_extra": {"example": {"id": "model_1"}}}

class GetStatusResponse(BaseModel):
    status: str

class UnloadRequest(LoadRequest):
    pass

class PredictRequest(BaseModel):
    id: str
    X: list[str]

    model_config = {"json_schema_extra": {"example": {"id": "model_1",
                                                      "X": ["text text 1", "text text 2"]}}}

class PredictResponse(BaseModel):
    predictions: list


class MLModelInListResponse(BaseModel):
    id: str
    type: MLModelType
    is_trained: bool = False


class MLModel(MLModelInListResponse):
    saved_model_file_path: Path | None = None
