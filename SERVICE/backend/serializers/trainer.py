from enum import Enum
from pathlib import Path

from pydantic import BaseModel
from typing import Literal

class MessageResponse(BaseModel):
    message: str


class ModelListResponse(BaseModel):
    models: list[dict]


class MLModelType(str, Enum):
    LogisticRegression = "logistic_regression"
    MultinomialNB = "multinomial_naive_bayes"
    LinearSVC = "linear_svc"


class VectorizerType(str, Enum):
    CountVectorizer = "bag_of_words"
    TfidfVectorizer = "tf_idf"


class MLModelConfig(BaseModel):
    id: str
    custom_tokenizer: Literal["spacy_lemma", None] = "spacy_lemma"
    vectorizer_type: VectorizerType = VectorizerType.CountVectorizer
    vectorizer_params: dict
    ml_model_type: MLModelType = MLModelType.LogisticRegression
    ml_model_params: dict

    model_config = {
        "json_schema_extra": {"example": {
            "X": ["text text 1", "text text 2"],
            "y": [0, 1],
            "config": {
                "id": "model_1",
                "custom_tokenizer": "spacy_lemma",
                "ml_model_type": "logistic_regression",
                "ml_model_params": {"C": 0.01},
                "vectorizer_type": "bag_of_words",
                "vectorizer_params": {"max_features": 10000}}}}}


class LoadRequest(BaseModel):
    id: str


class GetStatusResponse(BaseModel):
    status: str


class UnloadRequest(LoadRequest):
    pass


class PredictRequest(BaseModel):
    id: str
    X: list[str]

    model_config = {
        "json_schema_extra": {"example": {
            "id": "model_1", 
            "X": ["text text 1", "text text 2"]}}}


class PredictResponse(BaseModel):
    predictions: list[int]


class MLModelInListResponse(BaseModel):
    id: str
    type: MLModelType
    is_trained: bool = False


class MLModel(MLModelInListResponse):
    saved_model_file_path: Path | None = None
