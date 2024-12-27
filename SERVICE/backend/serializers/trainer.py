from enum import Enum
from pathlib import Path

from pydantic import BaseModel

class MessageResponse(BaseModel):
    message: str


class MLModelType(str, Enum):
    logistic_regression = "logistic_regression"
    multinomial_nb = "multinomial_naive_bayes"
    linear_svc = "linear_svc"


class VectorizerType(str, Enum):
    count_vectorizer = "bag_of_words"
    tfidf_vectorizer = "tf_idf"


class MLModelConfig(BaseModel):
    id: str
    spacy_lemma_tokenizer: bool = False
    vectorizer_type: VectorizerType
    vectorizer_params: dict
    ml_model_type: MLModelType
    ml_model_params: dict


class LoadRequest(BaseModel):
    id: str


class GetStatusResponse(BaseModel):
    status: str


class UnloadRequest(LoadRequest):
    pass


class PredictRequest(BaseModel):
    id: str
    X: list[str]


class PredictResponse(BaseModel):
    predictions: list[int]


class MLModelInListResponse(BaseModel):
    id: str
    type: MLModelType
    is_trained: bool = False


class MLModel(MLModelInListResponse):
    saved_model_file_path: Path | None = None


class PredictScoresResponse(BaseModel):
    scores: list[float]
