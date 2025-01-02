import re
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field, field_validator


class MessageResponse(BaseModel):
    """Pydantic model for the message response."""
    message: str


class MLModelType(str, Enum):
    """Enum for model type."""
    logistic_regression = "logistic_regression"
    multinomial_nb = "multinomial_naive_bayes"
    linear_svc = "linear_svc"


class VectorizerType(str, Enum):
    """Enum for vectorizer type."""
    count_vectorizer = "bag_of_words"
    tfidf_vectorizer = "tf_idf"


class MLModelConfig(BaseModel):
    """Pydantic model for the model config."""
    id: str
    spacy_lemma_tokenizer: bool = False
    vectorizer_type: VectorizerType
    vectorizer_params: dict
    ml_model_type: MLModelType
    ml_model_params: dict

    @field_validator("id")
    def validate_id(cls, v):
        """Validator for the ID field."""
        if not bool(re.compile(r"^[a-z0-9_]+(?:[-_][a-z0-9_]+)*$").match(v)):
            raise ValueError(
                "ID модели может состоять только из строчных латинских букв, "
                "цифр,  дефисов, и знаков подчеркивания"
            )
        return v


class LoadRequest(BaseModel):
    """Pydantic model for the load request."""
    id: str


class UnloadRequest(LoadRequest):
    """Pydantic model for the unload request."""
    pass


class PredictRequest(BaseModel):
    """Pydantic model for the predict request."""
    X: list[str] = Field(
        description="Кавычки в тексте необходимо экранировать"
    )


class PredictResponse(BaseModel):
    """Pydantic model for the predict response."""
    predictions: list[int]


class MLModelInListResponse(BaseModel):
    """Pydantic model for the model in list response."""
    id: str
    type: MLModelType
    is_trained: bool = False
    is_loaded: bool = False
    model_params: dict = Field(default_factory=dict)
    vectorizer_params: dict = Field(default_factory=dict)


class MLModel(MLModelInListResponse):
    """Pydantic model for the model."""
    saved_model_file_path: Path | None = None
