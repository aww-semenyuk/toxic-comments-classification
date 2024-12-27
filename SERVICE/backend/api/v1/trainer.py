import io
import json
from typing import Annotated

from fastapi import (
    APIRouter,
    HTTPException,
    Depends,
    UploadFile,
    File,
    Form,
    Path
)
from http import HTTPStatus

from pydantic import ValidationError
from starlette import status
from starlette.responses import StreamingResponse

from api.utils import extract_dataset_from_zip_file
from dependency import get_trainer_service
from exceptions import (
    ModelIDAlreadyExistsError,
    ModelNotFoundError,
    ModelNotLoadedError,
    ModelsLimitExceededError,
    InvalidFitPredictDataError,
    ActiveProcessesLimitExceededError,
    DefaultModelRemoveUnloadError,
    ModelNotTrainedError, ModelAlreadyLoadedError
)
from serializers.trainer import (
    MLModelConfig,
    LoadRequest,
    MessageResponse,
    UnloadRequest,
    PredictResponse,
    MLModelType,
    MLModelInListResponse,
    PredictRequest,
    VectorizerType
)
from services.trainer import TrainerService

router = APIRouter()


@router.get(
    "/",
    response_model=list[MLModelInListResponse],
    description="Получение списка моделей"
)
async def get_models(
    trainer_service: Annotated[TrainerService, Depends(get_trainer_service)]
):
    """Endpoint to get a list of models."""
    return await trainer_service.get_models()


@router.post(
    "/fit",
    status_code=HTTPStatus.OK,
    response_model=MessageResponse,
    description="Обучение новой модели"
)
async def fit(
    trainer_service: Annotated[TrainerService, Depends(get_trainer_service)],
    fit_file: Annotated[UploadFile, File()],
    id: Annotated[str, Form()],
    vectorizer_type: Annotated[VectorizerType, Form()],
    ml_model_type: Annotated[MLModelType, Form()],
    ml_model_params: Annotated[
        str,
        Form(description="Валидная JSON-строка")
    ] = "{}",
    spacy_lemma_tokenizer: Annotated[bool, Form()] = False,
    vectorizer_params: Annotated[
        str,
        Form(description="Валидная JSON-строка")
    ] = "{}"
):
    """Endpoint to train new model."""
    try:
        parsed_ml_model_params = json.loads(ml_model_params)
        parsed_vectorizer_params = json.loads(vectorizer_params)
    except json.decoder.JSONDecodeError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "Поля 'ml_model_params' и 'vectorizer_params' должны быть "
                "валидными JSON-строками."
            )
        )

    dataset = extract_dataset_from_zip_file(fit_file)

    try:
        return await trainer_service.fit_models(
            MLModelConfig(
                id=id,
                vectorizer_type=vectorizer_type,
                spacy_lemma_tokenizer=spacy_lemma_tokenizer,
                vectorizer_params=parsed_vectorizer_params,
                ml_model_type=ml_model_type,
                ml_model_params=parsed_ml_model_params
            ),
            dataset
        )
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=repr(e.errors()[0]["msg"])
        )
    except (
        ModelIDAlreadyExistsError,
        InvalidFitPredictDataError,
        ActiveProcessesLimitExceededError
    ) as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=e.detail
        )


@router.post(
    "/load",
    response_model=list[MessageResponse],
    description="Загрузка модели в пространство инференса"
)
async def load(
    request: LoadRequest,
    trainer_service: Annotated[TrainerService, Depends(get_trainer_service)]
):
    """Endpoint to load a model into the inference space."""
    try:
        return await trainer_service.load_model(request.id)
    except ModelNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=e.detail
        )
    except (
        ModelNotTrainedError,
        ModelsLimitExceededError,
        ModelAlreadyLoadedError
    ) as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=e.detail
        )


@router.post(
    "/unload",
    response_model=list[MessageResponse],
    description="Выгрузка модели из пространства инференса"
)
async def unload(
    request: UnloadRequest,
    trainer_service: Annotated[TrainerService, Depends(get_trainer_service)]
):
    """Endpoint to load a model into the inference space."""
    try:
        return await trainer_service.unload_model(request.id)
    except ModelNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=e.detail
        )
    except (ModelNotLoadedError, DefaultModelRemoveUnloadError) as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=e.detail
        )


@router.post(
    "/predict/{id}",
    response_model=PredictResponse,
    description="Предсказание модели"
)
async def predict(
    id: Annotated[str, Path()],
    request: PredictRequest,
    trainer_service: Annotated[TrainerService, Depends(get_trainer_service)]
):
    """Endpoint to make a prediction using a model."""
    try:
        return await trainer_service.predict(id, request)
    except ModelNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=e.detail
        )
    except (ModelNotLoadedError, InvalidFitPredictDataError) as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=e.detail
        )


@router.post(
    "/predict_scores/{id}",
    response_class=StreamingResponse,
    description="Получение данных для построения кривых обучения",
    response_description="CSV-файл с данными для построения кривых обучения"
)
async def predict_scores(
    id: Annotated[str, Path()],
    predict_file: Annotated[UploadFile, File()],
    trainer_service: Annotated[TrainerService, Depends(get_trainer_service)]
):
    """Endpoint to get the data for building learning curves."""
    dataset = extract_dataset_from_zip_file(predict_file)
    try:
        result = await trainer_service.predict_scores(id, dataset)

        buffer = io.StringIO()
        result.to_csv(buffer, index=False)
        buffer.seek(0)

        return StreamingResponse(
            buffer,
            media_type="text/csv",
            headers={
                "Content-Disposition": (
                    "attachment; filename=predicted_scores.csv"
                )
            }
        )
    except ModelNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=e.detail
        )
    except (ModelNotLoadedError, InvalidFitPredictDataError) as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=e.detail
        )


@router.delete(
    "/remove/{id}",
    response_model=list[MessageResponse],
    description="Удаление модели"
)
async def remove(
    id: Annotated[str, Path()],
    trainer_service: Annotated[TrainerService, Depends(get_trainer_service)]
):
    """Endpoint to remove a model."""
    try:
        return await trainer_service.remove_model(id)
    except ModelNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=e.detail
        )
    except DefaultModelRemoveUnloadError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=e.detail
        )


@router.delete(
    "/remove_all",
    response_model=MessageResponse,
    description="Удаление всех моделей"
)
async def remove_all(
    trainer_service: Annotated[TrainerService, Depends(get_trainer_service)]
):
    """Endpoint to remove all models."""
    return await trainer_service.remove_all_models()
