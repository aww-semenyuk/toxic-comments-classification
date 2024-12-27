import json
from typing import Annotated

from fastapi import APIRouter, HTTPException, Depends, UploadFile, File, Form
from http import HTTPStatus

from starlette import status

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
    ModelNotTrainedError
)
from serializers.trainer import (
    MLModelConfig,
    LoadRequest,
    GetStatusResponse,
    MessageResponse,
    UnloadRequest,
    PredictResponse,
    MLModelType,
    MLModelInListResponse,
    PredictRequest,
    VectorizerType,
    PredictScoresResponse
)
from services.trainer import TrainerService

router = APIRouter()


@router.post(
    "/fit",
    status_code=HTTPStatus.OK,
    response_model=MessageResponse
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
    except (
        ModelIDAlreadyExistsError,
        InvalidFitPredictDataError,
        ActiveProcessesLimitExceededError
    ) as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=e.detail
        )


@router.post("/load", response_model=list[MessageResponse])
async def load(
    request: LoadRequest,
    trainer_service: Annotated[TrainerService, Depends(get_trainer_service)]
):
    try:
        return await trainer_service.load_model(request.id)
    except ModelNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=e.detail
        )
    except (ModelNotTrainedError, ModelsLimitExceededError) as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=e.detail
        )


@router.get("/loaded_models", response_model=list[GetStatusResponse])
async def loaded_models(
    trainer_service: Annotated[TrainerService, Depends(get_trainer_service)]
):
    return await trainer_service.get_loaded_models()


@router.post("/unload", response_model=list[MessageResponse])
async def unload(
    request: UnloadRequest,
    trainer_service: Annotated[TrainerService, Depends(get_trainer_service)]
):
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


@router.post("/predict", response_model=PredictResponse)
async def predict(
    request: PredictRequest,
    trainer_service: Annotated[TrainerService, Depends(get_trainer_service)]
):
    try:
        return await trainer_service.predict(request)
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
    
@router.post("/predict_scores", response_model=list[PredictScoresResponse])
async def predict_scores(
    request: list[PredictRequest],
    trainer_service: Annotated[TrainerService, Depends(get_trainer_service)]
):
    try:
        return await trainer_service.predict_scores(request)
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


@router.get("/trained_models", response_model=list[MLModelInListResponse])
async def trained_models(
    trainer_service: Annotated[TrainerService, Depends(get_trainer_service)]
):
    return await trainer_service.get_trained_models()


@router.delete("/remove/{model_id}", response_model=list[MessageResponse])
async def remove(
    model_id: str,
    trainer_service: Annotated[TrainerService, Depends(get_trainer_service)]
):
    try:
        return await trainer_service.remove_model(model_id)
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


@router.delete("/remove_all", response_model=MessageResponse)
async def remove_all(
    trainer_service: Annotated[TrainerService, Depends(get_trainer_service)]
):
    return await trainer_service.remove_all_models()
