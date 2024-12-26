from typing import Annotated

from fastapi import APIRouter, HTTPException, Depends
from http import HTTPStatus

from starlette import status

from dependency import get_trainer_service
from exceptions import (
    ModelIDAlreadyExistsError,
    ModelNotFoundError,
    ModelNotLoadedError,
    ModelsLimitExceededError,
    InvalidFitPredictDataError,
    ActiveProcessesLimitExceededError
)
from serializers.trainer import (
    FitRequest,
    LoadRequest,
    GetStatusResponse,
    MessageResponse,
    UnloadRequest,
    PredictRequest,
    PredictResponse,
    PredictScoresResponse,
    ModelListResponse
)
from services.trainer import TrainerService

router = APIRouter()


@router.post(
    "/fit",
    status_code=HTTPStatus.OK,
    response_model=list[MessageResponse]
)
async def fit(
    request: list[FitRequest],
    trainer_service: Annotated[TrainerService, Depends(get_trainer_service)]
):
    try:
        return await trainer_service.fit_models(request)
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
    except ModelsLimitExceededError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=e.detail
        )


@router.get("/get_status", response_model=list[GetStatusResponse])
async def get_status(
    trainer_service: Annotated[TrainerService, Depends(get_trainer_service)]
):
    return await trainer_service.get_status()


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


@router.post("/predict", response_model=list[PredictResponse])
async def predict(
    request: list[PredictRequest],
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


@router.get("/list_models", response_model=list[ModelListResponse])
async def list_models(
    trainer_service: Annotated[TrainerService, Depends(get_trainer_service)]
):
    return await trainer_service.list_models()


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


@router.delete("/remove_all", response_model=list[MessageResponse])
async def remove_all(
    trainer_service: Annotated[TrainerService, Depends(get_trainer_service)]
):
    return await trainer_service.remove_all_models()
