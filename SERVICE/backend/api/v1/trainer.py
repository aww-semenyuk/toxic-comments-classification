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
    MLModelInListResponse
)
from services.trainer import TrainerService

router = APIRouter()


@router.post(
    "/fit",
    status_code=HTTPStatus.OK,
    response_model=MessageResponse
)
async def fit(
    id: Annotated[str, Form()],
    ml_model_type: Annotated[MLModelType, Form()],
    hyperparameters: Annotated[str, Form(description="Валидная JSON-строка")],
    fit_file: Annotated[UploadFile, File()],
    trainer_service: Annotated[TrainerService, Depends(get_trainer_service)]
):
    dataset = extract_dataset_from_zip_file(fit_file)

    try:
        parsed_hyperparameters = json.loads(hyperparameters)
    except json.decoder.JSONDecodeError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Поле 'hyperparameters' должно быть валидной JSON-строкой."
        )

    try:
        return await trainer_service.fit_models(
            MLModelConfig(
                id=id,
                ml_model_type=ml_model_type,
                hyperparameters=parsed_hyperparameters
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
    except (ModelNotLoadedError, DefaultModelRemoveUnloadError) as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=e.detail
        )


@router.post("/predict", response_model=PredictResponse)
async def predict(
    id: Annotated[str, Form()],
    predict_file: Annotated[UploadFile, File()],
    trainer_service: Annotated[TrainerService, Depends(get_trainer_service)]
):
    dataset = extract_dataset_from_zip_file(predict_file)
    try:
        return await trainer_service.predict(id, dataset)
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


@router.get("/list_models", response_model=list[MLModelInListResponse])
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
    except DefaultModelRemoveUnloadError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=e.detail
        )


@router.delete("/remove_all", response_model=list[MessageResponse])
async def remove_all(
    trainer_service: Annotated[TrainerService, Depends(get_trainer_service)]
):
    return await trainer_service.remove_all_models()
