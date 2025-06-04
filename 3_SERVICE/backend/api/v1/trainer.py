import io
import json
from ast import Param
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
    ModelNameAlreadyExistsError,
    ModelNotFoundError,
    ModelNotLoadedError,
    ModelsLimitExceededError,
    InvalidFitPredictDataError,
    ActiveProcessesLimitExceededError,
    DefaultModelRemoveUnloadError,
    ModelNotTrainedError, ModelAlreadyLoadedError
)
from serializers import (
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
from services import TrainerService

router = APIRouter()


@router.get(
    "/",
    response_model=list[MLModelInListResponse],
    description="Получение списка моделей"
)
async def get_models(
    trainer_service: Annotated[TrainerService, Depends(get_trainer_service)],
    is_dl: Annotated[bool | None, Param()] = None
):
    """Endpoint to get a list of models."""
    return await trainer_service.get_models(is_dl=is_dl)


@router.post(
    "/fit",
    status_code=HTTPStatus.OK,
    response_model=MessageResponse,
    description="Обучение новой модели"
)
async def fit(
    trainer_service: Annotated[TrainerService, Depends(get_trainer_service)],
    fit_file: Annotated[
        UploadFile,
        File(description=(
            "ZIP-архив с CSV-файлом. Файл должен содержать 2 столбца: "
            "`comment_text` (сырой текст) "
            "и `toxic` (бинарная метка токсичности)"
        ))
    ],
    name: Annotated[str, Form()],
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
                name=name,
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
            ModelNameAlreadyExistsError,
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
        return await trainer_service.load_model(request.name)
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
        return await trainer_service.unload_model(request.name)
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
    "/predict/{name}",
    response_model=PredictResponse,
    description="Предсказание модели"
)
async def predict(
    name: Annotated[str, Path()],
    request: PredictRequest,
    trainer_service: Annotated[TrainerService, Depends(get_trainer_service)]
):
    """Endpoint to make a prediction using a model."""
    try:
        return await trainer_service.predict(name, request)
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
    "/predict_scores/",
    response_class=StreamingResponse,
    description="Получение данных для построения кривых обучения",
    response_description="CSV-файл с данными для построения кривых обучения"
)
async def predict_scores(
    names: Annotated[
        str,
        Form(description="Список имен моделей через запятую (model_1,model_2)")
    ],
    predict_file: Annotated[
        UploadFile,
        File(description=(
            "ZIP-архив с CSV-файлом. Файл должен содержать 2 столбца: "
            "`comment_text` (сырой текст) и "
            "`toxic` (бинарная метка токсичности)"
        ))
    ],
    trainer_service: Annotated[TrainerService, Depends(get_trainer_service)]
):
    """Endpoint to get the data for building learning curves."""
    dataset = extract_dataset_from_zip_file(predict_file)
    try:
        result = await trainer_service.predict_scores(
            names.split(","),
            dataset
        )

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
    "/remove/{name}",
    response_model=list[MessageResponse],
    description="Удаление модели"
)
async def remove(
    name: Annotated[str, Path()],
    trainer_service: Annotated[TrainerService, Depends(get_trainer_service)]
):
    """Endpoint to remove a model."""
    try:
        return await trainer_service.remove_model(name)
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
