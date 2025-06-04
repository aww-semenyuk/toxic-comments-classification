import asyncio
import datetime as dt
import os

import cloudpickle
import pandas as pd
from fastapi import BackgroundTasks
from transformers import pipeline

from exceptions import (
    ModelNameAlreadyExistsError,
    ModelNotFoundError,
    ModelNotLoadedError,
    ModelsLimitExceededError,
    DefaultModelRemoveUnloadError,
    ModelNotTrainedError,
    ActiveProcessesLimitExceededError,
    ModelAlreadyLoadedError
)
from repository import BgTasksRepository, ModelsRepository
from serializers import (
    BGTaskSchema,
    BGTaskStatus,
    MessageResponse,
    PredictResponse,
    MLModelConfig,
    PredictRequest,
    MLModelInListResponse,
    MLModelCreateSchema,
    MLModelType
)
from serializers.utils.trainer import serialize_params
from services import BGTasksService
from services.utils.trainer import (
    train_and_save_model_task,
    get_dl_model_predictions
)
from settings import active_processes, app_config, logger, MODELS_DIR
from store import DEFAULT_MODELS_INFO


class TrainerService:
    """Service for training and managing models."""

    def __init__(
        self,
        models_repo: ModelsRepository,
        loaded_models: dict,
        bg_tasks_repo: BgTasksRepository,
        background_tasks: BackgroundTasks,
        bg_tasks_service: BGTasksService
    ):
        from main import app
        self.process_executor = app.state.process_executor

        self.models_repo = models_repo
        self.loaded_models = loaded_models
        self.bg_tasks_repo = bg_tasks_repo
        self.background_tasks = background_tasks
        self.bg_tasks_service = bg_tasks_service

    async def fit_models(
        self,
        model_config: MLModelConfig,
        fit_dataset: pd.DataFrame
    ) -> MessageResponse:
        """Train a new model."""
        if active_processes.value >= app_config.cores_cnt:
            raise ActiveProcessesLimitExceededError()

        model_name = model_config.name

        if await self.models_repo.get_model_by_name(model_name):
            raise ModelNameAlreadyExistsError(model_name)

        self.background_tasks.add_task(
            self._execute_fitting_task,
            model_name, model_config, fit_dataset
        )

        return MessageResponse(message=(
            f"Запущено обучение модели '{model_config.name}'."
        ))

    async def _execute_fitting_task(
        self,
        model_name: str,
        model_config: MLModelConfig,
        fit_dataset: pd.DataFrame
    ) -> None:
        """Execute the fitting task."""
        loop = asyncio.get_event_loop()

        bg_task_id = await self.bg_tasks_repo.create_task(
            BGTaskSchema(name=f"Обучение модели '{model_name}'"))
        await self.models_repo.create_model(MLModelCreateSchema(
            name=model_name,
            type=model_config.ml_model_type
        ))

        active_processes.value += 1
        try:
            (
                model_file_path,
                model_params,
                vectorizer_params
            ) = await asyncio.wait_for(
                loop.run_in_executor(
                    self.process_executor,
                    train_and_save_model_task,
                    model_config, fit_dataset
                ),
                timeout=1800
            )

            await self.models_repo.update_model_after_training(
                model_name=model_name,
                is_trained=True,
                model_params=serialize_params(model_params),
                vectorizer_params=serialize_params(vectorizer_params),
                saved_model_file_path=str(model_file_path),
            )

            status = BGTaskStatus.success
            result_msg = (
                f"Модель '{model_name}' успешно обучена."
            )
            active_processes.value -= 1
        except Exception as e:
            await self.models_repo.delete_model(model_name)

            status = BGTaskStatus.failure
            if isinstance(e, TimeoutError):
                result_msg = (
                    f"Превышено время обучения модели ({model_name}). "
                    "Задача остановлена."
                )
                logger.info(result_msg)
            else:
                result_msg = f"Ошибка при обучении модели '{model_name}': {e}."
                logger.error(result_msg)

            active_processes.value -= 1

        await self.bg_tasks_repo.update_task(
            task_uuid=bg_task_id,
            status=status,
            result_msg=result_msg,
            updated_at=dt.datetime.now()
        )
        await self.bg_tasks_service.rotate_tasks()

    async def load_model(self, model_name: str) -> list[MessageResponse]:
        """Load a model into memory."""
        model = await self.models_repo.get_model_by_name(model_name)
        if model.uuid in self.loaded_models:
            raise ModelAlreadyLoadedError(model_name)
        if len(self.loaded_models) >= (
            app_config.models_max_cnt + len(DEFAULT_MODELS_INFO)
        ):
            raise ModelsLimitExceededError()
        if not model:
            raise ModelNotFoundError(model_name)
        if not model.is_trained:
            raise ModelNotTrainedError(model_name)

        with open(model.saved_model_file_path, 'rb') as f:
            self.loaded_models[model.uuid] = cloudpickle.load(f)

        await self.models_repo.update_model_is_loaded(model_name, True)

        return [MessageResponse(
            message=f"Модель '{model_name}' загружена в память."
        )]

    async def unload_model(self, model_name: str) -> list[MessageResponse]:
        """Unload a model from memory."""
        model = await self.models_repo.get_model_by_name(model_name)
        if not model:
            raise ModelNotFoundError(model_name)
        if model.uuid not in self.loaded_models:
            raise ModelNotLoadedError(model_name)
        if model_name in DEFAULT_MODELS_INFO:
            raise DefaultModelRemoveUnloadError()

        self.loaded_models.pop(model.uuid, None)
        await self.models_repo.update_model_is_loaded(model_name, False)
        return [MessageResponse(
            message=f"Модель '{model_name}' выгружена из памяти."
        )]

    async def predict(
        self,
        model_name: str,
        predict_data: PredictRequest
    ) -> PredictResponse:
        """Make a prediction using a model."""
        model = await self.models_repo.get_model_by_name(model_name)

        if not model:
            raise ModelNotFoundError(model_name)
        if model.uuid not in self.loaded_models:
            raise ModelNotLoadedError(model_name)

        loaded_model = self.loaded_models.get(model.uuid)
        if model.is_dl_model:
            predictions = get_dl_model_predictions(
                model.type,
                loaded_model,
                predict_data.X
            )
        else:
            predictions = loaded_model.predict(predict_data.X).tolist()

        return PredictResponse(predictions=predictions)

    async def get_models(
        self,
        is_dl: bool = False
    ) -> list[MLModelInListResponse]:
        """Get a list of models."""
        response_list = []

        for model_info in await self.models_repo.get_models(is_dl=is_dl):
            response_list.append(
                MLModelInListResponse(
                    uuid=model_info.uuid,
                    name=model_info.name,
                    type=model_info.type,
                    is_trained=model_info.is_trained,
                    is_loaded=model_info.is_loaded,
                    model_params=serialize_params(model_info.model_params),
                    vectorizer_params=serialize_params(
                        model_info.vectorizer_params
                    )
                )
            )

        return response_list

    async def predict_scores(
        self,
        model_names: list[str],
        predict_dataset: pd.DataFrame
    ) -> pd.DataFrame:
        """Get prediction scores."""
        models = await self.models_repo.get_models_by_names(model_names)

        for model_name in model_names:
            if model_name not in [db_model.name for db_model in models]:
                raise ModelNotFoundError(model_name)
        for db_model in models:
            if db_model.uuid not in self.loaded_models:
                raise ModelNotLoadedError(db_model.name)

        results = []
        for db_model in models:
            loaded_model = self.loaded_models.get(db_model.uuid)

            X = predict_dataset["comment_text"]
            y_true = predict_dataset["toxic"]

            if db_model.is_dl_model:
                scores = get_dl_model_predictions(
                    db_model.type,
                    loaded_model,
                    X.tolist(),
                    return_scores=True
                )
            else:
                if hasattr(loaded_model, "predict_proba"):
                    scores = loaded_model.predict_proba(X)[:, 1]
                else:
                    scores = loaded_model.decision_function(X)

            results.append(pd.DataFrame({
                "model_name": db_model.name,
                "scores": scores,
                "y_true": y_true
            }))

        return pd.concat(results, ignore_index=True)

    async def remove_model(self, model_name: str) -> list[MessageResponse]:
        """Remove a model."""
        model = await self.models_repo.get_model_by_name(model_name)

        if not model:
            raise ModelNotFoundError(model_name)
        if model_name in DEFAULT_MODELS_INFO:
            raise DefaultModelRemoveUnloadError()

        saved_model_filepath = model.saved_model_file_path
        await self.models_repo.delete_model(model_name)
        self.loaded_models.pop(model_name, None)
        if os.path.isfile(saved_model_filepath):
            os.remove(saved_model_filepath)

        return [MessageResponse(message=f"Модель '{model_name}' удалена.")]

    async def remove_all_models(self) -> MessageResponse:
        """Remove all models."""
        deleted_models = await self.models_repo.delete_all_user_models()
        for model in deleted_models:
            file_path = model.saved_model_file_path
            if file_path and os.path.isfile(file_path):
                os.remove(file_path)
            self.loaded_models.pop(model.uuid, None)

        return MessageResponse(
            message="Все модели, кроме моделей по умолчанию, удалены."
        )

    async def create_and_load_models(self) -> None:
        """Create default models."""
        for model_name, model_info in DEFAULT_MODELS_INFO.items():
            saved_model_path = MODELS_DIR / "default" / model_info["filename"]
            is_dl_model = model_info["is_dl_model"]
            model_type = model_info["type"]

            model_params = {}
            vectorizer_params = {}
            if is_dl_model:
                if model_type == MLModelType.distilbert:
                    pipe = pipeline(
                        "text-classification",
                        model=saved_model_path,
                        tokenizer=model_info["tokenizer"]
                    )
                    model_params = pipe.model.config.to_dict()
                    vectorizer_params = pipe.tokenizer.init_kwargs
            else:
                with open(saved_model_path, "rb") as f:
                    pipe = cloudpickle.load(f)

                model_params = serialize_params(
                    pipe.named_steps["classifier"].get_params()
                )
                vectorizer_params = serialize_params(
                    pipe.named_steps["vectorizer"].get_params()
                )

            db_model = await self.models_repo.get_model_by_name(model_name)
            if db_model:
                db_model_uuid = db_model.uuid
            else:
                db_model_uuid = await self.models_repo.create_model(
                    MLModelCreateSchema(
                        name=model_name,
                        type=model_type,
                        is_dl_model=is_dl_model,
                        is_trained=True,
                        is_loaded=True,
                        model_params=model_params,
                        vectorizer_params=vectorizer_params,
                        saved_model_file_path=saved_model_path
                    )
                )

            self.loaded_models[db_model_uuid] = pipe

        for db_model in await self.models_repo.get_models():
            if db_model.name not in DEFAULT_MODELS_INFO and db_model.is_loaded:
                with open(db_model.saved_model_file_path, "rb") as f:
                    self.loaded_models[db_model.uuid] = cloudpickle.load(f)
