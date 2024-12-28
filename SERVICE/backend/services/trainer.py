import asyncio
import datetime as dt
import os
from uuid import UUID

import cloudpickle
import pandas as pd
from fastapi import BackgroundTasks

from exceptions import (
    ModelIDAlreadyExistsError,
    ModelNotFoundError,
    ModelNotLoadedError,
    ModelsLimitExceededError,
    DefaultModelRemoveUnloadError,
    ModelNotTrainedError,
    ActiveProcessesLimitExceededError,
    ModelAlreadyLoadedError
)
from serializers.background_tasks import BGTask, BGTaskStatus
from serializers.trainer import (
    MessageResponse,
    PredictResponse,
    MLModel,
    MLModelConfig,
    PredictRequest,
    MLModelInListResponse
)
from serializers.utils.trainer import serialize_params
from services.background_tasks import BGTasksService
from services.utils.trainer import train_and_save_model_task
from settings.app_config import active_processes, app_config
from settings.app_config import logger
from store import DEFAULT_MODELS_INFO


class TrainerService:
    """Service for training and managing models."""

    def __init__(
        self,
        models_store: dict[str, MLModel],
        loaded_models: dict,
        bg_tasks_store: dict[UUID, BGTask],
        background_tasks: BackgroundTasks,
        bg_tasks_service: BGTasksService
    ):
        from main import app
        self.process_executor = app.state.process_executor

        self.models = models_store
        self.loaded_models = loaded_models
        self.bg_tasks_store = bg_tasks_store
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

        model_id = model_config.id

        if model_id in self.models:
            raise ModelIDAlreadyExistsError(model_id)

        self.background_tasks.add_task(
            self._execute_fitting_task,
            model_id, model_config, fit_dataset
        )

        return MessageResponse(message=(
            f"Запущено обучение модели '{model_config.id}'."
        ))

    async def _execute_fitting_task(
        self,
        model_id: str,
        model_config: MLModelConfig,
        fit_dataset: pd.DataFrame
    ) -> None:
        """Execute the fitting task."""
        loop = asyncio.get_event_loop()

        bg_task = BGTask(name=f"Обучение модели '{model_id}'")
        bg_task_id = bg_task.uuid

        self.bg_tasks_store[bg_task_id] = bg_task
        self.models[model_id] = MLModel(
            id=model_id,
            type=model_config.ml_model_type
        )

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

            self.models[model_id] = MLModel(
                id=model_id,
                type=model_config.ml_model_type,
                is_trained=True,
                model_params=model_params,
                vectorizer_params=vectorizer_params,
                saved_model_file_path=model_file_path,
            )

            self.bg_tasks_store[bg_task_id].status = BGTaskStatus.success
            self.bg_tasks_store[bg_task_id].result_msg = (
                f"Модель '{model_id}' успешно обучена."
            )
            active_processes.value -= 1
        except Exception as e:
            self.models.pop(model_id, None)

            self.bg_tasks_store[bg_task_id].status = BGTaskStatus.failure

            if isinstance(e, TimeoutError):
                result_msg = (
                    f"Превышено время обучения модели ({model_id}). "
                    "Задача остановлена."
                )
                logger.info(result_msg)
            else:
                result_msg = f"Ошибка при обучении модели '{model_id}': {e}."
                logger.error(result_msg)

            self.bg_tasks_store[bg_task_id].result_msg = result_msg

            active_processes.value -= 1

        self.bg_tasks_store[bg_task_id].updated_at = dt.datetime.now()
        self.bg_tasks_service.rotate_tasks()

    async def load_model(self, model_id: str) -> list[MessageResponse]:
        """Load a model into memory."""
        if model_id in self.loaded_models:
            raise ModelAlreadyLoadedError(model_id)
        if len(self.loaded_models) >= (
            app_config.models_max_cnt + len(DEFAULT_MODELS_INFO)
        ):
            raise ModelsLimitExceededError()
        if model_id not in self.models:
            raise ModelNotFoundError(model_id)
        if not self.models[model_id].is_trained:
            raise ModelNotTrainedError(model_id)

        with self.models[model_id].saved_model_file_path.open('rb') as f:
            self.loaded_models[model_id] = cloudpickle.load(f)

        self.models[model_id].is_loaded = True

        return [MessageResponse(
            message=f"Модель '{model_id}' загружена в память."
        )]

    async def unload_model(self, model_id: str) -> list[MessageResponse]:
        """Unload a model from memory."""
        if model_id not in self.models:
            raise ModelNotFoundError(model_id)
        if model_id not in self.loaded_models:
            raise ModelNotLoadedError(model_id)
        if model_id in DEFAULT_MODELS_INFO:
            raise DefaultModelRemoveUnloadError()

        self.loaded_models.pop(model_id, None)
        self.models[model_id].is_loaded = False
        return [MessageResponse(
            message=f"Модель '{model_id}' выгружена из памяти."
        )]

    async def predict(
        self,
        model_id: str,
        predict_data: PredictRequest
    ) -> PredictResponse:
        """Make a prediction using a model."""
        if model_id not in self.models:
            raise ModelNotFoundError(model_id)
        if model_id not in self.loaded_models:
            raise ModelNotLoadedError(model_id)

        model = self.loaded_models.get(model_id)
        return PredictResponse(
            predictions=model.predict(predict_data.X).tolist()
        )

    async def get_models(self) -> list[MLModelInListResponse]:
        """Get a list of models."""
        response_list = []

        for model_info in self.models.values():
            response_list.append(
                MLModelInListResponse(
                    id=model_info.id,
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
        model_id: str,
        predict_dataset: pd.DataFrame
    ) -> pd.DataFrame:
        """Get prediction scores."""
        if model_id not in self.models:
            raise ModelNotFoundError(model_id)
        if model_id not in self.loaded_models:
            raise ModelNotLoadedError(model_id)

        model = self.loaded_models.get(model_id)

        X = predict_dataset["comment_text"]
        y_true = predict_dataset["toxic"]

        if hasattr(model, "predict_proba"):
            scores = model.predict_proba(X)[:, 1]
        else:
            scores = model.decision_function(X)

        return pd.DataFrame({
            "y_true": y_true,
            "scores": scores
        })

    async def remove_model(self, model_id: str) -> list[MessageResponse]:
        """Remove a model."""
        if model_id not in self.models:
            raise ModelNotFoundError(model_id)
        if model_id in DEFAULT_MODELS_INFO:
            raise DefaultModelRemoveUnloadError()

        saved_model_filepath = self.models[model_id].saved_model_file_path
        self.models.pop(model_id, None)
        self.loaded_models.pop(model_id, None)
        if os.path.isfile(saved_model_filepath):
            os.remove(saved_model_filepath)

        return [MessageResponse(message=f"Модель '{model_id}' удалена.")]

    async def remove_all_models(self) -> MessageResponse:
        """Remove all models."""
        model_to_remove_ids = [
            model_id for model_id in self.models
            if model_id not in DEFAULT_MODELS_INFO.keys()
        ]
        for model_id in model_to_remove_ids:
            model_info = self.models.get(model_id)
            if model_info:
                file_path = self.models[model_id].saved_model_file_path
                self.models.pop(model_id, None)

                if file_path and os.path.isfile(file_path):
                    os.remove(file_path)

            self.loaded_models.pop(model_id, None)

        return MessageResponse(
            message="Все модели, кроме моделей по умолчанию, удалены."
        )
