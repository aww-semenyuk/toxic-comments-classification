import asyncio
import os
import cloudpickle
from concurrent.futures import ProcessPoolExecutor

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
    MessageResponse,
    GetStatusResponse,
    PredictRequest,
    ModelListResponse,
    PredictResponse,
)
from utils.trainer import make_pipeline_from_config
from settings.app_config import AppConfig, active_processes


class TrainerService:
    def __init__(
        self,
        app_config: AppConfig,
        models_store: dict,
        loaded_models: dict
    ):
        self.models = models_store
        self.loaded_models = loaded_models
        self.app_config = app_config

        if not app_config.models_dir_path.exists():
            app_config.models_dir_path.mkdir(parents=True)

    async def fit_models(
        self, fit_data: list[FitRequest]
    ) -> list[MessageResponse]:
        unique_fit_items = {}
        for item in fit_data:
            if item.config.id not in unique_fit_items:
                unique_fit_items[item.config.id] = item
        unique_fit_items_list = list(unique_fit_items.values())

        for item in unique_fit_items_list:
            model_id = item.config.id
            if model_id in self.models:
                raise ModelIDAlreadyExistsError(model_id)

        fit_items_cnt = len(unique_fit_items_list)
        free_cores_cnt = self.app_config.cores_cnt - active_processes.value
        if fit_items_cnt > free_cores_cnt:
            raise ActiveProcessesLimitExceededError()

        active_processes.value += fit_items_cnt
        executor = ProcessPoolExecutor(max_workers=free_cores_cnt)
        loop = asyncio.get_running_loop()
        tasks = [
            loop.run_in_executor(executor, self._train_model, item)
            for item in unique_fit_items_list
        ]

        response_list = []
        for task, item in zip(
            asyncio.as_completed(tasks),
            unique_fit_items_list
        ):
            config = item.config
            model_id = config.id
            model_type = config.ml_model_type
            try:
                model = await task
                active_processes.value -= 1

                model_file_path = (
                    self.app_config.models_dir_path / f"{model_id}.cloudpickle"
                )
                with model_file_path.open('wb') as f:
                    cloudpickle.dump(model, f)

                self.models[model_id] = {
                    "type": model_type,
                    "saved_model_file_path": model_file_path
                }

                response_list.append(MessageResponse(
                    message=f"Model '{item.config.id}' trained and saved."
                ))
            except InvalidFitPredictDataError as e:
                response_list.append(MessageResponse(
                    message=f"Model '{item.config.id}' failed: {e.detail}"
                ))
            except Exception as e:
                response_list.append(MessageResponse(
                    message=(
                        f"Model '{item.config.id}' unexpected error: {str(e)}"
                    )
                ))

        return response_list

    def _train_model(self, data: FitRequest):
        try:
            pipe = make_pipeline_from_config(data.config)
            pipe.fit(data.X, data.y)
        except ValueError as e:
            raise InvalidFitPredictDataError(e.args[0])

        return pipe

    async def load_model(self, model_id: str) -> list[MessageResponse]:
        if len(self.loaded_models) == self.app_config.models_max_cnt:
            raise ModelsLimitExceededError()
        if model_id not in self.models:
            raise ModelNotFoundError(model_id)
        
        with self.models[model_id].get("saved_model_file_path").open('rb') as f:
            self.loaded_models[model_id] = cloudpickle.load(f)

        return [MessageResponse(message=f"Model '{model_id}' loaded.")]

    async def get_status(self) -> list[GetStatusResponse]:
        return [
            GetStatusResponse(status=f"Model '{model_id}' Status Ready")
            for model_id in self.loaded_models.keys()
        ]

    async def unload_model(self, model_id: str) -> list[MessageResponse]:
        if model_id not in self.models:
            raise ModelNotFoundError(model_id)

        self.loaded_models.pop(model_id)
        return [MessageResponse(message=f"Model '{model_id}' unloaded.")]

    async def predict(
        self,
        predict_data: list[PredictRequest]
    ) -> list[PredictResponse]:
        unique_predict_items = {}
        for item in predict_data:
            if item.id not in unique_predict_items:
                unique_predict_items[item.id] = item
        unique_predict_items_list = list(unique_predict_items.values())

        for item in unique_predict_items_list:
            model_id = item.id
            if model_id not in self.models:
                raise ModelNotFoundError(model_id)
            if model_id not in self.loaded_models:
                raise ModelNotLoadedError(model_id)

        model_preds = []
        for item in unique_predict_items_list:
            model = self.loaded_models.get(item.id)
            model_preds.append(PredictResponse(
                predictions=model.predict(item.X).tolist()
            ))

        return model_preds

    async def list_models(self) -> list[ModelListResponse]:
        models_list = [
            {"id": model_id, "type": model_info["type"]}
            for model_id, model_info in self.models.items()
        ]
        return [ModelListResponse(models=models_list)]

    async def remove_model(self, model_id: str) -> list[MessageResponse]:
        if model_id not in self.models:
            raise ModelNotFoundError(model_id)

        saved_model_filepath = self.models[model_id]["saved_model_file_path"]
        self.models.pop(model_id)
        self.loaded_models.pop(model_id, None)
        if os.path.isfile(saved_model_filepath):
            os.remove(saved_model_filepath)

        return [MessageResponse(message=f"Model '{model_id}' removed.")]

    async def remove_all_models(self) -> list[MessageResponse]:
        saved_model_file_paths = [
            model_info["saved_model_file_path"]
            for model_info in self.models.values()
        ]
        self.models.clear()
        self.loaded_models.clear()
        for file_path in saved_model_file_paths:
            if os.path.isfile(file_path):
                os.remove(file_path)

        return [MessageResponse(message="All models removed.")]
