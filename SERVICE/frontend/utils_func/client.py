import io
import json
import os
import zipfile
from typing import List, Any

import httpx

from logger_config import get_logger

logger = get_logger()

BASE_URL = os.environ.get('BACKEND_URL', 'http://localhost:8000') + "/api/v1"


async def train_model(
        data,
        model_id: str = 'default_logistic',
        model_type: str = 'linear_svc',
        model_params: dict = dict,
        vectorizer_type: str = 'bag_of_words',
) -> None:
    """
    Отправка запроса на обучение модели.
    """
    logger.info(
        f"train_model - Начало обучения модели "
        f"model_id={model_id} "
        f"model_type={model_type}"
    )
    async with httpx.AsyncClient() as client:
        try:
            files = {
                "fit_file": ("archive.zip", data, "application/zip")
            }
            data = {
                "id": model_id,
                "vectorizer_type": vectorizer_type,
                "ml_model_type": model_type,
                "ml_model_params": json.dumps(model_params)
            }
            response = await client.post(
                f"{BASE_URL}/models/fit",
                files=files,
                data=data,
            )
            response.raise_for_status()
            logger.info(
                f"train_model - Модель {model_id}  обучена: {response.json()}"
            )
        except httpx.HTTPStatusError as e:
            logger.error(
                f"train_model - Ошибка при обучении модели "
                f"model_id:{model_id}: "
                f"err:{e.response.json()}"
            )


async def get_list_models():
    """
    Список всех моделей.
    """
    logger.info("get_list_models - Получение списка моделей...")
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(f"{BASE_URL}/models/")
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            logger.error(
                f"get_list_models"
                f"Ошибка при получении списка моделей: "
                f"err: {e.response.json()}"
            )


async def load_model(model_id: str) -> str:
    """
    Загрузка модели на сервере.
    """
    logger.info(f" load_model - Загрузка модели model_id:{model_id}")
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                f"{BASE_URL}/models/load", json={"id": model_id}
            )
            response.raise_for_status()
            return ""
        except httpx.HTTPStatusError as e:
            logger.error(
                f"load_model - Ошибка при загрузке модели "
                f"model_id:{model_id} "
                f"err:{e.response.json()}"
            )
            return e.response.json()


async def unload_model(model_id: str) -> str:
    """
    Выгрузка модели.
    """
    logger.info("unload_model - Выгрузка модели...")
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                f"{BASE_URL}/models/unload", json={"id": model_id}
            )
            response.raise_for_status()
            return ""
        except httpx.HTTPStatusError as e:
            logger.error(
                f"unload_model - Ошибка при выгрузке модели "
                f"err:{e.response.json()}"
            )
            return e.response.json()


async def predict_model(id: str, X: List[str]) -> Any:
    """
    Predict модели.
    """
    logger.info(f"predict_model - Predict модели id:{id} X:{X}")
    data_json = {"X": X}
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                f"{BASE_URL}/models/predict/{id}", json=data_json
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            logger.error(
                f"Ошибка при Predict модели "
                f"id:{id} "
                f"X:{X} "
                f"err: {e.response.json()}"
            )
            return None


async def predict_scores_model(
    ids: str,
    zipped_csv: zipfile.ZipFile
) -> io.BytesIO | None:
    """
    Predict модели.
    """
    logger.info(f"predict_scores_model ids-{ids}")
    files = {
        "predict_file": ("archive.zip", zipped_csv, "application/zip")
    }
    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            response = await client.post(
                f"{BASE_URL}/models/predict_scores/",
                data={"ids": ids},
                files=files
            )
            response.raise_for_status()
            csv_content = response.content
            csv_data = io.BytesIO(csv_content)
            return csv_data
        except httpx.HTTPStatusError as e:
            logger.error(
                f"Ошибка при Predict Scores модели ids: {ids} "
                f"err: {e.response.json()}"
            )
            return None


async def remove_model(id: str) -> str:
    """
    Запрос на удаление модели.
    """
    logger.info(f"Удаление модели remove_model id: {id}")
    async with httpx.AsyncClient() as client:
        try:
            response = await client.delete(f"{BASE_URL}/models/remove/{id}")
            response.raise_for_status()
            return ""
        except httpx.HTTPStatusError as e:
            logger.error(
                f"Ошибка при remove_model запросе "
                f"id: {id} "
                f"err:{e.response.json()}"
            )
            return e.response.json()


async def remove_all_models():
    """
     Запрос на remove_all_models.
    """
    logger.info("delete_all_models - Удаление всех моделей")
    async with httpx.AsyncClient() as client:
        try:
            response = await client.delete(f"{BASE_URL}/models/remove_all")
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            logger.error(
                f"Ошибка при удалении всех моделей delete_all_models"
                f" err:{e.response.json()}"
            )


async def get_background_tasks() -> List[Any]:
    """
    Запрос на получение всех background_tasks.
    """
    logger.info("get_background_tasks - Получение всех background_tasks")
    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            response = await client.get(f"{BASE_URL}/tasks/")
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            logger.error(
                f"Ошибка при получении всех background_tasks"
                f" err: {e.response.json()}"
            )
            return []
