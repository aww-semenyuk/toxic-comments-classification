import json

import httpx
from typing import List, Any
from process_data import logging
import os

BASE_URL = os.environ.get('BACKEND_URL', 'http://localhost:8000') + "/api/v1"


async def train_model(data, model_id: str = 'default_logistic', model_type: str = 'linear_svc', model_params: dict = {}) -> None:
    """
    Отправка запроса на обучение модели.
    """
    logging.info(f"train_model - Начало обучения модели model_id={model_id} model_type={model_type}")
    async with httpx.AsyncClient() as client:
        try:
            files = {
                "fit_file": ("archive.zip", data, "application/zip")
            }
            data = {
                "id": model_id,
                "vectorizer_type": "bag_of_words",
                "ml_model_type": model_type,
                "ml_model_params": json.dumps(model_params)
            }
            response = await client.post(
                f"{BASE_URL}/models/fit",
                files=files,
                data=data,
            )
            response.raise_for_status()
            logging.info(f"train_model - Модель {model_id}  обучена: {response.json()}")
        except httpx.HTTPStatusError as e:
            logging.info(f"train_model - Ошибка при обучении модели {model_id}: {e.response.json()}")


async def get_list_models():
    """
    Список всех моделей.
    """
    logging.info("get_list_models - Получение списка моделей...")
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(f"{BASE_URL}/models/")
            response.raise_for_status()
            logging.info(f"get_list_models - Список моделей: {response.json()}")
            return response.json()
        except httpx.HTTPStatusError as e:
            logging.info(f"get_list_models - Ошибка при получении списка моделей: {e.response.json()}")


async def load_model(model_id: str):
    """
    Загрузка модели на сервере.
    """
    logging.info(f" load_model - Загрузка модели {model_id}...")
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(f"{BASE_URL}/models/load", json={"id": model_id})
            response.raise_for_status()
            logging.info(f"load_model - Модель {model_id} загружена: {response.json()}")
            return False
        except httpx.HTTPStatusError as e:
            logging.info(f"load_model - Ошибка при загрузке модели {model_id}: {e.response.json()}")
            return True


async def unload_model(model_id: str) -> bool:
    """
    Выгрузка модели.
    """
    logging.info("unload_model - Выгрузка модели...")
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(f"{BASE_URL}/models/unload", json={"id": model_id})
            response.raise_for_status()
            logging.info(f"unload_model - Модель выгружена: {response.json()}")
            return False
        except httpx.HTTPStatusError as e:
            logging.info(f"unload_model - Ошибка при выгрузке модели: {e.response.json()}")
            return True


async def predict_model(id: str, X: List[Any]) -> Any:
    """
    Predict модели.
    """
    logging.info("predict_model - Predict модели...")
    data_json = {"X": X}
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(f"{BASE_URL}/models/predict/{id}", json=data_json)
            response.raise_for_status()
            logging.info(f"Predict Модели {id}: {response.json()}")
            return response.json()
        except httpx.HTTPStatusError as e:
            logging.info(f"Ошибка при Predict модели {id}: {e.response.json()}")
            return None


async def remove_model(id: str) -> bool:
    """
    Запрос на удаление модели.
    """
    logging.info(f"Удаление модели remove_model id: {id}")
    async with httpx.AsyncClient() as client:
        try:
            response = await client.delete(f"{BASE_URL}/models/remove/{id}")
            response.raise_for_status()
            return None
        except httpx.HTTPStatusError as e:
            logging.info(f"Ошибка при remove_model запросе {id}: {e.response.json()}")
            return True


async def remove_all_models():
    """
     Запрос на remove_all_models.
    """
    logging.info(f"Удаление всех моделей delete_all_models")
    async with httpx.AsyncClient() as client:
        try:
            response = await client.delete(f"{BASE_URL}/models/remove_all")
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            logging.info(f"Ошибка при удалении всех моделей delete_all_models: {e.response.json()}")


async def get_background_tasks() -> List[Any]:
    """
    Запрос на получение всех background_tasks.
    """
    logging.info(f"Получение всех background_tasks")
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(f"{BASE_URL}/tasks/")
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            logging.info(f"Ошибка при получении всех background_tasks: {e.response.json()}")
            return []



