import httpx
from typing import List
from process_data import logging

BASE_URL = "http://127.0.0.1:8000/api/v1"


async def train_model(client: httpx.AsyncClient, model_id: str, model_type: str, X: List[List[float]], y: List[float]):
    """
    Отправка запроса на обучение модели.
    """
    print(f"Начало обучения модели {model_id} ({model_type})...")
    try:
        response = await client.post(
            f"{BASE_URL}/fit",
            json=[
                {
                    "X": X,
                    "y": y,
                    "config": {
                        "id": model_id,
                        "ml_model_type": model_type,
                        "hyperparameters": {"fit_intercept": True}
                    }
                }
            ],
        )
        response.raise_for_status()
        print(f"Модель {model_id} обучена: {response.json()}")
    except httpx.HTTPStatusError as e:
        print(f"Ошибка при обучении модели {model_id}: {e.response.json()}")


async def list_models(client: httpx.AsyncClient):
    """
    Список всех моделей.
    """
    print("Получение списка моделей...")
    try:
        response = await client.get(f"{BASE_URL}/list_models")
        response.raise_for_status()
        print(f"Список моделей: {response.json()}")
    except httpx.HTTPStatusError as e:
        print(f"Ошибка при получении списка моделей: {e.response.json()}")


async def load_model(client: httpx.AsyncClient, model_id: str):
    """
    Загрузка модели на сервере.
    """
    print(f"Загрузка модели {model_id}...")
    try:
        response = await client.post(f"{BASE_URL}/load", json={"id": model_id})
        response.raise_for_status()
        print(f"Модель {model_id} загружена: {response.json()}")
    except httpx.HTTPStatusError as e:
        print(f"Ошибка при загрузке модели {model_id}: {e.response.json()}")


async def unload_model(client: httpx.AsyncClient, model_id: str):
    """
    Выгрузка модели.
    """
    print("Выгрузка модели...")
    try:
        response = await client.post(f"{BASE_URL}/unload", json={"id": model_id})
        response.raise_for_status()
        print(f"Модель выгружена: {response.json()}")
    except httpx.HTTPStatusError as e:
        print(f"Ошибка при выгрузке модели: {e.response.json()}")


async def predict_model(client: httpx.AsyncClient, id: str, X: List[List[float]]):
    """
    Predict модели.
    """
    print("Predict модели...")
    try:
        response = await client.post(f"{BASE_URL}/predict", json={"id": "linear_1231", "X": X})
        response.raise_for_status()
        print(f"Predict Модели {id}: {response.json()}")
    except httpx.HTTPStatusError as e:
        print(f"Ошибка при Predict модели {id}: {e.response.json()}")


async def remove_model(client: httpx.AsyncClient, id: str):
    """
    Запрос на удаление модели.
    """
    logging.info(f"Удаление модели remove_model id: {id}")
    try:
        response = await client.delete(f"{BASE_URL}/remove/{id}")
        response.raise_for_status()
    except httpx.HTTPStatusError as e:
        logging.info(f"Ошибка при remove_model запросе {id}: {e.response.json()}")


async def remove_all_models(client: httpx.AsyncClient):
    """
     Запрос на remove_all_models.
    """
    logging.info(f"Удаление всех моделей delete_all_models")
    try:
        response = await client.delete(f"{BASE_URL}/remove_all")
        response.raise_for_status()
    except httpx.HTTPStatusError as e:
        logging.info(f"Ошибка при удалении всех моделей delete_all_models: {e.response.json()}")


async def get_background_tasks():
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



