from datetime import datetime
import hashlib
import time
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import f1_score
import pandas as pd
import os
import logging
from client import get_background_tasks, train_model, get_list_models, remove_all_models, remove_model, load_model, unload_model
import asyncio
import zipfile
import io

# Создаём папку для логов, если она не существует
log_dir = "logs/frontend"
os.makedirs(log_dir, exist_ok=True)

# Настраиваем логгер
log_file = os.path.join(log_dir, "app.log")
logging.basicConfig(
    filename=log_file,
    filemode="a",  # "a" для добавления логов, "w" для перезаписи
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO  # Уровень логирования: DEBUG, INFO, WARNING, ERROR, CRITICAL
)

# Установка обработчика для консоли (дополнительно)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logging.getLogger().addHandler(console_handler)

def is_data_correct(df):
    return {"toxic", "comment_text"}.issubset(df.columns)


import hashlib
import time

def generate_random_hash():
    """
    Генерирует случайный хеш на основе текущего времени.

    Returns:
        str: Первые 8 символов хеша.
    """
    # Получение текущего времени в формате UNIX timestamp
    current_time = str(time.time())

    # Генерация хеша на основе времени
    hash_object = hashlib.sha256(current_time.encode())
    hash_id = hash_object.hexdigest()[:8]

    return hash_id


def learn_logistic_regression(data, penalty='none', C='1.0', solver='liblinear', max_iter=1000):
    model_params = {
        'penalty': penalty,
        'C': C,
        'solver': solver,
        'max_iter': max_iter,
    }

    hash_id = generate_random_hash()
    model_id = f"{hash_id}_logistic"

    try:
        asyncio.run(train_model(data, model_id=model_id, model_type='logistic_regression', model_params=model_params))
        return None
    except Exception as e:
        logging.info(f"Ошибка обучения logistic_regression модели: {str(e)} Параметры: {model_params}")
        return True


def learn_LinearSVC_regression(data, C='1.0', penalty='l2', loss='squared_hinge', dual=True, class_weight=None, max_iter=1000):

    model_params = {
        'C': C,
        'penalty': penalty,
        'loss': loss,
        'dual': dual,
        'class_weight': class_weight,
        'max_iter': max_iter
    }

    hash_id = generate_random_hash()
    model_id = f"{hash_id}_linear_svc"

    try:
        asyncio.run(train_model(data, model_id=model_id, model_type='linear_svc', model_params=model_params))
        return None

    except Exception as e:
        logging.info(f"Ошибка обучения LinearSVC модели: {str(e)} Параметры: {model_params}")
        return True


def learn_naive_bayes(data, alpha=1.0, fit_prior=True):

    # Параметры модели MultinomialNB
    model_params = {
        'alpha': alpha,  # Параметр сглаживания
        'fit_prior': fit_prior  # Признак, указывающий следует ли использовать предположения о вероятностностях принадлежности к классам
    }

    hash_id = generate_random_hash()
    model_id = f"{hash_id}_naive_bayes"

    try:
        asyncio.run(train_model(data, model_id=model_id, model_type='multinomial_naive_bayes', model_params=model_params))
        return None
    except Exception as e:
        logging.info(f"Ошибка обучения Naive Bayes модели: {str(e)} Параметры: {model_params}")
        return True


def map_background_tasks() -> pd.DataFrame:
    res = asyncio.run(get_background_tasks())
    if not res:
        return pd.DataFrame()
    df = pd.DataFrame(res)
    df = df.drop(columns=["uuid"])
    df["updated_at"] = df["updated_at"].apply(
        lambda x: datetime.fromisoformat(x).strftime("%d %B %Y, %H:%M:%S")
    )
    df = df.rename(columns={"name": "Название задачи", "status": "Статус", "result_msg": "Актуальный статус результата", "updated_at": "Дата и время последнего изменения"})
    return df


# Функция, которая будет вызываться при нажатии кнопки
def delete_action(row_id) -> bool:
    err = asyncio.run(remove_model(row_id))
    return err


def load_model_action(row_id) -> bool:
    err = asyncio.run(load_model(row_id))
    return err


def unload_model_action(row_id) -> bool:
    err = asyncio.run(unload_model(row_id))
    return err


def map_current_models() -> pd.DataFrame:
    res = asyncio.run(get_list_models())
    df = pd.DataFrame(res)

    df = df.rename(columns={"is_trained": "Модель обучена", "is_loaded": "Модель загружена", "type": "Тип модели"})
    return df


def create_zip_from_csv(uploaded_file, zip_filename: str) -> bytes:
    """
    Функция для создания ZIP-архива из загруженного CSV-файла.

    :param uploaded_file: Загруженный файл через Streamlit (UploadedFile).
    :param zip_filename: Имя файла в архиве.
    :return: ZIP-архив в виде байтов.
    """
    zip_buffer = io.BytesIO()

    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        csv_content = uploaded_file.getvalue()
        zip_file.writestr(zip_filename, csv_content)

    return zip_buffer.getvalue()


def delete_all_models():
    asyncio.run(remove_all_models())