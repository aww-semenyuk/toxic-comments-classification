from datetime import datetime
from typing import Any

import pandas as pd

import logging
import asyncio
import zipfile
import io
import hashlib
import time

from utils_func.client import (
    get_background_tasks,
    train_model,
    get_list_models,
    remove_all_models,
    remove_model,
    load_model,
    unload_model,
    predict_model,
    predict_scores_model
)


def is_data_correct(df):
    """
    Проверяет датафрейм на корректность структуры.

    Args:
        df (pandas.DataFrame): Датафрейм с данными.

    Returns:
        bool: True, если столбцы "toxic" и "comment_text" есть, иначе False.
    """
    return {"toxic", "comment_text"}.issubset(df.columns)


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


def learn_logistic_regression(
        data,
        penalty='none',
        C='1.0',
        solver='liblinear',
        max_iter=1000
):
    model_params = {
        'penalty': penalty,
        'C': C,
        'solver': solver,
        'max_iter': max_iter,
    }

    hash_id = generate_random_hash()
    model_id = f"{hash_id}_logistic"

    try:
        asyncio.run(
            train_model(
                data,
                model_id=model_id,
                model_type='logistic_regression',
                model_params=model_params
            )
        )
        return None
    except Exception as e:
        logging.info(
            f"Ошибка обучения logistic_regression модели: err:{str(e)} "
            f"Параметры: {model_params}")
        return True


def learn_LinearSVC_regression(
        data,
        C='1.0',
        penalty='l2',
        loss='squared_hinge',
        dual=True,
        class_weight=None,
        max_iter=1000
):

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
        asyncio.run(
            train_model(
                data,
                model_id=model_id,
                model_type='linear_svc',
                model_params=model_params
            )
        )
        return None

    except Exception as e:
        logging.info(
            f"Ошибка обучения LinearSVC модели err:{str(e)} "
            f"Параметры: {model_params}"
        )
        return True


def learn_naive_bayes(data, alpha=1.0, fit_prior=True):

    # Параметры модели MultinomialNB
    model_params = {
        'alpha': alpha,  # Параметр сглаживания
        'fit_prior': fit_prior
    }

    hash_id = generate_random_hash()
    model_id = f"{hash_id}_naive_bayes"

    try:
        asyncio.run(
            train_model(
                data,
                model_id=model_id,
                model_type='multinomial_naive_bayes',
                model_params=model_params
            )
        )
        return None
    except Exception as e:
        logging.info(
            f"Ошибка обучения Naive Bayes модели err:{str(e)} "
            f"Параметры: {model_params}"
        )
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
    df = df.rename(
        columns={
            "name": "Название задачи",
            "status": "Статус",
            "result_msg": "Актуальный статус результата",
            "updated_at": "Дата и время последнего изменения"
        }
    )
    return df


# Функция, которая будет вызываться при нажатии кнопки
def delete_action(row_id) -> str:
    err = asyncio.run(remove_model(row_id))
    return err


def load_model_action(row_id) -> str:
    err = asyncio.run(load_model(row_id))
    return err


def unload_model_action(row_id) -> str:
    err = asyncio.run(unload_model(row_id))
    return err


def map_current_models() -> pd.DataFrame:
    res = asyncio.run(get_list_models())
    df = pd.DataFrame(res)

    df = df.rename(
        columns={
            "is_trained": "Модель обучена",
            "is_loaded": "Модель загружена",
            "type": "Тип модели"
        }
    )
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


def escape_quotes(text: str) -> str:
    """
    Экранирует кавычки внутри текста.

    Args:
        text (str): Входной текст.

    Returns:
        str: Текст с экранированными кавычками.
    """
    return text.replace('"', '\\"').replace("'", "\\'")


def predict_action(model_id, text) -> Any:
    formatted_text = escape_quotes(text)
    res = asyncio.run(predict_model(model_id, [formatted_text]))
    if res is not None and res.get('predictions') is not None:
        return res.get('predictions')[0]
    return None


def predict_scores_action(models_id, zipped_csv) -> Any:
    models_id_str = ",".join(map(str, models_id))
    csv_file = asyncio.run(predict_scores_model(models_id_str, zipped_csv))
    if csv_file is not None:
        return pd.read_csv(csv_file)
    return None


def format_df(df) -> Any:
    df_to_display = df.drop(columns=["Unnamed: 0"])
    return df_to_display
