from datetime import datetime

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
from client import get_background_tasks, train_model, get_list_models, remove_all_models
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


def learn_logistic_regression(data, penalty='none', C='1.0', solver='liblinear', max_iter=1000):
    model_params = {
        'penalty': penalty,
        'C': C,
        'solver': solver,
        'max_iter': max_iter,
        'random_state': 42,
    }

    try:
        asyncio.run(train_model(data))
        return None
    except Exception as e:
        logging.info(f"Ошибка обучения logistic_regression модели: {str(e)} Параметры: {model_params}")
        return True


def learn_LinearSVC_regression(data, C='1.0', penalty='l2', loss='squared_hinge', dual=True, class_weight=None, max_iter=1000):
    try:
        y = data['toxic']
        X_raw = data.drop('toxic', axis=1)

        model_params = {
            'C': C,
            'penalty': penalty,
            'loss': loss,
            'dual': dual,
            'class_weight': class_weight,
            'max_iter': max_iter,
            'random_state': 42
        }

        if y.nunique() > 2:
            y = pd.cut(y, bins=[-float('inf'), 0.5, float('inf')], labels=[0, 1])

        cat_features_mask = (X_raw.dtypes == "object").values

        X_train, X_test, y_train, y_test = train_test_split(X_raw, y, test_size=0.25, random_state=123)

        # Преобразование числовых столбцов
        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),  # Замена пропусков на среднее
            ('scaler', StandardScaler())  # Масштабирование признаков
        ])

        # Преобразование категориальных столбцов
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='NA')),  # Замена пропусков на 'NA'
            ('onehot', OneHotEncoder(handle_unknown='ignore'))  # OHE-кодирование
        ])

        # Объединяем преобразования с помощью ColumnTransformer
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, X_raw.columns[~cat_features_mask]),
                ('cat', categorical_transformer, X_raw.columns[cat_features_mask])
            ]
        )

        # Полный пайплайн с линейной регрессией
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', LinearSVC(**model_params))
        ])

        # 3. Обучение модели
        pipeline.fit(X_train, y_train)

        # 4. Оценка модели
        y_pred = pipeline.predict(X_test)
        f1 = f1_score(y_test, y_pred)
        accuracy = pipeline.score(X_test, y_test)

        logging.info(f"Модель LinearSVC обучена. Параметры: {model_params}")
        logging.info(f"f1 LinearSVC: {f1:.2f}")
        logging.info(f"Точность модели: {accuracy:.2f}")

        return f1, accuracy

    except Exception as e:
        logging.info(f"Ошибка обучения LinearSVC модели: {str(e)} Параметры: {model_params}")
        return None, None


def learn_naive_bayes(data, alpha=1.0, fit_prior=True):
    try:
        y = data['toxic']
        X_raw = data.drop('toxic', axis=1)

        # Параметры модели MultinomialNB
        model_params = {
            'alpha': alpha,  # Параметр сглаживания
            'fit_prior': fit_prior  # Признак, указывающий следует ли использовать предположения о вероятностностях принадлежности к классам
        }

        # Преобразование целевой переменной (бинаризация, если необходимо)
        if y.nunique() > 2:
            y = pd.cut(y, bins=[-float('inf'), 0.5, float('inf')], labels=[0, 1])

        cat_features_mask = (X_raw.dtypes == "object").values

        X_train, X_test, y_train, y_test = train_test_split(X_raw, y, test_size=0.25, random_state=123)

        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),  # Замена пропусков на среднее значение
            ('scaler', StandardScaler())  # Масштабирование признаков
        ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='NA')),  # Замена пропусков на 'NA'
            ('onehot', OneHotEncoder(handle_unknown='ignore'))  # One-hot кодирование
        ])

        # Объединяем преобразования с помощью ColumnTransformer
        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', categorical_transformer, X_raw.columns[cat_features_mask])  # Для категориальных признаков
            ]
        )

        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', MultinomialNB(**model_params))  # Используем MultinomialNB
        ])

        # Обучаем модель
        pipeline.fit(X_train, y_train)

        # Оценка модели
        y_pred = pipeline.predict(X_test)
        f1 = f1_score(y_test, y_pred)
        accuracy = pipeline.score(X_test, y_test)

        logging.info(f"Модель Naive Bayes (MultinomialNB) обучена. Параметры: {model_params}")
        logging.info(f"f1 Naive Bayes (MultinomialNB): {f1:.2f}")
        logging.info(f"accuracy Naive Bayes (MultinomialNB): {accuracy:.2f}")

        return f1, accuracy

    except Exception as e:
        logging.info(f"Ошибка обучения Naive Bayes модели: {str(e)} Параметры: {model_params}")
        return None


def map_background_tasks() -> pd.DataFrame:
    res = asyncio.run(get_background_tasks())
    df = pd.DataFrame(res)
    df = df.drop(columns=["uuid"])
    df["updated_at"] = df["updated_at"].apply(
        lambda x: datetime.fromisoformat(x).strftime("%d %B %Y, %H:%M:%S")
    )
    df = df.rename(columns={"name": "Название задачи", "status": "Статус", "result_msg": "Актуальный статус результата", "updated_at": "Дата и время последнего изменения"})
    return df


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