from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import r2_score
import pandas as pd
import os
import logging

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
    return {"target", "comment_text"}.issubset(df.columns)


def learn_logistic_regression(data):
    y = data['target']
    X_raw = data.drop('target', axis=1)

    # Convert continuous target to binary classes (example)
    if y.nunique() > 2:
        y = pd.cut(y, bins=[-float('inf'), 0.5, float('inf')], labels=[0, 1])

    cat_features_mask = (X_raw.dtypes == "object").values

    model = LogisticRegression()
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
        ('classifier', LogisticRegression(max_iter=1000, random_state=42))
    ])

    # 3. Обучение модели
    pipeline.fit(X_train, y_train)

    # 4. Оценка модели
    y_pred = pipeline.predict(X_test)
    r2 = r2_score(y_test, y_pred)

    logging.info(f"Модель logistic_regression обучена")
    logging.info(f"R2 logistic_regression: {r2}")

    return r2


def learn_LinearSVC_regression(data):
    y = data['target']
    X_raw = data.drop('target', axis=1)

    # Convert continuous target to binary classes (example)
    if y.nunique() > 2:
        y = pd.cut(y, bins=[-float('inf'), 0.5, float('inf')], labels=[0, 1])

    cat_features_mask = (X_raw.dtypes == "object").values

    model = LogisticRegression()
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
        ('classifier', LinearSVC())
    ])

    # 3. Обучение модели
    pipeline.fit(X_train, y_train)

    # 4. Оценка модели
    y_pred = pipeline.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    print(f'R2: {r2}')
    logging.info(f"Модель LinearSVC обучена")
    logging.info(f"R2 LinearSVC: {r2}")

    return r2