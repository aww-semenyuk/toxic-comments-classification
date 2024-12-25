from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.linear_model import LinearRegression, LogisticRegression

from exceptions import InvalidFitPredictDataError
from serializers.trainer import MLModelConfig, MLModelType


def prepare_fit_data() -> tuple[np.ndarray, np.ndarray]:
    X, y = make_classification(
        n_samples=10000000,
        n_features=20,
        n_classes=2,
        random_state=42
    )
    return X, y


def prepare_predict_data() -> np.ndarray:
    X, _ = make_classification(
        n_samples=100,
        n_features=20,
        n_classes=2,
        random_state=42
    )
    return X


def train_and_save_model_task(
    models_dir_path: Path,
    model_config: MLModelConfig,
    fit_dataset: pd.DataFrame
) -> Path:
    # TODO: Заменить затычку на настоящую функцию (1)
    X, y = prepare_fit_data()

    model_id = model_config.id
    model_type = model_config.ml_model_type
    try:
        if model_type == MLModelType.linear_regression:
            model = LinearRegression(**model_config.hyperparameters)
        else:
            model = LogisticRegression(**model_config.hyperparameters)
        model.fit(X, y)
    except ValueError as e:
        raise InvalidFitPredictDataError(e.args[0])

    safe_model_id = "".join(
        char for char in model_id if char.isalnum() or char in ('-', '_')
    ).rstrip()
    model_file_path = models_dir_path / f"{safe_model_id}.pkl"
    joblib.dump(model, model_file_path)

    return model_file_path
