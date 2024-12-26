import numpy as np
from sklearn.datasets import make_classification


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
