from typing import Optional

import numpy as np
from sklearn.metrics import accuracy_score


class MetricsClassification:
    def __init__(
        self,
        is_binary: bool = True,
        threshold: Optional[float] = None,
        class_size: Optional[int] = None,
    ):
        self.threshold = threshold

        # Metrics 방법 정의
        if is_binary:
            self.metrics = Binary(threshold=threshold)
        else:
            self.metrics = Multiclass(class_size=class_size)

    def __call__(
        self, predict: np.ndarray, label: np.ndarray, accuracy_only: bool = False
    ):
        return self.metrics(predict, label, accuracy_only)


class Binary:
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def __call__(
        self, predict: np.ndarray, label: np.ndarray, accuracy_only: bool = False
    ):
        predict = np.where(predict >= self.threshold, 1, 0)

        if accuracy_only:
            return accuracy_score(label, predict)


class Multiclass:
    def __init__(self, class_size: int):
        self.class_size = class_size

    def __call__(
        self, predict: np.ndarray, label: np.ndarray, accuracy_only: bool = False
    ):
        pass
