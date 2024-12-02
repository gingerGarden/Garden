from typing import Union

import numpy as np


def binary_probability(p: float) -> bool:
    """
    p의 확률로 true를 반환

    Args:
        p (float): True가 반환될 확률

    Returns:
        bool: True or False
    """
    return np.random.choice([True, False], p=[p, 1 - p], size=1).item()


def choose_one(array: np.ndarray) -> Union[float, int, str]:
    """
    array에서 무작위로 한개의 원소 반환

    Args:
        array (np.ndarray): 1개 이상의 원소로 이루어진 1차원 또는 다차원 numpy 배열

    Returns:
        Union[float, int, str]: 무작위로 선택된 값. 선택된 값은 float, int, str 등의 다양한 타입일 수 있다.
    """
    return np.random.choice(array, size=1).item()
