from typing import Any, Callable, List, Tuple, Union

import numpy as np
import scipy.stats as stats

from . import scaler


def linear(min: float = 0.0, max: float = 1.0, size: int = 50) -> np.ndarray:
    """
    min에서 max(포함)까지 size 크기의 선형 분포 생성

    Args:
        min (float, optional): 최소값인 실수. Defaults to 0.0.
        max (float, optional): 최대값인 실수. Defaults to 1.0.
        size (int, optional): 선형 분포의 크기. Defaults to 50.

    Returns:
        np.ndarray: min - max까지의 선형 분포
    """
    return np.linspace(start=min, stop=max, num=size, endpoint=True)


def normal(max: float = None, size: int = 50) -> np.ndarray:
    """
    0과 max(None인 경우 1)사이의 무작위 정규 분포를 생성한다.

    Args:
        max (float, optional): 무작위 정규 분포의 최댓값. Defaults to None.
        size (int, optional): 무작위 정규 분포의 크기. Defaults to 50.

    Returns:
        np.ndarray: 0과 max 사이의 무작위 정규 분포
    """
    basic = np.random.normal(0, 1, size)
    if max is not None:
        dist = scaler.min_max_scaling(basic) * max
    else:
        dist = basic
    return dist


def f(d1: int = 10, d2: int = 10, size: int = 100) -> np.ndarray:
    """
    무작위 f 분포를 생성한다.

    Args:
        d1 (int, optional): 분자 자유도(Numerator Degrees of Freedom). Defaults to 10.
        d2 (int, optional): 분모 자유도(Denominator Degrees of Freedom). Defaults to 10.
        size (int, optional): 무작위 f 분포의 크기. Defaults to 100.

    Returns:
        np.ndarray: 무작위 f 분포
    """
    return stats.f.rvs(d1, d2, size=size)
