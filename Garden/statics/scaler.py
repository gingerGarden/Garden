import numpy as np


def small_denominator(denominator: float, very_small_value=1e-8) -> float:
    """
    분모가 0인 경우, 매우 작은 값을 출력한다.

    Args:
        denominator (float): 실수인 분모
        very_small_value (float, optional): 분모가 0인 경우 출력되는 매우 작은 실수. Defaults to 1e-8.

    Returns:
        float: 분모가 0인 경우, very_small_value 출력
    """
    return very_small_value if denominator == 0 else denominator


def min_max(array: np.ndarray) -> np.ndarray:
    """
    넘파이 배열을 min-max scaling하여, 0과 1 사이의 실수로 변환한다.
    >>> max와 min이 동일한 경우, 나누기 연산을 방지하기 위해 아주 작은 값을 대신 사용.

    Args:
        array (np.ndarray): 변환하고자 하는 넘파이 배열

    Returns:
        np.ndarray: 0과 1 사이의 값으로 변환된 배열
    """
    min_score = np.min(array)
    max_score = np.max(array)
    denominator = small_denominator(max_score - min_score)  # 분모가 0인지 확인
    return (array - min_score) / denominator


def min_max_range_normalization(array: np.ndarray, min: float, max: float) -> np.ndarray:
    """
    min-max scaling의 공식을 확장을 이용해 데이터의 범위를 [min, max]로 변환
    >>> scaled_arr = ((arr - arr_min) / (arr_max - arr_min)) * (max_v - min_v) + min_v

    Args:
        array (np.ndarray): scaling 하고자 하는 넘파이 배열
        min (float): 변환하고자 하는 최솟값
        max (float): 변환하고자 하는 최댓값

    Returns:
        np.ndarray: [min, max]의 범위로 변환된 넘파이 배열
    """
    array_min = np.min(array)
    array_max = np.max(array)
    denominator = small_denominator(array_max - array_min)
    return (((array - array_min) / denominator) * (max - min)) + min


def ln(array: np.ndarray, very_small_value: float = 1e-8) -> np.ndarray:
    """
    입력된 배열을 자연로그 치환한다

    Args:
        array (np.ndarray): 대상 배열열
        very_small_value (float, optional): 0을 log 치환하는 것을 피하기 위해 더하는 매우 작은 값. Defaults to 1e-8.

    Returns:
        np.ndarray: 자연 로그 치환된 배열
    """
    min_score = np.min(array)
    if min_score < 0:
        array += np.abs(min_score) + very_small_value
    elif min_score == 0:
        array += very_small_value
    return np.log(array)


def robust(array: np.ndarray, q1_ratio: float = 0.25, q3_ratio: float = 0.75) -> np.ndarray:
    """
    Robust Scaling: 이상치에 민감하지 않은 데이터 정규화 기법으로, 사분위 범위(IQT)를 사용하여 이상치의 영향을 최소화하여 스케일링.
    >>> 중앙값과 IQR(Interquartile Range)를 사용하여 데이터를 보다 안정적으로 스케일링
        - 이상치가 많은 데이터에 적합
    >>> 분모가 0일 경우, 매우 작은 값을 추가하여 분모가 0이 되는 것을 방지

    Args:
        array (np.ndarray): 스케일링할 데이터의 배열(1차원 또는 다차원 numpy 배열).
        q1_ratio (float, optional): 1사분위 비율. Defaults to 0.25.
        q3_ratio (float, optional): 3사분위 비율. Defaults to 0.75.

    Returns:
        np.ndarray: 중앙값을 기준으로 정규화하고, IQR로 나눠 스케일링된 배열
    """
    iqr = np.quantile(array, q3_ratio) - np.quantile(array, q1_ratio)
    iqr = small_denominator(iqr)  # 분모가 0인지 확인
    q2_median = np.median(array)
    return (array - q2_median) / iqr


def max_abs(array: np.ndarray) -> np.ndarray:
    """
    MaxAbsScaler: 입력 배열의 값들을 절대값 기준으로 최대 절대값이 1.0이 되도록 스케일링.
    >>> Sparsity(희소성) 유지:
        - 데이터에 0이 포함된 경우, 변환 후에도 0을 유지
        - 스파스 행렬(대부분의 요소가 0인 행렬)의 희소성을 파괴하지 않을 수 있음.
        - 텍스트 분류, 클러스터링 등에서 중요한 의미를 가질 수 있음.
    >>> 분모가 0일 경우, 매우 작은 값을 추가하여 분모가 0이 되는 것을 방지

    Args:
        array (np.ndarray): 스케일링할 입력 데이터 배열, 1차원 또는 다차원 numpy 배열.

    Returns:
        np.ndarray: 각 값의 절대값이 1이하로 스케일링된 numpy 배열
    """
    denominator = small_denominator(np.max(np.abs(array)))  # 분모가 0인지 확인
    return array / denominator
