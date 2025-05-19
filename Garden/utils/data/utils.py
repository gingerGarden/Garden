from typing import List
import numpy as np
from collections import defaultdict


def get_duplicated_index(array: np.ndarray, unique_idx: bool = False) -> List[int]:
    """
    주어진 array에 대하여, 중복된 값들의 인덱스를 반환한다
    >>> 중복된 값 중 첫 번째 인덱스만 살리고, 나머지를 중복된 인덱스로 분류
    >>> array의 원소가 string인 경우, 성능이 높을 수 있음

    Args:
        array (np.ndarray): 입력 array
        unique_idx (bool, optional): True일 경우, 중복되지 않은 요소의 인덱스를 반환. Defaults to False.

    Returns:
        List[int]: 중복된 또는 고유한 요소들의 인덱스 리스트
    """
    positions = defaultdict(list)
    for idx, val in enumerate(array):
        positions[val].append(idx)
        
    # 중복된 idx (앞에 있는 index 살림)
    duplicated = [i for v in positions.values() if len(v) > 1 for i in v[1:]]
    
    if unique_idx:
        # 중복 제거한 idx
        return list(set(range(len(array))) - set(duplicated))
    return duplicated