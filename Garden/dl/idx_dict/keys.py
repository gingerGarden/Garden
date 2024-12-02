from typing import List, Union

import numpy as np
import pandas as pd


def make_basic_key_df(
    paths: Union[List[str], np.ndarray],
    labels: Union[List[str], np.ndarray],
    use_specific_column: bool = True,
    **kwargs: Union[List[str], np.ndarray],
) -> pd.DataFrame:
    """가장 기본적인 형태의 key_df를 생성한다 - path, label 2개의 컬럼으로 구성
    >>> path: 이미지 파일의 절대 경로
        label: 이미지 파일의 label

    Args:
        paths (Union[List[str], np.ndarray]): 파일의 절대 경로들을 list 또는 array로 입력
        labels (Union[List[str], np.ndarray]): 파일의 label을 list 또는 array로 입력
        use_specific_column (bool, optional): kwargs로 추가되는 컬럼을 make_basic_key_df의 고유 컬럼명으로 추가할지 여부. Defaults to True.
        **kwargs (Union[List[str], np.ndarray]): 추가적으로 들어갈 class(데이터 분할 시 추가 속성)
            - use_specific_column=False인 경우, 지정한 파라미터명이 새로운 컬럼명이 된다.

    Returns:
        pd.DataFrame: path, label 및 kwargs로 전달된 추가 class 컬럼이 포함된 DataFrame
    """
    data = {"path": paths, "label": labels}

    # kwargs로 전달된 데이터 추가
    for i, (column, value) in enumerate(kwargs.items()):
        if use_specific_column:
            data[f"class_{i}"] = value
        else:
            data[column] = value
    return pd.DataFrame(data)


def binary_label_convertor(
    array: Union[np.ndarray, pd.Series], positive_class: Union[str, int]
) -> np.ndarray:
    """
    array를 positive_class를 1로, 나머지는 0으로 이진 변환한다.

    Args:
        array (Union[np.ndarray, pd.Series]): positive_class와 다른 class들이 포함된 시리즈 또는 array
        postive_class (Union[str, int]): 이진 분류 시 기준(1)이 될 element의 class

    Returns:
        np.ndarray: 0과 1로 이진화된 1차원 배열
    """
    return np.where(array == positive_class, 1, 0)


def add_multiclass_onehot_columns(
    df: pd.DataFrame, column: str, key: str = "dummy", inplace: bool = False
) -> pd.DataFrame:
    """
    multiclass에 대하여 onehot embedding 결과를 기존 df에 추가하거나, 새로 출력함

    Args:
        df (pd.DataFrame): multiclass가 포함된 pd.DataFrame
        column (str): df에서 multiclass를 갖고 있는 column명
        key (str, optional): multiclass를 onehot column으로 생성 시, 추가되는 키(f"{column}_{key}"). Defaults to 'dummy'.
        inplace (bool, optional): 원본 DataFrame에 바로 반영할지 여부(False일 경우 복사본 반환). Defaults to False.

    Raises:
        ValueError: multiclass로 추가된 컬럼명({", ".join(duplicated_key)})이 기존 df에 이미 존재합니다. dummy의 값을 바꾸십시오.

    Returns:
        pd.DataFrame: multiclass를 onehot embedding하여 column으로 추가한 DataFrame(key_df에 추가 또는 복사)
    """
    df_copy = (
        df.copy(deep=True) if not inplace else df
    )  # inplace에 따라 df에 바로 추가할지 여부를 결정
    dummy_df = pd.get_dummies(
        df_copy[column], prefix=f"{column}_{key}", dtype=int
    )  # dummy 변수 생성

    # dummy_df의 column이 기존 column과 충돌나는 경우 ValueError 발생
    duplicated_key = set(dummy_df.columns) & set(df_copy.columns)
    if duplicated_key:
        columns_txt = ", ".join(duplicated_key)
        raise ValueError(
            f"multiclass로 추가된 컬럼명({columns_txt})이 기존 df에 이미 존재합니다. dummy의 값을 바꾸십시오."
        )

    # 컬럼 추가
    df_copy[dummy_df.columns] = dummy_df
    if not inplace:  # inplace가 False일 경우 df_copy를 따로 출력
        return df_copy
