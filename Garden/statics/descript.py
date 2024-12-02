from typing import Union

import numpy as np
import pandas as pd

from .utils import show_markdown_df


# 빈도표 출력
def frequency_table(
    array: Union[np.ndarray, pd.Series],
    num_digits: int = 3,
    percent: bool = False,
    markdown: bool = False,
) -> pd.DataFrame:
    """
    주어진 배열 또는 시리즈의 빈도표를 생성하는 함수

    Args:
        array (np.ndarray): 빈도표를 만들 배열이나 시리즈
        num_digits (int, optional): 비율 또는 퍼센트를 반올림할 소수점 자릿수. Defaults to 3.
        percent (bool, optional): 비율을 퍼센트로 변환하여 출력할지 여부. Defaults to False.
        markdown (bool, optional): table을 markdown으로 보여줄지 여부. Defaults to False.

    Raises:
        ValueError: array가 비어있는 경우 예외가 발생합니다.

    Returns:
        pd.DataFrame: 클래스, 빈도, 비율(또는 퍼센트)가 포함된 빈도표.
    """
    size = len(array)
    if size == 0:
        raise ValueError("빈 배열은 처리할 수 없습니다.")  # 경고문

    # 기본적인 빈도표 생성
    class_array, freq_array = np.unique(array, return_counts=True)
    freq_df = pd.DataFrame(
        {"class": class_array, "frequency": freq_array, "ratio": freq_array / size}
    )
    freq_df = freq_df.sort_values("class")
    freq_df = _add_total_record_for_frequency_table(freq_df, size)  # total record 추가

    # 퍼센트로 변경 여부
    if percent:
        _ratio_to_percent_in_frequency_table(freq_df, round=num_digits)
    else:
        freq_df["ratio"] = freq_df["ratio"].round(num_digits)

    if markdown:
        show_markdown_df(df=freq_df)  # markdown으로 출력할지 여부

    return freq_df


# 교차표 출력
def cross_table(
    df: pd.DataFrame,
    col1: str,
    col2: str,
    vertical: bool = True,
    percent: bool = True,
    ratio_style: int = 0,
    num_digits: int = 2,
    margin_txt: str = "total",
    markdown: bool = False,
):
    """
    교차표를

    Args:
        df (pd.DataFrame): 교차표 생성의 대상이 되는 pd.DataFrame
        col1 (str): 행 축에 들어갈 컬럼
        col2 (str): 열 축에 들어갈 컬럼
        vertical (bool, optional): 비율을 행 기준으로 생성할지 여부(False이면 열 기준으로 생성). Defaults to True.
        percent (bool, optional): 비율을 퍼센트로 변경할지 여부(False이면 비율로 생성). Defaults to True.
        ratio_style (int, optional): 비율을 어떤 방식으로 출력할지. Defaults to 0.
            - 0: 비율(퍼센트)를 교차표의 빈도 옆에 괄호로 추가
            - 1: 비율(퍼센트)를 교차표에 추가하지 않음
            - 2: 비율(퍼센트)만 교차표로 출력
        round (int, optional): 비율(퍼센트)의 반올림 자릿수. Defaults to 2.
        margin_txt (str, optional): 교차표의 margin(총합)의 컬럼과 인덱스 문자열. Defaults to 'total'.
        markdown (bool, optional): table을 markdown으로 보여줄지 여부. Defaults to False.

    Returns:
        pd.DataFrame: 교차표
    """
    _margin_check_for_cross_table(
        df, col1, col2, margin_txt
    )  # col1과 col2에 margin_txt가 있는지 확인

    # 기본적인 교차표 생성
    cross_df = pd.crosstab(df[col1], df[col2], margins=True, margins_name=margin_txt)
    # 비율 생성
    if ratio_style != 1:
        ratio_mat = _make_ratio_mat_for_cross_table(
            cross_df, vertical, percent, margin_txt
        )  # 비율(퍼센트) 행렬 계산
        ratio_mat = _ratio_style_for_cross_table(
            ratio_mat, percent, num_digits, ratio_style
        )  # 반올림 및 교차표에 맞게 형태 정의

        # 비율을 교차표에 붙인다.
        if ratio_style == 0:
            cross_df = _convert_mat_to_cross_df_style_for_cross_table(
                cross_df, mat=np.char.add(cross_df.values.astype("<U9"), ratio_mat)
            )
        # 비율을 교차표로 내보낸다.
        if ratio_style == 2:
            cross_df = _convert_mat_to_cross_df_style_for_cross_table(
                cross_df, mat=ratio_mat
            )

    if markdown:
        show_markdown_df(df=cross_df)  # markdown으로 출력할지 여부

    return cross_df


# frequency_table() 하위 함수
# ==================================================
def _add_total_record_for_frequency_table(
    freq_df: pd.DataFrame, size: int
) -> pd.DataFrame:
    """
    빈도표에 전체 합계 행을 추가하는 함수

    Args:
        freq_df (pd.DataFrame): frequency_table() 함수를 통해 생성된 빈도표 데이터프레임
        size (int): 빈도표의 대상이 되는 array나 시리즈의 크기

    Returns:
        pd.DataFrame: 합계 행이 추가된 빈도표
    """
    total_record = pd.DataFrame(
        [("total", size, 1.0)], columns=["class", "frequency", "ratio"]
    )
    return pd.concat([freq_df, total_record]).reset_index(drop=True)


def _ratio_to_percent_in_frequency_table(
    freq_df: pd.DataFrame, num_digits: int
) -> None:
    """
    빈도표의 비율을 퍼센트로 변환하는 함수

    Args:
        freq_df (pd.DataFrame): frequency_table() 함수를 통해 생성된 빈도표 데이터프레임
        round (int): 퍼센트를 반올림할 소수점 자릿수
    """
    freq_df["ratio"] = freq_df["ratio"] * 100
    freq_df.rename(columns={"ratio": "%"}, inplace=True)
    freq_df["%"] = freq_df["%"].round(num_digits)


# ==================================================


# cross_table() 하위 함수
# ==================================================
def _margin_check_for_cross_table(
    df: pd.DataFrame, col1: str, col2: str, margin_txt: str
):
    """
    교차표의 대상이 되는 column들의 element에 margin_txt와 동일한 문자가 있는 경우, ValueError 발생

    Args:
        df (pd.DataFrame): 교차표의 대상이 되는 DataFrames
        col1 (str): 교차표의 행 축 컬럼
        col2 (str): 교차표의 열 축 컬럼
        margin_txt (str): margin(total)에 들어가는 txt

    Raises:
        ValueError: margin에 대한 명칭이 class에 이미 포함되어 있습니다. margin_txt를 바꾸시오.
    """
    col1_class = np.unique(df[col1])
    col2_class = np.unique(df[col2])
    if (margin_txt in col1_class) | (margin_txt in col2_class):
        raise ValueError(
            f"{margin_txt}가 이미 {col1} 또는 {col2}에 포함되어 있습니다. margin_txt 값을 변경하십시오."
        )


def _make_ratio_mat_for_cross_table(
    cross_df: pd.DataFrame, vertical: bool, percent: bool, margin_txt: str
) -> np.ndarray:
    """
    교차표의 비율 행렬 생성

    Args:
        cross_df (pd.DataFrame): 교차표의 대상이 되는 DataFrame
        vertical (bool): 비율을 행 기준으로 생성할지 여부(False이면 열 기준으로 생성)
        percent (bool): 비율을 퍼센트로 변경할지 여부(False이면 비율로 생성)
        margin_txt (str): 교차표의 margin(총합)의 컬럼과 인덱스 문자열

    Returns:
        np.ndarray: 교차표의 비율 또는 퍼센트에 대한 행렬
    """
    # 비율 행렬 계산
    standard = (
        cross_df[[margin_txt]].values if vertical else cross_df.loc[margin_txt].values
    )
    ratio_mat = cross_df.values / standard

    if percent:
        ratio_mat = ratio_mat * 100  # 퍼센트 변환 여부
    return ratio_mat


def _ratio_style_for_cross_table(
    ratio_mat: np.ndarray, percent: bool, num_digits: int, ratio_style: int
) -> np.ndarray:
    """
    교차표에 비율을 어떠한 방식으로 추가할지
    >>> ratio_style == 1: 괄호를 붙임
        ratio_style != 1: 괄호를 붙이지 않음

    Args:
        ratio_mat (np.ndarray): 비율(퍼센트) 행렬
        percent (bool): 퍼센트 문자열을 추가할지 여부
        round (int): 반올림 자릿수
        ratio_style (int): 교차표에 비율을 어떠한 방식으로 추가할지

    Returns:
        np.ndarray: _description_
    """
    fstring = f".{num_digits}f"
    left_str = " (" if ratio_style == 0 else ""
    right_str = ("%)" if percent else ")") if ratio_style == 0 else ""
    ratio_vt = [left_str + format(i, fstring) + right_str for i in ratio_mat.flatten()]
    return np.array(ratio_vt).reshape(ratio_mat.shape)


def _convert_mat_to_cross_df_style_for_cross_table(
    cross_df: pd.DataFrame, mat: np.ndarray
) -> pd.DataFrame:
    """
    교차표로부터 산출된 행렬을 교차표와 동일한 형태로 변환

    Args:
        cross_df (pd.DataFrame): 기준이 되는 교차표
        mat (np.ndarray): 교차표로부터 산출된 행렬

    Returns:
        pd.DataFrame: 교차표의 형식으로 만들어진 행렬
    """
    return pd.DataFrame(mat, index=cross_df.index, columns=cross_df.columns)


# ==================================================
