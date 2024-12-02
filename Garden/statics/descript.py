from typing import Union, List, Any

import numpy as np
import pandas as pd

from .utils import show_markdown_df


def frequency_table(
        data: Union[List[Any], np.ndarray, pd.Series], 
        num_digits: int = 2,
        percent: bool = False,
        markdown: bool = False
    ) -> pd.DataFrame:
    """
    입력된 데이터(list, np.ndarray, pd.Series)에 대한 빈도표 출력
        - 'total' record 존재
        - 'ratio' 또는 '%' 컬럼 추가

    Args:
        data (Union[List[Any], np.ndarray, pd.Series]): 빈도표를 출력하고자 하는 범주형 데이터
        num_digits (int, optional): 실수의 소수점 자리 수. Defaults to 2.
        percent (bool, optional): 비율을 퍼센트로 출력 여부. Defaults to False.
        markdown (bool, optional): table을 markdown으로 보여줄지 여부. Defaults to False.

    Returns:
        pd.DataFrame: 빈도, 비율, 합이 포함된 빈도표
    """
    # data의 크기
    size = len(data)
    # data 크기 확인
    _FrequencyTable._check_size(size=size)
    # data 종류를 array로 고정
    data = _FrequencyTable._convert_type(data=data)
    # 기본적인 빈도표 생성
    freq_df = _FrequencyTable._basic_table(array=data, size=size)
    # total record 추가
    freq_df = _FrequencyTable._add_total_record(freq_df=freq_df, size=size)
    # ratio를 퍼센트로 변경 여부
    if percent:
        _FrequencyTable._ratio_to_percent(freq_df=freq_df, num_digits=num_digits)
    else:
        freq_df["ratio"] = freq_df["ratio"].round(num_digits)
    # markdown 출력 여부
    if markdown:
        show_markdown_df(df=freq_df)
    return freq_df
    

class _FrequencyTable:
    @staticmethod
    def _check_size(size: int):
        """
        size(data의 크기)가 0인지 확인

        Args:
            size (int): data의 크기

        Raises:
            ValueError: data의 크기가 0인 경우
        """
        if size == 0:
            raise ValueError("data의 크기가 0입니다. data를 확인하십시오.")
        
    @staticmethod
    def _convert_type(
            data: Union[List[Any], np.ndarray, pd.Series]
        ) -> np.ndarray:
        """
        list, np.ndarray, pd.Series로 입력된 data를 np.ndarray로 변환

        Args:
            data (Union[List[Any], np.ndarray, pd.Series]): 빈도표의 대상이 되는 data

        Raises:
            ValueError: 입력된 data가 list, np.ndarray, pd.Series에 해당하지 않는 경우

        Returns:
            np.ndarray: np.ndarray로 변형된 data
        """
        if isinstance(data, list):
            return np.array(data)
        elif isinstance(data, np.ndarray):
            return data
        elif isinstance(data, pd.Series):
            return data.values
        else:
            raise ValueError(f"data로 {type(data)}가 입력되었습니다. list, np.ndarray, pd.Series만 대상으로 합니다.")
        
    @staticmethod
    def _basic_table(array: np.ndarray, size: int) -> pd.DataFrame:
        """
        가장 기본적인 빈도표 생성

        Args:
            array (np.ndarray): 빈도표의 대상이 되는 array
            size (int): 빈도표의 크기

        Returns:
            pd.DataFrame: 빈도표
        """
        class_arr, freq_arr = np.unique(array, return_counts=True)
        freq_df = pd.DataFrame(
            {"class":class_arr, "frequency": freq_arr, "ratio": freq_arr / size}
        )
        return freq_df.sort_values("class")
    
    @staticmethod
    def _add_total_record(freq_df: pd.DataFrame, size: int) -> pd.DataFrame:
        """
        빈도표에 total 행 추가

        Args:
            freq_df (pd.DataFrame): 기본 빈도표
            size (int): data의 크기

        Returns:
            pd.DataFrame: total 행이 추가된 빈도표
        """
        total_record = pd.DataFrame(
            [("total", size, 1.0)], columns=["class", "frequency", "ratio"]
        )
        return pd.concat([freq_df, total_record]).reset_index(drop=True)
    
    @staticmethod
    def _ratio_to_percent(freq_df: pd.DataFrame, num_digits: int):
        """
        빈도표의 비율을 퍼센트로 변환

        Args:
            freq_df (pd.DataFrame): 비율이 추가된 빈도표
            num_digits (int): 퍼센트의 소수점 자리수
        """
        freq_df["ratio"] = freq_df["ratio"] * 100
        freq_df.rename(columns={"ratio": "%"}, inplace=True)
        freq_df["%"] = freq_df["%"].round(num_digits)


def cross_table(
        df: pd.DataFrame,
        col1: Union[str, int],
        col2: Union[str, int],
        vertical: bool = True,
        percent: bool = True,
        ratio_style: int = 1,
        num_digits: int = 2,
        margin_txt: str = "total",
        markdown: bool = False
    ) -> pd.DataFrame:
    """
    교차표 생성 및 출력

    Args:
        df (pd.DataFrame): 교차표 생성 대상이 되는 pd.DataFrame
        col1 (Union[str, int]): 행 축에 들어갈 컬럼
        col2 (Union[str, int]): 열 축에 들어갈 컬럼
        vertical (bool, optional): 비율을 행 기준으로 생성할지 여부(False이면 열 기준으로 생성). Defaults to True.
        percent (bool, optional): 비율을 퍼센트로 변경할지 여부(False이면 비율로 생성). Defaults to True.
        ratio_style (int, optional): 비율을 어떤 방식으로 출력할지. Defaults to 1.
            - 0: 비율(퍼센트)를 교차표에 추가하지 않음(교차표만 출력)
            - 1: 비율(퍼센트)를 교차표의 빈도 옆에 괄호로 추가
            - 2: 비율(퍼센트)만 교차표로 출력
        num_digits (int, optional): 비율(퍼센트)의 반올림 자릿수. Defaults to 2.
        margin_txt (str, optional): 교차표의 margin(총합)의 컬럼과 인덱스 문자열. Defaults to "total".
        markdown (bool, optional): table을 markdown으로 보여줄지 여부. Defaults to False.

    Returns:
        pd.DataFrame: 교차표
    """
    # col1과 col2에 margin_txt가 있는지 확인
    _CrossTable._margin_txt_check(df, col1, col2, margin_txt)
    # 기본적인 교차표 생성
    cross_df = pd.crosstab(df[col1], df[col2], margins=True, margins_name=margin_txt)
    # 비율 추가
    if ratio_style != 0:
        # 비율(퍼센트) 행렬
        ratio_mat = _CrossTable._make_ratio_mat(cross_df, vertical, percent, margin_txt)
        # 반올림 및 교차표에 맞게 형태 정의
        ratio_mat = _CrossTable._convert_mat_style(ratio_mat, percent, num_digits, ratio_style)
        # 비율을 교차표에 붙인다.
        if ratio_style == 1:
            ratio_mat = np.char.add(cross_df.values.astype("<U9"), ratio_mat)
        cross_df = pd.DataFrame(ratio_mat, index=cross_df.index, columns=cross_df.columns)
    # markdown 출력 여부
    if markdown:
        show_markdown_df(df=cross_df)
    return cross_df


class _CrossTable:
    
    @staticmethod
    def _margin_txt_check(df: pd.DataFrame, col1: str, col2: str, margin_txt: str):
        """
        교차표의 대상이 되는 column들의 element에 margin_txt와 동일한 문자가 있는지 확인

        Args:
            df (pd.DataFrame): 교차표의 대상이 되는 DataFrame
            col1 (str): 교차표의 행 축 컬럼
            col2 (str): 교차표의 열 축 컬럼
            margin_txt (str): margin(total)에 들어가는 string

        Raises:
            ValueError: col1, col2 모든 열에 margin_txt가 포함된 경우
            ValueError: col1 열에 margin_txt가 포함된 경우
            ValueError: col2 열에 margin_txt가 포함된 경우
        """
        mask_col1 = margin_txt in np.unique(df[col1])
        mask_col2 = margin_txt in np.unique(df[col2])
        if mask_col1 and mask_col2:
            raise ValueError(f"{margin_txt}가 {col1}, {col2}에 모두 포함되어 있습니다. margin_txt를 변경하십시오.")
        elif mask_col1:
            raise ValueError(f"{margin_txt}가 {col1}에 포함되어 있습니다. margin_txt를 변경하십시오.")
        elif mask_col2:
            raise ValueError(f"{margin_txt}가 {col2}에 포함되어 있습니다. margin_txt를 변경하십시오.")
        
    @staticmethod
    def _check_ratio_style(ratio_style: int):
        """
        ratio_style이 [0, 1, 2]에 포함되는지 확인

        Args:
            ratio_style (int): ratio_style

        Raises:
            ValueError: ratio_style이 [0, 1, 2]에 포함되지 않는 경우
        """
        ratio_style_list = [0, 1, 2]
        if ratio_style not in ratio_style_list:
            raise ValueError(f"ratio_style에 {ratio_style}이 입력되었습니다. {ratio_style_list}에 속하는 값을 입력하십시오.")
        
    @staticmethod
    def _make_ratio_mat(
            cross_df: pd.DataFrame, 
            vertical: bool, 
            percent: bool, 
            margin_txt: str
        ) -> np.ndarray:
        """
        교차표의 비율 행렬 생성

        Args:
            cross_df (pd.DataFrame): 기본 교차표
            vertical (bool): 비율을 행 기준으로 생성할지 여부(False이면 열 기준으로 생성)
            percent (bool): 비율을 퍼센트로 변경할지 여부(False이면 비율로 생성)
            margin_txt (str): 교차표의 margin(총합)의 컬럼과 인덱스 문자열

        Returns:
            np.ndarray: 교차표의 비율 또는 퍼센트에 대한 행렬
        """
        # 비율 행렬 계산
        standard = cross_df[[margin_txt]].values if vertical else cross_df.loc[margin_txt].values
        ratio_mat = cross_df.values / standard
        # percent 변환 여부
        if percent:
            ratio_mat = ratio_mat * 100
        return ratio_mat
    
    @staticmethod
    def _convert_mat_style(
            ratio_mat: np.ndarray,
            percent: bool,
            num_digits: int,
            ratio_style: int
        ) -> np.ndarray:
        """
        ratio_mat의 style을 변환
        >>> ratio_style == 1: 괄호를 붙임
            ratio_style != 1: 괄호를 붙이지 않음

        Args:
            ratio_mat (np.ndarray): 비율(퍼센트) 행렬
            percent (bool): 퍼센트 문자열(%)을 추가할지 여부
            num_digits (int): 반올림 자릿수
            ratio_style (int): 교차표에 비율을 어떠한 방식으로 추가할지

        Returns:
            np.ndarray: ratio_style에 맞게 변형된 비율(퍼센트) 행렬
        """
        left_str = " (" if ratio_style == 1 else ""
        right_str = ("%)" if percent else ")") if ratio_style == 1 else ""
        ratio_vt = [f"{left_str}{i:.{num_digits}f}{right_str}" for i in ratio_mat.flatten()]
        return np.array(ratio_vt).reshape(ratio_mat.shape)