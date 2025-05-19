import datetime
import inspect
import time
from collections import OrderedDict
from typing import Any, Callable, Dict, List, Union

import numpy as np


def format_with_rounding(value: float, num_digits: int) -> str:
    """
    value를 입력받아 num_digits 자릿수로 반올림한 문자열을 출력
        - float을 round하는 경우, 뒤가 0인 경우 잘림, string으로 출력하여 이렇게 되지 않도록 함.
        - ex) float에서 3.130은 3.13 으로 출력됨

    만약, None이 입력되는 경우, 해당 method가 적용되지 않는다.

    Args:
        value (float): 반올림 및 formating할 실수
        num_digits (int): 표현할 실수의 자릿수

    Returns:
        str: 특정 자릿수로 반올림하여 문자열로 출력
    """
    if value is not None:
        return f"{value:.{num_digits}f}"


def current_time(only_time: bool = True) -> str:
    """현재 시간(날짜)를 반환한다.

    Args:
        only_time (bool, optional): 시간만 반환할지 여부(False인 경우, 날짜와 함께 반환). Defaults to True.

    Returns:
        str: 현재 시간의 문자열
    """
    format = "%H:%M:%S" if only_time else "%Y.%m.%d %H:%M:%S"
    return datetime.datetime.today().strftime(format)


def time_checker(start: float) -> str:
    """start(float: time.time())부터 time_checker() 코드 실행까지 걸린 시간을 깔끔하게 출력
    Example: '0:01:55.60'

    Args:
        start (float): 소수초

    Returns:
        str: "시:분:초.밀리초" 형식의 문자열
    """
    # 소모 시간 측정
    end = time.time()
    second_delta = end - start
    result = decimal_seconds_to_time_string(decimal_s=second_delta)

    return result


def decimal_seconds_to_time_string(decimal_s: float) -> str:
    """소수 초 단위 시간을 받아 "시:분:초.밀리초" 형식의 문자열로 변환

    Args:
        decimal_s (_type_): 소수초
    Returns:
        str: "시:분:초.밀리초" 형식의 문자열
    """
    time_delta = datetime.timedelta(seconds=decimal_s)
    str_time_delta = str(time_delta).split(".")
    time1 = str_time_delta[0]
    if len(str_time_delta) == 1:
        time2 = "00"
    else:
        time2 = str_time_delta[1][:2]
    return f"{time1}.{time2}"


def make_numpy_arange(start: int, end: int, plus: bool = True) -> np.ndarray:
    """
    start, end 사이에 np.arange()를 적용
        - start와 end가 동일한 경우, end에 +1을 하거나 start에 -1을 하여 size가 반드시 1 이상인 array가 생성되게 한다.

    Args:
        start (int): np.arange의 start
        end (int): np.arange의 end
        plus (bool, optional): 기본적으로 end += 1이 적용되나, plus=False 경우 start에 -1함. Defaults to True.

    Returns:
        np.ndarray: np.arange()의 결과
    """
    if start == end:
        if plus:
            end += 1
        else:
            start -= 1
    return np.arange(start, end)


def list_flatten(lists: List[List[Any]]) -> List[Any]:
    """list 안에 list들이 혼합되어 있는 경우, 이를 1개의 list로 병합한다

    Args:
        lists (List[List[Any]]): list 안에 list가 들어 있는 list

    Returns:
        List[Any]: 1개의 list
    """
    stack = []
    for item in lists:
        if isinstance(item, list):
            stack.extend(list_flatten(item))
        else:
            stack.append(item)
    return stack


def get_method_parameters(
    fn: Callable, to_list: bool = False
) -> Union[OrderedDict, List]:
    """
    fn의 parameter들을 odict_keys 또는 list로 출력

    Args:
        fn (callable): 대상 method
        to_list (bool, optional): list로 출력할지 여부. Defaults to False.

    Returns:
        Union[OrderedDict, List]: method의 parameter들을 list 또는 odict_keys로 출력
    """
    # method의 signature
    sig = inspect.signature(fn)
    # method의 parameter 이름 목록 추출
    params = sig.parameters.keys()
    if to_list:
        return list(params)
    else:
        return params


def kwargs_filter(fn: Callable, params_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    kwargs로 입력된 params_dict에서 fn에 해당하는 parameter만 선택

    Args:
        fn (Callable): 대상 method
        params_dict (Dict[str, Any]): 대상 method의 parameter를 포함한 parameter의 dictionary

    Raises:
        ValueError: fn의 파라미터와 params_dict의 key가 전혀 일치 하지 않는 경우

    Returns:
        Dict[str, Any]: fn의 파라미터와 중 params_dict의 key와 일치하는 대상
    """
    # fn에 있는 모든 parameter들을 가져온다.
    params = get_method_parameters(fn)
    # params_dict에서 유효 parameter만 필터링한다.
    filtered_params_dict = {k: v for k, v in params_dict.items() if k in params}
    # filtered_params_dict의 크기가 0인지 확인.
    if len(filtered_params_dict) == 0:
        raise ValueError(
            f"입력된 params_dict는 fn의 파라미터에 해당하지 않습니다. 해당 fn은 {params}만 지원합니다."
        )
    else:
        return filtered_params_dict


def convert_byte_to_readable(byte_size:int, digit_number:int=2, lower:bool=True) -> str:
    """
    byte_size(byte)를 사람이 인식하기 쉬운 문자열로 변환한다.
    example) 2000 = "1.95 kb"

    Args:
        byte_size (int): byte
        digit_number (int, optional): 표현할 소수점 자리 수. Defaults to 2.
        lower (bool, optional): 단위를 소문자로 출력할지 여부. Defaults to True.

    Returns:
        str: 사람이 인식하기 쉬운 단위로 변환된 byte 문자열
    """
    _units = [
        'b', 'kb', 'mb', 'gb', 'tb', 'pb', 'eb', 'zb', 'yb'
    ] if lower else [
        'B', 'KB', 'MB', 'GB', 'TB', 'PB', 'EB', 'ZB', 'YB'
    ]

    for i, unit in enumerate(_units):
        if byte_size < 1024 or i == len(_units) - 1:    # 마지막 단위에 도달하면 그대로 반환
            return f"{byte_size:.{digit_number}f} {unit}"
        byte_size /= 1024