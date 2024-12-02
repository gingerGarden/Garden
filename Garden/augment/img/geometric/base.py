import warnings
from typing import Optional, Tuple

import cv2
import numpy as np

from ....utils.img.img import get_img_axis_size, size_tuple_type_error


def crop(img: np.ndarray, x: int, y: int, x_size: int, y_size: int) -> np.ndarray:
    """
    image를 crop 한다.

    Args:
        img (np.ndarray): 2D 이미지(흑백, RGB, RGBA 등)
        x (int): x축에서 자르기 시작할 위치
        y (int): y축에서 자르기 시작할 위치
        x_size (int): x축에서 자르기 시작할 위치부터 잘라낼 크기
        y_size (int): y축에서 자르기 시작할 위치부터 잘라낼 크기

    Returns:
        np.ndarray: crop된 이미지
    """
    return img[y:y + y_size, x:x + x_size, :]


def padding(
    img: np.ndarray,
    size_tuple: Optional[Tuple[int, int]] = None,
    pixel: Tuple[int, int, int] = (0, 0, 0),
) -> np.ndarray:
    """
    이미지를 padding한다.
    >>> size_tuple는 입력 이미지를 감싸는 형태의 padding일 때 사용하는 방법으로, size_tuple는 입력 이미지보다 커야한다.
        - size_tuple는 (new_x, new_y)로 이루어진 튜플로, new_x나 new_y가 원본 이미지보다 작은 경우 ValueError가 발생한다.
        - size_tuple가 None인 경우, 원본 이미지에서 가장 긴 축을 기준으로 padding 한다.

    Args:
        img (np.ndarray): padding할 이미지
        size_tuple (Optional[Tuple[int, int]], optional): 새로 padding할 이미지의 크기에 대한 튜플(x, y). Defaults to None.
        pixel (Tuple[int, int, int], optional): padding할 영역에 입력할 픽셀. Defaults to (0,0,0).

    Returns:
        np.ndarray: padding된 이미지
    """
    x, y = get_img_axis_size(img)
    # padding할 크기 계산
    x_pad_size, y_pad_size = _get_padding_size(x, y, size_tuple)
    if x_pad_size == 0 and y_pad_size == 0:
        return img
    else:
        # 각 방향 패딩 크기 계산
        top = int(np.ceil(y_pad_size / 2))
        bottom = int(y_pad_size - top)
        left = int(np.ceil(x_pad_size / 2))
        right = int(x_pad_size - left)
        # 패딩
        return cv2.copyMakeBorder(
            img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=pixel
        )


def padding_border_replicate(img: np.ndarray) -> np.ndarray:
    """
    img의 padding 방법으로, 이미지 가장자리의 pixel을 붙여넣는 border replicate 방법 사용

    Args:
        img (np.ndarray): 원본 이미지

    Returns:
        np.ndarray: border replicate로 padding된 이미지
    """
    x, y = get_img_axis_size(img=img)
    x_pad_size, y_pad_size = _get_padding_size(x, y, size_tuple=None)
    left = x_pad_size // 2
    right = x_pad_size - left
    top = y_pad_size // 2
    bottom = y_pad_size - top
    result = cv2.copyMakeBorder(
        img,
        top=top,
        bottom=bottom,
        left=left,
        right=right,
        borderType=cv2.BORDER_REPLICATE,
    )
    return result


def _get_padding_size(
    x: int, y: int, size_tuple: Optional[Tuple[int, int]]
) -> Tuple[int, int]:
    """
    패딩 크기 산출 - padding() 함수를 위해 사용

    Args:
        x (int): 원본 이미지의 x축 크기
        y (int): 원본 이미지의 y축 크기
        size_tuple (Optional[Tuple[int, int]]): padding할 이미지의 크기(원본 이미지보다 커야함)

    Returns:
        Tuple[int, int]: x축, y축의 패딩 크기
    """
    if size_tuple is not None:
        size_tuple_type_error(
            size_tuple=size_tuple
        )  # size_tuple이 (int, int) 인지 확인.
        # new_x, new_y는 x, y보다 반드시 크거나 같아야 한다.
        new_x, new_y = size_tuple
        _padding_size_warning(x, y, new_x, new_y)
        x_pad_size = new_x - x
        y_pad_size = new_y - y
    else:
        max_size = x if x >= y else y
        x_pad_size = max_size - x
        y_pad_size = max_size - y
    return x_pad_size, y_pad_size


def _padding_size_warning(x: int, y: int, new_x: int, new_y: int):
    """
    size_tuple이 None이 아니면서, 원본 이미지의 크기보다 각 축의 크기가 작은 경우 ValueError 출력
    """
    if (x > new_x) or (y > new_y):
        txt = f"입력된 이미지는 x={x}, y={y}로, 입력된 size_tuple의 x={new_x}, y={new_y}보다 큽니다. size_tuple를 수정하십시오."
        raise ValueError(txt)


def resize(img: np.ndarray, new_x: int, new_y: int, quality="low") -> np.ndarray:
    """
    img를 new_x, new_y의 크기로 조정
    >>> quality에 따라 보간법 설정 방법이 바뀜.
        - 'low', 'middle', 'high' 존재
    >>> 이미지를 확대 또는 축소 시키는 것에 따라 보간법이 바뀜.
        - 이미지의 축별로 커지고 작아지는 정도가 달라질 수 있으므로, 넓이로 비교

    Args:
        img (np.ndarray): 원본 이미지
        new_x (int): Resize할 x축 크기
        new_y (int): Resize할 y축 크기
        quality (str, optional): Resize할 보간법의 품질. Defaults to 'low'.

    Returns:
        np.ndarray: Resize된 이미지
    """
    # 이미지 넓이를 이용해 크기가 축소하는지 커지는지 확인
    x, y = get_img_axis_size(img)
    reduce = x * y >= new_x * new_y
    # Resize
    return cv2.resize(
        img, (new_x, new_y), interpolation=interpolation_quality(quality, reduce)
    )


def interpolation_quality(quality: str = "low", reduce: bool = False) -> int:
    """
    cv2의 보간 방법을 quality와 이미지의 확대 또는 축소에 맞게 출력
    >>> 'low':cv2.INTER_LINEAR
        'middle': cv2.INTER_AREA (축소) | cv2.INTER_CUBIC (확대)
        'high': cv2.INTER_LANCZOS4
    >>> 위 방법이 적합하지 않은 경우, 별도의 resize 코드를 짜는 것을 추천

    Args:
        quality (str, optional): 이미지의 품질. Defaults to 'low'.
            - 'low', 'middle', 'high'
        reduce (bool, optional): 이미지 축소 여부. Defaults to False.

    Returns:
        int: 품질에 해당하는 보간 방법
    """
    reduce_map = {
        "low": cv2.INTER_LINEAR,
        "middle": cv2.INTER_AREA,
        "high": cv2.INTER_LANCZOS4,
    }
    enlarge_map = {
        "low": cv2.INTER_LINEAR,
        "middle": cv2.INTER_CUBIC,
        "high": cv2.INTER_LANCZOS4,
    }
    quality_map = reduce_map if reduce else enlarge_map
    fn = quality_map.get(quality, cv2.INTER_LINEAR)
    if quality not in quality_map:
        txt = "You entered an incorrect quality, you must choose 'low', 'middle', or 'high'. Defaulting to 'low' quality."
        warnings.warn(txt)
    return fn
