import cv2
import numpy as np

from ....utils.img.img import check_rgb_or_rgba


def rgb_to_grayscale(img: np.ndarray, is_bgr: bool = False) -> np.ndarray:
    """
    RGB(BGR 또는 RGBA) 이미지를 Grayscale 이미지로 변환한다.

    Args:
        img (np.ndarray): BGR, RGB, RGBA 중 하나로 기대되는 이미지
        is_bgr (bool, optional): 입력 이미지가 BGR인 경우. Defaults to False.

    Returns:
        np.ndarray: 흑백 이미지
    """
    check_rgb_or_rgba(img)  # 입력 이미지가 BGR, RGB, RGBA 인지 확인한다.
    if img.shape[2] == 4:
        result = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
    else:
        if is_bgr:
            result = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            result = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return result
