from typing import Tuple

import cv2
import numpy as np

from ....utils.img.img import RGBTuple


class Circle:
    @staticmethod
    def make_mask(
        img: np.ndarray, center: Tuple[int, int], radius: int, none_value: int = 300
    ) -> np.ndarray:
        """
        이미지에 0과 none_value로 이루어진 mask를 생성한다.
        >>> 이미지는 위 마스크와 더해져서 바깥 부분을 제거하므로, uint8의 최대값인 255보다 큰 값이 none_value로 지정되어야 한다.

        Args:
            img (np.ndarray): 원본 이미지
            center_tuple (Tuple[int, int]): 이미지의 중심
            radius (int): 원의 반지름
            none_value (int, optional): 원 바깥 픽셀 값. Defaults to 300.

        Returns:
            np.ndarray: _description_
        """
        mask = np.full_like(img, fill_value=1)
        cv2.circle(mask, center, radius, color=0, thickness=-1)
        mask = mask.astype(np.int64)
        return np.where(mask == 1, none_value, mask)

    @staticmethod
    def add_mask(
        img: np.ndarray, mask: np.ndarray, none_value: int = 300, color: str = "random"
    ) -> np.ndarray:
        """
        이미지에 mask를 합쳐서 원본 이미지의 원 바깥 부분(none_value)에 color로 채운다.

        Args:
            img (np.ndarray): 원본 이미지
            mask (np.ndarray): 원의 안은 0, 바깥은 none_value로 찬 mask
            none_value (int, optional): 원 바깥 부분을 표기하는 pixel, 255보다 커야 함. Defaults to 300.

        Returns:
            np.ndarray: 원 바깥 부분이 self.color로 칠해진 이미지
        """
        add_mask_img = img.astype(np.int64) + mask
        add_mask_img = np.where(
            add_mask_img >= none_value, RGBTuple.get(color), add_mask_img
        )
        return add_mask_img.astype(np.uint8)
