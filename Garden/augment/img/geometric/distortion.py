import warnings
from typing import Optional, Tuple

import cv2
import numpy as np

from ....statics import distribution as dist
from ....statics import probability as prob
from ....statics import scaler
from ....utils.img.img import RGBTuple, get_img_axis_size, img_corners


class Perspective:
    def __init__(
        self,
        p: float = 1.0,
        min: float = -0.3,
        max: float = 0.1,
        how: int = 0,
        color: str = "random",
        distribution: Optional[np.ndarray] = None,
    ):
        """
        p의 확률로 이미지를 원근 왜곡한다.
        : 이미지의 각 꼭지점을 무작위로 음 또는 양의 계수를 곱해
        >>> 원근 왜곡의 처리 방법은 크게 2가지 종류에 따라 다르게 실시된다.
            - 1: 원근 왜곡으로 인해 이미지의 크기 변화를 캔버스의 크기에 반영
            - 2: 단순 원근 왜곡으로, 이미지 크기 변화를 반영하지 않음.
            - 0: 1, 2 둘 중 무작위 방법을 한 가지 선택.
            - 위 범위를 벗어나는 값이 입력되는 경우, 1의 방법으로 변환됨.
        Args:
            p (float, optional): 이미지를 원근 왜곡할 확률. Defaults to 1.0.
            min (float, optional): 원근 왜곡의 최소 비율. Defaults to -0.3.
            max (float, optional): 원근 왜곡의 최대 비율. Defaults to 0.1.
            how (int, optional): 원근 왜곡 후 처리 방법, 0, 1, 2. Defaults to 0.
            color (str, optional): 원근 왜곡 후 남은 영역 픽셀의 색. Defaults to 'random'.
            distribution (Optional[np.ndarray], optional): 이미지 왜곡 수준에 대한 분포. Defaults to None.
        """
        self.p = p
        self.how = how
        self._check_how()
        self.color = color

        if distribution is None:
            distribution = dist.linear(min=0, max=1, size=1000)
        self.dist = scaler.min_max_range_normalization(
            array=distribution, min=min, max=max
        )
        # crop mapping
        self.methods = {
            0: lambda: prob.binary_probability(p=0.5),
            1: lambda: True,
            2: lambda: False,
        }

    def __call__(self, img: np.ndarray) -> np.ndarray:
        """
        p의 확률로 이미지를 원근 왜곡

        Args:
            img (np.ndarray): 원본 이미지

        Returns:
            np.ndarray: 원근 왜곡된 이미지
        """
        if prob.binary_probability(p=self.p):
            extension = self.methods.get(self.how, self.methods[0])()
            img = self.transform(img, extension)
        return img

    def _check_how(self):
        if self.how not in [0, 1, 2]:
            warnings.warn(
                "how에는 0, 1, 2 3개의 정수 중 하나가 입력되어야 합니다! 기본값인 0으로 실행됩니다."
            )
            self.how = 0

    def transform(self, img: np.ndarray, extension: bool) -> np.ndarray:
        """
        원근 왜곡의 주요 알고리즘

        Args:
            img (np.ndarray): 원본 이미지
            extension (bool): 원근 왜곡으로 인해 이미지가 원본 범위를 벗어나는 것에 대하여 확장 여부

        Returns:
            np.ndarray: _description_
        """
        x, y = get_img_axis_size(img=img)
        # 이미지 좌표
        src_corners = img_corners(x=x, y=y)
        dst_corners = self._destination_corners(x, y)
        # 원근 변환 행렬
        trans_m = cv2.getPerspectiveTransform(src_corners, dst_corners)
        # 이미지 변환에 따른 확장 적용
        if extension:
            x, y = self._matrix_adjustment(m=trans_m, dst_corners=dst_corners)
        # 원근 변환 적용
        return cv2.warpPerspective(
            img, trans_m, (x, y), borderValue=RGBTuple.get(color=self.color)
        )

    def _destination_corners(self, x: int, y: int) -> np.ndarray:
        """
        원근 왜곡을 통한 이미지의 새로운 꼭지점

        Args:
            x (int): 기존 이미지의 x축 크기
            y (int): 기존 이미지의 y축 크기

        Returns:
            np.ndarray: 원근 왜곡 하였을 때, 이미지의 좌표
        """
        dst_points = np.float32(
            [
                [
                    self._random_points(x, reverse=False),
                    self._random_points(y, reverse=False),
                ],
                [
                    self._random_points(x, reverse=True),
                    self._random_points(y, reverse=False),
                ],
                [
                    self._random_points(x, reverse=False),
                    self._random_points(y, reverse=True),
                ],
                [
                    self._random_points(x, reverse=True),
                    self._random_points(y, reverse=True),
                ],
            ]
        )
        return dst_points

    def _random_points(self, x: int, reverse: bool = True) -> int:
        """
        무작위로 왜곡한 크기

        Args:
            x (int): 대상 축의 크기
            reverse (bool, optional): 역으로 크기 계산 여부. Defaults to True.

        Returns:
            int: 왜곡 크기
        """
        ratio = prob.choose_one(self.dist)
        return int(np.ceil(x * (1 - ratio))) if reverse else int(np.trunc(x * ratio))

    def _matrix_adjustment(
        self, m: np.ndarray, dst_corners: np.ndarray
    ) -> Tuple[int, int]:
        """
        원근 왜곡으로 인해 원본 범위를 벗어나는 현상을 막기 위해, 왜곡 행렬을 조정하고, 왜곡 후 이미지 크기 출력

        Args:
            m (np.ndarray): 원근 왜곡 행렬
            dst_corners (np.ndarray): 원근 왜곡으로 인해 변한 이미지의 꼭지점

        Returns:
            Tuple[int,int]: 원근 왜곡 후 이미지의 크기
        """
        x_min = np.min(dst_corners[:, 0])
        x_max = np.max(dst_corners[:, 0])
        y_min = np.min(dst_corners[:, 1])
        y_max = np.max(dst_corners[:, 1])
        # 변환 행렬 보정
        m[0, 2] += -x_min
        m[1, 2] += -y_min
        # 새 이미지 크기 보정
        return (int(x_max - x_min), int(y_max - y_min))
