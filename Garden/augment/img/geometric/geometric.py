import warnings
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np

from ....statics import distribution as dist
from ....statics import probability as prob
from ....statics import scaler
from ....utils.img.img import RGBTuple, get_img_axis_size, img_corners
from . import base


class Rotate:
    def __init__(
        self,
        p: float = 1.0,
        max_degree: int = 90,
        keep_window: bool = True,
        color: str = "random",
        distribution: Optional[np.ndarray] = None,
    ):
        """
        이미지를 회전한다.
        >>> distribution은 임의의 분포인 배열이다 - 기본적으로 선형 분포를 갖는다.
        >>> 회전 발생 후, 배경 영역의 색은 color의 문자열에 맞는 색으로 채워진다.
            - 'random', 'red'('r'), 'green'('g'), 'blue'('b'), 'yellow'('y'), 'sky'('s'), 'purple'('p'), 'white'('w'), 'black'('zero', 'z') 중 하나 선택 가능
        >>> 이미지 회전으로 인해 기존 윈도우의 바깥으로 이미지가 빠져나갈 수 있으며, 이는 keep_window로 잘림을 방지할 수 있다.

        Callable 함수가 주요 기능으로, Rotate class의 instance variable을 기반으로 이미지를 회전한다.
        ex) rotate_fn = Rotate(p=0.5, max_degree=45, keep_window=True)
            rotated_img = rotate_fn(img)

        Args:
            p (float, optional): 이미지 회전 확률. Defaults to 1.0.
            max_degree (int, optional): 이미지 회전의 최대 각도. Defaults to 90.
            keep_window (bool, optional): 이미지 회전으로 인한 이미지 잘림 현상을, 윈도우 크기를 늘려서 방지할지 여부. Defaults to True.
            color (str, optional): 회전 후 빈 공간을 채울 색상. 'random', 'red', 'green', 등 지정. Defaults to 'random'.
            distribution (Callable, optional): 무작위 각도의 분포. Defaults to dist.linear(min=0, max=1, size=100).
        """
        self.p = p
        self.keep_window = keep_window
        self.color = color
        if distribution is None:
            distribution = dist.linear(min=0, max=1, size=1000)
        self.degree_distribution = scaler.min_max_range_normalization(
            array=distribution, min=0, max=max_degree
        )

    def __call__(self, img: np.ndarray) -> np.ndarray:
        """
        self.p의 확률로 무작위 각도로 이미지를 회전한다.

        Args:
            img (np.ndarray): 입력 이미지

        Returns:
            np.ndarray: 회전 이미지
        """
        if prob.binary_probability(p=self.p):
            degree = prob.choose_one(array=self.degree_distribution)
            img = self.rotate(img, degree)
        return img

    def rotate(self, img: np.ndarray, degree: int) -> np.ndarray:
        """
        이미지 회전
        """
        x, y = get_img_axis_size(img=img)  # 이미지 무결성, 크기 출력
        center = (x // 2, y // 2)  # 이미지의 중심 계산
        trans_m = cv2.getRotationMatrix2D(
            center, degree, 1.0
        )  # 회전을 위한 변환 행렬 계산

        # 이미지 회전으로 인한 윈도우 크기 변환을 막기 위해 윈도우 크기와 변환 행렬 조정
        if self.keep_window:
            x, y = self._calculate_new_size(x, y, trans_m)

        rotated_img = cv2.warpAffine(
            img,
            trans_m,
            (x, y),
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=RGBTuple.get(color=self.color),
        )
        return rotated_img

    def _calculate_new_size(
        self, x: int, y: int, trans_m: np.ndarray
    ) -> Tuple[int, int]:
        """
        이미지 회전으로 인한 윈도우 바깥으로 이미지가 잘리는 현상을 방지하기 위해, 윈도우의 크기와 변환 행렬을 조정
        """
        # 회전 후 이미지 크기 계산 (최소한의 크기로 변경)
        cos_angle = np.abs(trans_m[0, 0])
        sin_angle = np.abs(trans_m[0, 1])
        # 새로운 이미지 크기 계산
        new_x = int((y * sin_angle) + (x * cos_angle))
        new_y = int((y * cos_angle) + (x * sin_angle))
        # 변환 행렬 보정
        trans_m[0, 2] += (new_x / 2) - x // 2
        trans_m[1, 2] += (new_y / 2) - y // 2
        return new_x, new_y


class Shear:
    def __init__(
        self,
        p: float = 1.0,
        min: float = -0.3,
        max: float = 0.3,
        how: int = 0,
        keep_window: int = 1,
        color: str = "random",
        distribution: Optional[np.ndarray] = None,
    ):
        """
        이미지를 Shear 변환한다.
        >>> distribution은 임의의 분포인 배열이다 - 기본적으로 선형 분포를 갖는다.
        >>> 회전 발생 후, 배경 영역의 색은 color의 문자열에 맞는 색으로 채워진다.
            - 'random', 'red'('r'), 'green'('g'), 'blue'('b'), 'yellow'('y'), 'sky'('s'), 'purple'('p'), 'white'('w'), 'black'('zero', 'z') 중 하나 선택 가능
        >>> Shear 변환은 x, y 축을 기준으로 변환하며, 이는 how를 이용하여 어떤 축을 기반으로 변환할지 정의할 수 있다.
            - 0: 무작위 확률로 x, y, xy축으로 변환, 1: x축 기준으로 변환, 2: y축 기준으로 변환, 3: x축, y축 기준으로 변환
        >>> Shear 변환으로 인해 기존 윈도우의 바깥으로 이미지가 빠져나갈 수 있으며, 이는 keep_window로 보정할 수 있다.
            - 0: 50% 확률로 보정 여부 무작위 설정, 1: 보정, 2: 보정하지 않음

        Callable 함수가 주요 기능으로, Shear class의 instance variable을 기반으로 이미지를 변환한다.
        ex) shear_fn = Shear(p=0.5, min=-0.3, max=0.3, how=1, keep_window=1, color='random')
            sheared_img = shear_fn(img)

        Args:
            p (float, optional): Shear 변환 확률. Defaults to 1.0.
            min (float, optional): Shear 보정의 최소값. Defaults to -0.3.
            max (float, optional): Shear 보정의 최대값. Defaults to 0.3.
            how (int, optional): Shear 변환의 축(0: 무작위, 1: x축, 2: y축, 3: xy축). Defaults to 0.
            keep_window (int, optional): Shear 변환으로 인한 이미지 밀림 보정 방법. Defaults to 1.
            color (str, optional): Shear 변환 후 빈 공간을 채울 색상. 'random', 'red', 'green', 등 지정. Defaults to 'random'.
            distribution (Callable, optional): 무작위 변환 크기의 분포. Defaults to dist.linear(min=0, max=1, size=1000).
        """
        self.p = p
        self.how = how
        self.keep_window = keep_window
        self.color = color
        if distribution is None:
            distribution = dist.linear(min=0, max=1, size=1000)
        self.dist = scaler.min_max_range_normalization(
            array=distribution, min=min, max=max
        )

        # mapping
        self.how_methods = {
            0: self._how_0_random_xy_factor,
            1: self._how_1_axis_x_factor,
            2: self._how_2_axis_y_factor,
            3: self._how_3_axis_xy_factor,
        }
        self.keep_window_bools = {
            0: lambda: prob.binary_probability(p=0.5),
            1: lambda: True,
            2: lambda: False,
        }

    def __call__(self, img: np.ndarray) -> np.ndarray:
        """
        self.p의 확률로 무작위로 shear 변환을 한다.

        Args:
            img (np.ndarray): 입력 이미지

        Returns:
            np.ndarray: Shear 변환된 이미지
        """
        if prob.binary_probability(p=self.p):
            factor_x, factor_y = self._get_xy_factors()  # x, y축의 shear 변환 크기
            adjust = self._get_keep_window_bool()  # 이미지 밀림 보정 여부
            sheard_img = self.shear_transformation(
                img=img, factor_x=factor_x, factor_y=factor_y, adjust=adjust
            )
        else:
            sheard_img = img
        return sheard_img

    def shear_transformation(
        self,
        img: np.ndarray,
        factor_x: float = 0.0,
        factor_y: float = 0.0,
        adjust: bool = True,
    ) -> np.ndarray:
        """
        img를 axis 축에 대하여 Shear transformation 한다.

        Args:
            img (np.ndarray): 원본 이미지
            factor_x (float, optional): x축의 변환 크기. Defaults to 0.0.
            factor_y (float, optional): y축의 변환 크기. Defaults to 0.0.
            adjust (bool, optional): 이미지 밀림 보정 여부. Defaults to True.

        Returns:
            np.ndarray: shear 변환된 이미지
        """
        # Shear 행렬 생성
        shear_m = self._shear_matrix(factor_x=factor_x, factor_y=factor_y)
        # Shear 행렬 적용으로 인한 이미지 크기 변환 보정
        x, y = (
            self._adjustment_transformation_img(img, shear_m)
            if adjust
            else get_img_axis_size(img)
        )
        # Shear 변환
        result = cv2.warpAffine(
            img,
            shear_m,
            (x, y),
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=RGBTuple.get(color=self.color),
        )
        return result

    def _shear_matrix(self, factor_x: float, factor_y: float) -> np.ndarray:
        """
        Shear 변환을 위한 변환 행렬 출력

        Args:
            factor_x (float): 변환에 대한 x값
            factor_y (float): 변환에 대한 y값

        Returns:
            np.ndarray: 변환 행렬
        """
        return np.float32([[1, factor_x, 0], [factor_y, 1, 0]])

    def _adjustment_transformation_img(
        self, img: np.ndarray, shear_m: np.ndarray
    ) -> Tuple[int, int]:
        """
        Shear 변환으로 인한 이미지 밀림 보정

        Args:
            img (np.ndarray): 원본 이미지
            shear_m (np.ndarray): shear 변환 행렬

        Returns:
            Tuple[int, int]: Shear 변환으로 인해 변경된 이미지의 크기
        """
        # 신규 좌표 계산
        new_corners = self._new_corners(img=img, shear_m=shear_m)
        # 각 축에 대한 Shear 행렬 보정치와 새로운 크기 계산
        shift_x, new_x = self._get_shift(new_corners, axis="x")
        shift_y, new_y = self._get_shift(new_corners, axis="y")
        # Shear 행렬에 시프트 적용
        shear_m[0, 2] += shift_x
        shear_m[1, 2] += shift_y
        return new_x, new_y

    def _new_corners(self, img: np.ndarray, shear_m: np.ndarray) -> np.ndarray:
        """
        Shear 변환으로 변하는 신규 이미지의 좌표 계산

        Args:
            img (np.ndarray): 원본 이미지
            shear_m (np.ndarray): Shear 변환 행렬

        Returns:
            np.ndarray: 원본 이미지의 각 꼭지점에 Shear 변환한 새로운 꼭지점 좌표
        """
        ori_corners = img_corners(img=img).reshape(-1, 1, 2)
        return cv2.transform(ori_corners, shear_m)

    def _get_shift(self, new_corners: np.ndarray, axis: str = "x") -> Tuple[float, int]:
        """
        새로운 좌표로 인한 이미지 밀림을 막기 위한 축별 Shear 행렬의 보정치와 새로운 이미지 크기 계산

        Args:
            new_corners (np.ndarray): Shear 변환으로 변경된 이미지의 꼭지점
            axis (str, optional): 변환 정도를 계산 대상의 축. Defaults to 'x'.

        Returns:
            Tuple[float, int]: 변환 최솟값과 새로운 이미지의 크기(축의 최대 길이)에 대한 Tuple
        """
        target_axis = 0 if axis == "x" else 1
        min_point = np.min(new_corners[:, 0, target_axis])
        max_point = np.max(new_corners[:, 0, target_axis])
        diff_min = max(0, -min_point)
        new_size = int(np.ceil(diff_min + max_point))
        return diff_min, new_size

    def _get_keep_window_bool(self) -> bool:
        """
        Shear 변환으로 인한 이미지 밀림에 대한 보정 여부 출력

        Returns:
            bool: 이미지 보정 여부
        """
        return self.keep_window_bools.get(self.keep_window, self.keep_window_bools[0])()

    # x_factor, y_factor 출력
    def _get_xy_factors(self) -> Tuple[float, float]:
        """
        self.how_method에서 self.how에 해당하는 value를 가지고 온다. default는 self._how_0_random_xy_factor로 0에 해당하는
        x, y를 출력한다.

        Returns:
            Tuple[float, float]: factor_x, factor_y로 shear 변환으로 각 축별로 변환할 크기를 출력
        """
        method = self.how_methods.get(
            self.how, self.how_methods[0]
        )  # dictionary.get(key, default)
        return method()

    def _how_0_random_xy_factor(self):
        axis = np.random.choice([1, 2, 3])
        return self.how_methods[axis]()

    def _how_1_axis_x_factor(self):
        return prob.choose_one(self.dist), 0.0

    def _how_2_axis_y_factor(self):
        return 0.0, prob.choose_one(self.dist)

    def _how_3_axis_xy_factor(self):
        return prob.choose_one(self.dist), prob.choose_one(self.dist)


class Translate:
    def __init__(
        self,
        p: float = 1.0,
        min: float = 0.05,
        max: float = 0.2,
        how=0,
        color="random",
        distribution: Optional[np.ndarray] = None,
    ):
        """
        이미지를 평행 이동 변환 한다.
        >>> distribution은 임의의 분포인 배열이다 - 기본적으로 선형 분포를 갖는다.
            - 이미지의 이동 거리 분포
        >>> 회전 발생 후, 배경 영역의 색은 color의 문자열에 맞는 색으로 채워진다.
            - 'random', 'red'('r'), 'green'('g'), 'blue'('b'), 'yellow'('y'), 'sky'('s'), 'purple'('p'), 'white'('w'), 'black'('zero', 'z') 중 하나 선택 가능
        >>> Translate 변환은 임의의 색(color)로 이동하고 남은 영역을 채우는 방식과 이미지 가장자리 픽셀로 채우는 방법 두 가지가 존재한다.
            - 0: 무작위로 두 방법 중 하나 선택, 1: 임의의 색으로 채움(color_padding()), 2: 가장자리 픽셀로 채움(border_replicate())

        Callable 함수가 주요 기능으로, Tranlate class의 instance variable을 기반으로 이미지를 변환한다.
        ex) translate_fn = Translate(p=0.5, min=0.05, max=0.2, how=0,color='random')
            translated_img = translate_fn(img)

        Args:
            p (float, optional): 이미지 변환 확률. Defaults to 1.0.
            min (float, optional): 최소 변환 거리의 비율. Defaults to 0.05.
            max (float, optional): 최대 변환 거리의 비율. Defaults to 0.2.
            how (int, optional): 변환 방법(0: 무작위, 1: 색깔, 2: 가장자리 픽셀). Defaults to 0.
            color (str, optional): 변환 후 빈 공간을 채울 색상. Defaults to 'random'.
            distribution (Optional[np.ndarray], optional): 무작위 변환 크기의 분포. Defaults to None.
        """
        self.p = p
        self.how = how
        self.color = color
        if distribution is None:
            distribution = dist.linear(min=0, max=1, size=1000)
        self.move_dist = scaler.min_max_range_normalization(
            array=distribution, min=min, max=max
        )

        # mapping
        self.style_methods = {
            0: self._random_translate_style,
            1: self.color_padding,
            2: self.border_replicate,
        }

    def __call__(self, img: np.ndarray) -> np.ndarray:
        """
        self.p의 확률로 Tranlate 변환을 한다.

        Args:
            img (np.ndarray): 입력 이미지(원본 이미지)

        Returns:
            np.ndarray: 변환된 이미지
        """
        if prob.binary_probability(p=self.p):
            fn = self.style_methods.get(self.how, self.style_methods[0])
            # 무작위 방법 정의
            if (self.how != 1) and (self.how != 2):
                fn = fn()
            translated_img = fn(img)
        else:
            translated_img = img
        return translated_img

    def _random_translate_style(self) -> Callable:
        """
        무작위 Translate 변환(평행 이동 변환) 방식을 정의한다.

        Returns:
            Callable: Tranlate 변환 메서드
        """
        return (
            self.color_padding
            if prob.binary_probability(p=0.5)
            else self.border_replicate
        )

    def color_padding(self, img: np.ndarray) -> np.ndarray:
        """
        평행 이동 후 self.color에 해당하는 색으로 남은 영역을 칠한다.

        Args:
            img (np.ndarray): 원본 이미지

        Returns:
            np.ndarray: 평행 이동한 이미지
        """
        x, y = get_img_axis_size(img)
        # 이동 행렬 생성
        translate_m = self._translate_matrix(img)
        # 평행 이동
        translated_imgs = cv2.warpAffine(
            img,
            translate_m,
            (x, y),
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=RGBTuple.get(color=self.color),
        )
        return translated_imgs

    def border_replicate(self, img: np.ndarray) -> np.ndarray:
        """
        이미지 가장자리의 픽셀로 채우는 방식으로 이미지의 평행 이동과 동일한 효과를 얻는다.

        Args:
            img (np.ndarray): 원본 이미지

        Returns:
            np.ndarray: 평행 이동한 이미지
        """
        x, y = get_img_axis_size(img)
        # border_replicate할 무작위 거리 계산
        move_x, move_y = self._calculate_move_range(x, y)
        left, right = (move_x, 0) if move_x >= 0 else (0, np.abs(move_x))
        top, bottom = (move_y, 0) if move_y >= 0 else (0, np.abs(move_y))
        # 경계 pixel을 복제하여 이동 거리를 채운다.
        replicated_img = cv2.copyMakeBorder(
            img,
            top=top,
            bottom=bottom,
            left=left,
            right=right,
            borderType=cv2.BORDER_REPLICATE,
        )
        return self._crop_border_replicate_img(replicated_img, x, y, move_x, move_y)

    def _translate_matrix(self, img: np.ndarray) -> np.ndarray:
        """
        color_padding() 메소드에서 평행 이동 행렬을 생성한다.

        Args:
            img (np.ndarray): 원본 이미지

        Returns:
            np.ndarray: 평행 이동 행렬
        """
        x, y = get_img_axis_size(img)
        move_x, move_y = self._calculate_move_range(x, y)
        translate_m = np.float32([[1, 0, move_x], [0, 1, move_y]])
        return translate_m

    def _crop_border_replicate_img(
        self, replicated_img: np.ndarray, x: int, y: int, move_x: int, move_y: int
    ) -> np.ndarray:
        """
        border_replicate() 메서드를 통해 가장자리 픽셀이 복사된 이미지를 crop하여, translate한 효과를 부여한다.

        Args:
            replicated_img (np.ndarray): 가장자리 픽셀이 복사된 이미지
            x (int): 원본 이미지의 x축 크기
            y (int): 원본 이미지의 y축 크기
            move_x (int): x축 이동거리
            move_y (int): y축 이동거리

        Returns:
            np.ndarray: 가장 자리 복사 거리를 고려하여 crop한 이미지
        """
        x_start = 0 if move_x >= 0 else np.abs(move_x)
        y_start = 0 if move_y >= 0 else np.abs(move_y)
        return base.crop(img=replicated_img, x=x_start, y=y_start, x_size=x, y_size=y)

    def _calculate_move_range(self, x: int, y: int) -> Tuple[int, int]:
        """
        Translation할 무작위 거리 계산

        Args:
            x (int): 원본 이미지의 x축 크기
            y (int): 원본 이미지의 y축 크기

        Returns:
            Tuple[int, int]: 각 축별 이동 거리
        """
        # 무작위 이동 거리 계산
        move_x = x * prob.choose_one(self.move_dist)
        move_y = y * prob.choose_one(self.move_dist)
        # 방향 정의
        if prob.binary_probability(p=0.5):
            move_x = move_x * -1
        if prob.binary_probability(p=0.5):
            move_y = move_y * -1
        # 정수 변환
        move_x = int(np.round(move_x, 0))
        move_y = int(np.round(move_y, 0))
        return (move_x, move_y)


class Scaling:
    def __init__(
        self,
        p: float = 1.0,
        min: float = 0.7,
        max: float = 1.3,
        how: int = 0,
        quality: str = "low",
        how_list: Optional[List[int]] = None,
        distribution: Optional[np.ndarray] = None,
    ):
        """
        이미지의 크기 조정을 한다.
        >>> distribution은 크기 비율에 대한 임의의 분포인 배열이다 - 기본적으로 선형 분포를 갖는다.
        >>> 크기 조정 방법은 how에 맞는 방법으로 이뤄진다.
            - 1: x,y 축 모두 비율 조정
            - 2: x, y축 모두 동일 비율로 조정
            - 3: x축만 조정
            - 4: y축만 조정
            - 0: 위 4가지 방법 중 하나를 무작위 선택
        >>> 이미지 Resize 과정의 보간은 quality에 해당하는 방법으로 보간된다.
            - 'low', 'middle', 'high'

        Callable 함수가 주요 기능으로, Scaling class의 instance variable을 기반으로 이미지의 축별 비율을 조정한다.
        ex) scaling_fn = Scaling(p=0.5, min=0.5, max=1.5, how=0, quality='high)
            scaled_img = scaling_fn(img)

        Args:
            p (float, optional): 이미지의 크기 조정 확률. Defaults to 1.0.
            min (float, optional): 이미지 축의 최소 비율. Defaults to 0.7.
            max (float, optional): 이미지 축의 최대 비율. Defaults to 1.3.
            how (int, optional): 축별 조정 방법. Defaults to 0.
            quality (str, optional): 보간 방법의 품질, 품질이 높을수록 연산이 오래걸릴 수 있음. Defaults to 'low'.
            how_list (Optional[List[int]], optional): None이 아닌 경우, how_list에 있는 방법 중 한 가지를 무작위로 선택. Defaults to None.
                - [0, 2, 3]인 경우, 0, 2, 3에 해당하는 3가지 방법 중 하나만 선택한다.
            distribution (Optional[np.ndarray], optional): 크기 비율의 분포. Defaults to None.
                - None인 경우, [min, max]의 선형 분포 생성.
        """
        self.p = p
        self.how = how
        self.how_list = how_list
        self.quality = quality
        if distribution is None:
            distribution = dist.linear(min=0, max=1, size=1000)
        self.dist = scaler.min_max_range_normalization(
            array=distribution, min=min, max=max
        )

        # mapping
        self.how_methods = {
            0: self._how_0_random,
            1: self._how_1_ratio,
            2: self._how_2_ratio,
            3: self._how_3_ratio,
            4: self._how_4_ratio,
        }
        self.how_method_keys = None
        self._make_how_method_keys()

    def __call__(self, img: np.ndarray) -> np.ndarray:
        """
        self.p의 확률로 무작위 비율로 이미지의 비율을 조정한다.

        Args:
            img (np.ndarray): 원본 이미지

        Returns:
            np.ndarray: 비율이 조정된 이미지
        """
        if prob.binary_probability(p=self.p):
            scaled_img = self.resize(img)
        else:
            scaled_img = img
        return scaled_img

    def resize(self, img: np.ndarray) -> np.ndarray:
        """
        이미지의 x, y축을 무작위 비율로 조정

        Args:
            img (np.ndarray): 원본 이미지

        Returns:
            np.ndarray: 무작위 비율로 resize된 이미지
        """
        x, y = get_img_axis_size(img=img)
        # 크기 조정 비율
        x_ratio, y_ratio = self._select_how()
        # 새 크기
        new_x = int(np.round(x * x_ratio, 0))
        new_y = int(np.round(y * y_ratio, 0))
        # scaling
        scaled_img = base.resize(img, new_x=new_x, new_y=new_y, quality=self.quality)
        return scaled_img

    def _select_how(self) -> Tuple[float, float]:
        """
        how_list가 None이 아닌 경우, how_list에 있는 방법으로 scaling 방법을 무작위로 선택

        Returns:
            Tuple[float, float]: x, y축의 크기 조정 비율
        """
        how = (
            np.random.choice(self.how_list, 1).item()
            if self.how_list is not None
            else self.how
        )
        return self.how_methods.get(how, self.how_methods[0])()

    def _make_how_method_keys(self):
        self.how_method_keys = list(self.how_methods.keys())
        self.how_method_keys.remove(0)

    def _how_0_random(self) -> Tuple[float, float]:
        how = prob.choose_one(self.how_method_keys)
        return self.how_methods[how]()

    def _how_1_ratio(self) -> Tuple[float, float]:
        return prob.choose_one(self.dist), prob.choose_one(self.dist)

    def _how_2_ratio(self) -> Tuple[float, float]:
        x = prob.choose_one(self.dist)
        return x, x

    def _how_3_ratio(self) -> Tuple[float, float]:
        return prob.choose_one(self.dist), 1.0

    def _how_4_ratio(self) -> Tuple[float, float]:
        return 1.0, prob.choose_one(self.dist)


class Resize:
    def __init__(
        self,
        size: int = 256,
        quality: str = "low",
        how: int = 2,
        how_list: Optional[List[int]] = None,
        color="random",
    ):
        """
        정방 행렬을 기준으로 이미지의 크기를 조정한다(정방 행렬이 아니어도 되는 경우, how=1로 한다).
        >>> how에 따라 resize 방식이 바뀐다.
            - 0: 무작위 방법, self.how_list에 있는 element에 해당하는 how 중 무작위로 한 가지 선택
            - 1: 비율 유지, 이미지의 넓이, 높이를 비율 유지하여 resize한다(장방 행렬이 될 수 있음)
            - 2: 비율 유지 - padding, 1로 resize하고 color로 padding
            - 3: 비율 유지 - border replicate, 1로 resize하고 가장자리의 pixel로 채움
            - 4: 비율 무시, 이미지의 가로, 세로 길이를 size에 맞게 resize

        Args:
            size (int, optional): 가장 긴 변의 길이(정방, 장방 행렬 모두). Defaults to 256.
            quality (str, optional): 이미지 보간법. Defaults to 'low'.
            how (int, optional): 이미지를 resize 하는 방법. Defaults to 2.
            how_list (List[int], optional): how=0인 경우, 선택 하는 방법, 0은 무작위 방법이므로 포함되면 안됩니다. Defaults to [2, 3, 4].
            color (str, optional): aspect_ratio_with_padding() 메서드에서 패딩에 들어가는 색. Defaults to 'random'.
        """
        self.size = size
        self.quality = quality
        self.how = how
        self.color = color
        self.how_list = [2, 3, 4] if how_list is None else how_list
        self.methods = {
            1: self.preserving_aspect_ratio,
            2: self.aspect_ratio_with_padding,
            3: self.aspect_ratio_with_border_replicate,
            4: self.distorting_aspect_ratio,
        }

    def __call__(self, img: np.ndarray) -> np.ndarray:
        """
        self.how에 해당하는 방법으로 이미지를 resize한다.

        Args:
            img (np.ndarray): 원본 이미지

        Returns:
            np.ndarray: resize된 이미지
        """
        if self._check_pass_img(img=img):
            return img

        if self.how == 0:
            return self._select_random()(img)
        else:
            return self.methods.get(self.how, self.methods[2])(img)

    def _select_random(self) -> Callable:
        """
        self.how=0인 경우, self.how_list에 있는 how 중 하나를 무작위로 선택하여 resize한다.

        Returns:
            Callable: resize할 메서드
        """
        how = np.random.choice(self.how_list, 1).item()
        if how == 0:
            fn = self.aspect_ratio_with_padding
            txt = "0 is the random option. Please exclude it from 'how_list'. The default value for the random setting is 3, and it will be resized accordingly."
            warnings.warn(txt)
            self.how_list.remove(0)
        else:
            fn = self.methods.get(how, self.methods[2])
        return fn

    def preserving_aspect_ratio(self, img: np.ndarray) -> np.ndarray:
        """
        이미지의 가로, 세로 비율을 유지한 상태로 크기 조정(비율 정보 유지)
        >>> 가장 큰 축의 크기를 self.size로 맞추고, 다른 축은 그에 맞는 비율로 조정

        Args:
            img (np.ndarray): 원본 이미지

        Returns:
            np.ndarray: 원본 비율이 유지된 상태로 크기가 조정된 이미지
        """
        x, y = get_img_axis_size(img)
        # 이미지의 최대 길이
        max_size = x if x >= y else y
        if self.size == max_size:
            return img
        else:
            new_x = int(np.round((x / max_size) * self.size, 0))
            new_y = int(np.round((y / max_size) * self.size, 0))
            return base.resize(img=img, new_x=new_x, new_y=new_y, quality=self.quality)

    def aspect_ratio_with_padding(self, img: np.ndarray) -> np.ndarray:
        """
        이미지의 최대 크기를 갖는 변을 self.size로 resize하고 다른 변은 그 비율에 맞게 조정한 후 padding
        >>> self.color에 해당하는 색으로 padding
        """
        return self._apply_padding(img, padding_type="color")

    def aspect_ratio_with_border_replicate(self, img: np.ndarray) -> np.ndarray:
        """
        이미지의 최대 크기를 갖는 변을 self.size로 resize하고 다른 변은 그 비율에 맞게 조정한 후 border replicate
        """
        return self._apply_padding(img, padding_type="border")

    def _apply_padding(self, img: np.ndarray, padding_type: str) -> np.ndarray:
        """
        self.preserving_aspect_ratio() 메서드로 이미지를 resize한 후 padding.
        >>> padding 방법은 2가지로 다음과 같음.
            - color: self.color로 빈 영역을 padding
            - border: border replicate로 padding

        Args:
            img (np.ndarray): 원본 이미지
            padding_type (str): 패딩 방식

        Returns:
            np.ndarray: resize후 패딩된 이미지
        """
        resized_img = self.preserving_aspect_ratio(img)
        if self._check_pass_img(img=resized_img):
            return resized_img
        else:
            if padding_type == "color":
                return base.padding(
                    img=resized_img, pixel=RGBTuple.get(color=self.color)
                )
            elif padding_type == "border":
                return base.padding_border_replicate(img=resized_img)

    def distorting_aspect_ratio(self, img: np.ndarray) -> np.ndarray:
        """
        이미지의 x, y축 비율을 무시하고 self.size에 맞게 resize

        Args:
            img (np.ndarray): 원본 이미지

        Returns:
            np.ndarray: resize된 이미지
        """
        x, y = get_img_axis_size(img)
        return base.resize(img, new_x=self.size, new_y=self.size, quality=self.quality)

    def _check_pass_img(self, img: np.ndarray) -> bool:
        """
        이미지의 x, y축이 이미 self.size와 동일하여 resize가 필요 없는지 확인

        Args:
            img (np.ndarray): 대상 이미지

        Returns:
            bool: True인 경우 pass 가능
        """
        x, y = get_img_axis_size(img)
        return (x == self.size) and (y == self.size)


class Flip:
    def __init__(self, p: float = 1.0, how_list: Optional[List[str]] = None):
        """
        이미지를 p의 확률로 flip 한다.
        >>> how_list에 따라 flip 방식이 바뀐다.
            - ['original', 'vertical', 'horizon', 'both'] 4가지 방법 존재
            - None 입력 시, 위 4가지 방법을 기본값으로 전달

        Args:
            p (float, optional): 이미지를 flip할 확률. Defaults to 1.0.
            how_list (Optional[List[str]], optional): 이미지를 flip할 방법. Defaults to None.
        """
        self.p = p
        self.default_how_list = ["original", "vertical", "horizon", "both"]
        # how_list 평가
        if how_list is None:
            how_list = self.default_how_list
        else:
            self._check_how_list(how_list)
        self.how_list = how_list
        # mapping
        self.how_methods = {
            "original": self.original,
            "horizon": self.horizon,
            "vertical": self.vertical,
            "both": self.both,
        }

    def __call__(self, img: np.ndarray) -> np.ndarray:
        """
        self.how_list에 있는 방법 중 하나로 flip한다.
        >>> 'original', 'vertical', 'horizon', 'both'에 해당하지 않는 방법이 how_list에 있는 경우, default인 'original'로 출력

        Args:
            img (np.ndarray): 원본 이미지

        Returns:
            np.ndarray: flip한 이미지
        """
        if prob.binary_probability(p=self.p):
            # how를 출력하고, how가 self.default_how_list에 포함되지 않는 경우, warning을 출력
            how = self._choose_how()
            fn = self.how_methods.get(how, self.how_methods["original"])
            fliped_img = fn(img)
        else:
            fliped_img = img
        return fliped_img

    def original(self, img: np.ndarray) -> np.ndarray:
        return img

    def vertical(self, img: np.ndarray) -> np.ndarray:
        return cv2.flip(img, 0)

    def horizon(self, img: np.ndarray) -> np.ndarray:
        return cv2.flip(img, 1)

    def both(self, img: np.ndarray) -> np.ndarray:
        return self.vertical(self.horizon(img))

    def _check_how_list(self, how_list: Optional[List[str]]):
        """
        how_list에 기본 값(self.dafault_how_list)에 해당 하는 값이 포함되어 있는지 확인

        Args:
            how_list (Optional[List[str]]): 입력된 how_list, None이 아닌 경우 해당 코드가 실행된다.

        Raises:
            ValueError: how_list가 list이지만 self.dafault_how_list에 있는 값이 하나도 포함되어 있지 않은 경우 발생
            TypeError: list가 입력되지 않은 경우 발생
        """
        txt = "how_list는 'original', 'vertical', 'horizon', 'both' 4개의 string이 포함된 list여야 합니다!"
        if isinstance(how_list, list):
            if len(set(self.default_how_list) & set(how_list)) == 0:
                raise ValueError(txt)
        else:
            raise TypeError(txt)

    def _choose_how(self) -> str:
        """
        self.how_list에 있는 무작위 원소 하나를 선택
        >>> self.dafault_how_list에 없는 방법이 선택되는 경우, 경고문구 출력

        Returns:
            str: 이미지를 flip할 방법
        """
        how = np.random.choice(self.how_list, size=1).item()
        if how not in self.default_how_list:
            txt = "입력하신 how_list에 'original', 'vertical', 'horizon', 'both'이 아닌 다른 값이 포함되어 있습니다. 기본 값인 'original'로 출력합니다."
            warnings.warn(txt)
        return how


class Crop:
    def __init__(
        self,
        p: float = 1.0,
        min: float = 0.1,
        max: float = 0.3,
        color: str = "random",
        crop_style: int = 0,
        crop_style_list: Optional[List[int]] = None,
        box_style: int = 0,
        box_style_list: Optional[List[int]] = None,
        distribution: np.ndarray = None,
    ):
        """
        p의 확률로 이미지를 Crop한다.
        >>> 이미지의 Crop은 총 2가지 종류에 따라 다르게 실시된다 - crop_style, box_style
        >>> crop_style: 이미지를 crop 하는 방식, 총 6가지
            - 1: color로 이미지를 color의 픽셀로 패딩하여 정방 행렬 형태로 만든 후, 중앙에서 crop
            - 2: color로 이미지를 color의 픽셀로 패딩하여 정방 행렬 형태로 만든 후, 무작위 위치에서 crop
            - 3: border replicated 방식으로 패딩하여 정방 행렬 형태로 만든 후, 중앙에서 crop
            - 4: border replicated 방식으로 패딩하여 정방 행렬 형태로 만든 후, 무작위 위치에서 crop
            - 5: 중앙에서 crop
            - 6: 무작위 위치에서 crop
            - 0: 위 6가지 방법 중 하나를 무작위로 선택(crop_style_list를 통해 취사 선택 가능)
        >>> box_style: crop하는 box의 형태, 총 3가지
            - 1: 이미지의 기존 비율을 유지하여 crop
            - 2: 이미지의 가장 작은 축을 기준으로 정방 행렬 형태로 crop
            - 3: 이미지의 모든 축에 대하여 무작위 비율로 crop
            - 0: 위 3가지 방법 중 하나를 무작위로 선택(box_style_list를 통해 취사 선택 가능)

        Args:
            p (float, optional): 이미지를 crop할 확률. Defaults to 1.0.
            min (float, optional): 이미지 crop의 최소 비율. Defaults to 0.1.
            max (float, optional): 이미지 crop의 최대 비율. Defaults to 0.3.
            color (str, optional): padding 시, 빈 영역을 채울 픽셀. Defaults to 'random'.
            crop_style (int, optional): 이미지를 crop 하는 방식. Defaults to 0.
                - default = [1, 2, 3, 4, 5, 6]
            crop_style_list (Optional[List[int]], optional): crop_style이 0인 경우, 무작위로 선택하는 crop_style의 경우의 수. Defaults to None.
                - None인 경우 모든 crop_style 중 한가지로 실행
            box_style (int, optional): 이미지 crop 시, box를 자르는 방식. Defaults to 0.
                - default = [1, 2, 3]
            box_style_list (Optional[List[int]], optional): box_style이 0인 경우, 무작위로 선택하는 box_style의 경우의 수. Defaults to None.
                - None인 경우 모든 box_style 중 한가지로 실행
            distribution (np.ndarray, optional): 이미지 crop의 비율에 대한 분포. Defaults to None.
                - None인 경우 선형 분포를 갖는다.
        """
        self.p = p
        self.color = color
        self.crop_style = crop_style
        self.box_style = box_style

        self.default_crop_style_list = [1, 2, 3, 4, 5, 6]
        self.default_box_style_list = [1, 2, 3]

        self.crop_style_list = crop_style_list or self.default_crop_style_list
        self.box_style_list = box_style_list or self.default_box_style_list

        # style과 style_list 검사
        self._check_error()

        self.methods = {
            1: (True, False, True),
            2: (True, False, False),
            3: (True, True, True),
            4: (True, True, False),
            5: (False, False, True),
            6: (False, False, False),
        }

        # crop 크기에 대한 default 분포 생성
        if distribution is None:
            distribution = dist.linear(min=0, max=1, size=1000)

        # box의 crop 방식에 대한 instance 생성
        self.box_ins = _BoxStyle(
            dist=scaler.min_max_range_normalization(
                array=distribution, min=min, max=max
            ),
            box_style=self.box_style,
            box_style_list=self.box_style_list,
        )

    def __call__(self, img: np.ndarray) -> np.ndarray:
        """
        p의 확률로 지정된 방식으로 이미지 crop

        Args:
            img (np.ndarray): 원본 이미지

        Returns:
            np.ndarray: crop된 이미지
        """
        if prob.binary_probability(p=self.p):
            padding, border_replicated, is_center = self._choose_crop_method()
            if padding:
                img = self.apply_padding(img, border_replicated=border_replicated)
            img = self.apply_crop(img, is_center=is_center)
        return img

    def _choose_crop_method(self) -> Tuple[bool, bool, bool]:
        """
        crop_style의 method를 정하는 bool을 self.methods에서 가져온다.

        Returns:
            Tuple[bool, bool, bool]: padding 여부, border_replicated 패딩 방식 여부, center crop 여부
        """
        if self.crop_style == 0:
            style = prob.choose_one(self.crop_style_list)
            return self.methods[style]
        else:
            return self.methods[self.crop_style]

    def apply_padding(
        self, img: np.ndarray, border_replicated: bool = True
    ) -> np.ndarray:
        """
        padding 적용
        """
        if border_replicated:
            return base.padding_border_replicate(img=img)
        else:
            return base.padding(img=img, pixel=RGBTuple.get(color=self.color))

    def apply_crop(self, img: np.ndarray, is_center: bool = True) -> np.ndarray:
        """
        crop 적용
        """
        x, y = get_img_axis_size(img)
        crop_x, crop_y = self.box_ins(x, y)
        # 시작점을 구한다.
        if is_center:
            start_x, start_y = self._center_start_point(
                x=x, y=y, new_x=crop_x, new_y=crop_y
            )
        else:
            start_x, start_y = self._random_start_point(
                x=x, y=y, new_x=crop_x, new_y=crop_y
            )
        return base.crop(img=img, x=start_x, y=start_y, x_size=crop_x, y_size=crop_y)

    def _center_start_point(
        self, x: int, y: int, new_x: int, new_y: int
    ) -> Tuple[int, int]:
        """
        center crop을 위한 시작 위치 계산

        Args:
            x (int): 원본 x축의 크기
            y (int): 원본 y축의 크기
            new_x (int): crop x축의 크기
            new_y (int): crop y축의 크기

        Returns:
            Tuple[int, int]: x, y축의 crop 시작 위치
        """
        # 시작점 = 중심점 - crop 크기의 반지름
        start_x = int(np.round(x / 2, 0)) - int(np.round(new_x / 2, 0))
        start_y = int(np.round(y / 2, 0)) - int(np.round(new_y / 2, 0))
        return (start_x, start_y)

    def _random_start_point(
        self, x: int, y: int, new_x: int, new_y: int
    ) -> Tuple[int, int]:
        """
        random crop을 위한 시작 위치 계산

        Args:
            x (int): 원본 x축의 크기
            y (int): 원본 y축의 크기
            new_x (int): crop x축의 크기
            new_y (int): crop y축의 크기

        Returns:
            Tuple[int, int]: x, y축의 crop 시작 위치
        """
        start_x = np.random.choice(np.arange(0, x - new_x))
        start_y = np.random.choice(np.arange(0, y - new_y))
        return (start_x, start_y)

    def _check_error(self):
        """
        style과 style_list에 대하여, default 값을 벗어나는지 여부에 대하여 오류 측정
        """
        _CropError.check_style(
            style=self.crop_style, valid_styles=self.default_crop_style_list, key="crop"
        )
        _CropError.check_style(
            style=self.box_style, valid_styles=self.default_box_style_list, key="box"
        )
        _CropError.check_list(
            style_list=self.crop_style_list,
            valid_styles=self.default_crop_style_list,
            key="crop",
        )
        _CropError.check_list(
            style_list=self.box_style_list,
            valid_styles=self.default_box_style_list,
            key="box",
        )


class _BoxStyle:
    def __init__(self, dist: np.ndarray, box_style: int, box_style_list: List[int]):
        """
        crop 되는 크기에 대하여 x, y축에 대하여 각각 크기를 출력

        Args:
            dist (np.ndarray): crop 크기의 분포
            box_style (int): crop 시 box의 유형
            box_style_list (List[int]): crop 시 box 유형들의 list
        """
        self.dist = dist
        self.box_style = box_style
        self.box_style_list = box_style_list

        self.methods = {1: self._keep_ratio, 2: self._square, 3: self._rectangle}

    def __call__(self, x: int, y: int) -> Tuple[int, int]:
        """
        이미지의 x, y축에 대하여 선택된(할) box_style에 대하여 crop할 크기를 반환

        Args:
            x (int): 원본 이미지의 x축
            y (int): 원본 이미지의 y축

        Returns:
            Tuple[int, int]: crop 이미지의 x, y축의 크기
        """
        if self.box_style == 0:
            style = prob.choose_one(self.box_style_list)
            return self.methods[style](x, y)
        else:
            return self.methods[self.box_style](x, y)

    def _keep_ratio(self, x: int, y: int) -> Tuple[int, int]:
        """
        이미지의 x,y축의 비율을 유지하여 crop할 크기 출력
        """
        ratio = prob.choose_one(self.dist)
        crop_x = int(np.round(x * (1 - ratio), 0))
        crop_y = int(np.round(y * (1 - ratio), 0))
        return (crop_x, crop_y)

    def _square(self, x: int, y: int) -> Tuple[int, int]:
        """
        이미지의 가장 작은 축을 기준으로 정방행렬 형태로 crop할 크기 출력
        """
        ratio = prob.choose_one(self.dist)
        min_size = x if x < y else y
        crop_size = int(np.round(min_size * (1 - ratio), 0))
        return (crop_size, crop_size)

    def _rectangle(self, x: int, y: int) -> Tuple[int, int]:
        """
        이미지의 x, y축 모두에서 무작위 비율로 crop할 크기 출력
        """
        ratio_x = prob.choose_one(self.dist)
        ratio_y = prob.choose_one(self.dist)
        crop_x = int(np.round(x * (1 - ratio_x), 0))
        crop_y = int(np.round(y * (1 - ratio_y), 0))
        return (crop_x, crop_y)


class _CropError:
    """
    Crop class의 crop_style, crop_style_list, box_style, box_style_list에 오류가 없는지 확인
    """

    @classmethod
    def check_list(
        self, style_list: List[int], valid_styles: List[int], key: str = "box"
    ):
        """
        style_list이 list인지, 기본값이 아닌 값이 존재하는지 확인

        Args:
            style_list (List[int]): crop, box에 대하여 선택될 수 있는 style의 list
            valid_styles (List[int]):crop, box에 대하여 모든 style의 list
            key (str, optional): 경고 문구 출력을 위한 key값(box인지 crop인지). Defaults to 'box'.

        Raises:
            TypeError: style_list가 list가 아님
            ValueError: style_list에 기본값이 아닌 값이 존재하는 경우
        """
        if not isinstance(style_list, list):
            raise TypeError(
                f"{key}의 style_list는 반드시 {valid_styles}에 포함된 값으로 구성된 list여야 합니다!"
            )

        invalid_items = [i for i in style_list if i not in valid_styles]
        if invalid_items:
            raise ValueError(
                f"{key}의 style_list에는 유효하지 않은 값 {invalid_items}가 포함되어 있습니다. 반드시 {valid_styles}에 포함된 값으로 구성되어야 합니다!"
            )

    @classmethod
    def check_style(self, style: int, valid_styles: List[int], key="box"):
        """
        style에 0을 포함한 기본값이 아닌 값이 존재하는지 확인

        Args:
            style (int): crop, box의 증강 방식에 대한 정수 값.
            valid_styles (List[int]): crop, box의 증강 방식으로 선택될 수 있는 모든 값(0 포함)
            key (str, optional): 경고 문구 출력을 위한 key값(box인지 crop인지). Defaults to 'box'.

        Raises:
            ValueError: style에 0을 포함한 기본값이 아닌 값이 존재하는 경우
        """
        extended_styles = [0] + valid_styles
        if style not in extended_styles:
            raise ValueError(
                f"{key}의 style은 반드시 {extended_styles}에 포함된 정수 중 하나로 선택되어야 합니다!"
            )
