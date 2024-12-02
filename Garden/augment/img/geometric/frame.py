import random
import warnings
from typing import Callable, Dict, List, Optional, Tuple

import GGStatify.distribution as dist
import GGStatify.probability as prob
import numpy as np
from GGStatify import scaler
from GGUtils.img.img import RGBTuple, get_img_axis_size, img_center

from ..mask import mask as Mask


class Frame:
    def __init__(self, p: float = 1.0, **kwargs):
        self.p = p
        self.fn_dict = kwargs
        self.fn_keys = list(self.fn_dict.keys())

    def __call__(self, img):
        if prob.binary_probability(p=self.p):
            fn = self.fn_dict[prob.choose_one(self.fn_keys)]
            return fn(img)
        return img


class Circle:
    def __init__(
        self,
        p: float = 1.0,
        min: float = 0.05,
        max: float = 0.2,
        how: int = 0,
        how_list: Optional[List] = [1, 2],
        color: str = "random",
        adjustment: float = 0.95,
        distribution: Optional[np.ndarray] = None,
    ):
        """
        이미지에 원으로 frame을 생성한다.
        >>> how: frame 생성 방식은 크게 4가지로 다음과 같다.
            - 1: 이미지의 가장 긴 변을 기준으로 이미지의 중심에 원을 그린다.
            - 2: 이미지의 가장 긴 변을 기준으로 무작위 위치에 원을 그린다.
            - 3: 이미지의 가장 짧은 변을 기준으로 이미지의 중심에 원을 그린다(이미지가 길수록 주요 정보 손실 위험 존재).
            - 4: 이미지의 가장 짧은 변을 기준으로 무작위 위치에 원을 그린다(이미지가 길수록 주요 정보 손실 위험 존재).

        Args:
            p (float, optional): 이미지에 원으로 된 frame을 생성할 확률. Defaults to 1.0.
            min (float, optional): frame 반지름의 최소 비율. Defaults to 0.05.
            max (float, optional): frame 반지름의 최대 비율. Defaults to 0.2.
            how (int, optional): 이미지에 frame을 생성하는 방법. Defaults to 0.
            how_list (Optional[List], optional): 무작위로 이미지에 frame 생성 시(how=0), frame을 그리는 방법. Defaults to [1,3].
                - 이미지가 긴 경우(직사각형), 3, 4 방법으로 frame을 그리면, 주요 Freature가 잘릴 수 있음.
                - 이미지가 정사각형에 가까울수록 1, 2, 3, 4 모든 방법을 사용해도 큰 문제가 없으나, 3과 4의 경우, 1과 2와 비율 차이만 존재.
                - 위 이유로 1, 2 방법을 기본값으로 하며, 해당 방법만 사용하는 것을 추천.
            color (str, optional): frame의 바깥 부분에 칠해지는 색. Defaults to 'random'.
            color (float, optional): 무작위로 중심 좌표를 그릴 때, 수치가 0.5에 가까울수록 중심 좌표가 이미지 중앙 근처로 간다. Defaults to 0.95.
                - how = 2에서 frame이 이미지를 가리는 정도에 대한 보정치이다.
                - 수치가 0.5에 가까워질수록 frame이 이미지를 가리지 않을 확률이 올라간다.
                - 0.5 이하인 경우, 범위를 벗어나므로 반드시 0.5를 초과해야한다.
            distribution (Optional[np.ndarray], optional): frame 비율에 대한 분포. Defaults to None.
        """
        self.p = p
        self.how = how
        self.color = color
        self.adjustment = adjustment
        if distribution is None:
            distribution = dist.linear(min=0, max=1, size=1000)
        self.dist = scaler.min_max_range_normalization(
            array=distribution, min=min, max=max
        )

        self.how_map = {
            0: self._how_0_extract_bools,
            1: lambda: (True, True, False),
            2: lambda: (True, False, True),
            3: lambda: (False, True, False),
            4: lambda: (False, False, True),
        }
        self.how_keys = None
        self._make_how_keys()

        self.how_list = self.how_keys if how_list is None else how_list

        # self.how, self.how_list check
        self._check_how()
        self._check_how_list()
        self._check_adjustment()

    def __call__(self, img: np.ndarray) -> np.ndarray:
        """
        p의 확률로 이미지에 how(how_list)의 방법으로 frame을 생성

        Args:
            img (np.ndarray): 원본 이미지

        Returns:
            np.ndarray: frame이 생성된 이미지
        """
        if prob.binary_probability(p=self.p):
            return self._make_frame(img)
        return img

    def _make_frame(self, img: np.ndarray) -> np.ndarray:
        """
        원본 이미지에 frame을 그리는 process
        """
        x, y = get_img_axis_size(img)
        # 반지름, 중심 좌표 계산
        is_max, make_half, is_random = self.how_map[self.how]()  # 이미지 변환 값
        radius = self._make_radius(x, y, is_max, make_half)
        center = self._make_center(x, y, radius, is_random)
        # 이미지에 mask를 생성하고 칠한다.
        mask = Mask.Circle.make_mask(img, center, radius)
        return Mask.Circle.add_mask(img, mask, color="random")

    def _make_radius(self, x: int, y: int, is_max: bool, make_half: bool) -> int:
        """
        반지름 계산
            - is_max인 경우, 가장 긴 변을 기준으로 반지름 계산(반대는 가장 짧은 변)
            - make_half인 경우, 반지름의 반지름을 계산하여, 반드시 이미지 안에 원의 frame이 존재하게 함
        """
        size = max(x, y) if is_max else min(x, y)
        if make_half:
            size = size / 2
        return int(np.trunc(size * (1 - prob.choose_one(self.dist))))

    def _make_center(
        self, x: int, y: int, radius: int, is_random: bool
    ) -> Tuple[int, int]:
        """
        중심 좌표 계산
            - is_random인 경우, 반드시 frame에 존재하는 위치에 대하여 중심 좌표 계산(반대는 이미지의 중심 좌표와 동일)
        """
        if is_random:
            new_x = self._choose_random_point(size=x, radius=radius)
            new_y = self._choose_random_point(size=y, radius=radius)
            return (new_x, new_y)
        else:
            return img_center(x=x, y=y)

    def _choose_random_point(self, size: int, radius: int) -> int:
        """
        한축에 대한 무작위 중심 좌표 추출
            - 변의 길이가 반지름보다 긴 경우, 변과 반지름의 차이 사이로 무작위 중심 좌표 추출
            - 변의 길이가 반지름보다 짧은 경우, 전체 범위 내에서 무작위 중심 좌표 추출
        """
        adjustment_radius = int(np.trunc(radius * self.adjustment))
        adjustment_size = int(np.trunc(size * self.adjustment))

        if size > radius:
            if prob.binary_probability(p=0.5):
                points = np.arange(0, size - adjustment_radius)
            else:
                if adjustment_radius == adjustment_size:
                    adjustment_size += 1
                points = np.arange(adjustment_radius, adjustment_size)
        else:
            points = np.arange(size - adjustment_size, adjustment_size)

        return prob.choose_one(array=points)

    def _how_0_extract_bools(self):
        """
        how가 0인 경우, how_list에 있는 방법 중 하나를 선택
        """
        how = prob.choose_one(self.how_list)
        return self.how_map[how]()

    def _make_how_keys(self):
        """
        how_map에서 0을 제외한 나머지 값을 list로 하여, 무작위 선택에 용이하게 함.
        """
        self.how_keys = list(self.how_map.keys())
        self.how_keys.remove(0)

    def _check_how(self):
        """
        how에 잘못된 값이 입력되었는지 확인
        """
        check_list = [0] + self.how_keys
        if self.how not in check_list:
            raise ValueError(
                f"how에 잘못된 값인 {self.how}가 입력되었습니다! {check_list}에 해당하는 값을 입력해야 합니다!"
            )

    def _check_how_list(self):
        """
        how_list에 잘못된 값이 입력되었는지 확인
        """
        if not isinstance(self.how_list, list):
            raise TypeError("how_list에는 반드시 list가 입력되어야 합니다!")
        if len(self.how_list) == 0:
            raise ValueError(
                "how_list에는 반드시 한 개 이상의 원소가 포함되어야 합니다!"
            )
        if len(set(self.how_list) - set(self.how_keys)) > 0:
            raise ValueError(
                f"how_list에는 {self.how_keys}에 포함되는 값만 입력되어야 합니다!"
            )

    def _check_adjustment(self):
        if self.adjustment == 1.0:
            warnings.warn(
                "adjustment에 1.0이 입력되는 경우, 에러가 발생할 수 있습니다. adjustment를 0.99로 수정합니다."
            )
            self.adjustment = 0.99
        if self.adjustment <= 0.5:
            raise ValueError(
                "ajustment에는 반드시 0.5를 초과하는 값이 입력되어야 하며, 0.5에 가까워질수록 frame이 이미지 안에 보이지 않을 수 있습니다!"
            )


class Square:
    def __init__(
        self,
        p: float = 1.0,
        min: float = 0.05,
        max: float = 0.3,
        how: int = 0,
        shuffle: int = 0,
        color: str = "random",
        distribution: Optional[np.ndarray] = None,
    ):
        """
        이미지에 네모로 frame을 생성한다.
        >>> how: frame 생성 방식은 6가지로 다음과 같다.
            - 1: 위, 아래, 왼쪽, 오른쪽 모두 무작위 비율로 frame을 그린다.
            - 2: (위, 아래), (왼쪽, 오른쪽) 2 세트에 대하여 무작위 비율로 frame을 그린다.
            - 3: 위, 아래, 왼쪽, 오른쪽에 대하여 최소 축을 기준으로 무작위 비율로 frame을 그린다.
            - 4: 위, 아래, 왼쪽, 오른쪽에 대하여 최대 축을 기준으로 무작위 비율로 frame을 그린다.
            - 5: 위, 아래에 대하여 각각 무작위 비율로 frame을 그린다.
            - 6: 왼쪽, 오른쪽에 대하여 각각 무작위 비율로 frame을 그린다.
            - 0: 위 6가지 방법 중 하나를 무작위로 선택한다.
        >>> shuffle: frame을 색칠하는 방법은 2가지로 다음과 같다.
            - 1: color가 'random'인 경우, 4가지 축을 모두 다른 색으로 무작위 순서로 칠한다.
            - 2: 4가지 축을 모두 같은 색으로 칠한다.
            - 0: 위 2가지 방법 중 하나를 무작위로 선택한다.

        Args:
            p (float, optional): 이미지에 frame을 생성할 확률. Defaults to 1.0.
            min (float, optional): frame의 최소 비율. Defaults to 0.05.
            max (float, optional): frame의 최대 비율. Defaults to 0.3.
            how (int, optional): 이미지에 frame을 그리는 방법. Defaults to 0.
            shuffle (int, optional): 이미지의 frame을 색칠하는 방법. Defaults to 0.
            color (str, optional): 이미지의 frame 색깔. Defaults to 'random'.
            distribution (Optional[np.ndarray], optional): 이미지의 frame 비율에 대한 분포. Defaults to None.
        """
        self.p = p
        self.color = color
        # size_map
        if distribution is None:
            distribution = dist.linear(min=0, max=1, size=1000)
        self.size_methods = _SquareSizeMethod(
            distribution=scaler.min_max_range_normalization(
                array=distribution, min=min, max=max
            )
        )
        self.how = self._error_check_how(
            value=how, valid_map=self.size_methods.size_map, key="size_map"
        )

        # shuffle_map
        self.shuffle_map = {
            0: lambda: prob.binary_probability(p=0.5),
            1: lambda: True,
            2: lambda: False,
        }
        self.shuffle = self._error_check_how(
            value=shuffle, valid_map=self.shuffle_map, key="shuffle"
        )

    def __call__(self, img: np.ndarray) -> np.ndarray:
        """
        p의 확률로 이미지에 대하여 네모로 된 frame을 생성한다.

        Args:
            img (np.ndarray): 원본 이미지

        Returns:
            np.ndarray: 네모 frame이 추가된 이미지
        """
        if prob.binary_probability(p=self.p):
            return self._make_frame(img)
        return img

    def _make_frame(self, img: np.ndarray) -> np.ndarray:
        """
        img에 frame을 생성하는 process
        """
        img = img.copy()
        x, y = get_img_axis_size(img)
        # how 설정 - size_map
        size_tuple = self.size_methods.size_map[self.how](x, y)
        size_dict = self._size_tuple_to_size_dict(size_tuple)
        # frame을 칠한다.
        self._fill_frame(img, x, y, size_dict)
        return img

    def _fill_frame(self, img: np.ndarray, x: int, y: int, size_dict: Dict[str, int]):
        """
        shuffle의 규칙에 맞게 frame을 칠한다.
        """
        # frame을 채우는 순서를 섞을지 여부
        shuffle = self.shuffle_map[self.shuffle]()
        fill_order = self._get_size_dict_keys(size_dict=size_dict, shuffle=shuffle)
        # default color 정의
        color = RGBTuple.get(self.color)
        # frame을 칠한다.
        for key in fill_order:
            current_color = RGBTuple.get(self.color) if shuffle else color
            self._fill_each_frame(img, x, y, size_dict, key, color=current_color)

    def _get_size_dict_keys(
        self, size_dict: Dict[str, int], shuffle: bool
    ) -> List[str]:
        """
        shuffle 여부에 따라 size_dict의 key를 섞어서 출력한다.
        """
        key_list = list(size_dict.keys())
        if shuffle:
            random.shuffle(key_list)
        return key_list

    def _fill_each_frame(
        self,
        img: np.ndarray,
        x: int,
        y: int,
        size_dict: Dict[str, int],
        key: str,
        color: Tuple[int, int, int],
    ):
        """
        key에 맞춰서 해당하는 frame에 color를 칠한다.
        """
        size = size_dict[key]
        slice_map = {
            "top": (slice(0, size), slice(None)),
            "bottom": (slice(y - size, y), slice(None)),
            "left": (slice(None), slice(0, size)),
            "right": (slice(None), slice(x - size, x)),
        }
        if len(img.shape) == 3:
            img[slice_map[key][0], slice_map[key][1], :] = color
        else:
            img[slice_map[key][0], slice_map[key][1]] = color[0]

    def _size_tuple_to_size_dict(
        self, size_tuple: Tuple[int, int, int, int]
    ) -> Dict[str, int]:
        """
        size_tuple(top, bottom, left, right)을 dictionary로 변환
        """
        return {
            "top": size_tuple[0],
            "bottom": size_tuple[1],
            "left": size_tuple[2],
            "right": size_tuple[3],
        }

    def _error_check_how(
        self, value: int, valid_map: Dict[int, Callable], key: str = "size_map"
    ):
        """
        self.how가 self.size_map나 self.shuffle에 self.shuffle_map에 벗어나는 값이 입력되었는지 확인
        """
        if value not in valid_map:
            txt = f"Invalid '{key}' value {value}. Valid values are {list(valid_map.keys())}. Using default (0)."
            warnings.warn(txt)
            return 0
        return value


class _SquareSizeMethod:
    def __init__(self, distribution):
        """
        이미지의 frame을 칠하는 방법들에 대한 방법들이 들어 있는 class
        - _Square에서 사용되는 주요 class
        """
        self.dist = distribution
        self.size_map = {
            0: self._how_0_frame_size,
            1: self._how_1_frame_size,
            2: self._how_2_frame_size,
            3: self._how_3_frame_size,
            4: self._how_4_frame_size,
            5: self._how_5_frame_size,
            6: self._how_6_frame_size,
        }
        self.size_map_keys = None
        self._make_size_map_keys()

    def _make_size_map_keys(self):
        """
        self.size_map의 key에서 0을 제외하여, self.size_maep_keys를 생성한다.
        """
        self.size_map_keys = list(self.size_map.keys())
        self.size_map_keys.remove(0)

    def _how_0_frame_size(self, x: int, y: int) -> Tuple[int, int, int, int]:
        """
        self.size_map의 0을 제외한 하나의 방법을 무작위로 선택하여 반환한다.
        """
        how = np.random.choice(self.size_map_keys)
        return self.size_map[how](x, y)

    def _how_1_frame_size(self, x: int, y: int) -> Tuple[int, int, int, int]:
        """
        top, bottom, left, right 모두 무작위 비율로 반환
        """
        return (
            self._random_size(y),
            self._random_size(y),
            self._random_size(x),
            self._random_size(x),
        )

    def _how_2_frame_size(self, x: int, y: int) -> Tuple[int, int, int, int]:
        """
        top, bottom을 세트로 left, right을 세트로 하여 각 세트에 대해 무작위 비율로 반환
        """
        return (self._random_size(y),) * 2 + (self._random_size(x),) * 2

    def _how_3_frame_size(self, x: int, y: int) -> Tuple[int, int, int, int]:
        """
        x, y축 중 가장 작은 변의 길이를 기준으로 동일한 무작위 비율로 반환
        """
        return (self._random_size(min(x, y)),) * 4

    def _how_4_frame_size(self, x: int, y: int) -> Tuple[int, int, int, int]:
        """
        x, y축 중 가장 큰 변의 길이를 기준으로 동일한 무작위 비율로 반환
        """
        return (self._random_size(max(x, y)),) * 4

    def _how_5_frame_size(self, x: int, y: int) -> Tuple[int, int, int, int]:
        """
        top, bottom 각각 무작위 비율로 반환(left, right는 0)
        """
        return (self._random_size(y), self._random_size(y), 0, 0)

    def _how_6_frame_size(self, x: int, y: int) -> Tuple[int, int, int, int]:
        """
        left, right 각각 무작위 비율로 반환(top, bottom는 0)
        """
        return (0, 0, self._random_size(x), self._random_size(x))

    def _random_size(self, size: int) -> int:
        """
        top, bottom, left, right의 무작위 비율로 frame 크기 정의
            - 각 축의 절반에 대해서만 정의하므로 adjustment로 나눔
        """
        adjustment = 2  # 각 축의 frame에 대한 것이므로, 절반을 고려하여 나눔
        return int(np.round((size * prob.choose_one(self.dist)) / adjustment, 0))
