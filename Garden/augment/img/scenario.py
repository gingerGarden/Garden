from typing import List, Optional

import numpy as np
from .geometric import distortion as distor
from .geometric import frame
from .geometric import geometric as geo
from ...statics import probability as prob


class RandomScenario:
    def __init__(self, *args, p: float = 0.5, p_list: Optional[List[int]] = None):
        """
        RandomScenario에 입력된 임의의 증강 알고리즘 중 하나를 p의 확률로 무작위 선택하여 적용
        >>> p_list
            - 입력된 임의의 증강 알고리즘들의 적용 비율을 조정하고자 하는 경우, p_list로 조정 가능
            - ex) test_ins = RandomScenario(fn1, fn2, fn3, p_list=[3, 2, 5])
                메서드 fn1, fn2, fn3에 대하여 3:2:5 비율로 무작위 선택
            - 입력된 증강 알고리즘의 수와 p_list의 크기는 동일해야함

        Args:
            p (float, optional): 무작위 증강 알고리즘 적용 확률. Defaults to 0.5.
            p_list (Optional[List[int]], optional): 증강 알고리즘 별. Defaults to None.
        """
        self.fn_tuple = args
        self.p = p
        self.p_list = self._make_p_list(p_list)

    def __call__(self, img: np.ndarray) -> np.ndarray:
        """
        Scenario에 입력된 증강 알고리즘 중 하나를 무작위로 적용

        Args:
            img (np.ndarray): 원본 이미지

        Returns:
            np.ndarray: 증강된 이미지
        """
        if prob.binary_probability(p=self.p):
            how = np.random.choice(self.p_list)
            return self.fn_tuple[how](img)
        return img

    def _make_p_list(self, p_list: List[int]) -> List[int]:
        """
        입력된 증강 알고리즘들의 확률
            - p_list가 None인 경우, self.fn_tuple의 index로 생성

        Args:
            p_list (List[int]): 증강 알고리즘 별 가중치에 대한 list

        Raises:
            ValueError: 증강 알고리즘의 수와 p_list의 크기가 동일하지 않은 경우

        Returns:
            List[int]: 증강 알고리즘 index의 list
        """
        if p_list is not None:
            if len(self.fn_tuple) != p_list:
                raise ValueError(
                    "Augmentation method와 p_list의 길이가 다릅니다! 길이를 확인하십시오!"
                )
            return [i for i, size in enumerate(p_list) for _ in range(size)]
        else:
            return [i for i, _ in enumerate(self.fn_tuple)]


class OrderScenario:
    def __init__(self, *args):
        """
        GGImgMorph의 증강 알고리즘들을 입력한 순서대로 img 증강 적용
        """
        self.fn_list = args

    def __call__(self, img: np.ndarray) -> np.ndarray:
        """
        Scenario에 입력된 증강 알고리즘 순서대로 img 증강

        Args:
            img (np.ndarray): 원본 이미지

        Returns:
            np.ndarray: 증강된 이미지
        """
        for fn in self.fn_list:
            img = fn(img)
        return img


# augment 예제
##############################
_color = "random"
_aug_1 = OrderScenario(geo.Flip(p=0.75), geo.Rotate(p=0.4, color=_color))
_aug_2 = RandomScenario(
    geo.Shear(color=_color, keep_window=0),
    geo.Translate(color=_color),
    geo.Scaling(how=0, how_list=[1, 3, 4]),
    distor.Perspective(min=-0.2, max=0.2, color=_color),
    frame.Circle(color=_color),
    frame.Square(min=0.1, max=0.3),
    geo.Resize(how=4),
    geo.Crop(p=1.0, crop_style=0, crop_style_list=[5, 6]),  # zoom
    p=0.8,
)
_aug_last = OrderScenario(geo.Crop(p=0.4, color=_color))
sample_scenario = OrderScenario(_aug_1, _aug_2, _aug_last)
