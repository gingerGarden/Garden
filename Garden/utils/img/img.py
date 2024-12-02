from typing import Any, Callable, List, Optional, Tuple

import cv2
import numpy as np
import torch

from ..img.viewer import Slider
from ..path import GetAbsolutePath


def get_rgb_img(img_path: str, to_rgb: bool = True) -> np.ndarray:
    """cv2를 이용하여 BGR 이미지를 RGB로 가지고 온다.

    Args:
        img_path (str): 이미지 파일의 경로
        to_rgb (bool, optional): cv2가 BGR로 기본적으로 가지고 오는 이미지를 RGB로 가지고 온다.
            Defaults to True.

    Returns:
        np.ndarray: cv2를 이용하여 가지고 온 이미지의 numpy 배열
    """
    img_array = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if to_rgb:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    return img_array


def get_img_axis_size(img: np.ndarray) -> Tuple[int, int]:
    """
    입력 이미지의 각 x, y축의 길이를 반환한다.
    >>> 입력 이미지는 2D 이미지인 흑백, RGB, RGBA 이미지 등을 대상으로 한다.
        만약, 해당 범위를 벗어나는 이미지가 입력되는 경우, ValueError가 발생한다.

    Args:
        img (np.ndarray): 2D 이미지로 기대되는 배열

    Raises:
        ValueError: 2D 이미지가 아닌 경우, ValueError 발생

    Returns:
        Tuple[int, int]: x, y 축의 길이 반환
    """
    shape_tuple = img.shape
    if len(shape_tuple) == 2:
        y, x = shape_tuple
    elif len(shape_tuple) == 3:
        y, x, _ = shape_tuple
    else:
        raise ValueError(
            "해당 함수는 2D 이미지(흑백, RGB, RGBA 등)를 대상으로 하고 있습니다. 입력된 배열은 해당 범위를 벗어납니다."
        )
    return x, y


def check_rgb_or_rgba(img: np.ndarray):
    """
    입력 이미지가 BGR, RGB, RGBA에 포함되는지 확인한다.

    Args:
        img (np.ndarray): BGR, RGB, RGBA에 해당하는지 shape으로 확인

    Raises:
        ValueError: 3개의 차원을 갖는 이미지이면서, 3번째 차원의 크기가 3 또는 4인지 확인. 해당하지 않는 경우 ValueError 발생
    """
    mask1 = len(img.shape) == 3
    mask2 = (img.shape[2] == 3) or (img.shape[2] == 4) if mask1 else False
    if not (mask1 and mask2):
        raise ValueError(
            "해당 함수의 입력 이미지는 BGR, RGB, RGBA를 대상으로 합니다. 입력 이미지의 shape을 확인해주십시오."
        )


def size_tuple_type_error(size_tuple: Optional[Tuple[int, int]]):
    """
    size_tuple은 이미지 크기에 대한 tuple로 x(int)와 y(int) 두 개의 정수로 구성된다.
    만약, 해당 구성을 벗어나는 경우 TypeError를 발생시킨다.

    Args:
        size_tuple (Optional[Tuple[int, int]]): (x, y) 두 개의 정수 원소로 구성된 Tuple

    Raises:
        TypeError: size_tuple의 형식에서 벗어나는 경우, 오류 발생
    """
    if size_tuple is not None:
        mask1 = isinstance(size_tuple, tuple)
        mask2 = len(size_tuple) == 2 if mask1 else False
        mask3 = (
            isinstance(size_tuple[0], int) and isinstance(size_tuple[1], int)
            if mask2
            else False
        )
        if not (mask1 & mask2 & mask3):
            raise TypeError(
                "size_tuple은 (int, int) 형식의 튜플 또는 None이 입력되어야 합니다."
            )


def img_corners(
    img: Optional[np.ndarray] = None,
    x: Optional[int] = None,
    y: Optional[int] = None,
    dtype: type = np.float32,
) -> np.ndarray:
    """
    입력 img의 x, y축을 기준으로 dtype으로 4개의 꼭지점을 넘파이 배열로 가지고 온다.

    Args:
        img (Optional[np.ndarray], optional): 원본 이미지. Defaults to None.
            - 이미지의 x, y축을 반환하는 get_img_axis_size() 메서드는 입력 이미지의 무결성을 확인하는 절차가 있으므로, 이것이 필요하지 않은 경우, x, y 값을 바로 입력
        x (Optional[int], optional): 이미지의 x축 크기. Defaults to None.
        y (Optional[int], optional): 이미지의 y축 크기. Defaults to None.
        dtype (type, optional): 이미지 꼭지점 좌표 배열의 dtype. Defaults to np.float32.

    Raises:
        ValueError: img 또는 x, y의 값이 None인 경우 발생

    Returns:
        np.ndarray: 이미지 꼭지점 행렬(4, 2) 좌표
    """
    if img is not None:
        x, y = get_img_axis_size(img=img)
    if (img is None) and (x is None or y is None):
        raise ValueError(
            "img, x, y의 모든 값들이 None으로 입력되지 않았습니다! img 또는 x, y의 값을 입력하십시오!"
        )
    return np.array([[0, 0], [x, 0], [0, y], [x, y]], dtype=dtype)


def img_center(
    img: Optional[np.ndarray] = None, x: Optional[int] = None, y: Optional[int] = None
) -> Tuple[int, int]:
    """
    이미지의 중심 좌표를 x, y축에 대하여 Tuple로 출력한다.

    Args:
        img (Optional[np.ndarray], optional): 원본 이미지. Defaults to None.
            - 이미지의 x, y축을 반환하는 get_img_axis_size() 메서드는 입력 이미지의 무결성을 확인하는 절차가 있으므로, 이것이 필요하지 않은 경우, x, y 값을 바로 입력
        x (Optional[int], optional): 이미지의 x축 크기. Defaults to None.
        y (Optional[int], optional): 이미지의 y축 크기. Defaults to None.

    Raises:
        ValueError: img 또는 x, y의 값이 None인 경우 발생

    Returns:
        Tuple[int, int]: 이미지의 중심 좌표 튜플(x, y)
    """
    if img is not None:
        x, y = get_img_axis_size(img=img)
    if (img is None) and (x is None or y is None):
        raise ValueError(
            "img, x, y의 모든 값들이 None으로 입력되지 않았습니다! img 또는 x, y의 값을 입력하십시오!"
        )
    return (x // 2, y // 2)


def img_to_tensor(
    img: np.ndarray,
    to_float: bool = True,
    max_pixel: float = 255,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    입력된 이미지를 Torch의 tensor 형식으로 변환
    >>> 일반 이미지: (Y, X, C)
        Torch 이미지: (C, Y, X)

    Args:
        img (np.ndarray): 원본 이미지
        to_float (bool, optional): 0에서 1사이의 실수로 변환 여부. Defaults to True.
        max_pixel (float, optional): 실수 변환을 위한 최대 픽셀 값. Defaults to 255.
            - RGB 또는 RGBA를 기본 이미지로 하므로, 기본값으로 255로 함.
        dtype (torch.dtype, optional): 실수의 dtype. Defaults to torch.float32.

    Returns:
        torch.Tensor: Torch 스타일로 변환된 이미지의 텐서
    """
    if to_float:
        img = img / max_pixel
    if len(img.shape) == 2:
        img = np.expand_dims(
            img, axis=-1
        )  # x, y축만 갖는 이미지(흑백 이미지)인 경우, 1차원의 채널을 추가함.
    # array를 tensor로 변환
    torch_img = torch.as_tensor(img, dtype=dtype)
    return torch_img.permute(2, 0, 1)  # (C, Y, X) 형식으로 변환


def tensor_to_img(
    tensor_img: torch.Tensor, to_int: bool = True, max_pixel: int = 255, dtype=np.uint8
) -> np.ndarray:
    """
    입력된 이미지 tensor를 numpy 배열 이미지로 변환
    >>> to_int가 True인 경우, max_pixel로 곱하여, 정수 이미지로 변환.

    Args:
        tensor_img (torch.Tensor): torch.Tensor type의 이미지.
        to_int (bool, optional): int로 변환 여부. Defaults to True.
        max_pixel (int, optional): int로 변환 시 곱하는 값. Defaults to 255.
        dtype (_type_, optional): int로 변환 된 이미지의 dtype. Defaults to np.uint8.

    Returns:
        np.ndarray: _description_
    """
    assert isinstance(tensor_img, torch.Tensor), "Input must be a torch.Tensor"

    if tensor_img.device != torch.device("cpu"):
        tensor_img = tensor_img.detach().cpu()

    np_img = tensor_img.numpy()
    np_img = np.transpose(np_img, (1, 2, 0))
    if to_int:
        return (np_img * max_pixel).astype(dtype)
    return np_img


class RGBTuple:
    @classmethod
    def get(cls, color: str = "random") -> Tuple[int, int, int]:
        """
        color에 해당하는 pixel tuple 출력

        Args:
            color (str, optional): color에 대한 string. Defaults to 'random'.
                color는 기본적으로 다음과 같은 string을 입력받는다. 아래 string의 범위를 벗어나는 경우,
                'random'으로 가지고 온다.
                >>> 'random', 'red'('r'), 'green'('g'), 'blue'('b'), 'yellow'('y'), 'sky'('s'),
                    'purple'('p'), 'white'('w'), 'black'('zero', 'z')

        Returns:
            Tuple[int, int, int]: 0~255까지의 정수인 R, G, B로 이루어진 Tuple
        """
        if color == "random":
            color = cls.random()
        if (color == "red") | (color == "r"):
            color = (255, 0, 0)
        elif (color == "green") | (color == "g"):
            color = (0, 255, 0)
        elif (color == "blue") | (color == "b"):
            color = (0, 0, 255)
        elif (color == "yellow") | (color == "y"):
            color = (255, 255, 0)
        elif (color == "sky") | (color == "s"):
            color = (0, 255, 255)
        elif (color == "purple") | (color == "p"):
            color = (255, 0, 255)
        elif (color == "white") | (color == "w"):
            color = (255, 255, 255)
        elif (color == "black") | (color == "zero") | (color == "z"):
            color = (0, 0, 0)
        else:
            color = cls.random()
        return color

    @classmethod
    def check(cls, target: Any) -> bool:
        """입력된 target이 RGB tuple에 해당하는지 확인

        Args:
            target (Any): RGB tuple이길 기대하는 입력값

        Returns:
            bool: target이 RGB tuple인지 여부
        """
        return (
            isinstance(target, tuple)
            and len(target) == 3
            and all(isinstance(i, int) for i in target)
            and all(0 <= i <= 255 for i in target)
        )

    @classmethod
    def random(cls) -> Tuple[int, int, int]:
        """0~255 사이의 정수 3개(RGB)로 이루어진 무작위 Tuple 생성

        Returns:
            Tuple[int, int, int]: RGB pixel tuple
        """
        r = np.random.choice(np.arange(0, 256), size=1).item()
        g = np.random.choice(np.arange(0, 256), size=1).item()
        b = np.random.choice(np.arange(0, 256), size=1).item()
        return (r, g, b)


class GetInDirectory:
    def __init__(
        self,
        dir_path: str,
        read_fn: Callable[..., Any] = get_rgb_img,
        img_extensions: List[str] = ["png", "jpeg", "jpg"],
        *args,
        **kwargs,
    ):
        """
        콜러블 메서드를 주 방법으로 사용하는 class로, dir_path 내 콜 메서드에서 입력받는 index에 해당하는 이미지를
        반환한다.

        Args:
            dir_path (str): 이미지들이 들어가 있는 디렉터리의 경로
            get_fn (Callable[..., Any]): 이미지를 가져오는 방법에 대한 임의의 함수. Defaults to get_rgb_img.
            img_extensions (List[str], optional): 이미지의 확장자. Defaults to ['png', 'jpeg', 'jpg'].
        """
        self.dir_path = dir_path
        self.read_fn = read_fn
        self.args = args  # tuple로 처리
        self.kwargs = kwargs  # dictionary로 처리

        # dir_path 내 image 파일들에 대하여 절대 경로를 가지고 온다.
        self.img_pathes = None
        self._make_img_list(parents_path=dir_path, extensions=img_extensions)

        # 모든 이미지들이 담긴 list - slide_viewer() 메서드 실행 시
        self.img_list = None

    def __call__(self, index: int) -> np.ndarray:
        """
        self.dir_path의 index에 해당하는 이미지를 self.read_fn 메서드를 사용하여 출력한다.

        Args:
            index (int): 이미지의 번호(문자에 대한 오름차순 기반)

        Returns:
            np.ndarray: 이미지의 배열
        """
        img_size = len(self.img_pathes)
        if index > img_size - 1 or index < 0:
            raise ValueError(
                f"입력된 index {index}는 총 이미지의 개수인 {img_size}의 범위를 넘어섭니다(index는 0부터이므로 {img_size-1}까지 입력). index를 수정하여 입력하십시오."
            )
        img_path = self.img_pathes[index]
        return self.read_fn(img_path, *self.args, **self.kwargs)

    def _make_img_list(self, parents_path: str, extensions: List[str]):
        """
        이미지들이 들어가 있는 디렉터리에서 이미지들의 절대 경로를 가지고 온다. 가져온 이미지들은 항상 동일한 결과가
        가지고 와지도록 오름차순 정렬한다.

        Args:
            parents_path (str): 이미지 파일들이 들어가 있는 부모 디렉터리의 경로
            extensions (List[str]): 이미지 파일들에 대한 확장자
        """
        get_files = GetAbsolutePath(extensions=extensions)
        img_array = get_files(parents_path=parents_path)
        self.img_pathes = np.sort(img_array)

    def _get_all_img(self):
        """
        모든 이미지들을 list에 담는다.
        """
        self.img_list = []
        for i in range(len(self.img_pathes)):
            self.img_list.append(self(index=i))

    def slide_viewer(
        self,
        title_color: str = "limegreen",
        title_size: int = 10,
        figsize: Tuple[int, int] = (8, 6),
    ):
        """
        img.viewer의 Slider 클래스를 이용하여 slider 생성
        """
        # self.img_list 생성
        self._get_all_img()
        # slider를 가지고 온다.
        Slider(
            imgs=self.img_list,
            titles=self.img_pathes,
            title_color=title_color,
            title_size=title_size,
            figsize=figsize,
        ).run()
