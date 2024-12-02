import time
import warnings
from typing import Callable, List

import numpy as np
from IPython.display import clear_output

from ...utils.img.img import get_rgb_img
from ...utils.img.viewer import show_img
from ...utils.path import GetAbsolutePath


class AugmentChecker:
    def __init__(
        self,
        dir_path: str,
        delay: float = 0.2,
        img_extensions: List[str] = ["png", "jpeg", "jpg"],
        img_read_fn: Callable = get_rgb_img,
        **kwargs,
    ):
        """
        이미지 증강 방법인 augment_fn을 size의 수만큼 확인하고, 변환되지 않는 경우가 존재한다면, 그 이미지의 idx를 출력한다.

        주요 기능 1) Callable
            - 해당 class는 Callable 기능을 주 요소로 사용하며, Callable 안에 이미지의 크기, augment_fn을 정의한다.

        주요 기능 2) self.get_target_img()
            - 해당 method는 idx에 해당하는 image의 numpy array를 출력한다.

        Args:
            dir_path (str): 이미지들이 들어있는 디렉터리의 경로
            delay (float, optional): 이미지 증강 후, 다음 이미지 출력까지 대기 시간. Defaults to 0.2.
            img_extensions (List[str], optional): 이미지 파일의 대상 확장자. Defaults to ['png', 'jpeg', 'jpg'].
            img_read_fn (Callable, optional): 이미지 파일을 numpy array로 읽어오는 method. Defaults to get_rgb_img.
            **kwargs: img_read_fn의 **kwargs
        """
        self.delay = delay  # 이미지 출력 대기 시간
        self.pathes_array = self._get_img_paths_array(
            dir_path, img_extensions
        )  # 이미지 경로의 array
        self.img_read_fn = img_read_fn
        self.kwargs = kwargs

    def __call__(
        self, size: int, augment_fn: Callable, random_choice: bool = True
    ) -> int:
        """
        self.dir_path에 있는 이미지들을 대상으로 augment_fn을 적용하고, 그 결과를 출력한다.
        >>> 만약, 정상적으로 동작하지 않는 경우, warnings을 발생시키고, 해당 이미지의 idx를 출력한다.

        Args:
            size (int): 반복 확인 갯수
            augment_fn (Callable): 이미지 증강 방법
            random_choice (bool, optional): 이미지를 복원 추출로 완전 무작위 추출하여 확인할지 여부. Defaults to True.
                - False인 경우, 순서대로 출력하며, size가 이미지의 전체 크기보다 큰 경우, 순서를 유지하여 반복하여 확인한다.

        Returns:
            int: augment_fn으로 변환이 되지 않은 이미지의 index
        """
        # 정해진 방법대로 index를 추출한다.
        idx_array = (
            self._random_index(size) if random_choice else self._order_choice(size)
        )

        for i, idx in enumerate(idx_array):

            img_path = self.pathes_array[idx]
            img = self.img_read_fn(img_path, **self.kwargs)

            try:
                aug_img = augment_fn(img)
                show_img(aug_img, title=f"{i+1}/{size}")
                time.sleep(self.delay)
                clear_output(wait=True)

            except Exception as e:
                # 예외 발생
                warnings.warn(f"Break!! target image index: {idx}, 예외 메시지: {e}", UserWarning)
                return idx

        return None

    def get_target_img(self, idx: int) -> np.ndarray:
        """
        idx에 해당하는 image의 np.ndarray를 출력한다.
        """
        img_path = self.pathes_array[idx]
        return self.img_read_fn(img_path, **self.kwargs)

    def _random_index(self, size: int) -> np.ndarray:
        """
        완전 무작위로 복원 추출하여 size만큼 index_array를 생성한다.
        """
        # 원본 이미지 경로의 index array
        origin_idx_array = np.arange(0, len(self.pathes_array))
        # 복원 추출하여 size만큼 index array 생성
        return np.random.choice(a=origin_idx_array, size=size, replace=True)

    def _order_choice(self, size: int) -> np.ndarray:
        """
        self.pathes_array 의 순서대로 size의 크기만큼 index_array를 생성한다.
        """
        # 원본 이미지 경로의 index array
        origin_idx_array = np.arange(0, len(self.pathes_array))
        # size 크기에 맞게 순서를 반복하여 추가
        # size가 원본 이미지의 수보다 작은 경우, 단순 slicing
        if len(origin_idx_array) >= size:
            return origin_idx_array[:size]
        # size가 원본 이미지 수보다 큰 경우, repeat_size만큼 순서대로 더 붙인 후, slicing한다.
        else:
            # 반복 횟수
            repeat_size = np.ceil(size / len(origin_idx_array))
            # 반복하여 붙인다.
            stack_array = np.array([], dtype=np.int64)
            for i in range(int(repeat_size)):
                stack_array = np.concatenate((stack_array, origin_idx_array), axis=0)
            # slicing
            return stack_array[:size]

    def _get_img_paths_array(
        self, dir_path: str, img_extensions: List[str]
    ) -> np.ndarray:
        """
        GetAbsolutePath() instance를 생성하고, img_extensions에 해당하는 이미지들의 절대 경로들을 가져온다.
        """
        get_files = GetAbsolutePath(extensions=img_extensions)
        return np.sort(get_files(dir_path))
