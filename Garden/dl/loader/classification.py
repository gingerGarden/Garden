from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils import data

from ...augment.img.geometric.geometric import Resize
from ...utils.img.img import get_rgb_img, img_to_tensor


class BasicDataset(data.Dataset):
    def __init__(
        self,
        key_list: List[Dict[str, Union[str, np.ndarray]]],
        augments: Optional[Callable] = None,
        path_key: str = "path",
        label_key: str = "label",
        # resize 관련 매개변수
        resize: Optional[int] = None,
        resize_quality: str = "low",
        resize_how: int = 0,
        resize_how_list: List[int] = [2, 3, 4],
        resize_padding_color: str = "random",
    ):
        """
        image에 대한 가장 기본적인 Dataset
        >>> key_list: 이미지 경로와 label의 list
            - [{'path':'이미지의_절대_경로_1', 'label':0}, {'path':'이미지의_절대_경로_2', 'label':0}, ...] 의 형태
            - GGDL 의 img_dict 기본 형식
            - 다른 key 값을 쓰는 경우, path_key와 label_key를 수정하여 반영 가능
        >>> augments: 이미지 증강 방법 메서드
            - 이미지만을 매개 변수로 입력받아 증강된 이미지를 출력하는 모든 메서드에 사용 가능
                ex) aug_img = fn(img)
            - GGImgMorph의 image 증강 메서드를 고려하여 개발
        >>> resize: 이미지 크기 조정(이미지를 정방 행렬로 조정, resize_how를 1로 설정하는 경우, 이미지 비율 유지)
            - resize는 기본적으로 from GGImgMorph.geometric.geometric import Resize의 Resize 클래스를 통해 동작
            - 'resize', 'resize_quality', 'resize_how', 'resize_padding_color' 4개의 파라미터는 이미지 resize에 대한 파라미터
            - reisze는 이미지의 크기로 입력하지 않는 경우, resize 하지 않음.
            - resize_how: 1(비율 유지), 2(padding), 3(border replicate), 4(이미지 가로, 세로 비율을 무시하고 resize)

        Args:
            key_list (List[Dict[str, Union[str, np.ndarray]]]): 각 이미지의 path, label 2가지가 dictionary 형태로 입력된 list
                - GGDL의 idx_dict 형태의 가장 최소 단위 데이터
            augments (Optional[Callable], optional): 이미지 증강 알고리즘 메서드. Defaults to None.
            path_key (str, optional): key_list의 각 이미지 레코드의 이미지 경로의 key. Defaults to 'path'.
            label_key (str, optional): key_list의 각 이미지 레코드의 이미지 label의 key. Defaults to 'label'.
            resize (Optional[int], optional): 이미지 resize 시, 이미지의 크기. Defaults to None.
            resize_quality (str): 이미지 resize 방식의 보간 방법. Defaults to 'low'.
                - 'low', 'middle', 'how'
            resize_how (int): resize 방법. Defaults to 0.
                - resize 방식
                - 1: 비율 유지, 이미지의 넓이, 높이를 비율 유지하여 resize한다(장방 행렬이 될 수 있음)
                - 2: 비율 유지 - padding, 1로 resize하고 color로 padding
                - 3: 비율 유지 - border replicate, 1로 resize하고 가장자리의 pixel로 채움
                - 4: 비율 무시, 이미지의 가로, 세로 길이를 size에 맞게 resize
            resize_how_list (List[int]): resize_how=0 인 경우, 무작위 선택되는 resize 방법. Defaults to [2, 3, 4].
            resize_padding_color (str): _description_. Defaults to "random".
                - resize_how = 2인 경우, 패딩 영역의 색
        """
        self.key_list = key_list
        self.path_key = path_key
        self.label_key = label_key
        self.augments = augments

        # resize가 None이 아닌 경우, GGImgMorph를 이용하여 이미지를 resize 한다.
        if resize is not None:
            self.resize = Resize(
                size=resize,
                quality=resize_quality,
                how=resize_how,
                how_list=resize_how_list,
                color=resize_padding_color,
            )
        else:
            self.resize = None

    def __len__(self):
        return len(self.key_list)

    def __getitem__(self, idx):
        path = self.key_list[idx][self.path_key]
        img = get_rgb_img(path)

        if self.augments is not None:
            img = self.augments(img)

        if self.resize is not None:
            img = self.resize(img)

        # 이미지를 실수 Tensor로 변환
        img = img_to_tensor(img)

        label = self.key_list[idx][self.label_key]
        return img, label


class GetLoader:
    def __init__(
        self,
        idx_dict: Dict[str, List[Dict[str, str]]],
        dataset_class: data.Dataset,
        batch_size: int = 16,
        workers: int = 0,
    ):
        """
        입력된 dataset_class로 DataLoader를 생성한다.
        >>> train, valid, test 각각 dataset의 설정이 다를 수 있으므로, Callable로 별도 설정하도록 함.

        Args:
            idx_dict (Dict[str, List[Dict[str, str]]]): idx_dict의 k에 대한 dictionary
            dataset_class (data.Dataset): data.Dataset의 class
            batch_size (int, optional): data.DataLoader의 batch_size. Defaults to 16.
            workers (int, optional): data.DataLoader의 num_workers. Defaults to 0.
        """
        self.idx_dict = idx_dict
        self.dataset_class = dataset_class
        self.batch_size = batch_size
        self.workers = workers

        # Loader 정의
        self.key_list = ["train", "valid", "test"]
        self.train = None
        self.test = None
        self.valid = None

    def __call__(self, key: str, **kwargs):
        """
        train, test, validation 각각 dataset의 **kwargs가 별도의 값이 입력될 수 있으므로, 별도의 **kwargs를 입력받아
        data.DataLoader 생성, 생성된 data.DataLoader는 instance variable(train, test, valid)로 변수 저장.

        Args:
            key_list (List[Dict[str, str]]): 파일의 경로, label로 구성된 dictionary들의 List
            augments (Optional[Callable], optional): 데이터 증강 방법. Defaults to None.
            key (str, optional): 해당 데이터가 train, test, valid 중 어디에 속하는지. Defaults to "train".

        Raises:
            ValueError: 입력된 key가 ["train", "valid", "test"] 에 포함되지 않는 경우 발생
        """
        # 입력한 key값이 잘못 입력 되지 않았는지 확인
        if key not in self.key_list:
            raise ValueError(
                f"입력한 {key}는 {self.key_list}에 포함되지 않습니다. key를 확인하십시오."
            )
        # dataset 생성
        dataset = self.dataset_class(key_list=self.idx_dict[key], **kwargs)
        # dataLoader 생성
        loader = self._make_loader(dataset, key)
        # key에 맞게 저장
        if key == "train":
            self.train = loader
        elif key == "test":
            self.test = loader
        elif key == "valid":
            self.valid = loader

    def _make_loader(self, dataset: data.Dataset, key: str) -> data.DataLoader:
        """
        key에 따라 parameter가 일부 다른 DataLoader 출력
        >>> train은 shuffle, drop_last True
        >>> valid, test는 shuffle, drop_last가 False
        """
        shuffle, drop_last = (True, True) if key == "train" else (False, False)
        loader = data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            pin_memory=torch.cuda.is_available(),
            num_workers=self.workers,
            shuffle=shuffle,
            drop_last=drop_last,
        )
        return loader
