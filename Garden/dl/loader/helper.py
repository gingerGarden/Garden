from typing import Optional, Tuple

import torch
from torch import nn
from torch.utils import data

from ...utils.img.img import tensor_to_img
from ...utils.img.viewer import show_img, show_imgs


def show_dataset_img(dataset: data.Dataset, index: int):
    """
    입력된 dataset에서 index에 해당하는 img와 img의 shape을 출력한다.

    Args:
        dataset (data.Dataset): imgDataset으로 인해 생성된 Dataset
        index (int): imgDataset의 이미지 index
    """
    img = tensor_to_img(dataset[index][0])
    show_img(img, title=f"index: {index}, img size: {img.shape}")


def show_batch_imgs(
    imgs: torch.Tensor,
    col: int = 4,
    img_size: Tuple[int, int] = (3, 3),
    show_number: bool = False,
):
    """
    Tensor로 만들어진 image를 모두 출력한다(Batch 대상)

    Args:
        imgs (torch.Tensor): Tensor로 변형된 이미지
        col (int, optional): 열 이미지의 수. Defaults to 4.
        img_size (Tuple[int, int], optional): 표시할 이미지의 크기(plt). Defaults to (3,3).
        show_number (bool, optional): 이미지 숫자를 표현할지 여부. Defaults to False.
    """
    img_list = [tensor_to_img(tensor_img=img) for img in imgs]
    show_imgs(img_list, col, img_size, show_number)


class LabelDtype:
    """
    손실 함수에 맞게 라벨의 dtype을 설정하는 클래스입니다.
    Callable 형태로 구현되어 입력된 torch.Tensor의 dtype을 변경합니다.

    Raises:
        ValueError: 지원하지 않는 손실 함수가 입력된 경우

    Returns:
        torch.Tensor: 손실 함수에 맞는 dtype으로 변경된 텐서
    """

    LONG_LOSS_FN = (nn.CrossEntropyLoss, nn.NLLLoss)
    FLOAT_LOSS_FN = (
        nn.MSELoss,
        nn.L1Loss,
        nn.BCELoss,
        nn.BCEWithLogitsLoss,
        nn.KLDivLoss,
        nn.HuberLoss,
        nn.MarginRankingLoss,
        nn.CosineEmbeddingLoss,
        nn.HingeEmbeddingLoss,
    )

    def __init__(
        self, loss_fn: nn.modules.loss = None, dtype: Optional[torch.dtype] = None
    ):
        """
        dtype이 정의 되어 있는 경우(dtype is not None), 그 dtype으로 출력된다.
        dtype이 정의 되어 있지 않고, loss_fn이 정의된 경우, 그 loss_fn에 맞는 dtype을 출력한다.

        Args:
            loss_fn (nn.modules.loss, optional): 손실함수. Defaults to None.
            dtype (Optional[torch.dtype], optional): 고정적으로 출력할 dtype. Defaults to None.
        """
        if dtype is not None:
            self.dtype = dtype
        elif loss_fn is not None:
            self.dtype = self._check_by_loss_fn(loss_fn)
        else:
            raise ValueError("dtype이나 loss_fn 중 하나는 반드시 지정해야 합니다.")

    def __call__(self, labels: torch.Tensor) -> torch.Tensor:
        """
        label(torch.Tensor)를 설정된 dtype으로 변경한다.

        Args:
            labels (torch.Tensor): label 텐서

        Returns:
            torch.Tensor: dtype이 변경된 label 텐서
        """
        return labels.to(self.dtype)

    def _check_by_loss_fn(self, loss_fn: nn.modules.loss) -> torch.dtype:
        """
        손실 함수에 맞는 torch dtype을 선택한다

        Args:
            loss_fn (nn.modules.loss): 손실 함수

        Raises:
            ValueError: 지원하지 않는 손실 함수가 입력된 경우

        Returns:
            torch.dtype: 손실 함수에 맞는 torch dtype
        """
        if isinstance(loss_fn, self.LONG_LOSS_FN):
            return torch.long
        elif isinstance(loss_fn, self.FLOAT_LOSS_FN):
            return torch.float
        else:
            raise ValueError(f"Unsupported loss function: {type(loss_fn).__name__}")
