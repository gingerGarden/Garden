from typing import Optional

from torch import optim
from torch.optim import lr_scheduler


def get_lr(
    optimizer: optim.Optimizer, scheduler: Optional[lr_scheduler._LRScheduler] = None
) -> float:
    """
    학습률을 출력함
        - scheduler 사용 시, scheduler를 이용하여, lr 출력

    Args:
        optimizer (optim.Optimizer): torch optimizer
        scheduler (Optional[lr_scheduler._LRScheduler], optional): torch scheduler. Defaults to None.

    Returns:
        float: learning rate
    """
    if scheduler is not None:
        return scheduler.get_last_lr()[0]
    else:
        return optimizer.param_groups[0]["lr"]
