import os
import shutil
from typing import Optional

import numpy as np
import torch


class EarlyStopping:
    def __init__(
        self,
        model: torch.nn.Module,
        path: str = "checkpoint.pt",
        patience: int = 5,
        delta: float = 0.0,
        target: str = "loss",
        auto_remove: bool = True,
        best_model_dir: Optional[str] = None,
        log_round: int = 5,
        verbose: bool = True,
    ):
        """
        target에 대하여 Early stopping을 한다. Early stopping의 대상은 기본적으로 validation set을 대상으로 하며,
        validation set이 존재하지 않는 경우, 실행하지 않는다.
        - epoch는 progressbar와 기준을 일치시키기 위해 1부터 시작 한다.

        early stopping 효과가 종료되는 위치에 self.end_point() 메서드를 위치해야한다.

        Args:
            model (torch.nn.Module): 대상 모델
            path (str, optional): early stopping 과정 중 최적의 모델 파라미터를 저장하는 경로. Defaults to 'checkpoint.pt'.
            patience (int, optional): target의 최소값 또는 최대값 갱신이 되지 않는동안 얼마나 추가 학습을 할지. Defaults to 5.
            delta (float, optional): 최적 값과의 최소 편차. Defaults to 0.0.
            target (str, optional): early stopping의 목적값. Defaults to 'loss'.
                - validation dataset에 대한 loss, accuarcy를 기본 대상으로 한다.
            auto_remove (bool, optional): early stopping 과정 중 생성된 모델 파라미터 파일을 학습 종료 후 삭제할지 여부. Defaults to True.
            log_round (int, optional): early stopping log의 소수점 자릿수. Defaults to 5.
            verbose (bool, optional): early stopping log 문자열을 출력할지 여부. Defaults to True.
                - False 인 경우, None 출력
        """
        # 고정된 instance variable
        self.model = model
        self.patience = patience
        self.delta = delta
        self.target = self._target_check(target)
        self.path = path
        self.auto_remove = auto_remove
        self.best_model_dir = best_model_dir
        self.log_round = log_round
        self.verbose = verbose

        # 변하는 instance variable
        self.best_score = np.Inf if self.target == "loss" else -np.Inf  # 최고 점수
        self.best_epoch = None  # 최고 Epoch
        self.stop = False  # Early stopping으로 학습을 종료할지 여부
        self.count = 0  # patience 대기
        self.k = None

    def __call__(self, epoch: int, score: float) -> str:
        """
        score에 대해서 early stopping 알고리즘을 진행한다.

        Args:
            epoch (int): 현재 epoch
            score (float): 현재 target score

        Returns:
            str: early stopping log
        """
        score = score + self.delta if self.target == "loss" else score - self.delta
        # score가 self.best_score보다 큰 경우(loss) 기존 모델을 저장
        if (
            self.best_score > score
            if self.target == "loss"
            else self.best_score < score
        ):
            self.count = 0
            return self._save_checkpoint(epoch, score)
        else:
            self.count += 1
            rest_epoch = self.patience - self.count
            if rest_epoch == 0:
                self.stop = True
            return f"[Early Stopping - Pass]: [best epoch]: {self.best_epoch} / [best score]: {self.best_score:.{self.log_round}f} / [extra epochs]: {rest_epoch}"

    def validation_score(
        self, loss: Optional[float] = None, acc: Optional[float] = None
    ) -> float:
        """
        validation의 현재 score 중, instance variable에 설정된 target에 해당하는 값을 출력

        Args:
            loss (Optional[float], optional): 손실값. Defaults to None.
            acc (Optional[float], optional): 정확도. Defaults to None.

        Returns:
            float: target에 해당하는 값
        """
        # 입력된 모든 parameter 들이 문제가 있는지 확인
        if loss is None and acc is None:
            raise ValueError("Both loss and accuracy cannot be None.")

        if self.target == "loss":
            return loss
        elif self.target == "accuracy":
            return acc

    def _target_check(self, target: str) -> str:
        """
        instance variable로 입력된 target이 대상 값 문자열이 맞는지 확인
        >>> target_list = ['loss', 'accuracy']

        Args:
            target (str): target_list에 포함된 문자열

        Raises:
            ValueError: target_list에 포함되지 않은 문자열이 입력된 경우

        Returns:
            str: target_list에 포함된 문자열인 경우, target 문자열을 그대로 출력
        """
        target_list = ["loss", "accuracy"]
        if target not in target_list:
            raise ValueError(
                f"target에 입력된 {target}는 지원하지 않습니다. 지원하는 target은 {target_list}에 해당합니다."
            )
        else:
            return target

    def _save_checkpoint(self, epoch: int, score: float) -> str:
        """
        모델의 파라미터를 저장한다.

        Args:
            epoch (float): 모델 파라미터 저장 당시의 epoch
            score (float): 모델 파라미터 저장 당시의 score

        Returns:
            str: early stopping의 log text
        """
        # save model parameter
        torch.save(self.model.state_dict(), self.path)
        # instance variable 갱신
        last_best_score = self.best_score
        self.best_score = score
        self.best_epoch = epoch + 1
        # log text
        if self.verbose:
            return f"[Early Stopping - Save]: [{self.target}]: [last]: {last_best_score:.{self.log_round}f} / [current]: {score:.{self.log_round}f} / [epoch]: {self.best_epoch}"
        else:
            return None

    def end_point(self) -> str:
        """
        모델의 학습 과정의 끝 부분에 추가하여, early stopping의 최선의 모델을 load하고 종료 log text를 출력

        Returns:
            str: 종료 log text
        """
        if self.stop:
            return self._load_checkpoint()
        else:
            return self._no_load_checkponit()

    def _load_checkpoint(self) -> str:
        """
        early stopping이 효과를 발휘하여, checkpoint를 load함.
        """
        # load model parameter
        self.model.load_state_dict(torch.load(self.path, weights_only=True))
        # checkpoint end point
        self._checkpoint_auto_remove_or_move()
        # log text
        if self.verbose:
            return f"[Early Stopping - Load]: [epoch]: {self.best_epoch} / [best score]: {self.best_score:.{self.log_round}f}"
        else:
            return None

    def _no_load_checkponit(self) -> str:
        """
        학습이 epochs에 도달하여 early stopping의 효과를 보기 전에 종료된 경우
        """
        # checkpoint end point
        self._checkpoint_auto_remove_or_move()
        # log text
        if self.verbose:
            return "[Early Stopping - Pass]: The training completed before Early Stopping could take effect."
        else:
            return None

    def _checkpoint_auto_remove_or_move(self):
        """
        early stopping 종료 후 행동.
        - if self.best_model_dir is not None
            : 최고 성능 모델을 self.best_model_dir로 이동
        - if self.auto_remove and self.best_model_dir is None
            : 최고 성능 모델을 제거
        """
        # move best model parameter file
        if self.best_model_dir is not None:
            best_model_path = f"{self.best_model_dir}/param_{self.k}.pt"
            shutil.move(self.path, best_model_path)
        # remove model parameter file
        if self.auto_remove and self.best_model_dir is None:
            os.remove(self.path)


class BackPropagation:
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        use_amp: bool = False,
        use_clipping: bool = False,
        max_norm: Optional[int] = None,
        device: Optional[str] = None,
    ):
        """
        Back propagation을 AMP(Automatic Mixed Precision), Gradient clipping 등을 고려하여 진행

        Args:
            optimizer (torch.optim.Optimizer): optimizer
            use_amp (bool, optional): AMP 사용 여부. Defaults to False.
            use_clipping (bool, optional): _description_. Defaults to False.
            max_norm (Optional[int], optional): _description_. Defaults to None.
            device (Optional[str], optional): 학습에 사용되는 device. Defaults to None.
        """
        self.optimizer = optimizer
        self.use_amp = use_amp
        self.grad_scaler = self._set_grad_scaler(device)
        self.use_clipping = use_clipping
        self.max_norm = max_norm
        self.params = [
            p for group in optimizer.param_groups for p in group["params"]
        ]  # optimizer의 파라미터 추출

    def __call__(self, loss: torch.Tensor):
        """
        BackPropagation에 설정된 방식되로 역전파 한다.

        Args:
            loss (torch.Tensor): loss_fn으로 부터 나온 loss
        """
        self.optimizer.zero_grad()  # gradient 0으로 초기화 - 이전 단계의 변화도가 누적되지 않도록 방지
        if self.use_amp:
            self._amp_backward(loss)
        else:
            self._normal_backward(loss)

    def _normal_backward(self, loss: torch.Tensor):
        """
        일반적인 역전파(backward)

        Args:
            loss (torch.Tensor): loss_fn으로 부터 나온 loss
        """
        loss.backward()
        if self.use_clipping:  # clipping 적용 여부
            self._gradint_clipping()
        self.optimizer.step()

    def _amp_backward(self, loss: torch.Tensor):
        """
        AMP가 설정된 경우, 역전파(backward)

        Args:
            loss (torch.Tensor): loss_fn으로 부터 나온 loss
        """
        # 손실 값을 스케일링하여 역전파 수행
        self.grad_scaler.scale(loss).backward()
        # clipping 적용 여부
        if self.use_clipping:
            self._gradint_clipping()
        # 스케일링된 그래디언트를 사용하여 옵티마이저의 매개변수 업데이트
        self.grad_scaler.step(self.optimizer)
        # 다음 학습 단계를 위한 스케일 팩터 업데이트
        self.grad_scaler.update()

    def _gradint_clipping(self):
        """
        gradient clipping 적용
        """
        # AMP가 적용되어 있는 경우, GradScaler를 역산 후 적용하여, 수치 안정성고가 임계값 유지를 한다.
        if self.use_amp:
            self.grad_scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(parameters=self.params, max_norm=self.max_norm)

    def _set_grad_scaler(self, device: str) -> Optional[torch.GradScaler]:
        """
        AMP를 사용하는 경우, GradScaler 생성

        Args:
            device (str): 학습에 사용되는 device

        Raises:
            ValueError: GradScaler를 생성해야 하나, device가 입력되지 않은 경우

        Returns:
            Optional[torch.GradScaler]: GradScaler 또는 None
        """
        if self.use_amp:
            if device is None:
                raise ValueError("device가 입력되지 않았습니다! device를 입력하십시오!")
            return torch.GradScaler(device=device)
        else:
            return None
