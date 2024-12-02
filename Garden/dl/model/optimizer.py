from typing import Any, Dict, List, Optional

import torch_optimizer
from torch import nn, optim
from torch.optim import lr_scheduler

from ...utils.utils import get_method_parameters, kwargs_filter


class OptimHelper:
    methods = {
        "SGD": optim.SGD,
        "Adam": optim.Adam,
        "Adagrad": optim.Adagrad,
        "RMSprop": optim.RMSprop,
        "AdamW": optim.AdamW,
        "Adadelta": optim.Adadelta,
        "Adamax": optim.Adamax,
        "ASGD": optim.ASGD,
        "LBFGS": optim.LBFGS,
        "NAdam": optim.NAdam,
        "RAdam": optim.RAdam,
        "Rprop": optim.Rprop,
        "SparseAdam": optim.SparseAdam,
        "Adafactor": torch_optimizer.Adafactor,
        "AdamP": torch_optimizer.AdamP,
        "DiffGrad": torch_optimizer.DiffGrad,
        "Lamb": torch_optimizer.Lamb,
        "NovoGrad": torch_optimizer.NovoGrad,
        "SWATS": torch_optimizer.SWATS,
        "AdaBound": torch_optimizer.AdaBound,
        "Yogi": torch_optimizer.Yogi,
        "Ranger": torch_optimizer.Ranger,
    }

    def __init__(self, name: str, hp_dict: Dict[str, Any]):
        """
        torch의 optimizer들을 사용하기 쉽게 정리한 class
        hp_dict의 key 중, optimizer의 parameter만 선택하여 입력한다.

        대상 optimizer 와 torch, torch_optimizer 예제 코드
        1. SGD (Stochastic Gradient Descent): 확률적 경사 하강법으로, 각 배치마다 모델의 매개변수를 업데이트합니다. 학습률과 모멘텀 등의 하이퍼파라미터를 조정하여 성능을 향상시킬 수 있습니다.
            >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        2. Adam (Adaptive Moment Estimation): 1차 및 2차 모멘트 추정을 활용하여 학습률을 적응적으로 조정하는 알고리즘입니다. 대부분의 경우 빠른 수렴 속도와 안정적인 성능을 제공합니다.
            >>> optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08)
        3. Adagrad (Adaptive Gradient Algorithm): 각 매개변수에 대해 학습률을 개별적으로 조정하여 드물게 나타나는 특징(feature)에 대해 학습을 촉진합니다. 그러나 학습률이 너무 작아질 수 있는 단점이 있습니다.
            >>> optimizer = torch.optim.Adagrad(model.parameters(), lr=0.01, lr_decay=0, weight_decay=0, initial_accumulator_value=0, eps=1e-10)
        4. RMSprop (Root Mean Square Propagation): Adagrad의 학습률 감소 문제를 해결하기 위해 지수 이동 평균을 사용하여 학습률을 조정합니다.
            >>> optimizer = torch.optim.RMSprop(model.parameters(), lr=0.01, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)
        5. AdamW: Adam 옵티마이저의 변형으로, 가중치 감쇠(weight decay)를 올바르게 처리하여 일반화 성능을 향상시킵니다.
            >>> optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False)
        6. Adadelta: Adagrad의 학습률 감소 문제를 해결하기 위해 학습률을 동적으로 조정하는 알고리즘입니다.
            >>> optimizer = torch.optim.Adadelta(model.parameters(), lr=1.0, rho=0.9, eps=1e-06, weight_decay=0)
        7. Adamax: Adam의 변형으로, 무한 노름(infinity norm)을 기반으로 학습률을 조정합니다.
            >>> optimizer = torch.optim.Adamax(model.parameters(), lr=0.002, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
        8. ASGD (Averaged Stochastic Gradient Descent): SGD의 변형으로, 모델의 일반화 성능을 향상시키기 위해 매개변수의 이동 평균을 사용합니다.
            >>> optimizer = torch.optim.ASGD(model.parameters(), lr=0.01, lambd=0.0001, alpha=0.75, t0=1000000.0, weight_decay=0)
        9. LBFGS (Limited-memory Broyden–Fletcher–Goldfarb–Shanno): 2차 미분 정보를 활용하는 최적화 알고리즘으로, 메모리 사용을 최소화하면서도 빠른 수렴을 제공합니다.
            >>> optimizer = torch.optim.LBFGS(model.parameters(), lr=1, max_iter=20, max_eval=None, tolerance_grad=1e-07, tolerance_change=1e-09, history_size=100, line_search_fn=None)
        10. NAdam (Nesterov-accelerated Adaptive Moment Estimation): Adam과 Nesterov 모멘텀을 결합한 알고리즘으로, 빠른 수렴과 안정성을 제공합니다.
            >>> optimizer = torch.optim.NAdam(model.parameters(), lr=0.002, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, momentum_decay=0.004)
        11. RAdam (Rectified Adam): Adam의 학습률 워밍업 문제를 해결하여 안정적인 학습을 지원합니다.
            >>> optimizer = torch.optim.RAdam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, degenerated_to_sgd=True)
        12. Rprop (Resilient Backpropagation): 각 매개변수에 대해 개별적인 학습률을 적용하여 학습 속도를 향상시킵니다.
            >>> optimizer = torch.optim.Rprop(model.parameters(), lr=0.01, etas=(0.5, 1.2), step_sizes=(1e-06, 50))
        13. SparseAdam: 희소 텐서에 특화된 Adam 옵티마이저로, 희소 데이터의 효율적인 학습에 사용됩니다.
            >>> optimizer = torch.optim.SparseAdam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08)
        14. Adafactor: 메모리 효율적인 Adam의 변형으로, 큰 모델 학습 시 유용합니다.
            >>> optimizer = torch_optimizer .Adafactor(model.parameters(), lr=1e-3, eps=(1e-30, 1e-3), clip_threshold=1.0, decay_rate=-0.8, beta1=None, weight_decay=0.0, scale_parameter=True, relative_step=True, warmup_init=False)
        15. AdamP: Adam의 변형으로, 과도한 가중치 증가를 억제하여 일반화 성능을 향상시킵니다.
            >>> optimizer = torch_optimizer.AdamP(model.parameters(), lr=1e-3, betas=(0.9, 0.999), weight_decay=1e-2)
        16. DiffGrad: 최근의 기울기 변화에 따라 학습률을 조정하는 알고리즘입니다.
            >>> optimizer = torch_optimizer.DiffGrad(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)
        17. Lamb (Layer-wise Adaptive Moments optimizer for Batch training): 대규모 분산 학습에 적합한 옵티마이저로, 레이어별 학습률 조정을 지원합니다.
            >>> optimizer = torch_optimizer.Lamb(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-6, weight_decay=0.01)
        18. NovoGrad: 메모리 사용량을 줄이면서도 빠른 수렴을 제공하는 옵티마이저입니다.
            >>> optimizer = torch_optimizer.NovoGrad(model.parameters(), lr=1e-3, betas=(0.95, 0.98), eps=1e-8, weight_decay=0)
        19. SWATS (Switching from Adam to SGD): 학습 초기에 Adam을 사용하고, 이후 SGD로 전환하는 알고리즘입니다.
            >>> optimizer = torch_optimizer.SWATS(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)
        20. AdaBound: Adam 옵티마이저의 변형으로, 학습 후반부에 학습률을 조정하여 더 안정적인 수렴을 유도합니다.
            >>> optimizer = torch_optimizer.AdaBound(model.parameters(), lr=1e-3, final_lr=0.1)
        21. Yogi: Adam의 변형으로, 학습 과정에서의 과도한 가중치 증가를 억제하여 안정적인 학습을 지원합니다.
            >>> optimizer = torch_optimizer.Yogi(model.parameters(), lr=1e-3)
        22. Ranger: RAdam과 Lookahead 옵티마이저를 결합한 알고리즘으로, 빠른 수렴과 일반화 성능 향상을 목표로 합니다.
            >>> optimizer = torch_optimizer.Ranger(model.parameters(), lr=1e-3)

        Args:
            name (str): optimizer의 이름
            hp_dict (Dict[str, Any]): optimizer에 입력될 paramter가 포함된 hyper parameter의 dictionary
                - **kwargs를 사용하므로 optimizer의 parameter key와 hp_dict의 key가 일치해야함. 중복된 key를 사용하면 안됨.
        """
        self.method = self._get_method(name)
        self.hp_dict = kwargs_filter(fn=self.method, params_dict=hp_dict)

    def __call__(self, param: nn.Module) -> optim.Optimizer:
        """
        사전 정의된 optimizer의 설정에 모델의 파라미터 입력

        Args:
            param (nn.Module): 모델 파라미터

        Returns:
            torch.optim.Optimizer: optimizer
        """
        return self.method(param, **self.hp_dict)

    def show_parameters(self, verbose: bool = False) -> List:
        """
        name에 입력된 optimizer의 모든 파라미터 출력

        Args:
            verbose (bool, optional): 파라미터들을 print 할지. Defaults to False.

        Returns:
            List: parameter의 list
        """
        params = get_method_parameters(fn=self.method, to_list=True)
        if verbose:
            print(f"{self.method}는 {list(params)}를 파라미터로 갖습니다.")
        return params

    def _get_method(self, name: str) -> optim.Optimizer:
        """
        name에 해당하는 optimizer를 선택함.

        Args:
            name (str): optimizer의 이름

        Raises:
            ValueError: name이 해당하지 않는 이름이 선택된 경우

        Returns:
            torch.optim: optimizer 객체
        """
        method_keys = list(self.methods.keys())
        if name not in method_keys:
            raise ValueError(
                f"입력한 name: {name}은 지원하지 않습니다. optimizer는 {method_keys}만 지원합니다."
            )
        else:
            return self.methods.get(name)


class SchedulerHelper:

    methods = {
        "LambdaLR": {
            "method": lr_scheduler.LambdaLR,
            "do_epoch": True,
            "step_param": None,
        },
        "MultiplicativeLR": {
            "method": lr_scheduler.MultiplicativeLR,
            "do_epoch": True,
            "step_param": None,
        },
        "StepLR": {"method": lr_scheduler.StepLR, "do_epoch": True, "step_param": None},
        "MultiStepLR": {
            "method": lr_scheduler.MultiStepLR,
            "do_epoch": True,
            "step_param": None,
        },
        "ConstantLR": {
            "method": lr_scheduler.ConstantLR,
            "do_epoch": True,
            "step_param": None,
        },
        "LinearLR": {
            "method": lr_scheduler.LinearLR,
            "do_epoch": True,
            "step_param": None,
        },
        "ExponentialLR": {
            "method": lr_scheduler.ExponentialLR,
            "do_epoch": True,
            "step_param": None,
        },
        "PolynomialLR": {
            "method": lr_scheduler.PolynomialLR,
            "do_epoch": True,
            "step_param": None,
        },
        "CosineAnnealingLR": {
            "method": lr_scheduler.CosineAnnealingLR,
            "do_epoch": True,
            "step_param": None,
        },
        "ChainedScheduler": {
            "method": lr_scheduler.ChainedScheduler,
            "do_epoch": None,
            "step_param": None,
        },
        "SequentialLR": {
            "method": lr_scheduler.SequentialLR,
            "do_epoch": None,
            "step_param": None,
        },
        "ReduceLROnPlateau": {
            "method": lr_scheduler.ReduceLROnPlateau,
            "do_epoch": True,
            "step_param": "score",
        },
        "CyclicLR": {
            "method": lr_scheduler.CyclicLR,
            "do_epoch": False,
            "step_param": None,
        },
        "OneCycleLR": {
            "method": lr_scheduler.OneCycleLR,
            "do_epoch": False,
            "step_param": None,
        },
        "CosineAnnealingWarmRestarts": {
            "method": lr_scheduler.CosineAnnealingWarmRestarts,
            "do_epoch": False,
            "step_param": "",
        },
    }

    def __init__(
        self,
        name: Optional[str] = None,
        hp_dict: Optional[Dict[str, Any]] = None,
        do_epoch: Optional[bool] = None,
    ):
        self.name = name
        self.method = None
        self.scheduler = None
        self.do_epoch = None
        if name is not None and hp_dict is not None:
            self._get_method_and_do_epoch(name, do_epoch)
            # scheduler method에 해당하는 key:value만 추출
            self.hp_dict = kwargs_filter(fn=self.method, params_dict=hp_dict)

    def __call__(self, optimizer: optim.Optimizer):
        """
        optimizer을 입력받아, self.scheduler 생성
            - self의 여러 method와 함꼐 사용되어야 유용하므로, self.scheduler로 사용

        Args:
            optimizer (optim.Optimizer): optimizer
        """
        if self.method is not None:
            self.scheduler = self.method(optimizer=optimizer, **self.hp_dict)
        else:
            self.scheduler = None

    def batch_step(self, current_training_ratio: float):
        """
        batch 단위에서 scheduler 적용 시

        Args:
            current_training_ratio (float): "CosineAnnealingWarmRestarts"를 위한 현재 학습 진행 상태.
            >>> epoch + i/iter_size
                ex) epoch = 5, i = 2, iter_size = 10
                    > 5.2 로, epoch와 iterator를 고려한 현재 학습 진행 상태를 알 수 있다.
        """
        if self.scheduler is not None:
            if not self.do_epoch:
                if self.name == "CosineAnnealingWarmRestarts":
                    self.scheduler.step(current_training_ratio)
                else:
                    self.scheduler.step()

    def epoch_step(self, score: float):
        """
        epoch 단위에서 scheduler 적용 시

        Args:
            score (float): "ReduceLROnPlateau"를 위한 대상 score
            >>> valid_loss 를 기본 대상으로 하며, valid_loss가 없는 경우, train_loss를 대상으로 한다.
        """
        if self.scheduler is not None:
            if self.do_epoch:
                if self.name == "ReduceLROnPlateau":
                    self.scheduler.step(score)
                else:
                    self.scheduler.step()

    def show_parameters(self, verbose: bool = False) -> List:
        """
        name에 입력된 optimizer의 모든 파라미터 출력

        Args:
            verbose (bool, optional): 파라미터들을 print 할지. Defaults to False.

        Returns:
            List: parameter의 list
        """
        params = get_method_parameters(fn=self.method, to_list=True)
        if verbose:
            print(f"{self.method}는 {list(params)}를 파라미터로 갖습니다.")
        return params

    def _get_method_and_do_epoch(self, name: str, do_epoch: Optional[bool]):
        """
        name에 해당하는 method와 do_epoch를 가져온다.

        Args:
            name (str): scheduler의 이름(torch.optim.lr_scheduler의 이름과 동일)
            do_epoch (Optional[bool]): epoch에서 적용되는 스케쥴러인지 여부(역은 batch)

        Raises:
            ValueError: 해당 class에서 지원하지 않는 name 입력 시
            ValueError: do_epoch를 별도 입력해야하는 경우
        """
        method_keys = list(self.methods.keys())
        if name not in method_keys:
            raise ValueError(
                f"입력한 name: {name}은 지원하지 않습니다. scheduler는 {method_keys}만 지원합니다."
            )
        else:
            self.method = self.methods[name]["method"]
            temp_do_epoch = self.methods[name]["do_epoch"]

            # temp_do_epoch가 None인 경우, 입력된 do_epoch로 입력
            if temp_do_epoch is None:
                if do_epoch is None:
                    raise ValueError(
                        "해당 scheduler는 do_epoch를 별도로 정의해아합니다!"
                    )
                else:
                    temp_do_epoch = do_epoch

            self.do_epoch = temp_do_epoch
