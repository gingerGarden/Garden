import re
from typing import Dict, Optional, Union

import numpy as np
import pandas as pd
import torch

from ...statics.descript import frequency_table


class Tuner:
    def __init__(
        self,
        model: torch.nn.Module,
        tuning_target: set = "layer",
        how: int = 0,
        freezing_ratio: Optional[float] = None,
        min_freezing_ratio: Optional[float] = None,
        max_freezing_ratio: Optional[float] = None,
        patience: Optional[int] = None,
        layer_key: Optional[str] = None,
        weight_ptn: Optional[str] = None,
        bias_ptn: Optional[str] = None,
        block_ptn: Optional[str] = None,
    ):

        # 학습에 사용되는 model
        self.model = model
        self.how = how

        # metadata
        self.target_list = [
            "layer",
            "block",
            "parameter",
        ]  # idx_df에서 idx가 될 수 있는 column의 list
        self.convert_key = None

        # Fine Tuning indexing을 위한 DataFrame과 index array 산출
        self.param_ins = ParameterIndex(model, weight_ptn, bias_ptn, block_ptn)
        self.idx_df = None
        self.idx_array = None
        self.make_idx_df(tuning_target)

        # Fine tuning 방법 정의
        self.methods = {
            0: FullFineTuning(self.model, self.idx_df, self.idx_array),
            1: FixedFeatureExtractor(
                self.model, self.idx_df, self.idx_array, layer_key=layer_key
            ),
            2: PartialLayerFreezing(
                self.model, self.idx_df, self.idx_array, freezing_ratio=freezing_ratio
            ),
            3: FreezeNUnfreeze(
                self.model,
                self.idx_df,
                self.idx_array,
                min=min_freezing_ratio,
                max=max_freezing_ratio,
                patience=patience,
            ),
        }
        self.method = None

    def __call__(self, epoch: int):
        if self.convert_key is None:
            self.method = self.methods.get(self.how, self.methods[0])
            self.convert_key = self.method(epoch)

        if self.convert_key:
            print("test_code")

    def target_idx_freezing(self, idx_array: np.ndarray, freezing: bool = True):
        """
        idx_array에 해당하는 parameter들의 학습 여부 결정

        Args:
            idx_array (np.ndarray): 조작하고자 하는 parameter들의 index array
            freezing (bool, optional): 대상 parameter들을 얼릴지 여부. Defaults to True.
        """
        # 이해하기 쉬운 스타일로 mask 설정
        mask = False if freezing else True
        # 대상 idx_array에 대해 적용
        for idx in idx_array:
            names = self.idx_df.loc[idx]["name"]
            # name이 1개인 경우, 2개 이상인 경우, 별도로 처리
            if isinstance(names, str):
                self.model.get_parameter(names).requires_grad = mask
            else:
                for name in names:
                    self.model.get_parameter(name).requires_grad = mask
            self.idx_df.loc[idx, "grad"] = mask

    def make_idx_df(self, tuning_target: str):
        """
        ParameterIndex class를 이용하여 idx_dict을 생성하고, 대상 parameter를 찾기 쉽도록 DataFrame으로 변형
        >>> tuning_target을 index로 설정
            tuning_target은 self.target_list에 속해야함

        Raises:
            ValueError: 입력된 tuning_target이 self.target_list에 없는 경우
        """
        # ParameterIndex instance를 사용하여, index_df를 생성
        idx_df = pd.DataFrame.from_dict(self.param_ins(), orient="index")
        idx_df = idx_df.reset_index(drop=False, names="parameter")

        if tuning_target not in self.target_list:
            txt = f"당신이 입력한 {tuning_target}은 {self.target_list}에 속하지 않습니다. 속하는 값으로 수정하십시오."
            raise ValueError(txt)
        else:
            self.idx_df = idx_df.set_index(tuning_target)
            self.idx_array = self.get_idx_array(idx_df=self.idx_df)

    def get_idx_array(self, idx_df: pd.DataFrame):
        """
        self.idx_df의 index에 대하여 index의 array를 추출
        """
        idx_array = idx_df.index.unique().values  # unique한 index 추출
        idx_array.sort()  # 오름차순 정렬
        return idx_array

    def check_freezing_parameter(self):
        """
        모델 파라미터에서 freezing되어 있는 파라미터에 대한 정보를 출력한다.

        ex) +---+-------+-----------+-------+
            |   | class | frequency | ratio |
            +---+-------+-----------+-------+
            | 0 | False |    315    | 0.89  |
            | 1 | True  |    37     | 0.11  |
            | 2 | total |    352    |  1.0  |
            +---+-------+-----------+-------+
            마지막 freezing layer: 314
        """
        stack = []
        end_of_freeze = None
        before_grad = None
        for key, value in self.param_ins().items():
            # 현재의 grad
            current_grad = value["grad"]

            if before_grad is None:
                before_grad = current_grad
            else:
                if before_grad != current_grad:
                    end_of_freeze = key - 1

            before_grad = current_grad
            # 현재의 grad를 추가
            stack.append(current_grad)

        _ = frequency_table(stack, markdown=True)
        print(f"마지막 freezing layer: {end_of_freeze}")


class FullFineTuning(Tuner):
    def __init__(self, model, idx_df, idx_array):
        """
        모든 parameter를 True로 한다.

        Full Fine-tuning (전체 모델 Fine-tuning)
        방법: 사전 학습된 모델의 모든 파라미터를 학습 가능 상태로 설정하고, 새로운 데이터셋에서 학습 진행
        장점: 모델의 모든 파라미터를 새롭게 업데이트하여 데이터셋에 최적화할 수 있음.
        단점: 계산 비용이 크고, 작은 데이터셋에서는 과적합의 위험이 큼.
        """
        self.model = model
        self.idx_df = idx_df
        self.idx_array = idx_array
        self.convert_key = True

    def __call__(self, _):
        self.target_idx_freezing(idx_array=self.idx_array, freezing=False)
        self.convert_key = False
        return self.convert_key


class FixedFeatureExtractor(Tuner):
    def __init__(self, model, idx_df, idx_array, layer_key: str):
        """
        Feature를 뽑을 Header를 제외한 모든 레이어를 학습.

        Fixed Feature Extractor (고정된 Feature 추출기) 의 변형
        방법: 사전 학습된 모델의 대부분 레이어를 고정시키고, 마지막 분류기 레이어만 학습.
        장점: 계산 비용이 낮으며, 빠르게 새로운 데이터셋에 맞출 수 있음.
        단점: 사전 학습된 Feature에만 의존하게 되어, 만약 새로운 데이터셋의 Feature가 크게 다르면 성능이 저하될 수 있음.
        """
        self.model = model
        self.idx_df = idx_df
        self.idx_array = idx_array
        self.layer_key = layer_key
        self.convert_key = True

    def __call__(self, _):
        # freezing 기준이 될 layer의 index를 가져온다.
        target_before_idx = self.layer_key_check_and_get_target_index()
        # freezing할 index들의 array
        freezing_idx_array = self.idx_array[:target_before_idx]
        # freezing
        self.target_idx_freezing(idx_array=freezing_idx_array, freezing=True)
        self.convert_key = False
        return self.convert_key

    def layer_key_check_and_get_target_index(self) -> int:
        """
        layer_key가 None이 아닌지, 포함된 layer가 존재하는지 확인하고 있다면 index를, 없는 경우 ValueError를 출력한다.
        """
        if self.layer_key is None:
            raise ValueError(
                "layer_key에 None이 입력되었습니다. layer_key는 header layer의 시작 이름의 일부로 구성되어야 합니다."
            )

        target_idx = self.find_before_extractor()
        if target_idx is None:
            raise ValueError(
                f"입력된 layer_key인 {self.layer_key}를 포함하는 parameter name이 존재하지 않습니다. self.idx_df를 통해 parameter들의 name을 확인하십시오."
            )

        return target_idx

    def find_before_extractor(self) -> int:
        """
        self.layer_key가 포함되어 있는 layer가 시작되는 index를 찾는다.
        """
        target_idx = None
        for idx in self.idx_df.index:
            names = self.idx_df.loc[idx, "name"]
            name = names if isinstance(names, str) else names.values[0]

            if re.search(self.layer_key, name, flags=0):
                target_idx = idx
                break
        return target_idx


class PartialLayerFreezing(Tuner):
    def __init__(self, model, idx_df, idx_array, freezing_ratio: float):
        """
        freezing_ratio를 기준으로 낮은 수준부터 freezing하여 학습을 하지 않도록 한다.

        Partial Layer Freezing (일부 레이어를 고정)
        방법: 모델의 일부 레이어를 고정(freeze)하고 나머지 레이어를 학습하여 새로운 데이터셋에 적응시킨다.
        장점: 기존의 특징을 유지하면서 학습 속도를 높이고 오버피팅을 방지할 수 있다.
        단점: 고정 비율이 높으면 새 데이터셋에 적응력이 떨어질 수 있어 비율 조정이 필요하다.
        """
        self.model = model
        self.idx_df = idx_df
        self.idx_array = idx_array
        self.freezing_ratio = freezing_ratio
        self.convert_key = True

    def __call__(self, _):
        # instance variable의 무결성 확인
        self.check_freezing_ratio()
        # freeze할 layer의 index
        freezing_idx_array = self.get_freeze_layer_index_array()
        # freezing
        self.target_idx_freezing(idx_array=freezing_idx_array, freezing=True)
        self.convert_key = False
        return self.convert_key

    def get_freeze_layer_index_array(self) -> np.ndarray:
        """
        self.freezing_ratio만큼 self.idx_array에서 freezing할 layer의 index를 가져온다.
        """
        freeze_size = int(np.round(len(self.idx_array) * self.freezing_ratio, 0))
        return self.idx_array[:freeze_size]

    def check_freezing_ratio(self):
        if self.freezing_ratio is None:
            raise ValueError(
                "freezing_ratio에 None이 입력되었습니다. freezing_ratio는 float을 입력해야합니다."
            )

        if not isinstance(self.freezing_ratio, float):
            raise ValueError(
                f"freezing_ratio는 float이 입력되어야 합니다. 현재 {self.freezing_ratio}가 입력되었습니다."
            )


class FreezeNUnfreeze(Tuner):
    def __init__(
        self,
        model: torch.nn.Module,
        idx_df: pd.DataFrame,
        idx_array: np.ndarray,
        min: float,
        max: float,
        patience: int,
    ):
        self.model = model
        self.idx_df = idx_df
        self.idx_array = idx_array
        self.min = min
        self.max = max
        self.patience = patience

        # metadata
        self.convert_key = True
        self.current_epoch = None

    def __call__(self, epoch):
        pass


class ParameterIndex:

    def __init__(
        self,
        model: torch.nn.Module,
        weight_ptn: Optional[str] = None,
        bias_ptn: Optional[str] = None,
        block_ptn: Optional[str] = None,
    ):
        """
        torch 기반 model의 model.named_parameters()를 기반으로 하여, parameter의 layer, block을 구분하는 index_dictionary를 생성한다.

        model의 parameter에 대하여 다음과 같은 방법을 통해 주요 metadata를 추출하여 index를 생성한다.
            1. paramter로부터 기본적인 정보를 가져온다.
            2. parameter를 weight, bias 존재 여부 확인
            3. parameter 순서와 weight, bias를 제외한 name이 이전과 동일한지 여부로 layer 구분
            4. 이전 layer와 현재 layer가 다르면서, self.block_ptn이 다른 경우, 다른 block으로 정의한다.

        Args:
            model (torch.nn.Module): torch 기반 모델
            weight_ptn (Optional[str], optional): 모델 parameter의 name에 대해서 weight의 정규식 패턴. Defaults to None.
                - None인 경우, torch model의 기본 형태인 `r'\\.weight$'`를 사용한다.
            bias_ptn (Optional[str], optional): 모델 parameter의 name에 대해서 bias의 정규식 패턴. Defaults to None.
                - None인 경우, torch model의 기본 형태인 `r'\\.bias$'`를 사용한다.
            block_ptn (Optional[str], optional): 모델 parameter의 name에 대해서 block의 정규식 패턴. Defaults to None.
                - None인 경우, torch model의 기본 형태인 `r'\\.\\d+\\.'`를 사용한다.
        """
        self.model = model
        self.weight_ptn = r"\.weight$" if weight_ptn is None else weight_ptn
        self.bias_ptn = r"\.bias$" if bias_ptn is None else bias_ptn
        self.block_ptn = r"\.\d+\." if block_ptn is None else block_ptn
        self.source = None

    def __call__(self) -> Dict[int, Dict[str, Union[str, bool, int]]]:
        """
        실행 메서드
        """
        # parameter로부터 가장 기본적인 정보를 가져온다.
        self.make_source_dict()
        # parameter를 weight, bias로 구분
        self.add_weight_or_bias()
        # weight, bias를 기준으로 layer 구분
        self.add_layer()
        # layer, 정규식을 이용하여 block 구분
        self.add_block()
        return self.source

    def make_source_dict(self):
        """
        parameter를 mapping하는 재료가 되는 dictionary 생성
        """
        stack_dict = dict()
        for i, (name, param) in enumerate(self.model.named_parameters()):
            stack_dict[i] = {
                "name": name,
                "grad": param.requires_grad,
            }
        self.source = stack_dict

    def add_weight_or_bias(self):
        """
        name을 기준으로 weight, bias를 나눈다.
        """
        re_weight = re.compile(self.weight_ptn)
        re_bias = re.compile(self.bias_ptn)

        for _, value in self.source.items():
            name = value["name"]
            value["weight"] = True if re_weight.search(name) else False
            value["bias"] = True if re_bias.search(name) else False

    def add_layer(self):
        """
        weight, bias를 제외한 name이 이전과 동일하면 동일한 layer로 취급한다.
        """
        i = -1  # 0부터 시작하도록 함.
        before_id = ""
        for _, value in self.source.items():
            if value["weight"]:
                current_id = re.sub(self.weight_ptn, "", value["name"])
            elif value["bias"]:
                current_id = re.sub(self.bias_ptn, "", value["name"])
            else:
                current_id = value["name"]

            # before_id랑 current_id가 다르면 i에 1을 더한다.
            if before_id != current_id:
                i += 1
                before_id = current_id
            value["layer"] = i

    def add_block(self):
        """
        이전 layer와 현재 layer가 다르면서, self.block_ptn이 다른 경우, 다른 block으로 정의한다.
        """
        before_block = None
        before_layer = None
        i = -1
        regex = re.compile(pattern=self.block_ptn)
        for _, value in self.source.items():

            # layer가 다를 때만 정규식 기반 block 확인을 한다.
            current_layer = value["layer"]
            if current_layer != before_layer:

                # 정규식 기반 block 확인
                searched = regex.search(string=value["name"])
                if searched:
                    current_block = searched.group()
                    if current_block != before_block:
                        before_block = current_block
                        i += 1
                else:
                    i += 1

            before_layer = current_layer  # before_layer 갱신
            value["block"] = i


# Layer-wise Unfreezing(선택적 레이어 Unfreeze)
# 방법: 먼저 마지막 분류기 레이어만 학습하고, 이후 특정 레이어를 점진적으로 해제하며 모델 전체를 학습합니다. 일반적으로 하위 레이어보다는 상위 레이어를 먼저 학습합니다.
# 장점: 모델의 중요한 저차원 특징을 유지하며, 상위 레이어의 고수준 특징을 새롭게 학습할 수 있습니다.
# 단점: 조정이 잘못되면 학습이 불안정해질 수 있습니다.

# Freeze and Unfreeze Strategy (고정과 해제 전략)
# 방법: 초기에는 대부분의 레이어를 고정시키고 분류기만 학습하다가, 점진적으로 레이어를 해제하면서 학습을 진행합니다. 이 방법은 특히 학습 초기 불안정함을 줄이기 위해 유용합니다.
# 장점: 학습이 안정적이며, 특정 단계마다 모델의 모든 부분을 최적화할 수 있습니다.
# 단점: 구현 복잡도가 높고, 반복적인 해제와 학습이 필요할 수 있습니다.


# Differential Learning Rate (차등 학습률)
# 방법: 모델의 각 레이어에 대해 다른 학습률을 설정합니다. 일반적으로 상위 레이어에는 더 높은 학습률을, 하위 레이어에는 낮은 학습률을 적용합니다.
# 장점: 사전 학습된 저수준 특징은 크게 수정하지 않으면서 고수준 특징을 효율적으로 조정할 수 있습니다.
# 단점: 각 레이어에 맞는 학습률을 설정하는 것이 복잡할 수 있습니다.

# Layer-wise Adaptive Learning Rates (레이어별 적응 학습률)
# 방법: 모델의 각 레이어에 대해 학습 중 성능에 따라 동적으로 학습률을 조정합니다.
# 장점: 모델이 특정 레이어에 대해 학습을 더 필요로 하는 경우 이를 반영할 수 있습니다.
# 단점: 학습률 조정 로직이 복잡하며, 계산 비용이 증가할 수 있습니다.


# Learning Rate Scheduler (학습률 스케줄러)
# 방법: 학습 과정에서 학습률을 점차 줄여가며 모델이 더 안정적으로 수렴하도록 합니다. StepLR, CosineAnnealingLR와 같은 스케줄러가 대표적입니다.
# 장점: 학습 과정 동안 성능을 높이는 데 유리하며, 최종 수렴 시 과적합을 방지할 수 있습니다.
# 단점: 학습 초기의 학습률 설정에 따라 학습이 너무 느려질 수도 있습니다.

# Regularization Techniques (정규화 기법)
# 방법: L2 정규화, Dropout 등을 적용하여 과적합을 방지하며 모델의 일반화 성능을 높일 수 있습니다.
# 장점: Fine-tuning 시 모델이 과적합되지 않도록 돕고, 일반화 성능을 높여줍니다.
# 단점: 정규화 파라미터를 잘못 설정하면 학습이 어려워질 수 있습니다.
