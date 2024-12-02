import os
import re
import warnings
from typing import Callable, List, Optional

import timm
from huggingface_hub import hf_hub_download
from torch import nn

from ...utils.module import GithubHelper
from ...utils.path import check_file_path_use_only_txt, file_download


class Classification(nn.Module):
    def __init__(
        self,
        model_name: str,
        pretrained: bool = True,
        class_size: Optional[int] = None,
        channel: int = 3,
        global_pooling: Optional[str] = None,
        custom_head_fn: Optional[Callable] = None,
    ):
        """
        Timm(Torch)을 기반으로 하는 기본적인 분류 모델 class
        >>> custom_head_fn
            - backbobe 모델의 구조를 훼손하지 않으면서 custom header를 추가 가능.
            - custom header의 입력 node의 크기는 backbone 모델의 기본 feature map 크기와 동일하도록, 입력 node의 크기를 임의의 변수 x로 입력받는 method로 추가
            - example)  def custom_header(x):
                            custom_head = nn.Sequential(
                                nn.Linear(x, 512),
                                nn.ReLU(),
                                nn.Dropout(0.5),
                                nn.Linear(512, 10)
                            )
                            return custom_head

        Args:
            model_name (str): timm 모델명
            pretrained (bool, optional): 사전 학습된 모델을 대상으로 할지 여부. Defaults to True.
            class_size (Optional[int], optional): 모델의 출력층. Defaults to None.
                - custom_head_fn이 None이 아닌 경우, class_size는 None으로 설정된다.
            channel (int, optional): 입력 이미지의 channel 크기. Defaults to 3.
            global_pooling (Optional[str], optional): Backbone 모델의 마지막 Global pooling 방식. Defaults to None.
                - Feature map을 고정된 크기의 벡터로 변환하는 것을 목적으로 함.
                - 'avg': Feature map의 평균적인 정보를 반영하므로, 이미지 전체의 균형 잡힌 Feature를 얻는데 유리
                - 'max': Feature map에서 가장 강한 Feature를 강조하므로, 중요한 영역을 더 잘 나타내고자 할 때 유리
                - '': Global pooling을 사용하지 않음 - Flatten layer로 처리
                - 기존 architecture와 일부 달라져 경고문이 출력될 수 있음
            custom_head_fn (Optional[Callable], optional): custom header의 function. Defaults to None.
        """
        # 부모 클래스를 초기화하여, torch.nn.Module의 메서드들을 사용할 수 있도록 함.
        super(Classification, self).__init__()

        # model 이름이 존재하는 경우, 기본 backbone 모델 설정
        self.backbone = None
        if TimmHelper.checker(model_name=model_name):
            self._set_backbone(
                model_name,
                pretrained,
                class_size,
                channel,
                global_pooling,
                custom_head_fn,
            )

        # custom_head 정의
        self.custom_head = None
        if custom_head_fn is not None:
            self._set_custom_head(custom_head_fn)

    def forward(self, x):
        x = self.backbone(x)
        if self.custom_head is not None:
            x = self.custom_head(x)
        return x

    def _set_backbone(
        self,
        model_name: str,
        pretrained: bool,
        class_size: Optional[int],
        channel: Optional[int],
        global_pooling: Optional[str],
        custom_head_fn: Optional[Callable],
    ):
        """
        Backbone 모델을 설정하는 메서드
        """
        if custom_head_fn is not None:
            class_size = 0

        self.backbone = timm.create_model(
            model_name=model_name,
            pretrained=pretrained,
            in_chans=channel,
            num_classes=class_size,
            global_pool=global_pooling,
        )

    def _set_custom_head(self, custom_head_fn: Callable):
        """
        Custom head를 설정하는 메서드
        """
        self.custom_head = custom_head_fn(self.backbone.num_features)


class TimmHelper:
    model_list = timm.list_models(pretrained=True)

    @classmethod
    def search(
        cls, pattern: str, match_exact: bool = False, ignore_case: bool = True
    ) -> List[str]:
        """
        정규표현식을 기반으로 timm에서 pattern에 해당하는 모델의 이름들을 가지고 온다.

        Args:
            pattern (str): pattern을 포함하거나 일치하는 모델을 가지고 온다.
            match_exact (bool, optional): pattern과 일치하는 모델을 대상으로 할지 여부. Defaults to False.
            ignore_case (bool, optional): 대소문자 무시 여부. Defaults to True.

        Raises:
            ValueError: 정규식 패턴에 해당하지 않는 값이 입력된 경우

        Returns:
            List[str]: pattern에 대한 모델들의 list
        """
        # 입력 pattern 무결성 확인
        try:
            flags = re.IGNORECASE if ignore_case else 0
            regex = re.compile(pattern=pattern, flags=flags)
        except re.error as e:
            raise ValueError(f"Invalid regular expression pattern: {pattern}") from e

        # 모델명 탐색
        fn = regex.fullmatch if match_exact else regex.search
        result = [model for model in cls.model_list if fn(string=model)]

        # 결과가 하나도 없는 경우 문구 출력
        if len(result) == 0:
            warnings.warn(
                "timm에 찾는 모델이 존재하지 않습니다. self.model_list로 전체 모델 확인이 가능합니다."
            )
        return result

    @classmethod
    def checker(cls, model_name: str) -> bool:
        """
        model_name이 Timm의 model 목록에 존재하는지 확인한다.

        Args:
            model_name (str): 사용하고자 하는 Timm 모델의 이름

        Raises:
            ValueError: 입력한 model_name이 존재하지 않는 경우

        Returns:
            bool: 입력한 model_name이 존재하는 경우 True 출력
        """
        if model_name not in cls.model_list:
            message = f"입력하신 model_name: {model_name}는 Timm에 존재하지 않습니다. 동일 모듈 내 TimmHelper.search() 메서드를 이용하면, 필요한 모델을 쉽게 찾을 수 있습니다."
            raise ValueError(message)
        else:
            return True

    @classmethod
    def download_parameter_from_hf(
        cls, repo_id: str, param_file: str = "pytorch_model.bin"
    ) -> str:
        """
        Hugging face에서 모델의 parameter를 다운로드한다.

        Args:
            repo_id (str): hugging face repo id
            param_file (str, optional): parameter file의 명칭. Defaults to "pytorch_model.bin".

        Raises:
            ConnectionError: parameter 다운로드 실패 시 발생

        Returns:
            str: parameter 파일 경로
        """
        try:
            param_path = hf_hub_download(repo_id=repo_id, filename=param_file)
            return param_path
        except Exception as e:
            raise ConnectionError(
                f"Parameter 다운로드 실패! Hugging face repo: '{repo_id}'가 다운로드 되지 않았습니다!"
            ) from e

    @classmethod
    def download_architecture_from_github(
        cls,
        url: str,
        parents_path: str,
        github_token: Optional[str] = None,
        is_package: Optional[bool] = None,
    ):
        """
        gihub의 url로부터 모델 아키텍처에 필요한 코드들을 모두 다운로드 한다.
        >>> is_package: 단일 파일로 받을지, 패키지 구조로 받을지 선택한다.
            - None인 경우, url을 기반으로 패키지 여부를 결정한다.
            - True: 깃허브 레포지토리에 있는 특정 패키지 내 모든 콘텐츠들을 다운로드 한다.
            - False: 깃허브 레포지토리 내에 있는 특정 python 파일을 다운로드 한다(Raw url을 대상으로 함).

        Args:
            url (str): 다운로드 받고자 하는 github 파일의 url
            parents_path (str): 데이터를 다운로드 하고자하는 부모 경로
            github_token (Optional[str], optional): github 토큰. Defaults to None.
            is_package (Optional[bool], optional): 다운로드 하고자 하는 파일이 package 구성인지 여부. Defaults to None.
        """
        # url에서 파일의 이름을 가지고 온다(패키지 또는 파일일 수 있음)
        last_word = url.rsplit("/", maxsplit=1)[1]

        # is_package가 None인 경우, last_word를 기반으로 file 여부를 확인
        if is_package is None:
            is_file = check_file_path_use_only_txt(last_word)
            is_package = False if is_file else True

        # download
        if is_package:
            GithubHelper.download_repo_directory(url, parents_path, token=github_token)
        else:
            file_path = os.path.join(parents_path, last_word)
            file_download(url, local_path=file_path)
