import re
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

SEED_LIST = [
    1000,
    2000,
    3000,
    4000,
    5000,
    6000,
    7000,
    8000,
    9000,
    1111,
    1101,
    2101,
    3101,
    4101,
    5101,
    6101,
    7101,
    8101,
    9101,
    2222,
    1202,
    2202,
    3202,
    4202,
    5202,
    6202,
    7202,
    8202,
    9202,
    3333,
    1303,
    2303,
    3303,
    4303,
    5303,
    6303,
    7303,
    8303,
    9303,
    4444,
    1404,
    2404,
    3404,
    4404,
    5404,
    6404,
    7404,
    8404,
    9404,
    5555,
    1504,
    2504,
    3504,
    4504,
    5504,
    6504,
    7504,
    8504,
    9504,
    6666,
    1604,
    2604,
    3604,
    4604,
    5604,
    6604,
    7604,
    8604,
    9604,
    7777,
    1704,
    2704,
    3704,
    4704,
    5704,
    6704,
    7704,
    8704,
    9704,
    8888,
    1804,
    2804,
    3804,
    4804,
    5804,
    6804,
    7804,
    8804,
    9804,
    9999,
    1904,
    2904,
    3904,
    4904,
    5904,
    6904,
    7904,
    8904,
    9904,
    9990,
]


class StratifiedIndexdict:

    def __init__(
        self,
        columns: List[str],
        is_binary: bool,
        path_column: str = "path",
        label_column: str = "label",
        onehot_key: str = "dummy",
        rounding_fn: Callable = np.ceil,
    ):
        """
        데이터의 sampling 방식, key_df를 dictionary로 바꾸는 방식에 대한 객체를 정의한다.

        Args:
            columns (List[str]): 층화 표본 추출의 기준이 되는 column의 list
            is_binary (bool): label이 이진 분류인지 여부
            path_column (str, optional): 이미지 데이터의 경로가 들어 있는 key_df의 컬럼명. Defaults to 'path'.
            label_column (str, optional): label이 들어 있는 key_df의 컬럼명. Defaults to 'label'.
            onehot_key (str, optional): 다중분류를 하는 경우 새로 생성되는 컬럼에 추가되는 텍스트. Defaults to 'dummy'.
            rounding_fn (Callable, optional): train, test, validation set을 비율로 나눌 때, 크기의 반올림 방식. Defaults to np.ceil.
        """
        # data의 sampling 방식
        self.sampling_ins = StratifiedSampling(columns=columns, rounding_fn=rounding_fn)
        # key_df를 dictionary로 변환하는 방식
        self.convert_ins = KeyDF2KeyDict(
            path_col=path_column,
            label_col=label_column,
            is_binary=is_binary,
            onehot_key=onehot_key,
        )

    def __call__(
        self,
        key_df: pd.DataFrame,
        k_fold_size: int = 1,
        test_ratio: float = 0.3,
        valid_ratio: Optional[float] = None,
        seed_list: List[int] = SEED_LIST,
        code_test: Optional[float] = None,
    ) -> Dict[int, Dict[str, List[Dict[str, Union[str, np.ndarray]]]]]:
        """
        설정한 방식대로 key_df를 나눈다.

        Args:
            key_df (pd.DataFrame): path, label이 포함된 분할 대상이 되는 데이터 프레임
            k_fold_size (int, optional):  몇 개의 idx_dict을 생성할지(k-fold stratified cross validation). Defaults to 1.
            test_ratio (float, optional): test set의 비율. Defaults to 0.3.
            valid_ratio (Optional[float], optional):  validation set의 비율. Defaults to None.
                - None인 경우, validation set은 생성되지 않는다.
            seed_list (List[int], optional): seed가 들어가 있는 list. Defaults to SEED_LIST.
            code_test (Optional[float], optional): key_df의 크기를 code_test의 비율만큼 축소해서 프로세스를 진행할지. Defaults to None.
                - code_test가 None인 경우, 전체 데이터로 프로세스를 진행한다.
                - code_test가 실수인 경우, 전체 데이터의 크기를 code_test의 비율만틈 축소하여 프로세스를 실행한다.

        Returns:
            Dict[int, Dict[str, List[Dict[str, Union[str, np.ndarray]]]]]: _description_
        """
        # seed_list 크기 확인
        self._check_seed_list_size(k_fold_size, seed_list)

        # code test
        key_df = self._code_test_sampling(key_df, code_test)

        # 데이터 분할
        result = dict()
        for i, seed in enumerate(seed_list[:k_fold_size]):
            result[i] = self.make_idx_dict(
                key_df=key_df, seed=seed, test_ratio=test_ratio, valid_ratio=valid_ratio
            )
        return result

    def make_idx_dict(
        self,
        key_df: pd.DataFrame,
        seed: int,
        test_ratio: float,
        valid_ratio: Optional[float],
    ) -> Dict[str, List[Dict[str, Union[str, np.ndarray]]]]:
        """
        'train', 'test', 'valid'(valid_ratio is not None)의 3(2)개의 key로 구성된 하위 idx_dict 생성

        Args:
            key_df (pd.DataFrame): 분할의 대상이 되는 DataFrame
            seed (int): 무작위 추출 시, seed
            test_ratio (float): test set의 비율
            valid_ratio (Optional[float]): validation set의 비율
                - None인 경우, 생성되지 않음.

        Returns:
            Dict[str, List[Dict[str, Union[str, np.ndarray]]]]: k-fold 중 1개에 대한 idx_dict
        """
        # train, valid set 분할
        test_key_df, rest_key_df = self.sampling_ins(
            df=key_df, ratio=test_ratio, seed=seed
        )
        result = {"test": self.convert_ins(key_df=test_key_df)}

        # valid_ratio가 정의된 경우, valid_set을 추가로 나눔
        if valid_ratio is not None:
            valid_key_df, train_key_df = self.sampling_ins(
                df=rest_key_df, ratio=valid_ratio, seed=seed
            )
            result["train"] = self.convert_ins(key_df=train_key_df)
            result["valid"] = self.convert_ins(key_df=valid_key_df)
        else:
            result["train"] = self.convert_ins(key_df=rest_key_df)

        return result

    def _check_seed_list_size(self, k_fold_size: int, seed_list: List[int]):
        """
        seed_list의 크기가 k_fold_size보다 큰지 확인

        Args:
            k_fold_size (int): idx_dict을 생성할 크기
            seed_list (List[int]): idx_dict을 생성할 seed의 list

        Raises:
            ValueError: seed_list의 크기보다 k_fold_size의 크기가 큰 경우
        """
        seed_size = len(seed_list)
        if k_fold_size > seed_size:
            raise ValueError(
                f"seed_list의 크기가 {seed_size}로, k_fold_size보다 작습니다. seed_list에 새로운 seed를 추가하십시오."
            )

    def _code_test_sampling(
        self, key_df: pd.DataFrame, code_test: Optional[float]
    ) -> pd.DataFrame:
        """
        code_test가 None이 아닌 0 초과 1미만의 실수인 경우(해당 경우는 util.code_test()에서 확인 됨)
        key_df의 크기를 code_test의 비율로, label의 비율을 고려하여 분할

        Args:
            key_df (pd.DataFrame): 원본 key_df
            code_test (Optional[float]): code_test 여부 및 code_test의 비율

        Returns:
            pd.DataFrame: code_test의 비율로 샘플링된 key_df
        """
        if code_test is not None:
            code_test_df, _ = self.sampling_ins(df=key_df, ratio=code_test, seed=1234)
            return code_test_df
        else:
            return key_df


class KeyDF2KeyDict:

    def __init__(
        self,
        path_col: str = "path",
        label_col: str = "label",
        is_binary: bool = True,
        onehot_key: str = "dummy",
    ):
        """
        key_df의 각 레코드를 Dict으로 변환하며, 이 Dict들은 List 안에 들어가 있다.
        >>> callable class로 train, test, validation의 key_df를 대상으로 path와 label에 대하여 key_dict을 생성함
            - binary: 각 레코드에 대해 {"path": path, "label": 스칼라 값} 형태로 출력.
                ex) [
                        {"path":path/filename1.jpg, "label":0},
                        {"path":path/filename2.jpg, "label":1}
                    ]
            - multiclass: 각 레코드에 대해 {"path": path, "label": np.ndarray(원-핫 벡터)} 형태로 출력.
                    [
                        {"path":path/filename1.jpg, "label":np.ndarray([0, 0, 1, 0])},
                        {"path":path/filename1.jpg, "label":np.ndarray([1, 0, 0, 0])}
                    ]

        Args:
            path_col (str, optional): key_df의 path의 컬럼명. Defaults to 'path'.
            label_col (str, optional): key_df의 label의 컬럼명. Defaults to 'label'.
            is_binary (bool, optional): key_df가 이진 분류를 대상으로 하는지 여부. Defaults to True.
                >>> 이진 분류인 경우, label은 스칼라.
                    다중 분류인 경우, label은 np.ndarray(1차원 배열)로 변환.
            onehot_key (str, optional): 다중 분류의 경우, key_df에서 원-핫 인코딩된 컬럼들의 접두사를 정의. Defaults to 'dummy'.
                >>> 다중 분류 시, 컬럼명은 f"{column}_{key}_{class}"로 구성된다.
                >>> class는 레이블 값의 클래스에 해당함.
        """
        self.path_col = path_col
        self.label_col = label_col
        self.is_binary = is_binary
        self.onehot_pattern = f"^{label_col}_{onehot_key}"

    def __call__(self, key_df: pd.DataFrame) -> List[Dict[set, Union[str, np.ndarray]]]:
        """
        클래스의 인스턴스를 함수처럼 호출할 수 있도록 정의된 메서드.
        key_df에 따라 이진 분류 또는 다중 분류에 맞는 key_dict 리스트를 반환한다.

        Args:
            key_df (pd.DataFrame): path와 label 정보를 포함한 DataFrame. 이 데이터프레임을 기반으로 key_dict 리스트를 생성한다.

        Returns:
            List[Dict[str, Union[str, np.ndarray]]]: 각 레코드를 딕셔너리로 변환한 리스트.
                - 이진 분류의 경우: {"path": 경로, "label": 스칼라 값} 형태.
                - 다중 분류의 경우: {"path": 경로, "label": np.ndarray(1차원 배열)} 형태.
        """
        return self._binary(key_df) if self.is_binary else self._multiclass(key_df)

    def _binary(self, key_df: pd.DataFrame) -> List[Dict[str, str]]:
        """
        이진 분류 key_df를 대상으로 key_dict을 생성하는 경우
        """

        def record_to_dict(x):
            path = x[self.path_col]
            label = x[self.label_col]
            return {"path": path, "label": label}

        return key_df.apply(record_to_dict, axis=1).to_list()

    def _multiclass(self, key_df: pd.DataFrame) -> List[Dict[str, np.ndarray]]:
        """
        다중 분류 key_df를 대상으로 key_dict을 생성하는 경우
        """

        def record_to_dict(idx):
            path = key_df.loc[idx, self.path_col]
            label_array = label_mat[idx]
            return {"path": path, "label": label_array}

        # self.onehot_pattern과 매칭되는 컬럼들을 list로 가지고 온다.
        label_columns = [
            i for i in key_df.columns if re.match(self.onehot_pattern, string=i)
        ]
        # self.onehot_pattern이 존재하지 않는 경우 경로문 출력
        if not label_columns:
            raise ValueError(f"No columns match the pattern {self.onehot_pattern}")
        label_mat = key_df[label_columns].values
        return list(map(record_to_dict, key_df.index))


class StratifiedSampling:
    def __init__(self, columns: List[str], rounding_fn: Callable = np.ceil):
        """
        key_df에 대하여 n개의 column에 대하여 층화 표본 추출 한다.

        >>> Callable class로 서로 다른 비율에 대하여, train, valdation, test 3개의 key_df를 생성하는 것을 주 목적으로 함.
            ex) # 전체 데이터의 20%를 test셋으로, 나머지의 10%를 validation set으로 하는 경우.
                this_ins = StratifiedSampling(columns=['label', 'class_0'])
                test_key_df, rest_key_df = this_ins(df=key_df, ratio=0.2)
                valid_key_df, train_key_df = this_ins(df=rest_key_df, ratio=0.1)

        Args:
            columns (List[str]): 층화 표본 추출할 대상이 되는 변수명
            rounding_fn (Callable, optional): 추출할 데이터의 크기 산정 시, 반올림 방식. Defaults to np.ceil.
        """
        self.columns = columns
        self.rounding_fn = rounding_fn
        self.df = None
        self.freq_df = None

    def __call__(
        self, df: pd.DataFrame, ratio: float, seed: Optional[int] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        df를 ratio의 비율로, seed의 난수로 완전 무작위 추출함(층화 표본 추출).

        Args:
            df (pd.DataFrame): 분할의 대상이 되는 key_df
            ratio (float): 분할의 비율
            seed (Optional[int], optional): 난수. Defaults to None.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: 추출된 key_df, 나머지 key_df
        """
        self._instance_variable(df, ratio, seed)  # 초기 변수 설정
        # 계층별 비율을 고려한 무작위 index에 대한 df
        choosed_idx = self._choose_random_index()  # index 추출
        choosed_df = self.df.loc[choosed_idx].reset_index(drop=True)
        # 나머지 index에 대한 df
        rest_idx = list(set(self.df.index) - set(choosed_idx))
        rest_df = self.df.loc[rest_idx].reset_index(drop=True)
        return choosed_df, rest_df

    def _instance_variable(self, df: pd.DataFrame, ratio: float, seed: int):
        """
        instance내 주요 변수(self.df, self.freq_df)와 난수 설정
        """
        if seed is not None:
            np.random.seed(seed)
        self.df = df
        self._freq_table(ratio)  # 빈도표 생성

    def _freq_table(self, ratio: float):
        """
        무작위 추출할 크기(choose)가 추가된 빈도표 생성
        """
        self.freq_df = pd.DataFrame(
            self.df[self.columns].value_counts().sort_index()
        )  # 빈도표 생성
        self.freq_df["choose"] = self.rounding_fn(self.freq_df["count"] * ratio).astype(
            np.int64
        )  # ratio가 반영된 choose 컬럼 생성

    def _choose_random_index(self) -> np.ndarray:
        """
        각 계층에 대하여, self.freq_df의 choose 값을 기반으로 key_df의 index를 완전 무작위 추출
        """
        stack = np.array([])
        for idx in self.freq_df.index:

            size = self.freq_df.loc[idx, "choose"]
            target_idx = self._get_target_idx(
                idx
            )  # freq_df의 index에 해당하는 df의 index를 가지고 온다
            if size > len(target_idx):
                size = len(
                    target_idx
                )  # 올림으로 인해 size가 실제 크기를 초과할 경우, size 수정

            choosed_idx = np.random.choice(target_idx, size=size, replace=False)
            stack = np.concatenate((stack, choosed_idx))
        return stack

    def _get_target_idx(self, idx: Tuple[str, ...]) -> List[int]:
        """
        freq_df의 index(columns)에 해당하는 index를 선택한다.
        """
        # result = set(self.df.index)
        mask = np.ones(len(self.df), dtype=bool)
        for i, column in enumerate(self.freq_df.index.names):
            mask = mask & (self.df[column] == idx[i])
        return self.df[mask].index.tolist()
