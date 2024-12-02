import re
from typing import List, Optional, Tuple, TypeAlias, Union

import numpy as np
import pandas as pd


class ModifyTexts:
    TextCollection: TypeAlias = Union[List[str], np.ndarray, pd.Series]

    @classmethod
    def replace(
        cls,
        texts: TextCollection,
        patterns: Union[str, List[str]],
        replacements: Union[str, List[str]] = "",
    ) -> TextCollection:
        """
        texts에 들어 있는 pattern들을 replacement의 list로 교체한다.

        Args:
            texts (TextCollection): 패턴들을 포함하고 있는 문자열의 모음
            patterns (Union[str, List[str]]): 교체할 패턴들
            replacements (Union[str, List[str]], optional): 교체될 문자열들. Defaults to ''.

        Returns:
            TextCollection: 패턴이 교체된 문자열의 모음
        """
        # texts에 대한 검증
        cls._check_texts(texts)
        # check patterns and replacements
        patterns, replacements = cls._check_patterns_and_replacements(
            patterns, replacements
        )

        # Apply each pattern-replacement pair
        def apply_patterns(txt: str) -> str:
            compiled_patterns = [
                (re.compile(p), r) for p, r in zip(patterns, replacements)
            ]
            for compiler, r in compiled_patterns:
                txt = compiler.sub(r, txt)
            return txt

        # check input type and process accordingly
        return cls._apply_to_texts(texts, apply_patterns)

    @classmethod
    def replace_without_specific_txt(
        cls,
        texts: TextCollection,
        pat: str,
        live_pat: str,
        except_end_size: Optional[int] = None,
    ) -> TextCollection:
        """
        texts에 들어 있는 pat에서 live_pat을 제외하고 제거한다.

        Args:
            texts (TextCollection): 패턴들을 포함하고 있는 문자열의 모음
            pat (str): 교체할 패턴
            live_pat (str): 교체하지 않을 패턴(교체할 패턴 내 존재해야하며, 해당 live_pat을 pat에 교체하는 방식)
            except_end_size (Optional[int], optional): 교체할 패턴에서 끝 부분을 살릴 크기. Defaults to None.
                - 0, None, int가 아닌 값 입력 시, 적용되지 않는다.
                - 교체하고자 하는 패턴의 끝 부분에 일부 문자열은 교체하고 싶지 않을 때 사용한다.
                - ex) ' (2014, Saunders_ Elsevier Health Sciences) ' 문자열을 ' (2014) '로 하고자 하는 경우, 2로 설정

        Returns:
            TextCollection: 패턴이 교체된 문자열의 모음
        """
        # texts에 대한 검증
        cls._check_texts(texts)

        # compiler 생성
        pat_comp = re.compile(pattern=pat)
        live_comp = re.compile(pattern=live_pat)

        def apply_patterns(text: str) -> str:
            # pat_comp로 패턴 탐색
            pat_match = pat_comp.search(string=text)
            if not pat_match:
                return text

            # pat_match에서 live_comp로 대체 패턴 탐색
            live_match = live_comp.search(string=pat_match.group(0))
            if not live_match:
                return text

            # 수정하고자 하는 pattern에서 제외 크기
            mask = (
                except_end_size is not None
                and except_end_size != 0
                and isinstance(except_end_size, int)
            )
            target_pat = (
                pat_match.group(0)[:-except_end_size] if mask else pat_match.group(0)
            )
            return re.sub(
                pattern=re.escape(target_pat), repl=live_match.group(0), string=text
            )

        # check input type and process accordingly
        return cls._apply_to_texts(texts, apply_patterns)

    @classmethod
    def compact_spaces(
        cls, texts: TextCollection, strip: bool = True
    ) -> TextCollection:
        """
        texts에 들어 있는 공백들을 깔끔하게 만든다.
        - 2개 이상의 공백을 1개의 공백으로 만든다.
        - 문자열 앞, 뒤에 있는 공백을 제거한다(strip=True)

        Args:
            texts (Union[List[str], np.ndarray, pd.Series]): 문자열의 모음
            strip (bool, optional): 문자열의 앞, 뒤 공백 제거 여부. Defaults to True.

        Returns:
            Union[List[str], np.ndarray, pd.Series]: 공백이 정리된 문자열의 모음
        """
        # texts에 대한 검증
        cls._check_texts(texts)

        def apply_patterns(txt: str) -> str:
            return (
                re.sub(r"\s+", " ", txt).strip() if strip else re.sub(r"\s+", " ", txt)
            )

        return cls._apply_to_texts(texts, apply_patterns)

    @classmethod
    def convert_case(
        cls,
        texts: TextCollection,
        upper: bool = False,
        lower: bool = False,
        capitalize: bool = False,
    ):
        # texts에 대한 검증
        cls._check_texts(texts)

        def apply_pattern(txt: str) -> str:
            if capitalize:
                return txt.capitalize()
            elif upper:
                return txt.upper()
            elif lower:
                return txt.lower()
            else:
                return txt

        return cls._apply_to_texts(texts, apply_pattern)

    @staticmethod
    def _check_texts(texts: TextCollection):
        """
        입력된 문자열들이 올바른지 확인한다.

        Args:
            texts (TextCollection): 문자열의 모음으로, list, pd.Series, np.ndarray 중 하나를 입력

        Raises:
            TypeError:
                - list: 문자열이 아닌 요소가 포함된 경우
                - pd.Series: dtype이 'object' 또는 'string'이 아닌 경우
                - np.ndarray: dtype이 'U' (유니코드), 'S' (바이트 문자열), 'O' (객체)가 아닌 경우
                - 그 외 지원되지 않는 타입이 입력된 경우

        Returns:
            bool: True if validation is successful.
        """
        if isinstance(texts, list):
            if not all(isinstance(txt, str) for txt in texts):
                raise TypeError("All elements in the list must be strings.")
        elif isinstance(texts, pd.Series):
            if texts.dtype not in ["object", "string"]:
                raise TypeError("All elements in the series must be strings(object).")
        elif isinstance(texts, np.ndarray):
            if texts.dtype.kind not in {"U", "S", "O"}:
                raise TypeError("All elements in the array must be strings(object).")
        else:
            raise TypeError("texts must be a list, Pandas Series, or Numpy ndarray.")

        return True

    @staticmethod
    def _check_patterns_and_replacements(
        patterns: Union[str, List[str]], replacements: Union[str, List[str]]
    ) -> Tuple[List[str], List[str]]:
        """
        patterns와 replacements에 입력된 데이터가 string인 경우, List로 크기 조정을 한다.

        Args:
            patterns (Union[str, List[str]]): pattern의 list 또는 str
            replacements (Union[str, List[str]]): replacement의 list 또는 str

        Raises:
            ValueError: patterns의 길이와 replacements의 길이가 일치하지 않는 경우

        Returns:
            Tuple[List[str], List[str]]: pattern과 replacement의 list
        """
        # convert single pattern/replacement to List for uniform processing
        if isinstance(patterns, str):
            patterns = [patterns]
        if isinstance(replacements, str):
            replacements = [replacements] * len(patterns)
        # validate Lengths of patterns and replacements
        if len(patterns) != len(replacements):
            raise ValueError("The number of patterns and replacements must match.")
        return patterns, replacements

    @staticmethod
    def _apply_to_texts(
        texts: TextCollection, apply_patterns: callable
    ) -> TextCollection:
        """
        List, np.ndarray, pd.Series로 이루어진 texts에 각 method 내 존재하는 apply_patterns 적용

        Args:
            texts (Union[List[str], np.ndarray, pd.Series]): list, np.ndarray, pd.Series 중 하나로
                이루어진 문자열의 모임
            apply_patterns (callable): cls.method 내, patterns, replacements를 순서대로 적용하는 method

        Raises:
            TypeError: list, np.ndarray, pd.Series가 아닌 데이터가 입력된 경우

        Returns:
            Union[List[str], np.ndarray, pd.Series]: apply_patterns이 적용된 list, np.ndarray, pd.Series
        """
        if isinstance(texts, pd.Series):
            return texts.apply(apply_patterns)
        elif isinstance(texts, np.ndarray):
            return np.vectorize(apply_patterns)(texts)
        elif isinstance(texts, list):
            return [apply_patterns(txt) for txt in texts]
        else:
            raise TypeError(
                "Input texts must be a list, numpy array, or pandas Series."
            )
