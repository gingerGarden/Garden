import os
import sys
import time
from collections import deque
from sys import stdout
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import psutil
from IPython.display import clear_output

from ..path import append_to_json
from ..utils import current_time, decimal_seconds_to_time_string


class ProgressBar:

    def __init__(
        self,
        header: str = "Progress Bar ",
        time_log: bool = True,
        memory_log: bool = False,
        step: int = 5,
        verbose: bool = True,
        bar_size: int = 40,
        save_path: Optional[str] = None,
        round: int = 3,
        done_txt="#",
        rest_txt=".",
    ):
        """
        Iterator의 진행 상태를 Progress Bar로 표현하고, Iterator 순환의 시간, 메모리 관련 정보를 출력한다.
        >>> memory_log의 주의사항


        예시) 임의의 test_list에 대하여, ProgressBar 적용 시

            test_list = ['a', 'b', 'c', 0, 5, 'd', 90, 'f']

            # ProgressBar 객체 생성
            pbar_ins = ProgressBar()
            for i, (bar_txt, value) in enumerate(pbar_ins(test_list)):      # i는 iteratorr의 숫자여야 한다.

                time.sleep(0.5)     # 0.5초씩 대기(예시 method)

                # iterator의 순환의 끝에 해당 self.end_of_iterator() 메서드를 입력하여, log를 출력하고, log의 text, dictionary를 반환한다.
                log_txt, log_dict = pbar_ins.end_of_iterator(progress_txt=bar_txt, i=i)


            # ---- 출력 예시 ----
            # Progress Bar [#########################################]( 8/ 8)
            # [time] spent: 0:00:04.02, eta: 0:00:00.50, 0.503s/it, current: 21:00:55
            # [memory] used:3.317 gb/31.254 gb(12.0 %), rest:27.495 gb, process:0.221 gb, cpu:0.2 %

        Args:
            header (str, optional): Progressbar의 가장 앞에 붙는 텍스트. Defaults to "Progress Bar ".
            time_log (bool, optional): Time log 생성 여부(Time class). Defaults to True.
            memory_log (bool, optional): Memory log 생성 여부(Memory class). Defaults to False.
            step (int, optional): Progress bar를 포함한 log text를 출력하는 iteration 간격. Defaults to 1.
                >>> 가장 처음과 마지막 iteration은 무조건 출력한다.
                >>> log text는 출력하지 않는 간격에선 None으로 출력한다.
                >>> log dict은 출력한다.
            verbose (bool, optional): log 출력 여부. Defaults to True.
            bar_size (int, optional): ProgressBar의 길이. Defaults to 40.
            save_path (str, optional): log가 저장될 파일의 경로. Defaults to None.
            round (str, optional): 소수점 자릿수. Defaults to 3.
            done_txt (str, optional): Progressbar에서 이미 진행된 부분을 표시하는 문자열. Defaults to "#".
            rest_txt (str, optional): Progressbar에서 진행되지 않은 부분을 표시하는 문자열. Defaults to ".".
        """
        self.header = header
        self.time_log = time_log
        self.memory_log = memory_log
        self.step = step
        self.verbose = verbose
        self.bar_size = bar_size
        self.save_path = save_path
        self.round = round
        self.done_txt = done_txt
        self.rest_txt = rest_txt

        # 객체 생성 시 고정 변수
        self.delta_deque = deque(maxlen=2)  # 순간 변화
        self.eta_deque = deque(maxlen=20)  # 평균 변화
        self.is_notebook = (
            "ipykernel" in sys.modules
        )  # ipykernel 모듈이 사용되는지 확인하여, 현재 환경이 Jupyter notebook인지 감지

        # Callable 시 고정 변수
        self.start_time = None  # iteration 시작 당시 시간
        self.iter_size = None  # iteration의 크기
        self.number_size = None  # [  1 / 1000] 의 정수 크기
        self.time_ins = None  # Time class의 instance
        self.memory_ins = None  # Memory class의 instance

    # Callable
    def __call__(self, iter: Iterable) -> Iterable[Tuple[Any, str]]:
        """
        Iterator를 입력받아, Iterator의 value와 progressbar의 text를 출력한다.

        Args:
            iter (Iterator): Progressbar를 출력하고 싶은 임의의 iterator

        Yields:
            Iterator[Tuple[Any, str]]: Iterator의 value와 progressbar string 출력
        """
        # Callable 시, instance variable 설정
        self._callable_initial_variable(iter)
        # Callable에 종속된 sub instance 생성
        self._declare_sub_instance()
        # iterator
        for i, item in enumerate(iter):
            bar_txt = self._progressbar(i)
            yield bar_txt, item

    # Iterator 끝에 추가하는 method
    def end_of_iterator(
        self,
        progress_txt: str,
        i: int,
        extra_log_dict: Optional[Dict[str, Any]] = None,
        end_of_txts: Optional[Union[str, List[str]]] = None,
        dont_save: bool = False,
    ) -> Tuple[str, Dict[str, Dict[str, Union[str, float]]]]:
        """
        Iterator의 끝에서 한 사이클 종료 시점의 시간, 메모리 관련 text를 출력하는 method
        해당 method를 ProgressBar instance의 iterator cycle의 끝에 추가하면, 한 cycle당 log 문구, dictionary를 출력받을 수 있음.

        Args:
            progress_txt (str): iterator에서 value와 함께 나오는 progress bar의 text
            i (int): iterator의 i번째 int
            end_of_txts (Optional[Union[str, List[str]]], optional): log text를 뒤에 붙이는 경우, 문자열 또는 문자열의 List로 가능. Defaults to None.
            dont_save (bool, optional): self.save_path의 여부와 상관 없이, 해당 self.end_of_iterator() 선언 당시, log가 저장되지 않게 할지 여부. Defaults to False.


        Returns:
            Tuple[str, Dict[str, Dict[str, Union[str, float]]]]: log의 문장, log의 dictionary
        """
        # method 시점에 Time, Memory의 Callable
        time_txt, time_log_dict = (
            self.time_ins(i) if self.time_ins is not None else (None, None)
        )
        mem_txt, mem_log_dict = (
            self.memory_ins() if self.memory_ins is not None else (None, None)
        )

        # log dict 저장
        log_dict = {"time": time_log_dict, "memory": mem_log_dict}
        if extra_log_dict is not None:
            log_dict["progress"] = extra_log_dict
        if self.save_path is not None and not dont_save:
            append_to_json(json_path=self.save_path, line=log_dict)

        # log text
        texts = [progress_txt]
        if time_txt is not None:
            texts.append(time_txt)
        if mem_txt is not None:
            texts.append(mem_txt)
        # end_of_txts 처리 - 끝에 붙는 txts
        self._add_end_of_txts(texts=texts, end_of_txts=end_of_txts)

        # 문구는 self.step마다 출력된다.
        log_txt = self._print_by_step(i, texts)

        return log_txt, log_dict

    # 초기 변수 설정
    def _callable_initial_variable(self, iter: Iterable):
        """
        instance의 변수들을 Callable을 선언 시를 기준으로 설정한다.

        설정 변수
            self.iter_size: iterator의 크기
            self.number_size: iterator의 크기의 문자열 크기(len(str(123)) = 3), 1을 추가하여 여백 생성
            self.start_time: Callable 선언하였을 시점의 시간
            self.delta_deque: iterator의 한 번 순회 시간을 알기 위해 delta_deque에 self.start_time 추가

        Args:
            iter (Iterator): Progressbar를 출력하고 싶은 임의의 iterator
        """
        self.iter_size = len(iter)
        self.number_size = len(str(self.iter_size)) + 1
        self.start_time = time.time()
        self.delta_deque.append(self.start_time)

    def _declare_sub_instance(self):
        """
        time, memory log에 대한 instance 생성
        """
        # Time instance 생성
        if self.time_log:
            self.time_ins = Time(
                delta_deque=self.delta_deque,
                eta_deque=self.eta_deque,
                iter_size=self.iter_size,
                start_time=self.start_time,
                round=self.round,
            )
        else:
            self.time_ins = None

        if self.memory_log:
            self.memory_ins = Memory(unit="gb", round=self.round)
        else:
            self.memory_ins = None

    def _progressbar(self, i: int) -> str:
        """
        iterator에 대하여, header, i, iteration의 크기 등을 기반으로 progressbar txt를 생성한다.
        example) header [##############...........................] (12/100)

        Args:
            header (str): progressbar 앞에 붙는 iterator의 주제
            i (int): iterator에서 출력된 순서(enumerate()로 출력된 n값)
            iter_size (int): iterator의 크기
            bar_size (int): progressbar의 크기. Defaults to 40.
            number_size (Optional[int], optional): progressbar의 정수로된 진행 상황의 정수 고정 크기. Defaults to None.
                - (    1/ 1000) 처럼 정수의 앞과 뒤의 크기를 맞추기 위한 parameter
                - None인 경우, iter_size를 기반으로 계산: len(str(iter_size)) + 1
                - +1은 여백을 위해 추가
            done_txt (str, optional): progressbar에서 진행된 부분의 txt. Defaults to "#".
            rest_txt (str, optional): progressbar에서 진행되지 않은 부분의 txt. Defaults to ".".

        Returns:
            str: _description_
        """
        corrected_i = (
            i + 1
        )  # i 는 0부터 시작하나, progressbar는 1부터 시작해야하므로, 이를 보정함
        # bar txt
        done_size = int(
            self.bar_size * corrected_i / self.iter_size
        )  # iter에서 진행된 크기
        rest_size = self.bar_size - done_size  # 진행되지 않은 크기
        bar_txt = f"{self.header}[{self.done_txt * (done_size+1)}{self.rest_txt * (rest_size)}]"
        # number txt
        count = (
            f"({corrected_i:>{self.number_size}}/{self.iter_size:>{self.number_size}})"
        )
        return bar_txt + count

    def _print_by_step(self, i: int, texts: List[str]) -> Optional[str]:
        """
        self._print_progressbar를 self.step 간격으로 출력(마지막 iterator는 무조건 출력)

        Args:
            i (int): 현재의 iterator
            texts (List[str]): 출력할 문구의 list

        Returns:
            Optional[str]: log sentence
        """
        if ((i + 1) % self.step == 0) or ((i + 1) == self.iter_size) or (i == 0):
            return self._print_progressbar(texts)
        else:
            return None

    def _print_progressbar(self, texts: List[str]) -> str:
        """
        log 문구 출력
        - verbose에 따라 sentence를 생성할지, 출력할지 여부가 달라짐
            0: sentence를 생성하고 출력
            1: sentence를 생성하고 출력하지 않음
            2: sentence를 생성하지 않음
        - verbose가 생성될 필요가 없다할지라도, log_dict는 생성되어 저장되어야 할 수 있음

        Args:
            texts (List[str]): 출력하고자 하는 문구들의 List
                >>> 해당 List는 element 단위로 내려쓰기 한다.

        Returns:
            str: 내려쓰기를 '\n'으로 연결한 log의 string
        """
        if self.verbose:
            sentence = ""
            for text in texts:
                sentence = sentence + text + "\n"
            # print
            stdout.write(sentence)
            stdout.flush()
            # Jupyter notebook 환경일 때만 clear_output 사용
            if self.is_notebook:
                clear_output(wait=True)
            return sentence
        else:
            return None

    def _add_end_of_txts(
        self, texts: List[str], end_of_txts: Union[str, List[str]]
    ) -> List[str]:
        """
        문자열의 마지막에 붙을 문자열의 List 추가
        - 마지막에 붙을 문자열은 반드시 str 또는 list여야 함

        Args:
            texts (List[str]): 기존 문자열
            end_of_txts (Union[str, List[str]]): 마지막에 붙일 문자열

        Raises:
            TypeError: str, list가 아닌 문자열이 입력 되는 경우

        Returns:
            List[str]: end_of_txts가 추가된 texts
        """
        if end_of_txts is not None:
            if isinstance(end_of_txts, str):
                texts.append(end_of_txts)
            elif isinstance(end_of_txts, list):
                texts = texts + end_of_txts
            else:
                dtype = type(end_of_txts)
                raise TypeError(
                    f"end_of_txts에 {dtype}이 입력되었습니다. str, list만 입력 가능합니다!"
                )


class Time:
    def __init__(
        self,
        delta_deque: deque,
        eta_deque: deque,
        iter_size: int,
        start_time: float,
        round: int,
    ):
        """
        ProgressBar class와 함께 사용되며, iterator 내부에서 시간 관련 정보를 측정한다.

        측정하는 정보는 다음과 같다.
            1. spent: iterator 시작부터 Callable까지 소모 시간
            2. eta: eta_deque를 기반으로 예상 완료 시간
            3. iter s/it: iterator의 평균 소모 시간
            4. current: 해당 log가 출력된 현재 시간
            5. eta_deque[-1]: Callable되었을 때, itarator의 소모 시간

        Args:
            delta_deque (deque): ProcessBar class에서 iterator 각 cycle에 대한 소모 시간 측정을 위한 deque
            eta_deque (deque): ProcessBar class에서 iterator 각 cycle의 소모 시간이 누적되는 deque
                - 예상 완료 시간 산출 목적
            iter_size (int): ProcessBar class에 입력된 iterator의 크기
            start_time (float): ProcessBar class에서 iterator가 시작되었을 때의 시간
            round (int): 소수점 자릿수
        """
        self.delta_deque = delta_deque
        self.eta_deque = eta_deque
        self.iter_size = iter_size
        self.start_time = start_time
        self.round = round

    def __call__(self, i: int) -> Tuple[str, Dict[str, Union[str, float]]]:
        """
        iterator 안에서 Callable 되었을 때를 기준으로 시간 관련 정보 생성

        생성 정보는 두 가지로 다음과 같다.
        1. text: [time] spend: 1 day, 6:28:55.33, eta: 0:00:00.00, 133.6778s/it, current: 2024.07.24 08:16:26]
        2. dictionary: {"current": "2024.07.24 08:16:26", "spent":"1 day, 6:28:55.33", "iteration":133.6778}

        Args:
            i (int): 현재 iterator의 순서

        Returns:
            Tuple[str, Dict[str, Union[str, float]]]: 시간 log text, 시간 log의 dictionary
        """
        # 시간 deque 추가
        self.delta_deque.append(time.time())  # callable 실행 시, 시간 추가
        self.eta_deque.append(
            float(self.delta_deque[1] - self.delta_deque[0])
        )  # 한 iterater의 소모 시간 추가
        # 주요 시간 정보 생성
        current, iter_mean, eta, spent = self._make_interest_time(i)
        # 정리 및 출력
        time_txt = self._make_to_time_txt(current, iter_mean, eta, spent)
        log_dict = self._log_dict(current, spent)
        return time_txt, log_dict

    def _make_interest_time(self, i: int) -> Tuple[str, float, float, float]:
        """
        관심 시간 변수들을 계산한다.
        current: 현재 시간
        iter_mean: eta_deque의 평균 시간(iterator의 평균 시간)
        eta: iterator 종료까지 예상 시간
        spent: iterator 시작부터 현재까지 소모된 시간
        """
        current = current_time(only_time=False)  # log 생성을 위한 현재 시간 문자열
        iter_mean = np.mean(self.eta_deque)  # eta_deque 기준, 1 iter 당 평균 시간 계산
        eta = (self.iter_size - i) * iter_mean  # 예상 남은 시간
        spent = (
            self.delta_deque[-1] - self.start_time
        )  # itorator 시작부터 callable까지 소모된 시간
        return current, iter_mean, eta, spent

    def _make_to_time_txt(
        self, current: str, iter_mean: float, eta: float, spent: float
    ):
        """
        self._make_interest_time()로 계산된 관심 시간 변수들을 text로 정리
        """
        spent_txt = decimal_seconds_to_time_string(decimal_s=spent)
        eta_txt = decimal_seconds_to_time_string(decimal_s=eta)
        clean_iter_mean = f"{iter_mean:.{self.round}f}"
        return f"[time] spent: {spent_txt}, eta: {eta_txt}, {clean_iter_mean}s/it, current: {current}"

    def _log_dict(self, current: str, spent: str):
        """
        주요 시간 정보들을 log로 저장
        """
        log_dict = {"current": current, "spent": spent, "iteration": self.eta_deque[-1]}
        return log_dict


class Memory:
    def __init__(self, unit: str = "gb", round: int = 3, process_id=os.getpid()):
        """
        ProgressBar class와 함께 사용되며(현 시점의 메모리 상태 출력 가능), Callable 되었을 때, memory 관련 정보를 출력한다.

        Args:
            unit (str, optional): memory 표현의 단위. Defaults to "gb".
                >>> 'kb', 'mb', 'gb', 'tb'
            round (int, optional): 소수점 자릿수. Defaults to 3.
            process_id (_type_, optional): 대상 process의 id. Defaults to os.getpid().
                >>> 별도로 설정하지 않는 경우, 해당 process의 id를 입력한다.
        """
        self.unit = unit
        self.round = round
        self.process_pid = process_id

    def __call__(self) -> Tuple[str, Dict[str, str]]:
        """
        Callable하였을 때, memory 정보를 출력

        Returns:
            Tuple[str, Dict[str, str]]: memory log text, memory log dictionary
        """
        mem_dict = self.current_memory()
        mem_txt = self._memory_txt(mem_dict)
        return mem_txt, mem_dict

    def current_memory(self):
        """
        현재 메모리 현황을 dictionary로 생성
        """
        p = psutil.virtual_memory()
        result = {
            "total": self.byte_convertor(p.total, unit=self.unit),
            "used": self.byte_convertor(p.used, unit=self.unit),
            "used_percent": f"{p.percent} %",
            "available": self.byte_convertor(p.available, unit=self.unit),
            "process": self.byte_convertor(self._process_memory(), unit=self.unit),
            "cpu": f"{psutil.cpu_percent()} %",
        }
        return result

    def byte_convertor(
        self, byte: int, unit: str, add_unit: bool = True
    ) -> Union[str, float]:
        """
        byte를 unit('kb', 'mb', 'gb', 'tb') 단위로 변환하여 출력한다.

        Args:
            byte (int): byte
            unit (str): 변환할 단위
            add_unit (bool, optional): 단위를 함께 출력할지 여부. Defaults to True.
                >>> True인 경우, '12.041 gb' 같은 식으로 출력

        Returns:
            Union[str, float]: add_unit 여부에 따라, 문자열(단위와 함께) 또는 실수로 출력
        """
        denominators = {"kb": 2**10, "mb": 2**20, "gb": 2**30, "tb": 2**40}
        # 분모 정의
        denominator = denominators.get(unit)
        # 단위 반영
        value = f"{byte/denominator:.{self.round}f}"
        result = f"{value} {self.unit}" if add_unit else value
        return result

    def _process_memory(self) -> int:
        """
        self.process_id와 자녀 process가 소모한 memory의 byte 출력

        Returns:
            int: byte
        """
        # get target process memory information
        p = psutil.Process(self.process_pid)
        # this process(parents) memory usage
        mem_use = p.memory_info().rss
        # add child process memory usage
        for child in p.children(recursive=True):
            mem_use += child.memory_info().rss
        return mem_use

    def _memory_txt(self, mem_dict: Dict[str, str]) -> str:
        """
        memory_dict을 문자열로 출력
        예시) [memory] used:2.953 gb/31.254 gb(10.9 %), rest:27.859 gb, process:0.221 gb, cpu:0.5 %
        """
        log_txt = (
            f"[memory] used:{mem_dict['used']}/{mem_dict['total']} ({mem_dict['used_percent']}),"
            f" rest:{mem_dict['available']}, process:{mem_dict['process']},"
            f" cpu:{mem_dict['cpu']}"
        )
        return log_txt
