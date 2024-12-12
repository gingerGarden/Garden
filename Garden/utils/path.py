import json
import os
import pickle
import shutil
import logging
import zipfile
from collections.abc import Callable
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import requests

from .utils import list_flatten


def do_or_load(
    savepath: str, fn: Callable, makes_new: bool = False, *args, **kwargs
) -> Any:
    """makes_new가 True거나 filepath에 fn의 결과로 인해 생성되는 파일이 없다면, fn을 동작하고 파일을 저장한다.
    만약 있다면, 그냥 파일을 가져온다.

    Args:
        filepath (str): fn의 결과로 인해 생성되는 데이터가 pickle로 저장될 경로
        fn (Callable): 임의의 함수
        makes_new (bool, optional): filepath에 결과가 있더라도 fn을 동작함. Defaults to False.

    Returns:
        Any: fn의 결과
    """
    if not os.path.exists(savepath) or makes_new:
        result = fn(*args, **kwargs)
        save_pickle(data=result, pickle_path=savepath)
    else:
        result = load_pickle(pickle_path=savepath)
    return result


def new_dir_maker(dir_path: str, makes_new=True):
    """dir_path 디렉터리 생성

    Args:
        dir_path (str): 디렉터리 경로
        makes_new (bool, optional): 디렉터리가 이미 존재하는 경우, 새로 만들지 여부. Defaults to True.
    """
    if os.path.exists(dir_path):
        if makes_new:
            remove_target_path(target_path=dir_path)
            os.mkdir(dir_path)
        else:
            pass
    else:
        os.mkdir(dir_path)


def save_pickle(data: Any, pickle_path: str):
    """data를 pickle_path 경로에 pickle로 저장

    Args:
        data (Any): 대상 데이터
        pickle_path (str): pickle의 경로
    """
    with open(pickle_path, "wb") as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)


def load_pickle(pickle_path: str) -> Any:
    """pickle_path 경로의 pickle을 불러옴

    Args:
        pickle_path (str): pickle 파일의 경로

    Returns:
        Any: pickle 안 데이터
    """
    with open(pickle_path, "rb") as f:
        return pickle.load(f)


def write_json(data: dict, json_path: str):
    """dictionary를 json_path에 저장
    >>> json에 입력 불가능한 python dtype이 있는 경우, 저장이 안됨

    Args:
        data (dict): dicationary
        json_path (str): 저장할 json 파일의 경로
    """
    with open(json_path, "w") as f:
        json.dump(data, f)


def read_json(json_path: str) -> dict:
    """json_path에서 dictionary를 읽어온다.

    Args:
        json_path (str): json 파일의 경로

    Returns:
        dict: json 파일 안에 있던 dictionary
    """
    with open(json_path, "r") as f:
        return json.load(f)


def make_null_json(
    path: str, empty: bool = True, null_list: bool = True, make_new: bool = False
):
    """
    비어있는 json을 생성한다(공백, list 또는 dictionary).

    Args:
        path (str): json 파일의 경로
        empty (bool, optional): 아무 것도 들어 있지 않은 json 파일 생성. Defaults to True.
        null_list (bool, optional): empty가 False인 경우, 비어 있는 list 또는 dictionary 생성. Defaults to True.
            - null_list == True인 경우, []
            - null_list == False인 경우, {}
        make_new (bool, optional): 새로 만들지 여부. Defaults to False.
    """
    if not os.path.exists(path) or make_new:
        if empty:
            # 빈 파일 생성
            with open(path, "w"):
                pass  # 아무것도 작성하지 않음
        else:
            data = [] if null_list else {}
            write_json(data, json_path=path)


def append_to_json(json_path: str, line: Union[Dict[str, Any], List[Any], str]):
    """
    json file에 line에 입력된 문자열 또는 dictionary를 추가.
    file이 없는 경우, 해당 json 파일 생성

    Args:
        json_path (str): json 파일의 경로
        line (Union[Dict[str, Any], List[Any], str])): json 파일에 append할 line
    """
    with open(
        json_path, "a", encoding="utf-8"
    ) as f:  # append 모드로, 새로운 데이터 파일 끝에 추가
        json.dump(
            line, f, ensure_ascii=False
        )  # ASCII 형식 대신 utf-8로 저장하여, 한글, 특수 문자 등 유지
        f.write("\n")  # line 간 구분을 위해 줄바꿈 추가
        
        
def unzip(file_path: str, unzip_name: Optional[str] = None, remove_zipfile: bool = False):
    """
    대상 file이 zipfile인 경우, 해당 zipfile을 해제한다.
        - 압축 해제 후, 디렉터리명 설정 가능(unzip_name is not None)
        - 압축 해제 후, zipfile 제거 여부(remove_zipfile = True)

    Args:
        file_path (str): zipfile로 의심되는 file의 경로
        unzip_name (Optional[str], optional): 압축 해제할 디렉터리명(동일 경로에 생성). Defaults to None.
        remove_zipfile (bool, optional): 압축 파일 제거 여부. Defaults to False.
    """
    # ZIP 파일 확인
    if not zipfile.is_zipfile(file_path):
        raise ValueError(f"{file_path}는 ZIP 파일이 아닙니다.")
    
    # 해제 디렉터리명 생성
    if unzip_name is None:
        extract_to_dir = os.path.splitext(file_path)[0]
    else:
        parents_path = os.path.dirname(parents_path)
        extract_to_dir = os.path.join(parents_path, unzip_name)
    
    # 해제 디렉터리가 이미 존재하는 경우, 정지
    if os.path.exists(extract_to_dir):
        raise RuntimeError(f"{extract_to_dir} 경로가 이미 존재합니다. unzip_name을 수정하십시오.")
        
    # 압축 해제
    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to_dir)
        
    # 압축 파일 제거
    if remove_zipfile:
        os.remove(file_path)        


def read_txt_file(file_path: str) -> str:
    """
    txt 파일의 내용을 읽어서 문자열로 가지고 온다.

    Args:
        file_path (str): txt 파일의 경로

    Returns:
        str: txt 파일의 내용
    """
    with open(file_path, "r") as file:
        content = file.read()
    return content


def write_txt_file(file_path: str, content: str) -> None:
    """
    content를 file_path에 txt파일로 저장한다.

    Args:
        file_path (str): txt 파일을 저장할 경로
        content (str): txt 파일의 내용
    """
    with open(file_path, "w") as file:
        file.write(content)


def get_file_name(file_path: str, no_extension: bool = True) -> str:
    """
    file_path로부터 file 이름을 가지고 온다.

    Args:
        file_path (str): file의 경로
        no_extension (bool, optional): 확장자를 함께 가져올지 여부. Defaults to True.

    Returns:
        str: file 이름
    """
    # error check
    if not isinstance(file_path, str):
        raise ValueError("file_path should be a string.")
    if not file_path:
        raise ValueError("file_path cannot be an empty string or None.")

    # main
    file_name = os.path.basename(file_path)
    if no_extension:
        file_name = os.path.splitext(file_name)[0]
    return file_name


def check_file_path_use_only_txt(txt: str) -> bool:
    """
    txt가 실제 파일 경로인지 확인
    - 문자열을 기반으로 하므로, 확장자가 존재하지 않는 파일의 경우 잘못된 결과가 출력될 수 있음

    Args:
        txt (str): file로 의심 가는 txt

    Returns:
        bool: file 여부
    """
    _, ext = os.path.splitext(txt)
    return ext != ""


def file_download(
    url: str, local_path: str, kb: int = 8, wait: int = 10, remove_old: bool = False
) -> bool:
    """
    url에 있는 file을 다운로드 한다.

    Args:
        url (str): file의 url
        local_path (str): url 파일을 저장한 파일의 경로
        kb (int, optional): 스트리밍으로 데이터를 받을 때 크기. Defaults to 8.
        wait (int, optional): 응답이 없는 경우 최대 대기 시간. Defaults to 10.
        remove_old (bool, optional): local_path가 이미 존재하는 경우, 기존 local_path 제거 여부. Defaults to False.

    Raises:
        ConnectionError: 타임아웃
        ConnectionError: 연결 오류
        ConnectionError: 요청 오류
        ConnectionError: 파일 저장 오류

    Returns:
        bool: 다운로드 여부
    """
    # local_path가 이미 존재하는 경우, 행동
    if os.path.exists(local_path):
        if not remove_old:
            logging.warning(f"'{local_path}'가 이미 존재합니다. 덮어쓰기를 원한다면 overwrite=True로 설정하십시오.")
            return False
        else:
            # local_path 제거
            if not remove_target_path(target_path=local_path):
                return False
        
    # url로부터 데이터를 다운로드 받는다.
    try:
        response = requests.get(url, stream=True, timeout=wait)
        response.raise_for_status()  # 400, 500 오류인지 확인

        with open(local_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * kb):
                if chunk:  # 빈 청크가 아닌 경우에만 쓴다.
                    f.write(chunk)
        return True
                    
    # 통신 오류 
    except requests.exceptions.Timeout:
        raise ConnectionError(f"[타임아웃] {url}에서 응답이 없습니다. {wait}초 내에 다시 시도하세요.")
    except requests.exceptions.ConnectionError:
        raise ConnectionError(f"[연결 오류] {url}에 연결할 수 없습니다.")
    except requests.exceptions.RequestException as e:
        raise ConnectionError(f"[요청 오류] {url}을(를) 다운로드할 수 없습니다. {e}")
    except OSError as e:
        raise ConnectionError(f"[파일 저장 오류] {local_path}에 파일을 저장할 수 없습니다. {e}")
    
    
def remove_target_path(target_path: str) -> bool:
    """
    파일 또는 디렉터리를 제거합니다.

    Args:
        target_path (str): 제거하고자 하는 파일 또는 디렉터리의 경로

    Returns:
        bool: 제거 성공 여부
    """
    if os.path.exists(target_path):
        try:
            if os.path.isfile(target_path):
                os.remove(target_path)
            # 디렉터리 내부가 비어있는지 확인
            else:
                if not os.listdir(target_path):
                    os.rmdir(target_path)
                else:
                    shutil.rmtree(target_path)
            return True
        except (PermissionError, OSError) as e:
            logging.warning(f"'{target_path}' 제거 실패: {e}")
            return False
    else:
        return False

    

class GetAbsolutePath:

    def __init__(self, extensions: Optional[Union[List[str], str]] = None):
        """
        콜러블 메서드(callable method)를 주 방법으로 사용되는 class로, 콜 메서드에서 입력받는 parents_path의
        하위 파일들의 부모 경로들을 가지고 온다.
        >>> instance variable인 extensions가 None이 아니라면, extensions에 해당하는 파일들만 출력한다.

        Args:
            extensions (List[str], optional): None이 아닌 경우, extensions list 내 확장자에 대한 파일들을
                출력한다. Defaults to None.
        """
        self.extensions = self._extensions_handler(extensions=extensions)

    def __call__(self, parents_path: str) -> np.ndarray:
        """parents_path의 하위 파일들의 절대 경로들을 배열로 가지고 온다.

        Args:
            parents_path (str): 부모 경로

        Returns:
            np.ndarray: 부모 경로 내부 파일들의 절대 경로에 대한 배열
        """
        # self.parents_path에 있는 모든 파일들의 절대 경로를 list로 가지고 온다.
        path_list = self.get_all_path(parents_path)
        # self.extension에 따라 별도의 처리
        if self.extensions is not None:
            return self._find_target_extension(path_list)
        else:
            return np.array(path_list)

    def get_all_path(self, parents_path: str) -> List[str]:
        """
        parents_path 하위 파일들의 절대 경로들을 list로 가지고 온다.

        Args:
            parents_path (str): 대상 부모 디렉터리의 경로

        Returns:
            List[str]: 하위 파일들의 절대 경로들이 들어가 있는 list
        """
        # from .utils import list_flatten/

        stack = []
        # os.walk를 통해 디렉터리 트리를 순차적으로 탐색하면서, 파일의 절대 경로를 list에 추가
        for root, _, files in os.walk(top=parents_path):
            sub_path_list = list_flatten(
                [os.path.join(os.path.abspath(root), file) for file in files]
            )
            stack.extend(sub_path_list)
        return stack

    def _extensions_handler(
        self, extensions: Optional[Union[List[str], str]]
    ) -> Optional[List[str]]:
        """
        입력된 extensions이 None이 아니라면 item을 소문자로 바꾼다.

        Args:
            extensions (Optional[Union[List[str], str]]): 확장자 string 또는 확장자 string들이 들어가 있는 list
                - None인 경우, None 출력

        Returns:
            Optional[List[str]]: 소문자로 변환된 확장자 list 또는 None
        """
        # None인 경우
        if extensions is None:
            return None
        # string 또는 list인 경우
        return [
            ext.lower()
            for ext in (extensions if isinstance(extensions, list) else [extensions])
        ]

    def _find_target_extension(self, path_list: List[str]) -> np.ndarray:
        """
        path_list에 대하여, self.extensions에 해당하는 파일의 절대 경로들만 가지고 온다.

        Args:
            path_list (List[str]): 조회의 대상이 될 절대 경로들이 들어가 있는 list

        Returns:
            np.ndarray: 대상 확장자들에 속하는 파일 절대 경로들에 대한 배열
        """
        # 확장자를 소문자로 만들어 별도의 컬럼으로 생성.
        path_df = self._make_path_df(path_list)
        # self.extensions에 해당하는 확장자만 선택.
        self._find_extensions_index(path_df)
        # mask에 해당하는 path만 출력
        return path_df[path_df["mask"]]["path"].values

    def _make_path_df(self, path_list: List[str]) -> pd.DataFrame:
        """
        path_list를 pd.DataFrame을 이용하여 전처리
        >>> pandas의 str 메서드는 정규식을 이용한 데이터 탐색에 특화되어 있으므로 pandas 사용

        Args:
            path_list (List[str]): parents_path 하위 파일들의 절대 경로

        Returns:
            pd.core.frame.DataFrame: path, extension 2 개의 컬럼으로 구성된 DataFrame, extension의 경우
            소문자로 변환되어, 대소문자에 의해 대상 파일을 놓치는 경우를 없게 함.
        """
        path_df = pd.DataFrame({"path": path_list})
        path_df["extension"] = path_df["path"].str.rpartition(".")[2].str.lower()
        return path_df

    def _find_extensions_index(self, path_df: pd.core.frame.DataFrame):
        """
        self.extensions에 해당하는 path의 index를 찾는다.

        Args:
            path_df (pd.core.frame.DataFrame): self._make_path_df()로 인해 생성된 path_df
        """
        mask = np.zeros(shape=path_df["extension"].values.shape)
        for extension in self.extensions:
            mask += (path_df["extension"] == extension).values
        path_df["mask"] = mask.astype(bool)  # mask 추가
