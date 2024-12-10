from typing import Optional, Dict, Union, List

import os
import warnings
import pandas as pd
from pprint import pprint

from ..path import load_pickle, new_dir_maker, file_download


class Get:
    datas = {
        "student-mat(tabular)":{
            "type":"tabular",
            "url":"https://github.com/gingerGarden/Garden/raw/refs/heads/main/sample_data/tabular/student-mat.pickle"
            },
        "student-por(tabular)":{
            "type":"tabular",
            "url":"https://github.com/gingerGarden/Garden/raw/refs/heads/main/sample_data/tabular/student-por.pickle"
        },
        "coco(image)":{
            "type":"zip",
            "url":{
                "images":"https://github.com/gingerGarden/Garden/raw/refs/heads/main/sample_data/image/coco/sample_images.zip",
                "annotations":"https://github.com/gingerGarden/Garden/raw/refs/heads/main/sample_data/image/coco/sample_annotaions.json"
            }
        }
    }
    tabular_key = {
        "metadata":"metadata",
        "attribute":"attribute",
        "data":"data"
    }
    
    # 경로 정보
    home_directory_path = os.path.expanduser("~")    
    library_dir = ".garden"
    data_dir = "data"
    parents_path = f"{home_directory_path}/{library_dir}/{data_dir}"
    
    @classmethod
    def show_data_list(cls) -> List[str]:
        """
        Garden에서 지원하는 샘플 데이터의 key list를 반환한다.

        Returns:
            List[str]: Garden에서 지원하는 데이터의 key list
        """
        return list(cls.datas.keys())
    
    @classmethod
    def _download(cls, key: str, remove_old: bool = False) -> bool:
        # 기초 디렉터리 생성(존재하지 않는 경우)
        if not os.path.exists(cls.parents_path):
            cls._make_base_directories()
        # sample data의 type
        sample_type = cls.datas[key]['type']
        urls = cls.datas[key]['url']
        # zip 파일이 존재하는 경우, zip을 푸는 코드 추가 개발할 것
            
            
    @classmethod
    def _make_base_directories(cls):
        """
        데이터 다운로드를 위한 가장 기본적인 디렉터리들을 생성한다.
        """
        new_dir_maker(f"{cls.home_directory_path}/{cls.library_dir}", makes_new=False)
        new_dir_maker(cls.parents_path, makes_new=False)
        
    @classmethod
    def _download_process(cls, url: str, remove_old: bool) -> bool:
        """
        url을 기반으로 데이터를 다운로드 받는다.

        Args:
            url (str): 대상 데이터의 url
            remove_old (bool): 기존 파일이 존재하는 경우, 기존 파일을 제거하고 다시 설치할지 여부

        Returns:
            bool: 다운로드 성공 여부
        """
        # url을 기반으로 파일 경로를 생성한다.
        file_name = url.split("/")[-1]
        file_path = f"{cls.parents_path}/{file_name}"
        # file 다운로드 및 다운로드 완료 여부 출력
        return file_download(url, local_path=file_path, remove_old=remove_old)
        
        
        
        
    # 재개발
    ###########################################################################
    @classmethod
    def load(
            cls, data_name: Optional[str] = None
        ) -> Dict[str, Dict[str, Union[str, pd.DataFrame]]]:
        """
        data_name에 해당하는 모든 데이터(메타 데이터, 데이터)를 가져온다.

        Args:
            data_name (Optional[str], optional): 데이터 이름. Defaults to None.
                - None이 입력되는 경우, 기본 값인 "student(math)"를 출력한다.

        Returns:
            Dict[str, Dict[str, Union[str, pd.DataFrame]]]: 데이터
        """
        if data_name is None:
            warnings.warn("key에 아무 값이 입력되지 않았습니다. default인 student(math) 데이터를 가져옵니다.")
            data_name = "student(math)"
        return load_pickle(pickle_path=cls.path_dict[data_name])
    
    @classmethod
    def data(cls, data_name: str) -> pd.DataFrame:
        """
        data_name에 해당하는 데이터만 가져온다.

        Args:
            data_name (str): 데이터 이름

        Returns:
            pd.DataFrame: 데이터
        """
        data = load_pickle(pickle_path=cls.path_dict[data_name])
        return data[cls.data_key]
    
    @classmethod
    def metadata(cls, data_name: str, verbose: bool = False) -> Dict[str, str]:
        """
        data_name에 해당하는 metadata를 가져온다.

        Args:
            data_name (str): 데이터 이름
            verbose (bool, optional): pprint로 보기 좋게 출력할지 여부. Defaults to False.

        Returns:
            Dict[str, str]: metadata의 dictionary
        """
        data = load_pickle(pickle_path=cls.path_dict[data_name])
        return cls._get_target_value(data=data, key=cls.metadata_key, verbose=verbose)
    
    @classmethod
    def attribute(cls, data_name: str, verbose: bool = False) -> Dict[str, str]:
        """
        data_name에 해당하는 attribute를 가져온다.

        Args:
            data_name (str): 데이터 이름
            verbose (bool, optional): pprint로 보기 좋게 출력할지 여부. Defaults to False.

        Returns:
            Dict[str, str]: attribute의 dictionary
        """
        data = load_pickle(pickle_path=cls.path_dict[data_name])
        return cls._get_target_value(data=data, key=cls.attribute_key, verbose=verbose)

    @classmethod
    def data_name_list(cls) -> List[str]:
        """
        sample data의 data name list를 출력한다.

        Returns:
            List[str]: data name의 List
        """
        return list(cls.path_dict.keys())
    
    @classmethod
    def _get_target_value(cls, data: Dict[str, str], key: str, verbose: bool) -> Dict[str, str]:
        """
        cls.metadata 와 cls.attribute의 하위 메서드
            - data의 dictionary에서 key에 해당하는 결과를 가져오고, pprint로 출력할지 여부를 정한다.

        Args:
            data (Dict[str, str]): metadata, attribute, data 3개의 key로 이루어진 데이터(dict)
            key (str): data(dict)에서 출력하고자 하는 key
            verbose (bool): pprint로 보기 좋게 출력할지 여부

        Returns:
            Dict[str, str]: key에 해당하는 데이터
        """
        result = data[key]
        if verbose:
            pprint(result)
        return result