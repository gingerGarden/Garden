from typing import Optional, Dict, Union, List

import os
import warnings
import pandas as pd
from pprint import pprint

from ..path import load_pickle



class Get:
    # package directory의 기본 경로
    package_dir = os.path.dirname(__file__)
    # data의 기본 경로
    path_dict = {
        "student(math)":f"{package_dir}/tabular/student-mat.pickle"
    }
    # metadata key
    metadata_key = "metadata"
    # attribute key
    attribute_key = "attribute"
    # data key
    data_key = "data"
    
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