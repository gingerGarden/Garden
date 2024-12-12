from typing import Optional, Dict, Union, List

import os
import zipfile
from ..path import load_pickle, new_dir_maker, file_download, unzip


class Get:
    data_dict = {
        "student-mat(tabular)":{
            "info":{"type":"file"},
            "urls":{"url1":"https://github.com/gingerGarden/Garden/raw/refs/heads/main/sample_data/tabular/student-mat.pickle"}
        },
        "student-por(tabular)":{
            "info":{"type":"file"},
            "urls":{"url1":"https://github.com/gingerGarden/Garden/raw/refs/heads/main/sample_data/tabular/student-por.pickle"}
        },
        "coco(images)":{
            "info":{"type":"directory", "dir_name":"coco"},
            "urls":{
                "url1":"https://github.com/gingerGarden/Garden/raw/refs/heads/main/sample_data/image/coco/sample_images.zip",
                "url2":"https://github.com/gingerGarden/Garden/raw/refs/heads/main/sample_data/image/coco/sample_annotaions.json"
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
        return list(cls.data_dict.keys())
    
    @classmethod
    def download_sample_data(cls, key: str, remove_old: bool):
        """
        key에 해당하는 샘플 데이터를 다운로드한다.
            - 데이터가 다운로드될 부모 경로가 존재하지 않는 경우, 이를 제거한다.

        Args:
            key (str): data_dict의 key
            remove_old (bool): 기존 데이터 제거 여부
        """
        # 기초 디렉터리 생성
        if not os.path.exists(cls.parents_path):
            cls._make_base_directories()
        
        # data를 다운로드 한다.
        _SampleDataDownload(
            parents_path=cls.parents_path, 
            key_dict=cls.data_dict[key],
            remove_old=remove_old
        ).run()
    
    @classmethod
    def _make_base_directories(cls):
        """
        데이터 다운로드를 위한 가장 기본적인 디렉터리들을 생성한다.
        """
        new_dir_maker(f"{cls.home_directory_path}/{cls.library_dir}", makes_new=False)
        new_dir_maker(cls.parents_path, makes_new=False)
    
    
    
class _SampleDataDownload:
    def __init__(self, parents_path: str, key_dict: Dict[str, str], remove_old: bool):
        self.parents_path = parents_path
        self.key_dict = key_dict
        self.remove_old = remove_old
        self.data_dir_path = None
        
    def run(self):
        
        if self.key_dict['info']['type'] == 'directory':
            # 데이터가 다운로드될 하위 디렉터리 생성
            self._make_sub_directory()
            # 하위 데이터 파일 다운로드
            self._directory_file_downloads()
            # zip 파일 압축 해제
            self._extract_zipfile()
        else:
            self._file_download(
                url=self.key_dict['urls']['url1'],
                dir_path=self.parents_path
            )
            
    def _make_sub_directory(self):
        """
        하위 데이터 파일들이 저장될 디렉터리 생성
        """
        # 디렉터리 경로 저장
        dir_name = self.key_dict["info"]["dir_name"]
        self.data_dir_path = f"{self.parents_path}/{dir_name}"
        # directory 생성
        makes_new = True if os.path.exists(self.data_dir_path) and self.remove_old else False
        new_dir_maker(dir_path=self.data_dir_path, makes_new=makes_new)
            
    def _directory_file_downloads(self):
        """
        urls에 있는 모든 url에 대하여 파일 다운로드
        """
        for _, url in self.key_dict['urls'].items():
            self._file_download(url, self.data_dir_path)
            
    def _extract_zipfile(self):
        """
        하위 디렉터리 내 zip file 존재 시, 압축 해제 및 zip file 제거
        """
        for file_name in os.listdir(self.data_dir_path):
            # file의 경로
            file_path = os.path.join(self.data_dir_path, file_name)
            # zip 파일이 존재하는 경우, 압축을 해제하고, zip 파일을 제거한다.
            if zipfile.is_zipfile(file_path):
                unzip(file_path, remove_zipfile=True)

    def _file_download(self, url: str, dir_path: str) -> bool:
        """
        file을 url로부터 다운로드 한다.
            - file 이름은 url의 / 가장 마지막에 있는 file명으로 한다.

        Args:
            url (str): 다운로드할 파일의 url
            dir_path (str): 파일이 다운로드될 디렉터리 경로

        Returns:
            bool: download 여부
        """
        # url을 기반으로 파일 경로를 생성한다.
        file_name = url.split("/")[-1]
        file_path = f"{dir_path}/{file_name}"
        # file 다운로드 및 다운로드 완료 여부 출력
        return file_download(url, local_path=file_path, remove_old=self.remove_old)
        
        
        

# class _Old:
#     @classmethod
#     def load(
#             cls, data_name: Optional[str] = None
#         ) -> Dict[str, Dict[str, Union[str, pd.DataFrame]]]:
#         """
#         data_name에 해당하는 모든 데이터(메타 데이터, 데이터)를 가져온다.

#         Args:
#             data_name (Optional[str], optional): 데이터 이름. Defaults to None.
#                 - None이 입력되는 경우, 기본 값인 "student(math)"를 출력한다.

#         Returns:
#             Dict[str, Dict[str, Union[str, pd.DataFrame]]]: 데이터
#         """
#         if data_name is None:
#             warnings.warn("key에 아무 값이 입력되지 않았습니다. default인 student(math) 데이터를 가져옵니다.")
#             data_name = "student(math)"
#         return load_pickle(pickle_path=cls.path_dict[data_name])
    
#     @classmethod
#     def data(cls, data_name: str) -> pd.DataFrame:
#         """
#         data_name에 해당하는 데이터만 가져온다.

#         Args:
#             data_name (str): 데이터 이름

#         Returns:
#             pd.DataFrame: 데이터
#         """
#         data = load_pickle(pickle_path=cls.path_dict[data_name])
#         return data[cls.data_key]
    
#     @classmethod
#     def metadata(cls, data_name: str, verbose: bool = False) -> Dict[str, str]:
#         """
#         data_name에 해당하는 metadata를 가져온다.

#         Args:
#             data_name (str): 데이터 이름
#             verbose (bool, optional): pprint로 보기 좋게 출력할지 여부. Defaults to False.

#         Returns:
#             Dict[str, str]: metadata의 dictionary
#         """
#         data = load_pickle(pickle_path=cls.path_dict[data_name])
#         return cls._get_target_value(data=data, key=cls.metadata_key, verbose=verbose)
    
#     @classmethod
#     def attribute(cls, data_name: str, verbose: bool = False) -> Dict[str, str]:
#         """
#         data_name에 해당하는 attribute를 가져온다.

#         Args:
#             data_name (str): 데이터 이름
#             verbose (bool, optional): pprint로 보기 좋게 출력할지 여부. Defaults to False.

#         Returns:
#             Dict[str, str]: attribute의 dictionary
#         """
#         data = load_pickle(pickle_path=cls.path_dict[data_name])
#         return cls._get_target_value(data=data, key=cls.attribute_key, verbose=verbose)

#     @classmethod
#     def data_name_list(cls) -> List[str]:
#         """
#         sample data의 data name list를 출력한다.

#         Returns:
#             List[str]: data name의 List
#         """
#         return list(cls.path_dict.keys())
    
#     @classmethod
#     def _get_target_value(cls, data: Dict[str, str], key: str, verbose: bool) -> Dict[str, str]:
#         """
#         cls.metadata 와 cls.attribute의 하위 메서드
#             - data의 dictionary에서 key에 해당하는 결과를 가져오고, pprint로 출력할지 여부를 정한다.

#         Args:
#             data (Dict[str, str]): metadata, attribute, data 3개의 key로 이루어진 데이터(dict)
#             key (str): data(dict)에서 출력하고자 하는 key
#             verbose (bool): pprint로 보기 좋게 출력할지 여부

#         Returns:
#             Dict[str, str]: key에 해당하는 데이터
#         """
#         result = data[key]
#         if verbose:
#             pprint(result)
#         return result