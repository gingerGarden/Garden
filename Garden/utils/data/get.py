from typing import Optional, Dict, Union, List, Tuple, Any

import os
import zipfile
import pandas as pd
from ..path import load_pickle, read_json, new_dir_maker, file_download, unzip


class _TabularAction:
    @classmethod
    def run(cls, data_path: str) -> Dict[str, Union[str, pd.DataFrame]]:
        data = load_pickle(data_path)
        info = data['metadata']
        data = data['data']
        return info, data
    
    
    
class _COCOAction:
    source_fils = "sample_annotaions.json"
    image_dir = "sample_images"
    
    @classmethod
    def run(cls, data_path: str):
        # source를 가져온다.
        source = read_json(os.path.join(data_path, cls.source_fils))
        
        # info
        source['info']['licenses'] = source['licenses']
        info = source['info']
        
        # data
        data = cls._make_data(
            parents_path=os.path.join(data_path, cls.image_dir),
            source=source
        )
        return info, data
    
    @classmethod
    def _make_data(cls, parents_path: str, source: Dict[str, Any]):
        for record_dict in source['images']:
            record_dict['file_name'] = os.path.join(parents_path, record_dict['file_name'])
            
        return {
            "images":source['images'], 
            "annotation":source['annotation'], 
            "categories":source['categories'], 
        }
        
        

class Get:
    data_dict = {
        "student-mat(tabular)":{
            "info":{"type":"file", "method":_TabularAction},
            "urls":{"url1":"https://github.com/gingerGarden/Garden/raw/refs/heads/main/sample_data/tabular/student-mat.pickle"},
        },
        "student-por(tabular)":{
            "info":{"type":"file", "method":_TabularAction},
            "urls":{"url1":"https://github.com/gingerGarden/Garden/raw/refs/heads/main/sample_data/tabular/student-por.pickle"}
        },
        "coco(images)":{
            "info":{"type":"directory", "dir_name":"coco", "method":_COCOAction},
            "urls":{
                "url1":"https://github.com/gingerGarden/Garden/raw/refs/heads/main/sample_data/image/coco/sample_images.zip",
                "url2":"https://github.com/gingerGarden/Garden/raw/refs/heads/main/sample_data/image/coco/sample_annotaions.json"
            }
        }
    }

    # 경로 정보
    home_directory_path = os.path.expanduser("~")
    library_dir = ".garden"
    data_dir = "data"
    parents_path = f"{home_directory_path}/{library_dir}/{data_dir}"
    
    @classmethod
    def sample_data(cls, key: str, remove_old: bool) -> Tuple[Dict, Dict]:
        # key가 지원 데이터 안에 포함되는지 확인
        if key not in list(cls.data_dict.keys()):
            raise ValueError(f"입력한 {key}는 샘플 데이터로 제공하지 않습니다. Get.show_sample_list()로 제공 데이터를 확인하십시오.")
        
        # 데이터 다운로드
        data_path = cls._get_data_path(key=key)
        if remove_old or not os.path.exists(data_path):
            cls._download_sample_data(key=key, remove_old=remove_old)
            
        # 데이터를 가져온다.
        info, data = cls.data_dict[key]['info']['method'].run(data_path)
        return info, data
                
    @classmethod
    def show_sample_list(cls) -> List[str]:
        """
        Garden에서 지원하는 샘플 데이터의 key list를 반환한다.

        Returns:
            List[str]: Garden에서 지원하는 데이터의 key list
        """
        return list(cls.data_dict.keys())
    
    @classmethod
    def _get_data_path(cls, key: str) -> str:
        """
        key의 데이터에 대해, 생성되는 대표 데이터 경로 출력

        Args:
            key (str): sample data의 key

        Returns:
            str: 대표 데이터 경로
        """
        if cls.data_dict[key]['info']['type'] == "directory":
            return os.path.join(cls.parents_path, cls.data_dict[key]['info']['dir_name'])
        else:
            file_name = cls.data_dict[key]['urls']['url1'].split("/")[-1]
            return os.path.join(cls.parents_path, file_name)
    
    @classmethod
    def _download_sample_data(cls, key: str, remove_old: bool):
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
        new_dir_maker(os.path.join(cls.home_directory_path, cls.library_dir), makes_new=False)
        new_dir_maker(cls.parents_path, makes_new=False)

    
    
class _SampleDataDownload:
    def __init__(self, parents_path: str, key_dict: Dict[str, str], remove_old: bool):
        """
        Sample data를 다운로드 받는다.
            - Sample data는 해당 코드의 github 레포지토리의 sample_data 디렉터리 하위에 존재한다.

        Args:
            parents_path (str): 데이터가 다운로드 될 부모 디렉터리 경로
            key_dict (Dict[str, str]): 데이터 다운로드의 key가 되는 값들이 존재하는 딕셔너리
            remove_old (bool): 기존 데이터가 존재한다면, 이를 제거하고 새로 받을지 여부
        """
        self.parents_path = parents_path
        self.key_dict = key_dict
        self.remove_old = remove_old
        self.data_dir_path = None
        
    def run(self):
        """
        Sample data를 다운로드 하는 프로세스 전체를 실행한다.
            - directory type과 그렇지 않은 타입은 별도로 진행된다.
            - directory type은 디렉터리를 생성 후, 하위 데이터 파일들을 다운로드하고, zip 파일이 있는 경우 압축 해제한다.
        """
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
        file_path = os.path.join(dir_path, file_name)
        # file 다운로드 및 다운로드 완료 여부 출력
        return file_download(url, local_path=file_path, remove_old=self.remove_old)