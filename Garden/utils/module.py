import hashlib
import importlib
import logging
import os
import re
import sys
import time
import types
from typing import List, Optional

import github
from github import Github

from .path import read_txt_file


def get_module_from_py(py_path: str, unique_module: bool = False) -> types.ModuleType:
    """
    py 파일로부터 module을 가지고 온다.
    module이 sys 위에 올라가 있는지 여부를 따지며, 존재하는 경우 기존 module을 가지고 온다.
        - module의 이름으로 이는 구분 되므로, unique_module=True인 경우, 모든 module명은 unique하게 생성되므로
          우연하게 일치 않는 이상, 동일 module을 생성하지 않는다.

    Args:
        py_path (str): py 파일의 경로
        unique_module (bool, optional): module 이름이 중복되지 않게 해시 사용 여부. Defaults to False.
            - True인 경우, hash 값으로 임의의 값을 생성하여 기존 module 이름과 중복되지 않게 함
            - False인 경우, 기존 모듈을 재사용함

    Raises:
        ValueError: py 파일 경로에 문제가 있는 경우

    Returns:
        types.ModuleType: py 파일의 module
    """
    # py_path 무결성 확인
    if (
        not os.path.exists(py_path)
        or not py_path.endswith(".py")
        or not os.path.isfile(py_path)
    ):
        raise ValueError(f"Invalid Python file path: {py_path}")

    # module 이름 생성
    if unique_module:
        # file 내용을 기반으로 해시를 생성 - 파일 경로로 생성하는 경우, 중복될 수도 있음.
        with open(py_path, "rb") as f:
            module_name = hashlib.md5(f.read()).hexdigest()
    else:
        from .path import get_file_name

        module_name = get_file_name(file_path=py_path)

    if module_name in sys.modules:
        return sys.modules[module_name]
    else:
        return _get_new_module(py_path, module_name)


def _get_new_module(py_path: str, module_name: str) -> types.ModuleType:
    """
    py 파일로부터 module_name 으로 모듈을 가져온다.

    Args:
        py_path (str): py 파일의 경로
        module_name (str): module 이름

    Raises:
        ImportError: module import가 되지 않는 경우 에러 발생

    Returns:
        types.ModuleType: py 파일의 모듈
    """
    # 1. spec 생성: 파일 경로와 모듈 이름을 이용해 모듈의 사양 생성
    spec = importlib.util.spec_from_file_location(name=module_name, location=py_path)
    # 2. spec을 기반으로 모듈 객체 생성
    module = importlib.util.module_from_spec(spec)
    # 3. 모듈을 sys.modules에 등록
    sys.modules[module_name] = module
    # 4. 모듈을 실행하여 사용할 수 있도록 로드
    try:
        spec.loader.exec_module(module=module)
    except Exception as e:
        raise ImportError(
            f"Failed to load or execute the module {module_name} from {py_path}: {e}"
        ) from e
    return module


def explore_modules_recursiverly(
    py_path: str,
    module_paths: Optional[List[str]] = None,
    visited: Optional[set] = None,
) -> List[str]:
    """
    py_path에 해당하는 py 파일이 참조하는 module 들의 py 파일의 경로들을 재귀적으로 탐색하여 가지고 온다.
    >>> module의 의존성을 고려할 때, 해당 module 들은 역순으로 동적으로 로드되어야 한다.
    >>> 탐색한 module들은 역순으로 module에 로드하여 의존성 문제가 발생하지 않도록 하는 것을 추천한다.

    Args:
        py_path (str): module이 포함된 python 파일의 경로
        module_paths (Optional[List[str]], optional): module이 들어있는 py 파일들의 경로. Defaults to None.
        visited (Optional[set], optional): 이미 방문한 module은 더 이상 방문하지 않도록 확인하는 용도. Defaults to None.

    Returns:
        List[str]: 의존성 module이 포함된 py 파일들의 list
    """
    # 탐색의 기본 프레임
    if module_paths is None:
        module_paths = []
    if visited is None:
        visited = set()
    # 현재 py_path에서 import 된 모듈 리스트를 가져온다.
    module_list = get_module_list_in_py_file(py_path=py_path, public_package=True)

    for module in module_list:
        if module in visited:  # 이미 방문한 모듈은 중복으로 탐색하지 않음
            continue

        visited.add(module)

        # module의 py 파일 경로
        module_path = get_module_path(py_path=py_path, module=module)

        # stack list에 추가
        module_paths.append(module_path)

        # 재귀적으로 하위 모듈 탐색
        explore_modules_recursiverly(
            py_path=module_path, module_paths=module_paths, visited=visited
        )
    return module_paths


def get_module_list_in_py_file(py_path: str, public_package: bool = True) -> List[str]:
    """
    py file에 import된 module들을 가지고 온다.
    >>> from ... import ... 와 import ... 를 대상으로 re로 가지고 온다.
    >>> public_package = True인 경우, .package 형식의 내부 package만 반환한다.

    Args:
        py_path (str): module 들을 가지고 오고자 하는 py file의 절대 경로
        public_package (bool): 배포되지 않은 내부 package module만 대상으로 출력할지 여부 Defaults to True.

    Returns:
        List[str]: import할 module의 list
    """
    # py file에 있는 모든 txt를 가지고 온다.
    content = read_txt_file(py_path)
    # txt에서 from ... import ..., import ... 에서 import할 module들을 가지고온다.
    imports = re.findall(
        r"^\s*(?:from\s+(\S+)\s+import\s+\S+|import\s+(\S+))", content, re.MULTILINE
    )
    # import할 모든 module의 txt를 가지고 온다.
    modules = [i[0] or i[1] for i in imports]
    # 순서를 유지한 상태로 중복 제거
    modules = list(dict.fromkeys(modules))
    # pip으로 서비스되는 module이 아니라, 개인 module에 대해서만 list를 가져올지 여부.
    if public_package:
        return [
            module for module in modules if re.match(pattern=r"^(\.)+", string=module)
        ]
    else:
        return list(modules)


def get_module_path(py_path: str, module: str) -> str:
    """
    py_path py 파일 경로를 기반으로 module py 파일의 경로를 가지고 온다.

    Args:
        py_path (str): module이 포함된 py 파일의 경로
        module (str): py 파일의 대상 module

    Returns:
        str: py 파일에 있는 module의 py 파일 경로
    """
    # module의 .의 갯수를 기반으로 first_py_path에서 module 파일의 경로를 정의한다.
    back_size = len(re.match(pattern=r"^(\.)+", string=module).group(1))
    directory_path = "/".join(py_path.rsplit("/", back_size)[:-1])
    # module의 이름을 기반으로 python 파일 이름 정의
    module_file = re.sub(r"^(\.)+", repl="", string=module) + ".py"
    return os.path.join(directory_path, module_file)


class GithubHelper:

    http_pattern = r"^https://github.com/"  # github 주소 - url 내 불필요한 앞부분
    tree_pattern = r"/tree/[^/]+/"  # tree 구조 - url 내 대상 디렉터리 구조
    wait = 60  # contents 다운로드 시 최대 대기 시간

    @classmethod
    def download_repo_directory(
        cls, url: str, parents_path: str, token: Optional[str] = None
    ):
        """
        Github 레포지터리의 특정 디렉터리에 있는 모든 파일과 폴더를 다운로드 한다.

        Ex) GithubHelper.download_files_from_specific_directory(
                url = "https://github.com/user/repo/tree/main/directory1/scripts",
                token = os.getenv('GITHUB_TOKEN'),      # 환경변수로 API 토큰을 받는 경우
                parents_path = "/home/usr/download"
            )

        Args:
            url (str): Github 레포지터리 내 디렉터리의 url
            parents_path (str): 디렉터리를 다운로드 할 부모 경로
            token (Optional[str], optional): Github API 토큰. Defaults to None.
                - None인 경우, 공개 레포지터리만 다운로드 가능하며, 다운로드 양의 제한이 있음.
        """
        repo = cls.get_repo_from_url(url=url)
        tree = cls.get_directory_tree(url=url)

        # Github instance 생성
        g = Github(token)
        repo_ins = g.get_repo(repo)

        # tree 내 파일 다운로드
        cls.download_contents(repo_ins, tree, parents_path)

    @classmethod
    def download_contents(
        cls,
        repo_ins: github.Repository.Repository,
        tree: str,
        parents_path: str,
        wait: Optional[int] = None,
        retry: int = 0,
        max_retry: int = 5,
        encoding: str = "utf-8",
    ):
        """
        repo_ins 를 사용하여, tree(Github contents)에 있는 파일들을 parents_path에 그 구조대로 다운로드 한다.
            - 만약, 내부에 디렉터리가 존재하는 경우, 자기 자신을 참조하여 내부 파일들을 다운로드한다.

        Args:
            repo_ins (Repository): github Repository 객체
            tree (str): Repository 내 디렉터리 구조
            parents_path (str): 디렉터리를 다운로드 할 부모 경로
            wait (int, optional): repo에서 content를 가져오지 못하는 경우 대기 시간. Defaults to None.
            retry (int, optional): content를 가져오기 위해 반복하는 현재 횟수. Defaults to 0.
                - content를 가져오는데 성공하는 경우 초기화
            max_retry (int, optional): content를 가져오기 위해 시도할 최대 횟수. Defaults to 5.
            encoding (str, optional): file을 쓸 때, encoding. Defaults to 'utf-8'.

        Raises:
            Exception: 파일이 다운로드 되지 않는 경우
        """
        wait = cls.wait if wait is None else wait

        # repo에 contents를 가져오지 못하는 경우, 재귀적으로 재접속
        try:
            contents = repo_ins.get_contents(tree)
            retry = 0
        except github.RateLimitExceededException:
            if retry < max_retry:
                logging.warning(
                    f"Rate limit exceeded. Waiting... Retry {retry+1}/{max_retry}"
                )
                time.sleep(wait)
                return cls.download_contents(
                    repo_ins,
                    tree,
                    parents_path,
                    wait=wait,
                    retry=retry + 1,
                    max_retry=max_retry,
                )
            else:
                logging.error("Maximum retries exceeded. Stopping process.")
                raise Exception("Rate limit retries exceeded. Process stopped.")

        # 재귀적으로 다운로드
        for content in contents:
            local_path = os.path.join(parents_path, content.path)

            if content.type == "file":
                cls.download_file(content, file_path=local_path, encoding=encoding)
            elif content.type == "dir":
                cls.download_contents(
                    repo_ins,
                    content.path,
                    parents_path,
                    wait=wait,
                    retry=0,
                    max_retry=max_retry,
                )

    @classmethod
    def download_file(
        cls,
        content: github.ContentFile.ContentFile,
        file_path: str,
        encoding: str = "utf-8",
    ):
        """
        Github에 있는 파일을 읽고 local에 쓴다.

        Args:
            content (github.ContentFile.ContentFile): Github content file
            file_path (str): 파일을 다운로드할 경로
            encoding (str, optional): 파일을 쓸 때 encoding. Defaults to 'utf-8'.
        """
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            file_content = content.decoded_content.decode()
            with open(file_path, "w", encoding=encoding) as f:
                f.write(file_content)
        except Exception as e:
            logging.warning(f"Error downloading {file_path}: {e}")

    @classmethod
    def get_repo_from_url(cls, url: str) -> str:
        """
        Github의 url에서 repo 문자열을 추출한다.

        Args:
            url (str): github url

        Returns:
            str: repo
        """
        repo = re.sub(pattern=cls.http_pattern, repl="", string=url)
        match = re.search(pattern=cls.tree_pattern, string=repo)
        if match:
            repo = repo[: match.start()]
        return repo

    @classmethod
    def get_directory_tree(cls, url: str) -> str:
        """
        Github의 url에서 디렉터리 구조 문자열을 추출한다.

        Args:
            url (str): github url

        Returns:
            str: / 로 연결된 디렉터리 구조
        """
        match = re.search(pattern=cls.tree_pattern, string=url)
        if match:
            return url[match.end():]
        else:
            raise ValueError(
                f"입력된 url에 github url의 tree 패턴인 {cls.tree_pattern}이 존재 하지 않습니다!"
            )
