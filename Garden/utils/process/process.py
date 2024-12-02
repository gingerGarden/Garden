import os
import sys


def get_verbose(verbose: bool) -> bool:
    """
    verbose가 True이면서, jupyter 환경일 때, True 출력

    Args:
        verbose (bool): process 절차에서 print할 것인지

    Returns:
        bool: process 절차에서 print할 것인지
    """
    # jupyter 환경인지 여부
    jupyter_mask = "ipykernel" in sys.modules
    # verbose = True 이면서, jupyter 환경일 때 True 출력
    return True if verbose and jupyter_mask else False


def get_process_id(verbose: bool = False) -> int:
    """
    해당 process의 id를 출력한다.

    Args:
        verbose (int): verbose = True 인 경우, 현 process id 출력

    Returns:
        int: process id
    """
    result = os.getpid()
    if verbose:
        print("해당 process의 id: ", result)
    return result
