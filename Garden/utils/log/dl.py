from typing import Dict, Optional

from ..path import new_dir_maker
from ..utils import format_with_rounding


class Log:
    def __init__(
        self,
        log_dir: str,
        process_id: int,
        model_type: str,
        save_log: bool = True,
        iter_save: bool = True,
        num_digits: int = 5,
        lr_num_digits: int = 6,
    ):
        """
        딥러닝 프로세스의 log

        Args:
            log_dir (str): log가 저장될 디렉터리의 경로
            process_id (int): 해당 프로세스의 id
            model_type (str): 딥러닝 모델의 종류 - 생성되는 log의 스타일 정의
            save_log (bool, optional): log를 저장할지 여부. Defaults to True.
            iter_save (bool, optional): iterator(batch) log 저장 여부. Defaults to True.
            num_digits (int, optional): log의 실수에 대하여 표현할 소수점 자릿수. Defaults to 5.
            lr_num_digits (int, optional): learning rate에 대하여 표현할 소수점 자릿수. Defaults to 6.
        """
        # fixed instance variable
        self.parents_path = f"{log_dir}/{process_id}"
        self.epoch = f"{self.parents_path}/epoch.json"  # epoch log file 경로(json)
        self.iterator = (
            f"{self.parents_path}/iterator.json"  # iterator log file 경로(json)
        )
        new_dir_maker(self.parents_path, makes_new=False)  # log 디렉터리 경로

        self.save_log = save_log
        self.iter_save = iter_save
        self.lr_num_digits = lr_num_digits

        # log txt의 유형 선택
        self.methods = {
            "classification": _Classification(num_digits=num_digits),
            "object_detection": _ObjectDetection(num_digits=num_digits),
            "segmentation": _Segmentation(num_digits=num_digits),
            "anomaly_detection": _AnomalyDetection(num_digits=num_digits),
        }
        self.method = None
        self._select_log_method(model_type=model_type)

        # 변화하는 instance variable
        self.k = None
        self.optimizer = None

    def epoch_log_txt(self, epoch_log_txt: Optional[str], bar_txt: str) -> str:
        """
        epoch에 대한 log_text
        >>> log_text는 기본적으로 ProgressBar class에서 출력되는 bar_txt임.
            epoch == 0일 때, epoch에 대한 log 지표가 없음
            epoch >= 1일 때, 현 epoch가 아닌 전 epoch에 대한 log 지표가 출력되어야 함

        Args:
            epoch_log_txt (Optional[str]): 해당 메서드를 통해 출력되어야 하는 log_txt
                - None인 경우, bar_txt를 그대로 출력
            bar_txt (str): ProgressBar class의 bar_txt

        Returns:
            str: epoch_log_txt
        """
        return (
            f"{bar_txt}\n\n{'===='*20}\n\n"
            if epoch_log_txt is None
            else f"{epoch_log_txt}\n{'===='*20}\n\n"
        )

    def log_txt(self, lr, **kwargs) -> str:
        """
        self를 선언하였을 때, model_type에 해당하는 self.method가 정의되고, self.method에 맞는 파라미터를 입력하여 log 문구 출력

        Returns:
            str: self.method에 맞는 log 문구
                ex) Classificaition의 예
                    [Metrics] [Train]: acc: 0.920, loss: 0.024 / [valid]: acc: 0.890, loss: 0.035
        """
        lr_log = f"[learning rate]: {lr:.{self.lr_num_digits}f}"
        performance_log = self.method.log_text(**kwargs)
        return f"{lr_log}\n{performance_log}"

    def log_dict(self, upper_i, i, lr, **kwrags) -> Dict[str, Dict[str, float]]:
        """
        self를 선언하였을 때, model_type에 해당하는 self.method가 정의되고, self.method에 맞는 파라미터를 입력하여 log dictionary 출력

        Returns:
            Dict[str, Dict[str, float]]: self.method에 맞는 log dictionary
                ex) Classificaition의 예
                    {'upper_i':upper_i, 'i':i, 'loss':{'train':loss, 'valid':valid_loss}, 'acc':{'train':acc, 'valid':valid_acc}}
                    >>> 'upper_i'는 상위 단계로, iterator에서는 epoch, epochs에서는 k에 해당
                    >>> 'i'는 현재 iterator의 iteration 순서를 가리킴(epoch 또는 iteration)
        """
        result = self.method.log_dict(**kwrags)
        result["meta"] = {"upper_i": upper_i, "i": i, "lr": lr}
        return result

    def _select_log_method(self, model_type: str):
        """
        log의 출력 방식 정의

        Args:
            model_type (str): model의 종류
            >>> 'classification', 'object_detection', 'segmentation', 'anomaly_detection'

        Raises:
            ValueError: _description_
        """
        if model_type not in self.methods:
            raise ValueError(
                f"입력한 model_type: {model_type}은 해당 Log에서 지원하지 않습니다. 지원하는 model_type은 {list(self.methods.keys())}입니다."
            )
        self.method = self.methods.get(model_type)


class _Classification:
    def __init__(self, num_digits: int):
        """
        Classification 모델의 log 스타일

        Args:
            num_digits (int): 실수의 소숫점 자릿수
        """
        self.num_digits = num_digits

    def log_text(
        self,
        train_acc: float,
        train_loss: float,
        valid_acc: Optional[float] = None,
        valid_loss: Optional[float] = None,
        test_acc: Optional[float] = None,
        test_loss: Optional[float] = None,
    ) -> str:
        """
        log의 text 생성 방식
        - train의 acc, loss는 반드시 있음.
        - valid 와 test는 없을 수 있으며, 없는 경우에 맞는 txt 출력

        Args:
            train_acc (float): train accuracy
            train_loss (float): train loss
            valid_acc (Optional[float], optional): validation accuracy. Defaults to None.
            valid_loss (Optional[float], optional): validation loss. Defaults to None.
            test_acc (Optional[float], optional): test accuracy. Defaults to None.
            test_loss (Optional[float], optional): test accuracy. Defaults to None.

        Returns:
            str: log_txt
        """
        train_acc = format_with_rounding(value=train_acc, num_digits=self.num_digits)
        train_loss = format_with_rounding(value=train_loss, num_digits=self.num_digits)
        valid_acc = format_with_rounding(value=valid_acc, num_digits=self.num_digits)
        valid_loss = format_with_rounding(value=valid_loss, num_digits=self.num_digits)
        test_acc = format_with_rounding(value=test_acc, num_digits=self.num_digits)
        test_loss = format_with_rounding(value=test_loss, num_digits=self.num_digits)

        valid_mask = valid_acc is not None or valid_loss is not None
        test_mask = test_acc is not None or test_loss is not None

        if valid_mask and not test_mask:
            return f"[Metrics] [Train]: acc: {train_acc}, loss: {train_loss} / [valid]: acc: {valid_acc}, loss: {valid_loss}"
        elif not valid_mask and test_mask:
            return f"[Metrics] [Train]: acc: {train_acc}, loss: {train_loss} / [Test]: acc: {test_acc}, loss: {test_loss}"
        elif valid_mask and test_mask:
            return f"[Metrics] [Train]: acc: {train_acc}, loss: {train_loss} / [valid]: acc: {valid_acc}, loss: {valid_loss} / [Test]: acc: {test_acc}, loss: {test_loss}"
        else:
            return f"[Metrics] [Train]: acc: {train_acc}, loss: {train_loss}"

    def log_dict(
        self,
        loss: float,
        acc: Optional[float] = None,
        valid_acc: Optional[float] = None,
        valid_loss: Optional[float] = None,
    ) -> Dict[str, Dict[str, Optional[float]]]:
        """
        log의 dictionary 생성 방식

        Args:
            loss (float): train set의 loss - 반드시 존재함
            acc (Optional[float], optional): train set의 accuracy. Defaults to None.
                - batch iterator에서 존재하지 않음.
            valid_acc (Optional[float], optional): _description_. Defaults to None.
                - batch iterator에서 존재하지 않음.
                - validation set이 존재하지 않는 경우, 존재하지 않음
            valid_loss (Optional[float], optional): _description_. Defaults to None.
                - batch iterator에서 존재하지 않음.
                - validation set이 존재하지 않는 경우, 존재하지 않음

        Returns:
            Dict[str, Dict[str, Optional[float]]]: accuracy, loss의 dictionary
        """
        return {
            "loss": {"train": loss, "valid": valid_loss},
            "acc": {"train": acc, "valid": valid_acc},
        }


class _ObjectDetection:
    def __init__(self, num_digits):
        self.num_digits = num_digits


class _Segmentation:
    def __init__(self, num_digits):
        self.num_digits = num_digits


class _AnomalyDetection:
    def __init__(self, num_digits):
        self.num_digits = num_digits
