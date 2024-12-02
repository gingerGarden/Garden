from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils import data

from ....utils.log.progressbar import ProgressBar
from ....utils.path import make_null_json, save_pickle
from .utils import get_lr


class Classification:
    def __init__(
        self,
        model,
        loader,
        optimizer,
        back_propagation,
        early_stopping,
        option,
        epoch_header: str = "Epoch    ",
        iter_header: str = "Iterator ",
    ):
        # 입력받는 instance variable
        self.model = model
        self.loader = loader
        self.optimizer = optimizer
        self.back_propagation = back_propagation
        self.early_stopping = early_stopping
        self.option = option

        self.epoch_header = epoch_header
        self.iter_header = iter_header

        # 생성되는 instance variable
        self.early_stopping_mask = (
            self.loader.valid is not None and self.early_stopping is not None
        )

        # epoch 내에서 ProgressBar의 instance
        self.pbar_ins = ProgressBar(
            header=self.epoch_header,
            time_log=True,
            memory_log=True,
            step=1,
            verbose=option.verbose,
            save_path=self.option.log_ins.epoch,
        )

    def fit(self):
        pass

    def _fit_iterator(self, epoch: int, epoch_log_txt: str) -> Tuple[float, float]:
        """
        batch 단위로 모델을 학습한다.
            - 역전파
            - 스케쥴러 조정
            - log 생성(batch 단위에 대해서만 재생성 - 그 이상은 축적하지 않음)

        Args:
            epoch (int): 현재의 epoch
            epoch_log_txt (str): 이전 epoch 당시의 log_txt

        Returns:
            Tuple[float, float]: train_loss, train_accuracy
        """
        # iterator 당 log 파일을 새로 갱신한다.
        make_null_json(path=self.option.log_ins.iterator, empty=True, make_new=True)

        pbar_ins = ProgressBar(
            header=self.iter_header,
            time_log=True,
            memory_log=False,
            step=5,
            verbose=self.option.verbose,
            save_path=self.option.log_ins.iterator,
        )
        stack_output = np.array([], dtype=np.float32)
        stack_label = np.array([], dtype=np.float32)
        loss_list = []

        # 학습 모드로 설정
        self.model.train()
        iter_size = len(self.loader.train)
        for i, (pbar_txt, (imgs, labels)) in enumerate(pbar_ins(self.loader.train)):

            # load to device
            imgs = imgs.to(self.option.device)
            # label dtype을 loss_fn에 맞게 수정 및 shape 등 변형
            labels = (
                self.option.label_dtype_fn(labels).reshape(-1, 1).to(self.option.device)
            )
            # 모델 학습 및 평활된 numpy 배열 출력
            loss, output_array, label_array = self._fit_and_flatten_in_one_iter(
                imgs, labels
            )
            # scheduler 조정
            self.option.scheduler_helper.batch_step(epoch + i / iter_size)
            # 결과 정리
            loss_list.append(loss.item())
            stack_output = np.concatenate((stack_output, output_array), axis=0)
            stack_label = np.concatenate((stack_label, label_array), axis=0)
            # Iterator 종료 시점을 기준으로 log 출력
            lr = get_lr(
                optimizer=self.optimizer,
                scheduler=self.option.scheduler_helper.scheduler,
            )
            perform_log_dict = self.option.log_ins.log_dict(
                upper_i=epoch, i=i, lr=lr, loss=loss.item()
            )
            _, _ = pbar_ins.end_of_iterator(
                progress_txt=f"{epoch_log_txt}{pbar_txt}",
                i=i,
                extra_log_dict=perform_log_dict,
            )

        # iterator의 loss 평균, Accuracy출력
        train_loss = np.mean(loss_list)
        train_accuracy = self.option.metrics(
            predict=stack_output, label=stack_label, accuracy_only=True
        )
        return train_loss, train_accuracy

    def _fit_and_flatten_in_one_iter(
        self, imgs: torch.Tensor, labels: torch.Tensor
    ) -> Tuple[torch.Tensor, np.ndarray, np.ndarray]:
        """
        모델을 학습하고, 추론된 결과와 label을 1차원 numpy 배열로 출력한다.

        Args:
            imgs (torch.Tensor): Tensor 이미지
            labels (torch.Tensor): Tensor label

        Returns:
            Tuple[torch.Tensor, np.ndarray, np.ndarray]: loss, img 배열, label 배열
        """
        # AMP
        with torch.autocast(
            device_type=self.option.device, enabled=self.option.use_amp
        ):
            output = self.model(imgs)
            loss = self.option.loss_fn(output, labels)

        # back propagation
        self.back_propagation(loss)

        # output과 label을 평활하고 numpy 배열로 변환
        output = self.output_handler(
            tensor=output, extra_activation_fn=self.option.extra_activation_fn
        )
        labels = self.output_handler(tensor=labels)

        return loss, output, labels

    def _epoch_log(
        self,
        bar_txt: str,
        es_log_txt: str,
        epoch: int,
        train_acc: float,
        train_loss: float,
        valid_acc: Optional[float],
        valid_loss: Optional[float],
    ) -> str:
        """
        한 epoch에 대한 log 생성 및 저장
        - 모델 성능(train, valid)에 대한 log text
        - 모델 성능(train, valid)에 대한 log dictionary
        - epoch에 대하여, bar_txt와 함께 출력할 log_txt(epoch_log_txt)

        Args:
            bar_txt (str): progressbar의 진행 상황에 대한 text
            es_log_txt (str): early stopping 진행 상황에 대한 text
            epoch (int): 현재 epoch
            train_acc (float): train accuracy
            train_loss (float): train loss
            valid_acc (Optional[float]): validation accuracy
            valid_loss (Optional[float]): validation loss

        Returns:
            str: 현재 epoch에 대한 log text로, 다음 epoch 진행 동안 이전의 epoch에 대한 log 문자열이 출력됨.
        """
        # learning rate
        lr = get_lr(
            optimizer=self.optimizer, scheduler=self.option.scheduler_helper.scheduler
        )

        # 모델 성능에 대한 log txt
        perform_txt = self.option.log_ins.log_txt(
            lr=lr,
            train_acc=train_acc,
            train_loss=train_loss,
            valid_acc=valid_acc,
            valid_loss=valid_loss,
        )
        # 모델 성능에 대한 log dictionary
        perform_log_dict = self.option.log_ins.log_dict(
            upper_i=self.option.log_ins.k,
            i=epoch,
            lr=lr,
            acc=train_acc,
            loss=train_loss,
            valid_acc=valid_acc,
            valid_loss=valid_loss,
        )
        # Epoch에 대하여, 출력할 log txt
        epoch_log_txt, _ = self.pbar_ins.end_of_iterator(
            progress_txt=bar_txt + "\n" + perform_txt,
            i=epoch,
            end_of_txts=es_log_txt,
            extra_log_dict=perform_log_dict,
        )
        return epoch_log_txt

    @torch.inference_mode()
    def inference(self, loader: data.DataLoader) -> Tuple[float, float]:
        """
        입력 받은 loader를 이용하여, 추론한다.

        Args:
            loader (data.DataLoader): validation set 또는 test set의 DataLoader

        Returns:
            Tuple[float, float]: 추론 결과의 loss, accuracy
        """
        stack_output = np.array([], dtype=np.float32)
        stack_label = np.array([], dtype=np.float32)
        loss_list = []

        # 추론 모드로 설정
        self.model.eval()
        for imgs, labels in loader:

            imgs = imgs.to(self.option.device)
            labels = (
                self.option.label_dtype_fn(labels).reshape(-1, 1).to(self.option.device)
            )
            # 모델 추론 및 평활된 numpy 배열 출력
            loss, output_array, label_array = self._inference_and_flatten(imgs, labels)
            # 결과 정리
            loss_list.append(loss.item())
            stack_output = np.concatenate((stack_output, output_array), axis=0)
            stack_label = np.concatenate((stack_label, label_array), axis=0)

        # inference의 loss 평균, Accuracy출력
        eval_loss = np.mean(loss_list)
        eval_accuracy = self.option.metrics(
            predict=stack_output, label=stack_label, accuracy_only=True
        )
        # metric을 위한 결과 지표
        output_dict = {"predict": stack_output, "label": stack_label}
        return eval_loss, eval_accuracy, output_dict

    def _inference_and_flatten(
        self, imgs: torch.Tensor, labels: torch.Tensor
    ) -> Tuple[torch.Tensor, np.ndarray, np.ndarray]:
        """
        추론 후 데이터를 평활하는 과정
        - torch.no_grad(): 를 통해 기울기가 변하지 않도록 함.
        - torch.cuda.synchronize(): 배치마다 GPU 연산이 종료된 후, CPU가 사용되도록 하여, 정확한 결과가 나오도록 함.
        - self.output_handler(): 결과를 1차원으로 평활시키며, 필요에 따라 추가 활성화 함수를 통과시킴.

        Args:
            imgs (torch.Tensor): _description_
            labels (torch.Tensor): _description_

        Returns:
            Tuple[torch.Tensor, np.ndarray, np.ndarray]: loss, 추론 결과, label
        """
        # GPU를 사용하는 동안 추론
        with torch.no_grad():
            output = self.model(imgs)
            loss = self.option.loss_fn(output, labels)

        # 각 배치마다 동기화 GPU 사용이 종료된 후 CPU 사용
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # output과 label을 평활하고 numpy 배열로 변환
        output = self.output_handler(
            tensor=output, extra_activation_fn=self.option.extra_activation_fn
        )
        labels = self.output_handler(tensor=labels)

        return loss, output, labels

    def output_handler(
        self, tensor: torch.Tensor, extra_activation_fn: Optional[str] = None
    ) -> np.ndarray:
        """
        모델 추론 결과를 1차원 numpy 배열로 변환
        >>> 모델의 추론 결과를 추가 활성화 함수(sigmoid, softmax)에 통과시킬 필요가 있다면, 그에 해당하는 활성화 함수에 통과시킴

        Args:
            output (torch.Tensor): 모델의 추론 결과
            extra_activation_fn (Optional[str], optional): 추가 활성화 함수. Defaults to None.
                - 'sigmoid', 'softmax' 존재

        Returns:
            np.ndarray: 1 차원 배열로 변환된 모델의 추론 결과
        """
        # extra_activateion_fn이 존재하는 경우, 그에 해당하는 활성화 함수를 통과 시킴.
        act_fns = {
            "sigmoid": torch.sigmoid,
            "softmax": lambda x: torch.softmax(x, dim=1),
        }
        act_fn = act_fns.get(extra_activation_fn, None)
        if act_fn is not None:
            tensor = act_fn(tensor)

        # numpy로 전환
        np_array = tensor.to("cpu").detach().numpy()
        # 1 차원으로 평활
        return np_array.flatten()

    def end_of_fit(
        self,
        epoch: int,
        bar_txt: str,
        train_acc: float,
        train_loss: float,
        valid_acc: Optional[float],
        valid_loss: Optional[float],
    ) -> Dict[str, Union[np.ndarray, List[str]]]:
        """
        학습이 완료된 후, 프로세스
            - early stopping 종료에 대한 log text 생성
            - test set에 대한 inference
            - test set의 inference 결과와 주요 지표 저장
            - epoch 종료 시, Test set을 포함한 log 생성 및 출력

        Args:
            epoch (int): 학습이 종료되었을 당시의 epoch
            bar_txt (str): epoch에 대한 progress bar의 txt
            train_acc (float): train set의 accuracy
            train_loss (float): train set의 loss
            valid_acc (Optional[float]): validation set의 accuracy
            valid_loss (Optional[float]): validation set의 loss

        Returns:
            Dict[str, Union[np.ndarray, List[str]]]: test set의 inference 결과
        """
        # early stopping 종료
        if self.early_stopping_mask:
            es_log_txt = self.early_stopping.end_point()

        # test set에 대한 evaluate
        test_loss, test_acc, test_output_dict = self.inference(loader=self.loader.test)

        # 결과 저장
        self.save_result_dict(test_output_dict)

        # epoch 종료 시, Test set에 대한 Log 생성
        lr = get_lr(self.optimizer, scheduler=self.option.scheduler_helper.scheduler)
        perform_txt = self.option.log_ins.log_txt(
            lr=lr,
            train_acc=train_acc,
            train_loss=train_loss,
            valid_acc=valid_acc,
            valid_loss=valid_loss,
            test_acc=test_acc,
            test_loss=test_loss,
        )

        # epoch 종료 시, 최종 log 출력
        _, _ = self.pbar_ins.end_of_iterator(
            progress_txt=bar_txt + "\n" + perform_txt,
            i=epoch,
            end_of_txts=es_log_txt,
            dont_save=True,
        )
        return test_output_dict

    def save_result_dict(self, test_output_dict: Dict[str, np.ndarray]):
        """
        모델 pipeline의 최종 결과를 저장한다

        Args:
            test_output_dict (Dict[str, np.ndarray]): test dataset에 대한 추론 결과
        """
        # test set에 대한 결과 및 metadata 저장
        test_output_dict["path"] = [i["path"] for i in self.loader.idx_dict["test"]]
        result_dict = {
            "metadata": {
                "model_name": self.option.model_name,
                "device": self.option.device,
                "hp_dict": self.option.hp_dict,
                "use_amp": self.option.use_amp,
                "use_clipping": self.option.use_clipping,
            },
            "inference": test_output_dict,
        }
        file_path = f"{self.option.results_parents}/{self.option.process_id}/{self.option.log_ins.k}.pickle"
        save_pickle(data=result_dict, pickle_path=file_path)
