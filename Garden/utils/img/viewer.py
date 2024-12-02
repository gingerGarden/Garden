from typing import List, Tuple, Union

import ipywidgets
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display
from ipywidgets import widgets


def show_img(
    img: np.ndarray,
    title: str = None,
    title_color: str = "limegreen",
    titlesize: int = 15,
    figsize: Tuple[int, int] = (8, 6),
):
    """
    matplotlib.pyplot을 사용하여 이미지를 출력한다.
    >>> 이미지의 형태가 흑백 이미지인 경우, 흑백으로 출력한다.

    Args:
        img (np.ndarray): 출력하고자 하는 이미지의 형태
        title (str, optional): 이미지의 제목. Defaults to None.
        title_color (str, optional): 제목의 색. Defaults to 'limegreen'.
        titlesize (int, optional): 제목의 크기. Defaults to 15.
        figsize (Tuple[int, int], optional): 이미지 크기에 대한 tuple. Defaults to (8,6).
    """
    if not isinstance(img, np.ndarray):
        raise TypeError(
            "입력 데이터는 np.ndarray 타입이어야 하며, 이미지를 기대합니다."
        )

    plt.figure(figsize=figsize)
    if (len(img.shape) == 2) or (img.shape[-1] == 1):
        plt.imshow(img, cmap="gray")
    else:
        plt.imshow(img)
    plt.axis("off")
    if title is not None:
        plt.title(title, fontsize=titlesize, pad=20, color=title_color)
    plt.show()


def show_imgs(
    img_list: List[np.ndarray],
    col: int = 4,
    img_size: Tuple[int, int] = (3, 3),
    show_number: bool = False,
):
    """
    image(np.ndarray)로 구성된 list를 col의 갯수만큼 서브 이미지를 생성하여 보기 쉽게 만듦.

    Args:
        img_list (List[np.ndarray]): image들로 구성된 List
        col (int, optional): 한 줄에 몇 개의 이미지로 구성되게 할 것인지. Defaults to 4.
        img_size (Tuple[int, int], optional): 각 이미지의 크기. Defaults to (3,3).
        show_number (bool, optional): 각 이미지의 번호를 각 이미지에 붙일지 여부. Defaults to False.
    """
    row = int(np.ceil(len(img_list) / col))

    # figsize 설정
    fig_width = col * img_size[0]
    fig_height = row * img_size[1]
    plt.figure(figsize=(fig_width, fig_height))

    for i, img in enumerate(img_list):

        plt.subplot(row, col, i + 1)
        plt.imshow(img)
        if show_number:
            plt.title(i)
        plt.axis("off")

    plt.tight_layout(pad=0)
    plt.show()


class Slider:
    def __init__(
        self,
        imgs: Union[List[np.ndarray], np.ndarray],
        titles: Union[List[str], None] = None,
        title_color: str = "limegreen",
        title_size: int = 10,
        figsize: Tuple[int, int] = (8, 6),
    ):
        """
        imgs를 Slider로 출력한다.
        >>> imgs는 List또는 이미지들에 대한 다차원 배열이다.
        >>> titles는 각 이미지들의 제목들에 대한 list 또는 1차원 배열이다.

        해당 class는 run() 메서드를 사용하여, 주요 메서드를 실행한다.
        ex) Slider(imgs).run()

        Args:
            imgs (Union[List[np.ndarray], np.ndarray]): n개의 이미지들 (n>=1)
            titles (Union[List[str], None], optional): n개의 각 이미지에 대한 타이틀. Defaults to None.
            title_color (str, optional): 타이틀의 색. Defaults to 'limegreen'.
            title_size (int, optional): 타이틀의 크기. Defaults to 10.
            figsize (Tuple[int, int], optional): 이미지의 크기. Defaults to (8, 6).
        """
        if titles is not None and len(titles) != len(imgs):
            raise ValueError("이미지와 타이틀의 개수가 일치하지 않습니다.")

        self.imgs = imgs
        self.titles = titles
        self.title_size = title_size
        self.title_color = title_color
        self.figsize = figsize

    def run(self):
        """
        Slider 클래스의 메인 동작
        """
        # 각 layout을 가지고 온다.
        slider, int_box = self._layout()
        # 슬라이더와 텍스트 입력 박스의 값을 동기화
        widgets.jslink((slider, "value"), (int_box, "value"))
        display(slider)
        # 이미지 표시 함수
        ipywidgets.interact(self._show_img_use_index, idx=int_box)

    def _show_img_use_index(self, idx):
        """
        index에 해당하는 이미지 출력
        """
        img = self.imgs[idx]
        idx_txt = f"{idx+1}/{len(self.imgs)}"
        title = f"{idx_txt}: {self.titles[idx]}" if self.titles is not None else idx_txt
        show_img(
            img,
            title=title,
            title_color=self.title_color,
            fontsize=self.title_size,
            figsize=self.figsize,
        )

    def _layout(self, value: int = 0):
        """
        Widget의 layout
        """
        slider = widgets.IntSlider(
            value=value,
            min=0,
            max=len(self.imgs) - 1,
            description="Slide index",
            layout=widgets.Layout(width="700px"),
        )
        int_box = widgets.IntText(
            value=value, description="Index: ", layout=widgets.Layout(width="200px")
        )
        return slider, int_box
