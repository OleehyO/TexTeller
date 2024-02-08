import cv2
import numpy as np
from typing import List
from PIL import Image


def convert2rgb(image_paths: List[str]) -> List[np.ndarray]:
    # 输出的np.ndarray的格式为：[H, W, C]（通道在第三维），通道的排列顺序为RGB
    processed_images = []

    for path in image_paths:
        # 读取图片
        image = cv2.imread(path, cv2.IMREAD_UNCHANGED)

        if image is None:
            print(f"Image at {path} could not be read.")
            continue

        # 检查图片是否使用 uint16 类型
        if image.dtype == np.uint16:
            raise ValueError(f"Image at {path} is stored in uint16, which is not supported.")

        # 获取图片通道数
        channels = 1 if len(image.shape) == 2 else image.shape[2]

        # 如果是 RGBA (4通道), 转换为 RGB
        if channels == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)

        # 如果是 I 模式 (单通道灰度图), 转换为 RGB
        elif channels == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        # 如果是 BGR (3通道), 转换为 RGB
        elif channels == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        processed_images.append(image)

    return processed_images