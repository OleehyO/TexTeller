import torch
import random
import numpy as np
import cv2

from torchvision.transforms import v2
from typing import List, Union
from PIL import Image

from ...globals import (
    OCR_IMG_CHANNELS,
    OCR_IMG_SIZE,
    OCR_FIX_SIZE,
    IMAGE_MEAN, IMAGE_STD,
    MAX_RESIZE_RATIO, MIN_RESIZE_RATIO
)


general_transform_pipeline = v2.Compose([
    v2.ToImage(),    # Convert to tensor, only needed if you had a PIL image
                        #+返回一个List of torchvision.Image，list的长度就是batch_size
                        #+因此在整个Compose pipeline的最后，输出的也是一个List of torchvision.Image
                        #+注意：不是返回一整个torchvision.Image，batch_size的维度是拿出来的
    v2.ToDtype(torch.uint8, scale=True),  # optional, most input are already uint8 at this point
    v2.Grayscale(),  # 转灰度图（视具体任务而定）

    v2.Resize(       # 固定resize到一个正方形上
        size=OCR_IMG_SIZE - 1,  # size必须小于max_size 
        interpolation=v2.InterpolationMode.BICUBIC,
        max_size=OCR_IMG_SIZE,
        antialias=True
    ),

    v2.ToDtype(torch.float32, scale=True),  # Normalize expects float input
    v2.Normalize(mean=[IMAGE_MEAN], std=[IMAGE_STD]),

    # v2.ToPILImage()  # 用于观察转换后的结果是否正确（debug用）
])


def trim_white_border(image: np.ndarray):
    # image是一个3维的ndarray，RGB格式，维度分布为[H, W, C]（通道维在第三维上）

    # # 检查images中的第一个元素是否是嵌套的列表结构
    # if isinstance(image, list):
    #     image = np.array(image, dtype=np.uint8)

    # 检查图像是否为RGB格式，同时检查通道维是不是在第三维上
    if len(image.shape) != 3 or image.shape[2] != 3:
        raise ValueError("Image is not in RGB format or channel is not in third dimension")

    # 检查图片是否使用 uint8 类型
    if image.dtype != np.uint8:
        raise ValueError(f"Image should stored in uint8")

    # 创建与原图像同样大小的纯白背景图像
    h, w = image.shape[:2]
    bg = np.full((h, w, 3), 255, dtype=np.uint8)

    # 计算差异
    diff = cv2.absdiff(image, bg)

    # 只要差值大于1，就全部转化为255
    _, diff = cv2.threshold(diff, 1, 255, cv2.THRESH_BINARY)

    # 把差值转灰度图
    gray_diff = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)
    # 计算图像中非零像素点的最小外接矩阵
    x, y, w, h = cv2.boundingRect(gray_diff) 

    # 裁剪图像
    trimmed_image = image[y:y+h, x:x+w]

    return trimmed_image


def padding(images: List[torch.Tensor], required_size: int):
    images = [  
        v2.functional.pad(
            img,
            padding=[0, 0, required_size - img.shape[2], required_size - img.shape[1]]
        )
        for img in images
    ]
    return images


def random_resize(
    images: List[np.ndarray], 
    minr: float, 
    maxr: float
) -> List[np.ndarray]:
    # np.ndarray的格式：3维，RGB格式，维度分布为[H, W, C]（通道维在第三维上）

    # # 检查images中的第一个元素是否是嵌套的列表结构
    # if isinstance(images[0], list):
    #     # 将嵌套的列表结构转换为np.ndarray
    #     images = [np.array(img, dtype=np.uint8) for img in images]

    if len(images[0].shape) != 3 or images[0].shape[2] != 3:
        raise ValueError("Image is not in RGB format or channel is not in third dimension")

    ratios = [random.uniform(minr, maxr) for _ in range(len(images))]
    return [
        cv2.resize(img, (int(img.shape[1] * r), int(img.shape[0] * r)), interpolation=cv2.INTER_LANCZOS4)  # 抗锯齿
        for img, r in zip(images, ratios)
    ]


def general_transform(images: List[np.ndarray]) -> List[torch.Tensor]:
    # 裁剪掉白边
    images = [trim_white_border(image) for image in images]
    # general transform pipeline
    images = general_transform_pipeline(images)  # imgs: List[PIL.Image.Image]
    # padding to fixed size
    images = padding(images, OCR_IMG_SIZE)
    return images


def train_transform(images: List[Image.Image]) -> List[torch.Tensor]:
    assert OCR_IMG_CHANNELS == 1 , "Only support grayscale images for now"
    assert OCR_FIX_SIZE == True, "Only support fixed size images for now"

    # random resize first
    images = [np.array(img.convert('RGB')) for img in images]
    images = random_resize(images, MIN_RESIZE_RATIO, MAX_RESIZE_RATIO)
    return general_transform(images)


def inference_transform(images: List[np.ndarray]) -> List[torch.Tensor]:
    assert OCR_IMG_CHANNELS == 1 , "Only support grayscale images for now"
    assert OCR_FIX_SIZE == True, "Only support fixed size images for now"

    return general_transform(images)


if __name__ == '__main__':
    from pathlib import Path
    from .helpers import convert2rgb
    base_dir = Path('/home/lhy/code/TeXify/src/models/ocr_model/model')
    imgs_path = [
        base_dir / '1.jpg',
        base_dir / '2.jpg',
        base_dir / '3.jpg',
        base_dir / '4.jpg',
        base_dir / '5.jpg',
        base_dir / '6.jpg',
        base_dir / '7.jpg',
    ]
    imgs_path = [str(img_path) for img_path in imgs_path]
    imgs = convert2rgb(imgs_path)
    # res = train_transform(imgs)
    # res = [v2.functional.to_pil_image(img) for img in res]
    res = random_resize(imgs, 0.5, 1.5)
    pause = 1

