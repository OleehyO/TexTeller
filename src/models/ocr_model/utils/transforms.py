import torch
import random
import numpy as np
import cv2

from torchvision.transforms import v2
from typing import List
from PIL import Image

from models.globals import (
    FIXED_IMG_SIZE,
    IMAGE_MEAN, IMAGE_STD,
    MAX_RESIZE_RATIO, MIN_RESIZE_RATIO
)

general_transform_pipeline = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.uint8, scale=True),
    v2.Grayscale(),
    v2.Resize(
        size=FIXED_IMG_SIZE - 1,
        interpolation=v2.InterpolationMode.BICUBIC,
        max_size=FIXED_IMG_SIZE,
        antialias=True
    ),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[IMAGE_MEAN], std=[IMAGE_STD]),
])


def trim_white_border(image: np.ndarray):
    if len(image.shape) != 3 or image.shape[2] != 3:
        raise ValueError("Image is not in RGB format or channel is not in third dimension")

    if image.dtype != np.uint8:
        raise ValueError(f"Image should stored in uint8")

    h, w = image.shape[:2]
    bg = np.full((h, w, 3), 255, dtype=np.uint8)
    diff = cv2.absdiff(image, bg)

    _, diff = cv2.threshold(diff, 1, 255, cv2.THRESH_BINARY)
    gray_diff = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)
    x, y, w, h = cv2.boundingRect(gray_diff) 

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
    if len(images[0].shape) != 3 or images[0].shape[2] != 3:
        raise ValueError("Image is not in RGB format or channel is not in third dimension")

    ratios = [random.uniform(minr, maxr) for _ in range(len(images))]
    return [
        cv2.resize(img, (int(img.shape[1] * r), int(img.shape[0] * r)), interpolation=cv2.INTER_LANCZOS4)  # 抗锯齿
        for img, r in zip(images, ratios)
    ]


def general_transform(images: List[np.ndarray]) -> List[torch.Tensor]:
    images = [trim_white_border(image) for image in images]
    images = general_transform_pipeline(images)
    images = padding(images, FIXED_IMG_SIZE)
    return images


def train_transform(images: List[Image.Image]) -> List[torch.Tensor]:
    images = [np.array(img.convert('RGB')) for img in images]
    images = random_resize(images, MIN_RESIZE_RATIO, MAX_RESIZE_RATIO)
    return general_transform(images)


def inference_transform(images: List[np.ndarray]) -> List[torch.Tensor]:
    return general_transform(images)
