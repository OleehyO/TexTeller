import torch
import random
import numpy as np
import cv2

from torchvision.transforms import v2
from typing import List, Union
from PIL import Image
from collections import Counter

from ...globals import (
    IMG_CHANNELS,
    FIXED_IMG_SIZE,
    IMAGE_MEAN, IMAGE_STD,
    MAX_RESIZE_RATIO, MIN_RESIZE_RATIO
)
from .ocr_aug import ocr_augmentation_pipeline

# train_pipeline = default_augraphy_pipeline(scan_only=True)
train_pipeline = ocr_augmentation_pipeline()

general_transform_pipeline = v2.Compose([
    v2.ToImage(),    
    v2.ToDtype(torch.uint8, scale=True),  # optional, most input are already uint8 at this point
    v2.Grayscale(),

    v2.Resize(
        size=FIXED_IMG_SIZE - 1,
        interpolation=v2.InterpolationMode.BICUBIC,
        max_size=FIXED_IMG_SIZE,
        antialias=True
    ),

    v2.ToDtype(torch.float32, scale=True),  # Normalize expects float input
    v2.Normalize(mean=[IMAGE_MEAN], std=[IMAGE_STD]),

    # v2.ToPILImage()
])


def trim_white_border(image: np.ndarray):
    if len(image.shape) != 3 or image.shape[2] != 3:
        raise ValueError("Image is not in RGB format or channel is not in third dimension")

    if image.dtype != np.uint8:
        raise ValueError(f"Image should stored in uint8")

    corners = [tuple(image[0, 0]), tuple(image[0, -1]),
               tuple(image[-1, 0]), tuple(image[-1, -1])]
    bg_color = Counter(corners).most_common(1)[0][0]
    bg_color_np = np.array(bg_color, dtype=np.uint8)
    
    h, w = image.shape[:2]
    bg = np.full((h, w, 3), bg_color_np, dtype=np.uint8)

    diff = cv2.absdiff(image, bg)
    mask = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    threshold = 15
    _, diff = cv2.threshold(mask, threshold, 255, cv2.THRESH_BINARY)

    x, y, w, h = cv2.boundingRect(diff) 

    trimmed_image = image[y:y+h, x:x+w]

    return trimmed_image


def add_white_border(image: np.ndarray, max_size: int) -> np.ndarray:
    randi = [random.randint(0, max_size) for _ in range(4)]
    pad_height_size = randi[1] + randi[3]
    pad_width_size  = randi[0] + randi[2]
    if (pad_height_size + image.shape[0] < 30):
        compensate_height = int((30 - (pad_height_size + image.shape[0])) * 0.5) + 1
        randi[1] += compensate_height
        randi[3] += compensate_height
    if (pad_width_size + image.shape[1] < 30):
        compensate_width = int((30 - (pad_width_size + image.shape[1])) * 0.5) + 1
        randi[0] += compensate_width
        randi[2] += compensate_width
    return v2.functional.pad(
        torch.from_numpy(image).permute(2, 0, 1),
        padding=randi,
        padding_mode='constant',
        fill=(255, 255, 255)
    )


def padding(images: List[torch.Tensor], required_size: int) -> List[torch.Tensor]:
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


def rotate(image: np.ndarray, min_angle: int, max_angle: int) -> np.ndarray:
    # Get the center of the image to define the point of rotation
    image_center = tuple(np.array(image.shape[1::-1]) / 2)

    # Generate a random angle within the specified range
    angle = random.randint(min_angle, max_angle)

    # Get the rotation matrix for rotating the image around its center
    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)

    # Determine the size of the rotated image
    cos = np.abs(rotation_mat[0, 0])
    sin = np.abs(rotation_mat[0, 1])
    new_width = int((image.shape[0] * sin) + (image.shape[1] * cos))
    new_height = int((image.shape[0] * cos) + (image.shape[1] * sin))

    # Adjust the rotation matrix to take into account translation
    rotation_mat[0, 2] += (new_width / 2) - image_center[0]
    rotation_mat[1, 2] += (new_height / 2) - image_center[1]

    # Rotate the image with the specified border color (white in this case)
    rotated_image = cv2.warpAffine(image, rotation_mat, (new_width, new_height), borderValue=(255, 255, 255))

    return rotated_image


def ocr_aug(image: np.ndarray) -> np.ndarray:
    if random.random() < 0.2:
        image = rotate(image, -5, 5)
    image = add_white_border(image, max_size=25).permute(1, 2, 0).numpy()
    image = train_pipeline(image)
    return image


def train_transform(images: List[Image.Image]) -> List[torch.Tensor]:
    assert IMG_CHANNELS == 1 , "Only support grayscale images for now"

    images = [np.array(img.convert('RGB')) for img in images]
    # random resize first
    images = random_resize(images, MIN_RESIZE_RATIO, MAX_RESIZE_RATIO)
    images = [trim_white_border(image) for image in images]

    # OCR augmentation
    images = [ocr_aug(image) for image in images]

    # general transform pipeline
    images = [general_transform_pipeline(image) for image in  images]
    # padding to fixed size
    images = padding(images, FIXED_IMG_SIZE)
    return images


def inference_transform(images: List[Union[np.ndarray, Image.Image]]) -> List[torch.Tensor]:
    assert IMG_CHANNELS == 1 , "Only support grayscale images for now"
    images = [np.array(img.convert('RGB')) if isinstance(img, Image.Image) else img for img in images]
    images = [trim_white_border(image) for image in images]
    # general transform pipeline
    images = [general_transform_pipeline(image) for image in  images]  # imgs: List[PIL.Image.Image]
    # padding to fixed size
    images = padding(images, FIXED_IMG_SIZE)

    return images
