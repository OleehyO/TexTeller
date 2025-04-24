from collections import Counter
from typing import List, Union

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import v2

from texteller.constants import (
    FIXED_IMG_SIZE,
    IMG_CHANNELS,
    IMAGE_MEAN,
    IMAGE_STD,
)
from texteller.logger import get_logger


_logger = get_logger()


def readimgs(image_paths: list[str]) -> list[np.ndarray]:
    """
    Read and preprocess a list of images from their file paths.

    This function reads each image from the provided paths, handles different
    bit depths (converting 16-bit to 8-bit if necessary), and normalizes color
    channels to RGB format regardless of the original color space (BGR, BGRA,
    or grayscale).

    Args:
        image_paths (list[str]): A list of file paths to the images to be read.

    Returns:
        list[np.ndarray]: A list of NumPy arrays containing the preprocessed images
                         in RGB format. Images that could not be read are skipped.
    """
    processed_images = []
    for path in image_paths:
        image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError(f"Image at {path} could not be read.")
        if image.dtype == np.uint16:
            _logger.warning(f"Converting {path} to 8-bit, image may be lossy.")
            image = cv2.convertScaleAbs(image, alpha=(255.0 / 65535.0))

        channels = 1 if len(image.shape) == 2 else image.shape[2]
        if channels == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        elif channels == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif channels == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        processed_images.append(image)

    return processed_images


def trim_white_border(image: np.ndarray) -> np.ndarray:
    if len(image.shape) != 3 or image.shape[2] != 3:
        raise ValueError("Image is not in RGB format or channel is not in third dimension")

    if image.dtype != np.uint8:
        raise ValueError(f"Image should stored in uint8")

    corners = [tuple(image[0, 0]), tuple(image[0, -1]), tuple(image[-1, 0]), tuple(image[-1, -1])]
    bg_color = Counter(corners).most_common(1)[0][0]
    bg_color_np = np.array(bg_color, dtype=np.uint8)

    h, w = image.shape[:2]
    bg = np.full((h, w, 3), bg_color_np, dtype=np.uint8)

    diff = cv2.absdiff(image, bg)
    mask = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    threshold = 15
    _, diff = cv2.threshold(mask, threshold, 255, cv2.THRESH_BINARY)

    x, y, w, h = cv2.boundingRect(diff)

    trimmed_image = image[y : y + h, x : x + w]

    return trimmed_image


def padding(images: List[torch.Tensor], required_size: int) -> List[torch.Tensor]:
    images = [
        v2.functional.pad(
            img, padding=[0, 0, required_size - img.shape[2], required_size - img.shape[1]]
        )
        for img in images
    ]
    return images


def transform(images: List[Union[np.ndarray, Image.Image]]) -> List[torch.Tensor]:
    general_transform_pipeline = v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.uint8, scale=True),
            v2.Grayscale(),
            v2.Resize(
                size=FIXED_IMG_SIZE - 1,
                interpolation=v2.InterpolationMode.BICUBIC,
                max_size=FIXED_IMG_SIZE,
                antialias=True,
            ),
            v2.ToDtype(torch.float32, scale=True),  # Normalize expects float input
            v2.Normalize(mean=[IMAGE_MEAN], std=[IMAGE_STD]),
        ]
    )

    assert IMG_CHANNELS == 1, "Only support grayscale images for now"
    images = [
        np.array(img.convert("RGB")) if isinstance(img, Image.Image) else img for img in images
    ]
    images = [trim_white_border(image) for image in images]
    images = [general_transform_pipeline(image) for image in images]
    images = padding(images, FIXED_IMG_SIZE)

    return images
