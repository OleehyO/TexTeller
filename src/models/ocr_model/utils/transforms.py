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
    v2.ToImage(),    # Convert to tensor, only needed if you had a PIL image
                     #+返回一个List of torchvision.Image，list的长度就是batch_size
                     #+因此在整个Compose pipeline的最后，输出的也是一个List of torchvision.Image
                     #+注意：不是返回一整个torchvision.Image，batch_size的维度是拿出来的
    v2.ToDtype(torch.uint8, scale=True),  # optional, most input are already uint8 at this point
    v2.Grayscale(),  # 转灰度图（视具体任务而定）

    v2.Resize(       # 固定resize到一个正方形上
        size=FIXED_IMG_SIZE - 1,  # size必须小于max_size 
        interpolation=v2.InterpolationMode.BICUBIC,
        max_size=FIXED_IMG_SIZE,
        antialias=True
    ),

    v2.ToDtype(torch.float32, scale=True),  # Normalize expects float input
    v2.Normalize(mean=[IMAGE_MEAN], std=[IMAGE_STD]),

    # v2.ToPILImage()  # 用于观察转换后的结果是否正确（for debug）
])


def trim_white_border(image: np.ndarray):
    # np.ndarray: 3D, RGB format, dimension distribution: [H, W, C] (channel dimension is on the third dimension)

    # check if image is in RGB format and channel is in third dimension
    if len(image.shape) != 3 or image.shape[2] != 3:
        raise ValueError("Image is not in RGB format or channel is not in third dimension")

    # check if image is stored in uint8
    if image.dtype != np.uint8:
        raise ValueError(f"Image should stored in uint8")

    corners = [tuple(image[0, 0]), tuple(image[0, -1]),
               tuple(image[-1, 0]), tuple(image[-1, -1])]
    bg_color = Counter(corners).most_common(1)[0][0]
    bg_color_np = np.array(bg_color, dtype=np.uint8)
    
    # create a background image with the same size and color as the original image
    h, w = image.shape[:2]
    bg = np.full((h, w, 3), bg_color_np, dtype=np.uint8)

    # compute the difference between the original image and the background image
    diff = cv2.absdiff(image, bg)
    mask = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    # cut off those that are close to the background color as well
    threshold = 15 
    _, diff = cv2.threshold(mask, threshold, 255, cv2.THRESH_BINARY)

    # Calculate the minimum bounding matrix of non-zero pixels in an image
    x, y, w, h = cv2.boundingRect(diff) 

    # crop image
    trimmed_image = image[y:y+h, x:x+w]

    return trimmed_image if 0 not in trimmed_image.shape else image    


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
    # np.ndarray: 3D, RGB format, dimension distribution: [H, W, C] (channel dimension is on the third dimension)

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
    # random rotate in [-5, 5] degree with 20% probability
    if random.random() < 0.2:
        image = rotate(image, -5, 5)
    # add white border
    image = add_white_border(image, max_size=25).permute(1, 2, 0).numpy()
    # data augmentation
    image = train_pipeline(image)
    return image


def train_transform(images: List[Image.Image], categories: List[str]) -> List[torch.Tensor]:
    assert IMG_CHANNELS == 1 , "Only support grayscale images for now"

    images = [np.array(img.convert('RGB')) for img in images]
    # random resize first
    images = random_resize(images, MIN_RESIZE_RATIO, MAX_RESIZE_RATIO)
    # crop white border
    images = [trim_white_border(image) for image in images]

    # OCR augmentation
    # images = [ocr_aug(image) for image in images]
    images = [ocr_aug(image) if not category.endswith("nature") else image for image, category in zip(images, categories)]

    # general transform pipeline
    images = [general_transform_pipeline(image) for image in images]

    # padding to fixed size
    images = padding(images, FIXED_IMG_SIZE)
    return images


def inference_transform(images: List[Union[np.ndarray, Image.Image]]) -> List[torch.Tensor]:
    assert IMG_CHANNELS == 1 , "Only support grayscale images for now"
    images = [np.array(img.convert('RGB')) if isinstance(img, Image.Image) else img for img in images]
    # crop white border
    images = [trim_white_border(image) for image in images]
    # general transform pipeline
    images = [general_transform_pipeline(image) for image in  images]  # imgs: List[PIL.Image.Image]
    # padding to fixed size
    images = padding(images, FIXED_IMG_SIZE)

    return images


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
    res = random_resize(imgs, 0.5, 1.5)
    pause = 1

