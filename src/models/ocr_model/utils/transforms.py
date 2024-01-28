import torch
import torchvision

from torchvision.transforms import v2
from PIL import ImageChops, Image
from typing import Any, Dict, List

from ....globals import OCR_IMG_CHANNELS, OCR_IMG_SIZE, OCR_FIX_SIZE, IMAGE_MEAN, IMAGE_STD


def trim_white_border(image: Image.Image):
    if image.mode == 'RGB':
        bg_color = (255, 255, 255)
    elif image.mode == 'RGBA':
        bg_color = (255, 255, 255, 255)
    elif image.mode == 'L':
        bg_color = 255
    else:
        raise ValueError("Unsupported image mode")
    # 创建一个与图片一样大小的白色背景
    bg = Image.new(image.mode, image.size, bg_color)
    # 计算原图像与背景图像的差异。如果原图像在边框区域与左上角像素颜色相同，那么这些区域在差异图像中将是黑色的。
    diff = ImageChops.difference(image, bg)
    # 这一步增强差异图像中的对比度，使非背景区域更加明显。这对确定边界框有帮助，但参数的选择可能需要根据具体图像进行调整。
    diff = ImageChops.add(diff, diff, 2.0, -100)
    # 找到差异图像中非黑色区域的边界框。如果找到，原图将根据这个边界框被裁剪。
    bbox = diff.getbbox()
    if bbox:
        return image.crop(bbox)


def train_transform(images: List[Image.Image]) -> List[torch.Tensor]:
    assert OCR_IMG_CHANNELS == 1 , "Only support grayscale images for now"
    assert OCR_FIX_SIZE == True, "Only support fixed size images for now"
    images = [trim_white_border(image) for image in images]
    transforms = v2.Compose([
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

    images = transforms(images)  # imgs: List[PIL.Image.Image]
    images = [  
        v2.functional.pad(
            img,
            padding=[0, 0, OCR_IMG_SIZE - img.shape[2], OCR_IMG_SIZE - img.shape[1]]
        )
        for img in images
    ]
    return images


def inference_transform(images: List[Image.Image]) -> List[torch.Tensor]:
    assert OCR_IMG_CHANNELS == 1 , "Only support grayscale images for now"
    assert OCR_FIX_SIZE == True, "Only support fixed size images for now"
    return train_transform(images)
