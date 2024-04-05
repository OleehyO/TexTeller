import torch
from torchvision.transforms import v2

from PIL import Image, ImageChops
from ...globals import (
    IMAGE_MEAN, IMAGE_STD, 
    LABEL_RATIO,
    RESIZER_IMG_SIZE,
    NUM_CHANNELS
)

from typing import (
    Any,
    List,
    Dict,
)


def trim_white_border(image: Image):
    if image.mode == 'RGB':
        bg_color = (255, 255, 255)
    elif image.mode == 'RGBA':
        bg_color = (255, 255, 255, 255)
    elif image.mode == 'L':
        bg_color = 255
    else:
        raise ValueError("Unsupported image mode")
    bg = Image.new(image.mode, image.size, bg_color)
    diff = ImageChops.difference(image, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    if bbox:
        return image.crop(bbox)


def preprocess_fn(samples: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
    imgs = samples['pixel_values']
    imgs = [trim_white_border(img) for img in imgs]
    labels = [float(img.height * LABEL_RATIO) for img in imgs]

    assert NUM_CHANNELS == 1, "Only support grayscale images"
    transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.uint8, scale=True),
        v2.Grayscale(),
        v2.Resize(
            size=RESIZER_IMG_SIZE - 1,  # size必须小于max_size 
            interpolation=v2.InterpolationMode.BICUBIC,
            max_size=RESIZER_IMG_SIZE,
            antialias=True
        ),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[IMAGE_MEAN], std=[IMAGE_STD]),
    ])
    imgs = transform(imgs)
    imgs = [
        v2.functional.pad(
            img,
            padding=[0, 0, RESIZER_IMG_SIZE - img.shape[2], RESIZER_IMG_SIZE - img.shape[1]]
        )
        for img in imgs
    ]

    res = {'pixel_values': imgs, 'labels': labels}
    return res


if __name__ == "__main__":  # unit test
    import datasets
    data = datasets.load_dataset("/home/lhy/code/TeXify/src/models/resizer/train/dataset/dataset.py").shuffle(seed=42)
    data = data.with_transform(preprocess_fn)
    train_data, test_data = data['train'], data['test']

    inpu = train_data[:10]
    pause = 1
