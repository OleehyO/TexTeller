import cv2
import numpy as np
from typing import List


def convert2rgb(image_paths: List[str]) -> List[np.ndarray]:
    processed_images = []
    for path in image_paths:
        image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if image is None:
            print(f"Image at {path} could not be read.")
            continue
        if image.dtype == np.uint16:
            print(f'Converting {path} to 8-bit, image may be lossy.')
            image = cv2.convertScaleAbs(image, alpha=(255.0/65535.0))

        channels = 1 if len(image.shape) == 2 else image.shape[2]
        if channels == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        elif channels == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif channels == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        processed_images.append(image)

    return processed_images
