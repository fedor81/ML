import os
import random

import cv2
import numpy as np
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from PIL import Image, ImageEnhance

from ..utils import extra_augs
from ..utils.datasets import CustomImageDataset
from ..utils.visualization import show_multiple_augmentations
from .standard_augmentations import ROOT


class CustomMotionBlur:
    """Случайная размытие движения"""

    def __init__(self, max_kernel_size=15):
        self.max_kernel_size = max_kernel_size

    def __call__(self, img):
        kernel_size = random.choice([9, 15, 21])
        angle = random.randint(0, 360)

        # Создаем ядро для размытия движения
        kernel = np.zeros((kernel_size, kernel_size))
        center = kernel_size // 2
        cv2.line(kernel, (center, 0), (center, kernel_size - 1), 1, 1)
        kernel = kernel / np.sum(kernel)

        # Применяем поворот ядра
        M = cv2.getRotationMatrix2D((center, center), angle, 1)
        kernel = cv2.warpAffine(kernel, M, (kernel_size, kernel_size))

        # Применяем размытие
        img_np = np.array(img)
        blurred = cv2.filter2D(img_np, -1, kernel)
        return Image.fromarray(blurred)


class CustomRandomPerspective:
    """Случайное перспективное преобразование"""

    def __init__(self, distortion_scale=0.5):
        self.distortion_scale = distortion_scale

    def __call__(self, img):
        width, height = img.size

        # Начальные точки (углы изображения)
        startpoints = [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]]

        # Конечные точки (со случайным смещением)
        endpoints = []
        for x, y in startpoints:
            new_x = x + random.randint(
                -int(width * self.distortion_scale), int(width * self.distortion_scale)
            )
            new_y = y + random.randint(
                -int(height * self.distortion_scale), int(height * self.distortion_scale)
            )
            endpoints.append([new_x, new_y])

        return F.perspective(img, startpoints, endpoints)


class CustomRandomBrightnessContrast:
    """Случайная яркость и контрастность"""

    def __init__(self, brightness_range=(0.7, 1.3), contrast_range=(0.7, 1.3)):
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range

    def __call__(self, img):
        # Применяем яркость
        enhancer = ImageEnhance.Brightness(img)
        brightness_factor = random.uniform(*self.brightness_range)
        img = enhancer.enhance(brightness_factor)

        # Применяем контраст
        enhancer = ImageEnhance.Contrast(img)
        contrast_factor = random.uniform(*self.contrast_range)
        return enhancer.enhance(contrast_factor)


# Сравнение с extra_augs
def compare_extra_augs():
    dataset = CustomImageDataset(os.path.join(ROOT, "data", "train"), transform=None)
    image, label = dataset.random()

    compare_augs = [
        (
            (
                "GaussianNoise",
                transforms.Compose([transforms.ToTensor(), extra_augs.AddGaussianNoise()]),
            ),
            (
                "MotionBlur",
                transforms.Compose([CustomMotionBlur(max_kernel_size=5), transforms.ToTensor()]),
            ),
        ),
        (
            (
                "AutoContrast",
                transforms.Compose([transforms.ToTensor(), extra_augs.AutoContrast()]),
            ),
            (
                "RandomBrightnessContrast",
                transforms.Compose([CustomRandomBrightnessContrast(), transforms.ToTensor()]),
            ),
        ),
        (
            (
                "ElasticTransform",
                transforms.Compose([transforms.ToTensor(), extra_augs.ElasticTransform()]),
            ),
            (
                "RandomPerspective",
                transforms.Compose([CustomRandomPerspective(), transforms.ToTensor()]),
            ),
        ),
    ]

    for compare in compare_augs:
        (extra_name, extra_aug), (my_name, my_aug) = compare

        show_multiple_augmentations(
            transforms.ToTensor()(image),
            augmented_imgs=[extra_aug(image), my_aug(image)],
            titles=[extra_name, my_name],
            save_path=os.path.join(
                ROOT, "results", "custom_augmentations", f"{extra_name}_{my_name}.png"
            ),
        )


if __name__ == "__main__":
    compare_extra_augs()
