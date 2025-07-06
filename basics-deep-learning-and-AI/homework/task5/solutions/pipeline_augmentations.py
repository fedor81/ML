import os
from typing import Callable

import torchvision.transforms as transforms
from PIL import Image

from ..utils.datasets import CustomImageDataset
from .standard_augmentations import ROOT
from ..utils.visualization import show_multiple_augmentations


class AugmentationPipeline:
    def __init__(self, **kwargs):
        self.augmentations = kwargs

    def add_augmentation(self, name: str, aug: Callable):
        """Добавляет аугментацию в пайплайн"""
        self.augmentations[name] = aug

    def remove_augmentation(self, name: str):
        """Удаляет аугментацию из пайплайна"""
        if name in self.augmentations:
            del self.augmentations[name]

    def apply(self, image: Image.Image) -> Image.Image:
        """Применяет все аугментации к изображению"""
        img = image.copy()
        for aug_name, aug_func in self.augmentations.items():
            img = aug_func(img)
        return img

    def get_augmentations(self) -> dict[str, Callable]:
        """Возвращает словарь всех аугментаций"""
        return self.augmentations.copy()

    def __call__(self, *args, **kwargs):
        self.apply(*args, **kwargs)


def light_augmentation_pipeline() -> AugmentationPipeline:
    pipeline = AugmentationPipeline(
        flip=transforms.RandomHorizontalFlip(p=0.3),
        jitter=transforms.ColorJitter(brightness=0.1),
        blur=transforms.GaussianBlur(kernel_size=3),
    )
    return pipeline


def medium_augmentation_pipeline() -> AugmentationPipeline:
    pipeline = AugmentationPipeline(
        flip=transforms.RandomHorizontalFlip(p=0.5),
        jitter=transforms.ColorJitter(brightness=0.2, contrast=0.2),
        rotate=transforms.RandomRotation(15),
        random_grayscale=transforms.RandomGrayscale(p=0.3),
    )
    return pipeline


def heavy_augmentation_pipeline() -> AugmentationPipeline:
    pipeline = AugmentationPipeline(
        flip=transforms.RandomHorizontalFlip(p=0.8),
        jitter=transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3),
        rotate=transforms.RandomRotation(30),
        blur=transforms.GaussianBlur(kernel_size=3),
        elastic=transforms.ElasticTransform(alpha=250.0),
        random_affine=transforms.RandomAffine(
            degrees=0, translate=(0.3, 0.3), scale=(0.5, 1.5), shear=30
        ),
        random_grayscale=transforms.RandomGrayscale(p=0.5),
        solarize=transforms.RandomSolarize(threshold=192.0, p=0.3),
    )
    return pipeline


def run():
    dataset = CustomImageDataset(os.path.join(ROOT, "data", "train"), transform=None)
    pipelines = [
        ("Light", light_augmentation_pipeline()),
        ("Medium", medium_augmentation_pipeline()),
        ("Heavy", heavy_augmentation_pipeline()),
    ]
    to_tensor = transforms.ToTensor()
    original, label = dataset.random()
    augmented = []

    for _, pipeline in pipelines:
        augmented.append(to_tensor(pipeline.apply(original)))

    show_multiple_augmentations(
        to_tensor(original),
        augmented,
        titles=(name for name, _ in pipelines),
        save_path=os.path.join(ROOT, "results", "pipeline.png"),
    )


if __name__ == "__main__":
    run()
