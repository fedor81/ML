import os
import torch
from PIL import Image
from torchvision import transforms

from ..utils.datasets import CustomImageDataset
from ..utils.visualization import (
    show_multiple_augmentations,
)

ROOT = "./basics-deep-learning-and-AI/homework/task5/"


def run():
    dataset = CustomImageDataset(
        os.path.join(ROOT, "data", "train"), transform=transforms.ToTensor()
    )
    classes = dataset.get_class_names()
    images = {class_name: dataset.first(class_name)[0] for class_name in classes}

    # Аугментации
    standard_augs = [
        (
            "RandomHorizontalFlip",
            transforms.Compose([transforms.RandomHorizontalFlip(p=1.0)]),
        ),
        (
            "RandomCrop",
            transforms.Compose([transforms.RandomCrop(200, padding=20)]),
        ),
        (
            "ColorJitter",
            transforms.Compose(
                [
                    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1),
                ]
            ),
        ),
        (
            "RandomRotation",
            transforms.Compose([transforms.RandomRotation(degrees=30)]),
        ),
        (
            "RandomGrayscale",
            transforms.Compose([transforms.RandomGrayscale(p=1.0)]),
        ),
        (
            "AllTogether",
            transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(p=1.0),
                    transforms.RandomCrop(size=200, padding=20),
                    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1),
                    transforms.RandomRotation(degrees=30),
                    transforms.RandomGrayscale(p=1.0),
                ]
            ),
        ),
    ]

    # Применение и отображение
    for class_name, original in images.items():
        augmented = []
        titles = []

        for name, aug_transform in standard_augs:
            aug_img = aug_transform(original)
            augmented.append(aug_img)
            titles.append(name)

        show_multiple_augmentations(
            original,
            augmented,
            titles,
            save_path=os.path.join(ROOT, "results", "standard_augmentations", f"{class_name}.png"),
        )


if __name__ == "__main__":
    run()
