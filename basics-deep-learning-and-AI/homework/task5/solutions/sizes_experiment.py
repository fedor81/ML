import gc
import os
import time
from typing import Callable

import psutil
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from PIL import Image
from prettytable import PrettyTable
from tqdm import tqdm

from ..utils.datasets import CustomImageDataset
from .standard_augmentations import ROOT
from ...task4.utils.visualization_utils import makedirs_if_not_exists


def sizes_experiment():
    sizes = [
        (32, 32),
        (64, 64),
        (128, 128),
        (224, 224),
        (512, 512),
        (1024, 1024),
    ]
    augmentations = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.GaussianBlur(kernel_size=3),
            transforms.RandomPosterize(bits=2),
        ]
    )
    count = 100

    load_times = []
    aug_times = []
    memory_usage = []

    for size in sizes:
        dataset = CustomImageDataset(
            os.path.join(ROOT, "data", "train"), target_size=size, transform=None
        )
        images = []

        # Начальное потребление памяти
        gc.collect()
        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss / (1024**2)  # в MB

        # Загрузка изображений
        start_time = time.time()
        for i in tqdm(range(count), desc=f"Loading images: {size}"):
            image, label = dataset[i]
            images.append(image)
        load_times.append(time.time() - start_time)

        # Применение аугментаций
        start_time = time.time()
        for image in tqdm(images, desc=f"Applying augmentations: {size}"):
            augmentations(image)
        aug_times.append(time.time() - start_time)

        # Замеры памяти
        mem_after = process.memory_info().rss / (1024**2)
        mem_usage = mem_after - mem_before
        memory_usage.append(mem_usage)

    # Вывод результатов
    print(f"\nЗагрузка и применение аугментаций к {count} изображениям:\n")

    sizes = [f"{size[0]}x{size[1]}" for size in sizes]
    table = PrettyTable(["Size", "Load time", "Augmentations time", "Memory usage"])
    table.float_format = ".2"

    for row in zip(sizes, load_times, aug_times, memory_usage):
        table.add_row(row)

    print(table)

    # Вывод графиков
    plt.figure(figsize=(15, 5))

    # График времени загрузки
    plt.subplot(1, 3, 1)
    plt.plot(sizes, load_times, "o-")
    plt.title("Время загрузки 100 изображений")
    plt.xlabel("Размер изображения (px)")
    plt.ylabel("Время (сек)")
    plt.grid(True)

    # График времени аугментации
    plt.subplot(1, 3, 2)
    plt.plot(sizes, aug_times, "o-")
    plt.title("Время аугментации 100 изображений")
    plt.xlabel("Размер изображения (px)")
    plt.ylabel("Время (сек)")
    plt.grid(True)

    # График использования памяти
    plt.subplot(1, 3, 3)
    plt.plot(sizes, memory_usage, "o-")
    plt.title("Использование памяти")
    plt.xlabel("Размер изображения (px)")
    plt.ylabel("Память (MB)")
    plt.grid(True)

    plt.tight_layout()
    path = os.path.join(ROOT, "results", "sizes_experiment.png")
    makedirs_if_not_exists(path)
    plt.savefig(path)
    plt.show()


if __name__ == "__main__":
    sizes_experiment()
