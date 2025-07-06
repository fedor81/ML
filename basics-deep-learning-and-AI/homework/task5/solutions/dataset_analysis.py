import os

import numpy as np
from matplotlib import pyplot as plt
from prettytable import PrettyTable

from ...task4.utils.visualization_utils import makedirs_if_not_exists
from ..utils.datasets import CustomImageDataset
from .standard_augmentations import ROOT


def print_dataset_info():
    path = os.path.join(ROOT, "data", "train")
    dataset = CustomImageDataset(path)
    classes = dataset.get_class_names()

    class_counts: dict[int:int] = {}  # Подсчет количества изображений в каждом классе
    widths = []
    heights = []

    for image, label in dataset:
        class_counts[label] = class_counts.get(label, 0) + 1
        width, height = image.size
        widths.append(width)
        heights.append(height)

    table = PrettyTable(classes)
    table.add_row([class_counts[dataset.class_to_idx[class_name]] for class_name in classes])

    print(f"\nКоличество изображений в {path}:\n")
    print(table)

    print("\nОбщая статистика по размерам:\n")
    print(f"Всего изображений: {len(widths)}")
    print(f"Минимальный размер: {min(widths)}x{min(heights)}")
    print(f"Максимальный размер: {max(widths)}x{max(heights)}")
    print(f"Средний размер: {np.mean(widths):.0f}x{np.mean(heights):.0f}")

    # Графики распределения размеров
    plt.figure(figsize=(15, 5))

    # Ширина
    plt.subplot(1, 2, 1)
    plt.hist(widths, bins=30, color="skyblue", edgecolor="black")
    plt.title("Распределение ширины изображений")
    plt.xlabel("Ширина (пиксели)")
    plt.ylabel("Количество")

    # Высота
    plt.subplot(1, 2, 2)
    plt.hist(heights, bins=30, color="salmon", edgecolor="black")
    plt.title("Распределение высоты изображений")
    plt.xlabel("Высота (пиксели)")
    plt.ylabel("Количество")

    plt.tight_layout()
    path = os.path.join(ROOT, "results", "dataset_info", "size.png")
    makedirs_if_not_exists(path)
    plt.savefig(path)
    plt.show()

    # Визуализация по классам
    plt.figure(figsize=(15, 7))
    counts = [class_counts[dataset.class_to_idx[class_name]] for class_name in classes]
    plt.bar(classes, counts, color="purple")
    plt.title("Количество изображений по классам")
    plt.xlabel("Класс")
    plt.ylabel("Количество")
    plt.xticks(rotation=45)

    plt.tight_layout()
    path = os.path.join(ROOT, "results", "dataset_info", "classes.png")
    makedirs_if_not_exists(path)
    plt.savefig(path)
    plt.show()


if __name__ == "__main__":
    print_dataset_info()
