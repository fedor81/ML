import os

import torch
import torchvision
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision import models, transforms

from ...task4.utils import training_utils, visualization_utils
from ..utils.datasets import CustomImageDataset
from .standard_augmentations import ROOT


train_transform = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(p=0.5),  # Случайное горизонтальное отражение
        transforms.RandomRotation(15),  # Случайный поворот на ±15 градусов
        transforms.ColorJitter(
            brightness=0.2,  # Случайное изменение яркости
            contrast=0.2,  # Случайное изменение контраста
            saturation=0.2,  # Случайное изменение насыщенности
            hue=0.1,  # Случайное изменение оттенка (в пределах ±0.1)
        ),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)
val_transform = transforms.Compose([transforms.ToTensor()])

train_dataset = CustomImageDataset(
    os.path.join(ROOT, "data", "train"), target_size=(224, 224), transform=train_transform
)
val_dataset = CustomImageDataset(
    os.path.join(ROOT, "data", "test"), target_size=(224, 224), transform=val_transform
)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# Загрузка предобученной модели
model_name = "resnet18"
model = models.resnet18(weights="IMAGENET1K_V1")
model.fc = torch.nn.Linear(model.fc.in_features, len(train_dataset.get_class_names()))

# Обучение и валидация
result = training_utils.train_model_v2(
    model,
    model_name,
    train_loader,
    val_loader,
    end_epoch=100,
    save_folder=os.path.join(ROOT, "weights", model_name),
    patience=10,
)

# Отображение результатов
table = visualization_utils.ResultsTable()
table.add_row(result)
print(table)

# Отображение графиков
plt.figure(figsize=(12, 5))

# График функции потерь
plt.subplot(1, 2, 1)
plt.plot(result.train_losses, label="Train Loss")
plt.plot(result.test_losses, label="Validation Loss")
plt.title("Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

# График точности
plt.subplot(1, 2, 2)
plt.plot(result.train_accs, label="Train Accuracy")
plt.plot(result.test_accs, label="Validation Accuracy")
plt.title("Training and Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)

plt.tight_layout()
path = os.path.join(ROOT, "results", f"{model_name}.png")
visualization_utils.makedirs_if_not_exists(path)
plt.savefig(path)
plt.show()
