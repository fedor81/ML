import os
import time

import torch
import torch.nn as nn
from matplotlib import pyplot as plt

from ..task3.utils.datasets_utils import get_mnist_loaders, get_cifar_loaders
from .homework_cnn_vs_fc_comparison import ROOT
from .models.cnn_models import ResidualConfigurable
from .models.custom_layers import (
    BottleneckBlock,
    L1Conv2d,
    ResidualBlock,
    SpatialAttention,
    StochasticPool2d,
    Swish,
    WideBlock,
)
from .utils.visualization_utils import makedirs_if_not_exists, ResultsTable
from .utils import training_utils

# --------------------------------------------------------------------------------------------------
# Задание 3.1


def compare_layers():
    x = torch.randn(32, 64, 128, 128)  # [batch, channels, height, width]

    print("\nСравнение сверточных слоев")
    conv = nn.Conv2d(64, 128, 3, padding=1)
    l1conv = L1Conv2d(64, 128, 3, padding=1)

    start = time.time()
    _ = conv(x)
    print(f"Standard Conv: {time.time() - start:.4f}s")

    start = time.time()
    _ = l1conv(x)
    print(f"L1 Conv: {time.time() - start:.4f}s")

    print("\nСравнение Attention")
    test_input = torch.randn(1, 64, 32, 32)
    attn = SpatialAttention(64)
    output = attn(test_input)
    print(f"Input shape: {test_input.shape}, Output shape: {output.shape}")

    # Сравнение функции активации с ReLU
    x = torch.linspace(-5, 5, 100)
    swish = Swish()
    relu = nn.ReLU()

    plt.figure(figsize=(10, 5))
    plt.plot(x.numpy(), swish(x).numpy(), label="Swish")
    plt.plot(x.numpy(), relu(x).numpy(), label="ReLU")
    plt.title("Swish vs ReLU Activation")
    plt.legend()
    plt.grid()
    save_path = os.path.join(ROOT, "plots", "swish_vs_relu.png")
    makedirs_if_not_exists(save_path)
    plt.savefig(save_path)
    plt.show()

    print("\nСравнение StochasticPool и MaxPool")
    x = torch.randn(32, 64, 128, 128)  # [batch, channels, height, width]
    stoch_pool = StochasticPool2d(kernel_size=2, stride=2)
    max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

    start = time.time()
    output_max = max_pool(x)
    print(f"MaxPool: {time.time() - start:.4f}s, output shape: {output_max.shape}")

    # Тестирование StochasticPool
    start = time.time()
    output_stoch = stoch_pool(x)
    print(f"StochasticPool: {time.time() - start:.4f}s, output shape: {output_stoch.shape}")

    # Проверка диапазона значений
    print("\nMaxPool output range:", output_max.min().item(), output_max.max().item())
    print("StochasticPool output range:", output_stoch.min().item(), output_stoch.max().item())


# --------------------------------------------------------------------------------------------------
# Задание 3.2


def compare_residual_blocks(epochs=10):
    train_loader, test_loader = get_cifar_loaders()

    models = [
        ("BasicBlock", lambda: ResidualConfigurable(ResidualBlock, [2, 2, 2])),
        ("BottleneckBlock", lambda: ResidualConfigurable(BottleneckBlock, [2, 2, 2])),
        ("WideBlock", lambda: ResidualConfigurable(WideBlock, [2, 2, 2])),
    ]

    # Модели получились более легковесными, в задании 2.2,
    # поэтому для них хватает места в оперативной памяти
    results: list[training_utils.TrainResult] = []
    table = ResultsTable()

    for name, model_fn in models:
        model = model_fn()
        result = training_utils.train_model_v2(
            model,
            name,
            train_loader,
            test_loader,
            end_epoch=epochs,
            save_folder=os.path.join(ROOT, "weights", name),
        )
        table.add_row(result)
        results.append(result)

    print(table)

    # График точности
    plt.subplot(2, 1, 1)
    for res in results:
        plt.plot(res.test_accs, label=res.name)
    plt.title("Test Accuracy by Block Type")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    # График потерь
    plt.subplot(2, 1, 2)
    for res in results:
        plt.plot(res.train_losses, label=res["name"])
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    path = os.path.join(ROOT, "plots", "block_comparison.png")
    makedirs_if_not_exists(path)
    plt.savefig(path)
    plt.show()


if __name__ == "__main__":
    compare_residual_blocks(epochs=1)
