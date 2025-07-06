import os

import matplotlib.pyplot as plt


def plot_training_history(history, save_path=None):
    """Визуализирует историю обучения"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(history["train_losses"], label="Train Loss")
    ax1.plot(history["test_losses"], label="Test Loss")
    ax1.set_title("Loss")
    ax1.legend()

    ax2.plot(history["train_accs"], label="Train Acc")
    ax2.plot(history["test_accs"], label="Test Acc")
    ax2.set_title("Accuracy")
    ax2.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)

    plt.show()


def plot_results(results: dict, save_path=None):
    """Выводит 4 графика (train/test loss/accuracy) на одном изображении"""

    # Создаем фигуру с 2x2 субплoтами
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 10))
    fig.suptitle("Training and Validation Metrics", fontsize=16)

    # Определяем какие графики будем отображать и их расположение
    metrics = [
        ("train_losses", "Train Loss", 0, 0),
        ("train_accs", "Train Accuracy", 0, 1),
        ("test_losses", "Test Loss", 1, 0),
        ("test_accs", "Test Accuracy", 1, 1),
    ]

    for metric, title, row, col in metrics:
        ax = axes[row][col]

        for model_name, result in results.items():
            ax.plot(result[metric], label=model_name)

        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(title)
        ax.legend()
        ax.grid(True)

    plt.tight_layout()

    if save_path:
        dir_path = os.path.dirname(save_path)
        if dir_path and not os.path.exists(save_path):
            os.makedirs(dir_path, exist_ok=True)
        plt.savefig(save_path)
        print(f"Графики сохранены в {save_path}")

    plt.show()


def count_parameters(model):
    """Подсчитывает количество параметров модели"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
