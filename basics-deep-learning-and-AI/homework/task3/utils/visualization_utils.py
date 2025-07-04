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


def plot_results(results: dict, save_folder: str = None):
    """Args:
    results: словарь вида {
        "model_name": {
            "train_losses": list[int],
            "train_accs": list[int],
            "test_losses": list[int],
            "test_accs": list[int],
            "train_time": list[int],
        }
    }"""

    graphics = ["train_losses", "train_accs", "test_losses", "test_accs"]

    for graphic in graphics:
        plt.figure(figsize=(12, 6))

        for model_name, result in results.items():
            plt.plot(
                result[graphic],
                label=model_name,
            )

        plt.xlabel("Epoch")
        plt.ylabel(graphic)
        plt.title(graphic)
        plt.legend()
        plt.grid(True)

        if save_folder:
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
                print(f"Создан каталог для сохранения графиков: {save_folder}")
            plt.savefig(f"{save_folder}{graphic}.png")

    plt.tight_layout()
    plt.show()


def count_parameters(model):
    """Подсчитывает количество параметров модели"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
