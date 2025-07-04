import numpy as np
import seaborn as sns
import torch
from matplotlib import pyplot as plt

from utils.datasets_utils import get_cifar_loaders, get_mnist_loaders
from utils.experiment_utils import train_model
from utils.model_utils import FullyConnectedModel
from utils.visualization_utils import (
    plot_results,
)

# --------------------------------------------------------------------------------------------------
# Задание 3.1

train_loader, test_loader = get_cifar_loaders()
input_size = 32 * 32 * 3
num_classes = 10


def compare_regularization(
    epochs=10,
    save_folder=None,
    models={
        "none": FullyConnectedModel(
            input_size,
            num_classes,
            layers=[
                {"type": "linear", "size": 128},
                {"type": "linear", "size": 128},
            ],
        ),
        "dropout_0.1": FullyConnectedModel(
            input_size,
            num_classes,
            layers=[
                {"type": "linear", "size": 128},
                {"type": "dropout", "rate": 0.1},
                {"type": "linear", "size": 128},
                {"type": "dropout", "rate": 0.1},
            ],
        ),
        "dropout_0.3": FullyConnectedModel(
            input_size,
            num_classes,
            layers=[
                {"type": "linear", "size": 128},
                {"type": "dropout", "rate": 0.3},
                {"type": "linear", "size": 128},
                {"type": "dropout", "rate": 0.3},
            ],
        ),
        "dropout_0.5": FullyConnectedModel(
            input_size,
            num_classes,
            layers=[
                {"type": "linear", "size": 128},
                {"type": "dropout", "rate": 0.5},
                {"type": "linear", "size": 128},
                {"type": "dropout", "rate": 0.5},
            ],
        ),
        "batchnorm": FullyConnectedModel(
            input_size,
            num_classes,
            layers=[
                {"type": "linear", "size": 128},
                {"type": "batch_norm"},
                {"type": "linear", "size": 128},
                {"type": "batch_norm"},
            ],
        ),
        "both": FullyConnectedModel(
            input_size,
            num_classes,
            layers=[
                {"type": "linear", "size": 128},
                {"type": "batch_norm"},
                {"type": "dropout", "rate": 0.1},
                {"type": "linear", "size": 128},
                {"type": "batch_norm"},
                {"type": "dropout", "rate": 0.1},
            ],
        ),
        "l2": FullyConnectedModel(
            input_size,
            num_classes,
            layers=[
                {"type": "linear", "size": 128},
                {"type": "linear", "size": 128},
            ],
        ),
    },
):
    """Сравнивает техники регуляризации"""

    results = {}

    # Инициализация модели и оптимизатора
    for name, model in models.items():
        print(f"Обучение модели {name}...")
        weight_decay = 0.01 if name == "l2" else 0  # L2 регуляризация

        history = train_model(model, train_loader, test_loader, epochs, weight_decay)

        # Получаем веса модели для анализа
        weights = model.get_weights()
        history["weights"] = weights
        history["weight_mean"] = np.mean(np.abs(weights))
        history["weight_std"] = np.std(weights)
        results[name] = history

    plot_results(results, save_folder=save_folder)

    # Распределение весов
    plt.figure()
    for name in models.keys():
        sns.kdeplot(results[name]["weights"], label=name)
    plt.xlabel("Weight Values")
    plt.ylabel("Density")
    plt.title("Distribution of Weight Values")
    plt.savefig(save_folder + "weights_distribution.png")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Таблица с результатами
    print("\nСравнение техник регуляризации:")
    print(f"{'Regularization':<15} | {'Accuracy':<8} | {'Weight Mean':<12} | {'Weight Std':<10}")
    print("-" * 55)
    for name, history in results.items():
        print(
            f"{name:<15} | {max(history['test_accs']):.4f}   | {history['weight_mean']:.6f}    | {history['weight_std']:.6f}"
        )


# if __name__ == "__main__":
#    compare_regularization(epochs=10, save_folder="./basics-deep-learning-and-AI/homework/task3/plots/task3/")


# --------------------------------------------------------------------------------------------------
# Задание 3.2

if __name__ == "__main__":
    compare_regularization(
        epochs=15,
        save_folder="./basics-deep-learning-and-AI/homework/task3/plots/task3/2/",
        models={
            "dropout_adaptive": FullyConnectedModel(
                input_size,
                num_classes,
                layers=[
                    {"type": "linear", "size": 128},
                    {"type": "dropout", "rate": 0.5},
                    {"type": "linear", "size": 128},
                    {"type": "dropout", "rate": 0.4},
                    {"type": "linear", "size": 128},
                    {"type": "dropout", "rate": 0.3},
                ],
            ),
            "batchnorm_adaptive": FullyConnectedModel(
                input_size,
                num_classes,
                layers=[
                    {"type": "linear", "size": 128},
                    {"type": "batch_norm", "momentum": 0.1},
                    {"type": "linear", "size": 128},
                    {"type": "batch_norm", "momentum": 0.3},
                    {"type": "linear", "size": 128},
                    {"type": "batch_norm", "momentum": 0.5},
                ],
            ),
            "combined": FullyConnectedModel(
                input_size,
                num_classes,
                layers=[
                    {"type": "linear", "size": 128},
                    {"type": "batch_norm", "momentum": 0.1},
                    {"type": "dropout", "rate": 0.5},
                    {"type": "linear", "size": 128},
                    {"type": "batch_norm", "momentum": 0.3},
                    {"type": "dropout", "rate": 0.3},
                    {"type": "linear", "size": 128},
                    {"type": "batch_norm", "momentum": 0.5},
                    {"type": "dropout", "rate": 0.1},
                ],
            ),
        },
    )
