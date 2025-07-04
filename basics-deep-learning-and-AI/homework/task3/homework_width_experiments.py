import pandas as pd
import seaborn as sns
import torch
from matplotlib import pyplot as plt

from utils.datasets_utils import get_cifar_loaders, get_mnist_loaders
from utils.experiment_utils import train_model
from utils.model_utils import FullyConnectedModel
from utils.visualization_utils import (
    count_parameters,
    plot_results,
    plot_training_history,
)

# --------------------------------------------------------------------------------------------------
# Задание 2.1


def compare_layers(
    layers=[(64, 32, 16), (256, 128, 64), (1024, 512, 256), (2048, 1024, 512)], epochs=15
):
    """Сравнивает модели с разным количеством нейронов"""
    data = [
        # ("MNIST", get_cifar_loaders(), 32 * 32 * 3, 10),
        ("CIFAR", get_mnist_loaders(), 28 * 28, 10)
    ]
    results = {}
    for dataset_name, (train_loader, test_loader), input_size, num_classes in data:
        dataset_results = {}

        for layers_sizes in layers:
            print(f"\nТренировка модели с {layers_sizes} нейронами на {dataset_name}")
            model_layers = []

            for layer_size in layers_sizes:
                model_layers.append({"type": "linear", "size": layer_size})

            model = FullyConnectedModel(
                input_size=input_size, num_classes=num_classes, layers=model_layers
            )

            history = train_model(model, train_loader, test_loader, epochs=epochs)

            # Сохраняем данные модели
            history["count_params"] = count_parameters(model)
            history["layers_sizes"] = layers_sizes

            dataset_results[
                f"{dataset_name}_{str(layers_sizes).strip("()").replace(", ", "_")}"
            ] = history

        # Строим графики по датасету
        results[dataset_name] = dataset_results
        plot_results(
            dataset_results,
            save_folder="./basics-deep-learning-and-AI/homework/task3/plots/task2/",
        )

    # Таблица со всеми результатами
    print("\nСравнение моделей:")

    for dataset_name, dataset_results in results.items():
        print(f"\nДатасет: {dataset_name}\n")
        print(
            f"| {'Sizes':<12} | {'Count params':<10} | {'Train Loss':<10} | {'Train Acc':<10} | {'Test Loss':<10} | {'Test Acc':<10} | {'Time':<10}"
        )
        for results in dataset_results.values():
            train_time = results["train_time"]
            model = str(results["layers_sizes"])
            train_loss, train_acc, test_loss, test_acc = (
                min(results["train_losses"]),
                max(results["train_accs"]),
                min(results["test_losses"]),
                max(results["test_accs"]),
            )
            print(
                f"| {model:<10} | {results['count_params']:<10}   | {train_loss:.4f}     | {train_acc:.4f}     | {test_loss:.4f}     | {test_acc:.4f}     | {train_time:.4f}"
            )


# --------------------------------------------------------------------------------------------------
# Задание 2.2


def grid_search(
    depth_options=[1, 2, 3, 4],
    width_options={
        "const_64": lambda index: 64,
        "const_128": lambda index: 128,
        "expand_16": lambda index: 16 * (index + 1),
        "expand_32": lambda index: 32 * (index + 1),
        "narrow_256": lambda index: 256 // ((index + 1) * 2),
        "narrow_512": lambda index: 512 // ((index + 1) * 2),
    },
    epochs=10,
):
    """Перебирает все комбинации параметров для поиска лучшей модели на датасете MNIST"""

    train_loader, test_loader = get_mnist_loaders()
    results = []

    for depth in depth_options:
        for width_type, width_func in width_options.items():
            print("Тренировка модели с глубиной", depth, "и схемой ширины", width_type)
            model_layers = []

            for i in range(depth):
                model_layers.append({"type": "linear", "size": width_func(i)})
                model_layers.append({"type": "relu"})

            model = FullyConnectedModel(input_size=28 * 28, num_classes=10, layers=model_layers)
            history = train_model(model, train_loader, test_loader, epochs=epochs, patience=3)

            history["depth"] = depth
            history["width_type"] = width_type
            history["Accuracy"] = max(history["test_accs"])

            results.append(history)

    results_df = pd.DataFrame(results)

    # Тепловая карта
    pivot_table = results_df.pivot(index="depth", columns="width_type", values="Accuracy")
    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot_table, annot=True, fmt=".3f", cmap="YlGnBu", cbar_kws={"label": "Accuracy"})
    plt.title("Точность модели в зависимости от глубины и схемы ширины")
    plt.xlabel("Схема ширины слоев")
    plt.ylabel("Глубина сети (количество слоев)")
    plt.savefig("./basics-deep-learning-and-AI/homework/task3/plots/task2/heatmap.png")
    plt.show()


if __name__ == "__main__":
    grid_search()
