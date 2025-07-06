import os

import numpy as np
from .models.cnn_models import KernelizableCNN, ShallowCNN, MediumCNN, DeepCNN, CNNWithResidual
from ..task3.utils.datasets_utils import get_mnist_loaders, get_cifar_loaders
from .utils import comparison_utils, visualization_utils, training_utils
from .homework_cnn_vs_fc_comparison import ROOT


# --------------------------------------------------------------------------------------------------
# Задание 2.1


def compare_kernels(epochs=10):
    train_loader, test_loader = get_mnist_loaders()
    in_channels = 1
    out_channels = 10

    configs = {
        "3x3": [{"out": 64, "kernel": 3}, {"out": 64, "kernel": 3}],
        "5x5": [{"out": 16, "kernel": 5}, {"out": 32, "kernel": 5}],
        "7x7": [{"out": 8, "kernel": 7}, {"out": 16, "kernel": 7}],
        "mixed": [{"out": 32, "kernel": 1}, {"out": 64, "kernel": 3}],
    }
    models = {
        name: KernelizableCNN(layers, in_channels, out_channels) for name, layers in configs.items()
    }

    results = comparison_utils.compare_models(
        models,
        train_loader,
        test_loader,
        models_save_folder=os.path.join(ROOT, "weights", "task2"),
        graphic_save_path=os.path.join(ROOT, "plots", "task2", "kernels_comparison.png"),
        epochs=epochs,
    )

    for model_name, config in configs.items():
        results[model_name]["rf"] = comparison_utils.calculate_rf(config)
        visualization_utils.visualize_activations(
            models[model_name],
            train_loader,
            model_name,
            save_path=os.path.join(ROOT, "plots", "task2", f"activations_{model_name}.png"),
        )

    visualization_utils.print_results_table(results, {"rf": "Рецептивное поле"})


# if __name__ == "__main__":
#     compare_kernels(epochs=10)

# --------------------------------------------------------------------------------------------------
# Задание 2.2


def compare_depths(epochs=10):
    train_loader, test_loader = get_cifar_loaders()
    in_channels = 3
    num_classes = 10

    models = {
        "conv_2": lambda: ShallowCNN(in_channels, num_classes),
        "conv_4": lambda: MediumCNN(in_channels, num_classes),
        "conv_6": lambda: DeepCNN(in_channels, num_classes),
        "residual": lambda: CNNWithResidual(in_channels, num_classes),
    }

    results = []
    for name, model_fn in models.items():
        model = model_fn()
        result = training_utils.train_analyze(model, name, train_loader, test_loader, epochs=epochs)
        results.append(result)

    visualization_utils.visualize_results_depth_comparison(
        results,
        test_loader,
        save_folder=os.path.join(ROOT, "results", "architecture_analysis"),
    )

    # Вывод финальных метрик
    print("\nСравнение моделей:")
    print(
        f"{'Model':<20} | {'Test Acc':<10} | {'Time (s)':<10} | {'Grad Mean':<10} | {'Grad Min/Max'}"
    )
    print("-" * 70)
    for res in results:
        grad_mean = np.mean(res["gradient_stats"]["mean"])
        grad_min = np.mean(res["gradient_stats"]["min"])
        grad_max = np.mean(res["gradient_stats"]["max"])
        print(
            f"{res['name']:<20} | {max(res['test_accs']):.4f}     | "
            f"{res['time']:.2f}      | {grad_mean:.2e}    | "
            f"{grad_min:.2e}/{grad_max:.2e}"
        )


if __name__ == "__main__":
    compare_depths(epochs=2)
