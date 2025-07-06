import os
from ..task3.utils.datasets_utils import get_cifar_loaders, get_mnist_loaders
from ..task3.utils.model_utils import FullyConnectedModel
from .models.cnn_models import CNNWithResidual, SimpleCNN, RegularizedResidual
from .utils import comparison_utils, visualization_utils

ROOT = "basics-deep-learning-and-AI/homework/task4/"

# --------------------------------------------------------------------------------------------------
# Задание 1.1


def compare_mnist_fn_vs_cnn(epochs=10):
    mnist_train, mnist_test = get_mnist_loaders()
    input_channels = 1
    input_size = 28
    num_classes = 10

    models = {
        "FullyConnected": FullyConnectedModel(
            input_channels * input_size * input_size,
            num_classes,
            layers=[
                {"type": "linear", "size": 128},
                {"type": "relu"},
                {"type": "linear", "size": 128},
                {"type": "relu"},
                {"type": "linear", "size": 128},
            ],
        ),
        "SimpleCNN": SimpleCNN(input_channels, num_classes),
        "ResidualCNN": CNNWithResidual(input_channels, num_classes),
    }
    results = comparison_utils.compare_models(
        models,
        mnist_train,
        mnist_test,
        epochs=epochs,
        models_save_folder=os.path.join(ROOT, "weights/"),
        graphic_save_path=os.path.join(ROOT, "plots/task1/mnist.png"),
    )
    # Вывод таблицы результатов
    visualization_utils.print_results_table(results, {"count_params": "Count Params"})


# --------------------------------------------------------------------------------------------------
# Задание 1.2


def compare_cifar_fn_vs_cnn(epochs=10):
    train_loader, test_loader = get_cifar_loaders()
    input_channels = 3
    input_size = 32
    num_classes = 10

    models = {
        "FullyConnected": FullyConnectedModel(
            input_channels * input_size * input_size,
            num_classes,
            layers=[
                {"type": "linear", "size": 512},
                {"type": "relu"},
                {"type": "linear", "size": 256},
                {"type": "relu"},
                {"type": "linear", "size": 128},
                {"type": "relu"},
                {"type": "linear", "size": 64},
                {"type": "relu"},
            ],
        ),
        "ResidualCNN": CNNWithResidual(input_channels, num_classes),
        "ResidualWithReg": RegularizedResidual(input_channels, num_classes),
    }
    results = comparison_utils.compare_models(
        models,
        train_loader,
        test_loader,
        epochs=epochs,
        models_save_folder=os.path.join(ROOT, "weights/"),
        graphic_save_path=os.path.join(ROOT, "plots/task1/cifar.png"),
    )

    visualization_utils.print_results_table(results, {"count_params": "Count Params"})

    # Вывод градиентов
    # for name, result in results.items():
    #     pprint(result["gradients"])

    visualization_utils.show_confusion_matrix(
        models,
        test_loader,
        classes=("plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"),
        save_path=os.path.join(ROOT, "plots/task1/cifar_confusion.png"),
    )


if __name__ == "__main__":
    compare_cifar_fn_vs_cnn(epochs=10)
