import torch

from utils.datasets_utils import get_mnist_loaders
from utils.model_utils import FullyConnectedModel
from utils.experiment_utils import train_model
from utils.visualization_utils import plot_results, plot_training_history

# --------------------------------------------------------------------------------------------------
# Задание 1.1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

train_loader, test_loader = get_mnist_loaders()
epochs = 15


def compare_depths(depths=[1, 2, 3, 5, 7]):
    """Сравнивает модели разной глубины"""
    results = {}

    for depth in depths:
        print(f"\nTraining model with {depth} layers\n")

        # Сборка модели
        layers = []
        for i in range(depth - 1):
            layers.append({"type": "linear", "size": 128})
            layers.append({"type": "relu"})

        model = FullyConnectedModel(
            input_size=784,
            num_classes=10,
            layers=layers,
        )

        # Обучение
        history = train_model(model, train_loader, test_loader, epochs=epochs, device=str(device))
        results[depth] = history

    plot_results(
        {f"depth_{depth}": history for depth, history in results.items()},
        save_folder="./basics-deep-learning-and-AI/homework/task3/plots/",
    )

    # Таблица с результатами
    print("\nСравнение моделей:")
    print(f"{'Layers':<10} | {'Train Acc':<10} | {'Test Acc':<10} | {'Time (s)':<10}")
    print("-" * 45)

    for depth in depths:
        train_acc = results[depth]["train_accs"][-1]
        test_acc = results[depth]["test_accs"][-1]
        time_taken = results[depth]["train_time"]
        print(f"{depth:<10} | {train_acc:.4f}     | {test_acc:.4f}     | {time_taken:.2f}")


# --------------------------------------------------------------------------------------------------
# Задание 1.2


def train_model_with_batch_norm_dropout():
    layers_count = 7
    layers = []

    for i in range(layers_count - 1):
        layers.append({"type": "linear", "size": 128})
        layers.append({"type": "batch_norm"})
        layers.append({"type": "relu"})
        layers.append({"type": "dropout", "rate": 0.25})

    model = FullyConnectedModel(
        input_size=784,
        num_classes=10,
        layers=layers,
    )

    history = train_model(model, train_loader, test_loader, epochs=epochs)
    plot_training_history(
        history,
        save_path="./basics-deep-learning-and-AI/homework/task3/plots/depth_7_norm_dropout.png",
    )
    print(f"Время обучения: {history['train_time']} секунд")
