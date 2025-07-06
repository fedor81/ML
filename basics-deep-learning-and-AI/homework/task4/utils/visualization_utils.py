import os
from typing import Sequence
import seaborn as sns
import torch
from matplotlib import pyplot as plt
from prettytable import PrettyTable
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

from .training_utils import TrainResult
from ...task3.utils.visualization_utils import count_parameters


def print_results_table(results: dict[str, dict], extra_columns: dict[str:str]):
    """Выводит таблицу результатов обучения моделей
    Args:
        extra_columns: {'example_param': 'Column Name'}"""
    # Сохраняем порядок ключей, так как словарь его не гарантирует
    extra_keys = tuple(extra_columns.keys())
    columns = [
        "Model",
        "Train Loss",
        "Train Acc",
        "Test Loss",
        "Test Acc",
        "Train Time (s)",
        "Infer Time (s)",
    ]
    columns.extend(extra_columns[key] for key in extra_keys)
    table = PrettyTable(columns)
    table.float_format = ".4"

    for name, history in results.items():
        row = [
            name,
            min(history["train_losses"]),
            max(history["train_accs"]),
            min(history["test_losses"]),
            max(history["test_accs"]),
            history["train_time"],
            history["inference_time"],
        ]
        # Добавляем дополнительные значения
        for key in extra_keys:
            if key in history:
                row.append(history[key])
            else:
                row.append(None)
        table.add_row(row)

    print(table)


class ResultsTable(PrettyTable):
    def __init__(self, **kwargs):
        columns = [
            "Model",
            "Train Loss",
            "Train Acc",
            "Test Loss",
            "Test Acc",
            "Train Time (s)",
            "Infer Time (s)",
            "Count Params",
        ]
        super().__init__(columns, **kwargs)
        self.float_format = ".4"

    def add_row(self, result: TrainResult):
        row = [
            result.name,
            min(result.train_losses),
            max(result.train_accs),
            min(result.test_losses),
            max(result.test_accs),
            result.train_time,
            result.min_inference_time,
            count_parameters(result.model),
        ]
        super().add_row(row)


def makedirs_if_not_exists(path):
    dir_path = os.path.dirname(path)
    if dir_path and not os.path.exists(path):
        os.makedirs(dir_path, exist_ok=True)


def show_confusion_matrix(
    models: dict[str : torch.nn.Module], test_loader, classes: list[str], save_path=None
):
    """Выводит confusion matrix моделей
    Args:
        models: словарь - model_name: model"""

    # Создаем фигуру с несколькими subplots
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 12))
    fig.suptitle("Confusion Matrices Comparison", fontsize=16)

    # Проходим по всем моделям и строим их confusion matrix
    for i, (model_name, model) in enumerate(models.items()):
        row = i // 2
        col = i % 2
        ax = axes[row, col]

        sns.heatmap(
            _get_confusion_matrix(model, test_loader),
            annot=True,
            fmt="d",
            xticklabels=classes,
            yticklabels=classes,
            ax=ax,
        )
        ax.set_title(model_name)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")

    # Если моделей меньше 4, скрываем пустые subplots
    for i in range(len(models), 4):
        row = i // 2
        col = i % 2
        axes[row, col].axis("off")

    plt.tight_layout()

    if save_path:  # Сохранение изображения
        makedirs_if_not_exists(save_path)
        plt.savefig(save_path)
    plt.show()


def _get_confusion_matrix(model, testloader, device="cpu"):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(testloader, desc="Inference"):
            inputs = inputs.to(torch.device(device))
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return confusion_matrix(all_labels, all_preds)


def visualize_activations(model, loader, model_name, save_path=None, device="cpu"):
    """Визуализирует активации первого слоя модели"""
    model = model.to(device)
    sample = next(iter(loader))[0][0:1].to(device)  # Для визуализации
    model.eval()
    activations = []

    def hook(module, input, output):
        activations.append(output.detach())

    handle = model.features[0].register_forward_hook(hook)
    with torch.no_grad():
        model(sample)
    handle.remove()

    num_channels = activations[0].shape[1]
    num_plots = min(16, num_channels)

    plt.figure(figsize=(12, 6))
    for i in range(num_plots):
        plt.subplot(4, 4, i + 1)
        plt.imshow(activations[0][0, i].cpu().numpy())
        plt.axis("off")
    plt.suptitle(f"Активации первого слоя {model_name}")
    if save_path:
        makedirs_if_not_exists(save_path)
        plt.savefig(save_path)
    plt.show()


def visualize_results_depth_comparison(results: list[dict], test_loader, save_folder=None):
    """Выводит графики: точности, градиентов, времени обучения, feature maps"""

    plt.figure(figsize=(18, 12))

    # График точности
    plt.subplot(2, 2, 1)
    for res in results:
        plt.plot(res["test_accs"], label=res["name"])
    plt.title("Test Accuracy by Model Depth")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    # График градиентов
    plt.subplot(2, 2, 2)
    for res in results:
        plt.plot(res["gradient_stats"]["mean"], label=res["name"])
    plt.title("Average Gradient Magnitude")
    plt.xlabel("Iteration")
    plt.ylabel("Gradient Mean")
    plt.yscale("log")
    plt.legend()

    # Сравнение времени обучения
    plt.subplot(2, 2, 3)
    times = [res["time"] for res in results]
    names = [res["name"] for res in results]
    plt.bar(names, times)
    plt.title("Training Time Comparison")
    plt.ylabel("Time (seconds)")
    plt.xticks(rotation=45)
    if save_folder:
        makedirs_if_not_exists(save_folder)
        plt.savefig(os.path.join(save_folder, "results_depth_comparison.png"))

    # Визуализация feature maps
    plt.subplot(2, 2, 4)
    sample_img = next(iter(test_loader))[0][0]
    for res in results:
        if "conv_6" in res["name"]:
            model = res["model"].cpu()
            model.eval()
            _ = model(sample_img.unsqueeze(0))
            fmaps = model.feature_maps[0][0]  # Первый слой

            # Показываем первые 8 карт признаков
            fig, axes = plt.subplots(1, 8, figsize=(15, 2))
            for i in range(8):
                axes[i].imshow(fmaps[i].numpy(), cmap="viridis")
                axes[i].axis("off")
            plt.suptitle(f'Feature Maps - {res["name"]} (First Layer)')
            if save_folder:
                makedirs_if_not_exists(save_folder)
                plt.savefig(os.path.join(save_folder, "feature_maps_" + res["name"] + ".png"))
            plt.tight_layout()
            plt.show()
            break

    plt.tight_layout()
    plt.show()
