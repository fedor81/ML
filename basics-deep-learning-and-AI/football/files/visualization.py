import os
from pprint import pprint
from prettytable import PrettyTable
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np
import torch

from ...homework.task4.utils.visualization_utils import makedirs_if_not_exists
from ...homework.task3.utils.visualization_utils import count_parameters
from . import train_utils


ROOT_DIR = "./basics-deep-learning-and-AI/football/"


def show_table(result: train_utils.TrainResult, model):
    table = PrettyTable(
        ["Best Epoch", "Train Loss", "Test Loss", "Train Time", "Inference Time", "Parameters"]
    )
    table.add_row(
        [
            result.best_epoch,
            min(result.train_losses),
            min(result.test_losses),
            result.train_time,
            result.min_inference_time,
            count_parameters(model),
        ]
    )
    print(table)


def plot(result):
    """Показывает график loss"""
    plt.figure()
    plt.plot(result.train_losses, label="Train Loss")
    plt.plot(result.test_losses, label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    save_path = os.path.join(ROOT_DIR, "plots", "loss.png")
    makedirs_if_not_exists(save_path)
    plt.savefig(save_path)
    plt.show()


def plot_outcome_matrix(y_true, y_pred):
    outcomes_true = np.sign(y_true[:, 0] - y_true[:, 1])
    outcomes_pred = np.sign(y_pred[:, 0] - y_pred[:, 1])

    cm = confusion_matrix(outcomes_true, outcomes_pred, labels=[1, 0, -1])

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        xticklabels=["Home Wins", "Ничья", "Away Wins"],
        yticklabels=["Home Wins", "Ничья", "Away Wins"],
    )
    plt.xlabel("Предсказание")
    plt.ylabel("Истина")


def show_metrics(model, data_loader: torch.utils.data.DataLoader, device="cpu", save_folder=None):
    model.eval()
    y_pred = []
    y_true = []

    with torch.no_grad():  # отключаем вычисление градиентов
        for inputs, labels in data_loader:
            predicted = model(inputs)

            y_pred.extend(predicted.cpu().numpy())
            y_true.extend(labels.cpu().numpy())

    y_pred = np.array(y_pred)
    y_true = np.array(y_true)

    metrics = train_utils.calculate_metrics(y_true, y_pred)
    pprint(metrics)

    plot_predictions(y_true, y_pred, save_folder=save_folder)  # Визуализация предсказаний
    plot_outcome_matrix(y_true, y_pred, save_folder=save_folder)
    plot_enhanced_predictions(y_true, y_pred, save_folder=save_folder)
    plt.show()


def _to_list(outcomes: dict):
    return [outcomes["home wins"], outcomes["draw"], outcomes["away wins"]]


def plot_outcome_matrix(y_true, y_pred, save_folder=None):
    # Округляем предсказания
    y_pred = np.round(y_pred).clip(0)

    # Преобразуем в разницу голов
    diff_true = y_true[:, 0] - y_true[:, 1]
    diff_pred = y_pred[:, 0] - y_pred[:, 1]

    # Определяем классы
    true_outcomes = np.sign(diff_true)
    pred_outcomes = np.sign(diff_pred)

    # Находим реально присутствующие классы
    present_labels = np.unique(true_outcomes)
    labels = [1, 0, -1]  # Победа хозяев, ничья, победа гостей

    # Фильтруем только присутствующие метки
    valid_labels = [label for label in labels if label in present_labels]

    # Строим матрицу только для существующих классов
    cm = confusion_matrix(true_outcomes, pred_outcomes, labels=valid_labels)

    # Создаем подписи
    label_names = ["Home Wins", "Draw", "Away Wins"]

    # Визуализация
    plt.figure(figsize=(6, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", xticklabels=label_names, yticklabels=label_names, cmap="Blues"
    )
    plt.xlabel("Предсказание")
    plt.ylabel("Истина")
    plt.title("Confusion Matrix")
    if save_folder:
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        plt.savefig(os.path.join(save_folder, "confusion_matrix.png"))


def plot_predictions(y_true, y_pred, save_folder=None):
    plt.figure()
    plt.scatter(y_true[:, 0], y_pred[:, 0], alpha=0.3, label="Home")
    plt.scatter(y_true[:, 1], y_pred[:, 1], alpha=0.3, label="Away")
    plt.plot([0, 6], [0, 6], "r--")
    plt.xlabel("Реальный счет")
    plt.ylabel("Предсказанный счет")
    plt.legend()
    plt.title("Распределение предсказаний и реальных счетов")
    if save_folder:
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        plt.savefig(os.path.join(save_folder, "predictions.png"))


def plot_enhanced_predictions(y_true, y_pred, save_folder=None):
    plt.figure(figsize=(12, 10))

    # Основной scatter plot с плотностью точек
    plt.subplot(2, 2, (1, 3))
    hb = plt.hexbin(
        x=np.concatenate([y_true[:, 0], y_true[:, 1]]),
        y=np.concatenate([y_pred[:, 0], y_pred[:, 1]]),
        gridsize=20,
        cmap="viridis",
        norm=LogNorm(),
        mincnt=1,
    )
    plt.colorbar(hb, label="Количество матчей")
    plt.plot([0, 6], [0, 6], "r--", alpha=0.5)
    plt.xlabel("Реальный счет")
    plt.ylabel("Предсказанный счет")
    plt.title("Плотность предсказаний (логарифмическая шкала)")
    plt.xlim(-0.5, 6.5)
    plt.ylim(-0.5, 6.5)

    # Гистограмма ошибок для домашней команды
    plt.subplot(2, 2, 2)
    errors_home = y_pred[:, 0] - y_true[:, 0]
    sns.histplot(errors_home, bins=20, kde=True, color="blue", alpha=0.5)
    plt.axvline(x=0, color="r", linestyle="--")
    plt.xlabel("Ошибка предсказания")
    plt.title("Распределение ошибок (дома)")

    # Гистограмма ошибок для гостевой команды
    plt.subplot(2, 2, 4)
    errors_away = y_pred[:, 1] - y_true[:, 1]
    sns.histplot(errors_away, bins=20, kde=True, color="orange", alpha=0.5)
    plt.axvline(x=0, color="r", linestyle="--")
    plt.xlabel("Ошибка предсказания")
    plt.title("Распределение ошибок (гости)")

    plt.tight_layout()

    # Добавляем текстовую статистику
    stats_text = (
        f"Общая статистика (n={len(y_true)}):\n"
        f"MAE домашние: {np.mean(np.abs(errors_home)):.2f}\n"
        f"MAE гости: {np.mean(np.abs(errors_away)):.2f}\n"
        f"Точность исхода: {np.mean(np.sign(y_true[:,0]-y_true[:,1]) == np.sign(y_pred[:,0]-y_pred[:,1])):.1%}"
    )
    plt.gcf().text(0.5, 0.02, stats_text, ha="center", bbox=dict(facecolor="white", alpha=0.8))

    if save_folder:
        os.makedirs(save_folder, exist_ok=True)
        plt.savefig(
            os.path.join(save_folder, "enhanced_predictions.png"), dpi=300, bbox_inches="tight"
        )
