import gc
import os
import time

import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
    precision_score,
    recall_score,
    f1_score,
)


from ...homework.task4.utils.training_utils import get_last_save_path
from ...homework.task3.utils.experiment_utils import save_model, load_model
from ...homework.task4.utils import training_utils


class TrainResult(training_utils.TrainResult):
    def __init__(self, model, name):
        super().__init__(model, name)
        self.best_loss = float("inf")
        self.best_epoch = None

    def state_dict(self):
        return {
            "name": self.name,
            "train_losses": self.train_losses,
            "test_losses": self.test_losses,
            "train_time": self.train_time,
            "min_inference_time": self.min_inference_time,
            "best_loss": self.best_loss,
            "best_epoch": self.best_epoch,
        }

    def load_state_dict(self, state_dict: dict):
        self.name = state_dict.get("name")
        self.train_losses = state_dict.get("train_losses", [])
        self.test_losses = state_dict.get("test_losses", [])
        self.train_time = state_dict.get("train_time", 0)
        self.min_inference_time = state_dict.get("min_inference_time", float("inf"))
        self.best_loss = state_dict.get("best_loss", float("inf"))
        self.best_epoch = state_dict.get("best_epoch", None)

    def record_train(self, loss, *, train_time):
        self.train_losses.append(loss)
        self.train_time += train_time

    def record_eval(self, loss, *, inference_time):
        self.test_losses.append(loss)
        self.min_inference_time = min(self.min_inference_time, inference_time)

        if self.best_loss > loss:
            self.best_loss = loss
            self.best_epoch = len(self.test_losses)

    def print_losses(self):
        print(f"Train Loss: {self.train_losses[-1]:.4f}")
        print(f"Test Loss: {self.test_losses[-1]:.4f}")
        print("-" * 100)

    def early_stop(self) -> bool:
        """Проверяет нужно ли останавливать обучение"""
        if self.test_losses:
            return self.test_losses[-1] > self.best_loss
        return False


def calculate_metrics(y_true, y_pred):
    # Для общего счета
    mae = mean_absolute_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)

    # Округляем предсказания
    y_pred = np.round(y_pred).clip(0)

    # Определяем исход матча
    true_outcomes = np.sign(y_true[:, 0] - y_true[:, 1])
    pred_outcomes = np.sign(y_pred[:, 0] - y_pred[:, 1])

    # Доступные классы в данных
    classes = np.unique(true_outcomes)  # [-1, 0, 1]

    # Рассчитываем precision и recall для каждого класса
    precision = precision_score(
        true_outcomes, pred_outcomes, labels=classes, average=None, zero_division=0
    )
    recall = recall_score(
        true_outcomes, pred_outcomes, labels=classes, average=None, zero_division=0
    )

    f1 = f1_score(true_outcomes, pred_outcomes, labels=classes, average=None)

    # Создаем словари с метриками для каждого класса
    precision_dict = {}
    recall_dict = {}
    f1_dict = {}

    for i, cls in enumerate(classes):
        label = "home wins" if cls == 1 else "draw" if cls == 0 else "away wins"
        precision_dict[label] = precision[i]
        recall_dict[label] = recall[i]
        f1_dict[label] = f1[i]

    return {
        "MAE": mae,
        "RMSE": rmse,
        "Precision": precision_dict,
        "Recall": recall_dict,
        "True outcomes": _to_dict(true_outcomes),
        "Predicted outcomes": _to_dict(pred_outcomes),
        "F1": f1_dict,
    }


def _to_dict(outcomes):
    return {
        "home wins": np.sum(outcomes == 1),
        "draw": np.sum(outcomes == 0),
        "away wins": np.sum(outcomes == -1),
    }


def run_epoch(
    model,
    data_loader,
    criterion,
    optimizer=None,
    device="cpu",
    is_test=False,
):
    if is_test:
        model.eval()
    else:
        model.train()

    epoch_loss = 0

    for batch_idx, (data_dict, target) in enumerate(
        tqdm(data_loader, desc="Training" if not is_test else "Testing")
    ):
        data_dict, target = {k: v.to(device) for k, v in data_dict.items()}, target.to(device)

        if not is_test and optimizer is not None:
            optimizer.zero_grad()

        output = model(data_dict)
        loss = criterion(output, target)

        if not is_test and optimizer is not None:
            loss.backward()
            optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(data_loader)


def load_last_save(
    model, result: TrainResult, optimizer, scheduler, save_folder: str, delete_old_weights=False
) -> dict | None:
    """Загружает последнюю модель если существует
    Returns: state_dict"""
    last_model_path = get_last_save_path(save_folder)
    if last_model_path:
        print(f"Loading model from {last_model_path}")
        try:
            state_dict = load_model(model, last_model_path)
        except RuntimeError as e:
            print("Error loading model:", e)

            if delete_old_weights:
                print("Deleting old weights...")

                # Удаление старых весов
                for file in os.listdir(save_folder):
                    if file.endswith(".pt"):
                        os.remove(os.path.join(save_folder, file))
        else:
            if "result" in state_dict and result:
                result.load_state_dict(state_dict["result"])
            if "optimizer" in state_dict and optimizer:
                optimizer.load_state_dict(state_dict["optimizer"])
            if "scheduler" in state_dict and scheduler:
                scheduler.load_state_dict(state_dict["scheduler"])
            return state_dict
    return None


def train_model(
    model,
    train_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    end_epoch=10,
    lr=0.001,
    patience=5,
    save_folder=None,
    delete_old_weights=False,
    clean_memory=False,
) -> TrainResult:
    """
    Тренирует модель и сохраняет результаты в TrainResult.
    Если модель уже обучена, то загружает последнюю сохраненную модель и продолжает обучение до end_epoch.

    Args:
        end_epoch: продолжает обучение до указанного количества эпох
        patience: Early stopping с указанным количеством эпох без улучшения
        delete_old_weights: Удалить старые веса модели, если их невозможно загрузить
        clean_memory: Чистит кэш GPU и Python на каждой эпохе
        save_folder: Сохраняет модель каждую эпоху в указанной папке
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    result = TrainResult(model, None)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=5)
    criterion = torch.nn.HuberLoss(delta=0.5)  # Менее чувствителен к выбросам

    patience_counter = 0
    start_epoch = 0

    # Загрузка сохранения
    state_dict = load_last_save(
        model,
        result,
        optimizer,
        scheduler=scheduler,
        save_folder=save_folder,
        delete_old_weights=delete_old_weights,
    )
    if state_dict:
        start_epoch = state_dict.get("epoch", start_epoch)
        patience_counter = state_dict.get("patience", patience_counter)

    for epoch in range(start_epoch, end_epoch):
        if not result.early_stop():  # Early stopping
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

        print(f"Epoch {epoch+1}/{end_epoch}:")

        # train
        start_epoch_time = time.time()
        train_loss = run_epoch(model, train_loader, criterion, optimizer, device, is_test=False)
        train_time = time.time() - start_epoch_time

        # evaluate
        inference_start_time = time.time()
        test_loss = run_epoch(
            model, test_loader, criterion, optimizer=None, device=device, is_test=True
        )
        inference_time = time.time() - inference_start_time
        scheduler.step(test_loss)

        result.record_train(train_loss, train_time=train_time)
        result.record_eval(test_loss, inference_time=inference_time)

        if save_folder:  # Сохранение модели
            save_model(
                model,
                os.path.join(save_folder, f"epoch_{epoch + 1}.pt"),
                result=result.state_dict(),
                optimizer=optimizer.state_dict(),
                scheduler=scheduler.state_dict(),
                epoch=epoch,
                patience=patience,
            )

        if clean_memory:
            torch.cuda.empty_cache()  # Очистка кэша CUDA
            gc.collect()  # Запуск сборщика мусора Python

        result.print_losses()

    return result
