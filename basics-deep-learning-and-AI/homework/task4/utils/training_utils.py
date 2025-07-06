import gc
import os
import time

import numpy as np
import torch
from tqdm import tqdm

from ...task3.utils.experiment_utils import run_epoch, save_model, load_model


def train_analyze(model, model_name, train_loader, test_loader, epochs=10, save_folder=None):
    """Функция для обучения и анализа градиентов"""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Для анализа градиентов
    gradient_stats = {"mean": [], "std": [], "max": [], "min": []}

    train_losses = []
    test_accs = []
    start_time = time.time()

    gc.collect()
    torch.cuda.empty_cache()

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for inputs, labels in tqdm(train_loader, desc=f"Training {model_name}"):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            # Анализ градиентов
            grad_means = []
            grad_stds = []
            grad_maxs = []
            grad_mins = []

            for param_name, param in model.named_parameters():
                if "weight" in param_name and param.grad is not None:
                    grad = param.grad.abs().cpu().numpy()
                    grad_means.append(np.mean(grad))
                    grad_stds.append(np.std(grad))
                    grad_maxs.append(np.max(grad))
                    grad_mins.append(np.min(grad[grad > 0]))

            gradient_stats["mean"].append(np.mean(grad_means))
            gradient_stats["std"].append(np.mean(grad_stds))
            gradient_stats["max"].append(np.mean(grad_maxs))
            gradient_stats["min"].append(np.mean(grad_mins))

            optimizer.step()
            running_loss += loss.item()

        # Оценка на тестовом наборе
        test_acc = evaluate(model, test_loader, device)
        train_loss = running_loss / len(train_loader)

        train_losses.append(train_loss)
        test_accs.append(test_acc)

        if save_folder:
            save_model(
                model,
                os.path.join(save_folder, f"epoch_{epoch+1}.pt"),
                train_loss=train_loss,
                test_acc=test_acc,
                optimizer=optimizer.state_dict(),
            )

        print(
            f"{model_name} | Epoch {epoch+1}/{epochs} | "
            f"Loss: {train_loss:.4f} | Test Acc: {test_acc:.4f}"
        )

        del inputs, labels, outputs
        gc.collect()
        torch.cuda.empty_cache()

    training_time = time.time() - start_time

    return {
        "name": model_name,
        "train_losses": train_losses,
        "test_accs": test_accs,
        "time": training_time,
        "gradient_stats": gradient_stats,
        "model": model,
    }


def evaluate(model, testloader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in tqdm(testloader, desc="Evaluating"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return correct / total


class TrainResult:
    def __init__(self, model, name):
        """Args:
        model: прежде чем передавать model, перенесите на его device"""
        self.model = model
        self.name = name
        self.train_losses = []
        self.train_accs = []
        self.test_losses = []
        self.test_accs = []
        self.train_time = 0
        self.min_inference_time = float("inf")

    def record_inference_time(self, time):
        self.min_inference_time = min(time, self.min)

    def state_dict(self):
        return {
            "name": self.name,
            "train_losses": self.train_losses,
            "train_accs": self.train_accs,
            "test_losses": self.test_losses,
            "test_accs": self.test_accs,
            "train_time": self.train_time,
            "min_inference_time": self.min_inference_time,
        }

    def load_state_dict(self, state_dict):
        self.name = state_dict["name"]
        self.train_losses = state_dict["train_losses"]
        self.train_accs = state_dict["train_accs"]
        self.test_losses = state_dict["test_losses"]
        self.test_accs = state_dict["test_accs"]
        self.train_time = state_dict["train_time"]
        self.min_inference_time = state_dict["min_inference_time"]

    def record_train(self, train_loss, train_acc, train_time):
        self.train_losses.append(train_loss)
        self.train_accs.append(train_acc)
        self.train_time += train_time

    def record_eval(self, test_loss, test_acc, eval_time):
        self.test_losses.append(test_loss)
        self.test_accs.append(test_acc)
        self.min_inference_time = min(eval_time, self.min_inference_time)


def get_last_save_path(save_folder: str | None) -> str | None:
    last_model_path = None
    if save_folder and os.path.exists(save_folder):
        saved_files = [
            f for f in os.listdir(save_folder) if f.startswith("epoch_") and f.endswith(".pt")
        ]
        if saved_files:
            # Извлекаем номера эпох из имен файлов
            epochs_numbers = [int(f.split("_")[1].split(".")[0]) for f in saved_files]
            last_epoch = max(epochs_numbers)
            last_model_path = os.path.join(save_folder, f"epoch_{last_epoch}.pt")
    return last_model_path


def train_model_v2(
    model,
    model_name,
    train_loader,
    test_loader,
    end_epoch=10,
    lr=0.001,
    patience=5,
    weight_decay=0,
    save_folder=None,
    clean_memory=False,
) -> TrainResult:
    """
    Тренирует модель и сохраняет результаты в TrainResult.
    Если модель уже обучена, то загружает последнюю сохраненную модель и продолжает обучение до end_epoch.
    Применяет Early stopping.
    Сохраняет модель каждую эпоху в save_folder.
    Чистит кэш GPU и Python при каждом вызове функции, если clean_memory=True.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    result = TrainResult(model, model_name)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_loss = float("inf")
    patience_counter = 0
    start_epoch = 0

    # Заргузка последней модели по пути save_folder, если существует
    last_model_path = get_last_save_path(save_folder)
    if last_model_path:
        print(f"Loading model from {last_model_path}")

        state_dict = load_model(model, last_model_path)
        start_epoch = state_dict.get("epoch", start_epoch)

        if "result" in state_dict:
            result.load_state_dict(state_dict["result"])
        if "optimizer" in state_dict:
            optimizer = optimizer.load_state_dict(state_dict["optimizer"])

    for epoch in range(start_epoch, end_epoch):
        print(f"Epoch {epoch+1}/{end_epoch}:")

        start_epoch_time = time.time()
        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, optimizer, device, is_test=False
        )
        train_time = time.time() - start_epoch_time
        result.record_train(train_loss, train_acc, train_time)

        # evaluate
        inference_start_time = time.time()
        test_loss, test_acc = run_epoch(model, test_loader, criterion, None, device, is_test=True)
        inference_time = time.time() - inference_start_time
        result.record_eval(test_loss, test_acc, inference_time)

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
        print("-" * 100)

        if save_folder:  # Сохранение модели
            save_model(
                model,
                os.path.join(save_folder, f"epoch_{epoch + 1}.pt"),
                result=result,
                optimizer=optimizer.state_dict(),
                epoch=epoch,
            )

        if clean_memory:
            torch.cuda.empty_cache()  # Очистка кэша CUDA
            gc.collect()  # Запуск сборщика мусора Python

        # Early stopping
        if test_loss < best_loss:
            best_loss = test_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    return result
