import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


# --------------------------------------------------------------------------------------------------
# Задание 1
class LinearRegression(torch.nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, 1)
        self.l1 = 0.01  # Коэффициент L1
        self.l2 = 0.01  # Коэффициент L2

    def forward(self, x):
        return self.linear(x)

    def regularization_loss(self):
        l1_loss = sum(p.abs().sum() for p in self.parameters())  # Модель штрафуется за сумму весов
        l2_loss = sum(
            p.pow(2).sum() for p in self.parameters()  # Модель штрафуется за сумму квадратов весов
        )
        return self.l1 * l1_loss + self.l2 * l2_loss


class RegressionDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def train_model(model, loader, criterion, optimizer, epochs=1000, patience=10):
    """Обучение модели с использованием Early Stopping."""
    best_loss = float("inf")
    patience_counter = 0
    losses = []

    for epoch in range(epochs):
        total_epoch_loss = 0

        for i, (batch_X, batch_y) in enumerate(loader):
            optimizer.zero_grad()
            y_pred = model(batch_X)

            # loss = MSE + регуляризация
            loss = criterion(y_pred, batch_y)
            reg_loss = model.regularization_loss()
            total_loss = loss + reg_loss

            total_loss.backward()
            optimizer.step()

            total_epoch_loss += total_loss.item()

        avg_epoch_loss = total_epoch_loss / len(loader)
        losses.append(avg_epoch_loss)

        # Early stopping
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {avg_epoch_loss:.4f}")

    plt.plot(losses)
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    os.makedirs(
        "./basics-deep-learning-and-AI/homework/task2/plots", exist_ok=True
    )  # Сохранение графика
    plt.savefig(
        f"./basics-deep-learning-and-AI/homework/task2/plots/{model.__class__.__name__}_loss.png"
    )
    plt.show()


def train_linear_regression():
    """Тренирует модель линейной регрессии"""
    from sklearn.datasets import make_regression

    # Генерация данных
    X, y = make_regression(n_samples=1000, n_features=20, noise=0.1, random_state=42)
    X = torch.FloatTensor(X)
    y = torch.FloatTensor(y).view(-1, 1)

    dataset = RegressionDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = LinearRegression(X.shape[1])
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    train_model(model, dataloader, criterion, optimizer)


# --------------------------------------------------------------------------------------------------
# Задание 2
class LogisticRegression(torch.nn.Module):
    """Многоклассовая логистическая регрессия"""

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)


def train_logistic_regression(model, X_train, y_train, epochs=1000):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")


def calculate_metrics(y_true, y_pred, num_classes):
    """Вычисляет метрики precision, recall и F1, confusion matrix"""
    confusion = torch.zeros(num_classes, num_classes)
    for t, p in zip(y_true, y_pred):
        confusion[t.long(), p.long()] += 1

    # Precision, Recall, F1
    precision = torch.diag(confusion) / (confusion.sum(0) + 1e-8)
    recall = torch.diag(confusion) / (confusion.sum(1) + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    macro_precision = precision.mean().item()
    macro_recall = recall.mean().item()
    macro_f1 = f1.mean().item()

    return macro_precision, macro_recall, macro_f1, confusion


def predict_logistic_regression(model, X_test, y_test):
    """Оценка модели"""
    with torch.no_grad():
        # Предсказания для тестового набора
        outputs = model(X_test)
        _, predicted = torch.max(outputs.data, 1)

        # Конвертируем тензоры в numpy массивы для sklearn метрик
        num_classes = len(torch.unique(y_test))
        precision, recall, f1, cm = calculate_metrics(y_test, predicted, num_classes)

        # Метрики
        print("\nClassification Metrics:\n")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print(f"Confusion Matrix:\n{cm}")


# --------------------------------------------------------------------------------------------------
# Обучение
if __name__ == "__main__":
    train_linear_regression()


if __name__ == "__main__":
    X, y = make_classification(
        n_samples=1000, n_features=20, n_classes=2, n_informative=10, random_state=42
    )
    # Разделение на train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train, X_test = torch.FloatTensor(X_train), torch.FloatTensor(X_test)
    y_train, y_test = torch.LongTensor(y_train), torch.LongTensor(y_test)

    model = LogisticRegression(X_train.shape[1], len(torch.unique(y_train)))

    train_logistic_regression(model, X_train, y_train)  # Обучение модели
    predict_logistic_regression(model, X_test, y_test)  # Оценка модели
