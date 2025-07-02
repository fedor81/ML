import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split

# --------------------------------------------------------------------------------------------------
# Задание 1

# Подготовка данных
california = fetch_california_housing()
X = StandardScaler().fit_transform(california.data)
y = california.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)
y_train = torch.FloatTensor(y_train).view(-1, 1)
y_test = torch.FloatTensor(y_test).view(-1, 1)


# Модель для испытаний
class LinearRegression(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x)


def train_with_params(model, criterion, optimizer, train_loader, epochs=100, patience=10):
    """Функция для обучения с разными параметрами"""
    best_loss = float("inf")
    patience_counter = 0
    train_loss = []
    model.train()

    for epoch in range(epochs):
        epoch_loss = 0

        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        train_loss.append(avg_loss)

        # Early stopping
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}, best loss: {best_loss}")
                break
    return train_loss


# Результаты экспериментов
results = []

# Варианты параметров
learning_rates = [0.001, 0.01, 0.1]
batch_sizes = [16, 32, 64]
optimizers = {"SGD": optim.SGD, "Adam": optim.Adam, "RMSprop": optim.RMSprop}

# Проведение экспериментов
for lr in learning_rates:
    for batch_size in batch_sizes:
        for opt_name, opt_fn in optimizers.items():
            print(f"Эксперимент с LR={lr}, BS={batch_size}, Opt={opt_name}")

            # Создание DataLoader
            train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

            # Инициализация модели и оптимизатора
            model = LinearRegression(X_train.shape[1])
            criterion = nn.MSELoss()
            optimizer = opt_fn(model.parameters(), lr=lr)

            # Обучение
            losses = train_with_params(model, criterion, optimizer, train_loader)

            # Оценка на тестовом наборе
            model.eval()
            with torch.no_grad():
                y_pred = model(X_test)
                test_loss = criterion(y_pred, y_test).item()

            # Сохранение результатов
            results.append(
                {
                    "Learning Rate": lr,
                    "Batch Size": batch_size,
                    "Optimizer": opt_name,
                    "Final Train Loss": losses[-1],
                    "Test Loss": test_loss,
                    "Loss History": losses,
                }
            )

# Визуализация результатов

# Создаем DataFrame для анализа
df_results = pd.DataFrame(results)

# График 1: Сравнение оптимизаторов
plt.figure(figsize=(12, 6))

for opt_name in optimizers.keys():
    subset = df_results[df_results["Optimizer"] == opt_name]
    plt.plot(subset["Learning Rate"], subset["Test Loss"], "o-", label=opt_name)

plt.xscale("log")
plt.xlabel("Learning Rate (log scale)")
plt.ylabel("Test Loss")
plt.title("Сравнение оптимизаторов при разных LR")
plt.legend()
plt.grid(True)
plt.savefig("./basics-deep-learning-and-AI/homework/task2/plots/experiments_optimizers.png")
plt.show()

# График 2: Влияние размера батча
plt.figure(figsize=(12, 6))

for batch in batch_sizes:
    subset = df_results[df_results["Batch Size"] == batch]
    plt.plot(subset["Learning Rate"], subset["Test Loss"], "o-", label=f"Batch {batch}")

plt.xscale("log")
plt.xlabel("Learning Rate (log scale)")
plt.ylabel("Test Loss")
plt.title("Влияние размера батча при разных LR")
plt.legend()
plt.grid(True)
plt.savefig("./basics-deep-learning-and-AI/homework/task2/plots/experiments_batch_size.png")
plt.show()

# График 3: Тепловая карта результатов
pivot_table = df_results.pivot_table(
    index=["Batch Size", "Learning Rate"], columns="Optimizer", values="Test Loss"
)

plt.figure(figsize=(10, 8))
sns.heatmap(pivot_table, annot=True, fmt=".2f", cmap="YlGnBu")
plt.title("Тестовая ошибка по комбинациям параметров")
plt.savefig("./basics-deep-learning-and-AI/homework/task2/plots/experiments_heatmap.png")
plt.show()

# График 4: лосс для лучших комбинаций
best_combinations = df_results.sort_values("Test Loss").head(3)

plt.figure(figsize=(12, 6))

for i, (_, row) in enumerate(best_combinations.iterrows()):
    plt.plot(
        row["Loss History"],
        label=f"LR={row['Learning Rate']}, BS={row['Batch Size']}, Opt={row['Optimizer']}",
    )

plt.xlabel("Epoch")
plt.ylabel("Train Loss")
plt.title("Кривые обучения лучших комбинаций")
plt.legend()
plt.grid(True)
plt.savefig("./basics-deep-learning-and-AI/homework/task2/plots/experiments_best_combinations.png")
plt.show()

# Таблица с результатами
print("\nТоп-5 комбинаций параметров:")
print(
    df_results.sort_values("Test Loss")
    .head(5)[["Learning Rate", "Batch Size", "Optimizer", "Test Loss"]]
    .to_string(index=False)
)


# --------------------------------------------------------------------------------------------------
# Задание 2

# Не сделано
