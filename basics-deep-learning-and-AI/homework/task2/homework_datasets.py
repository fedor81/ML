import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import DataLoader, Dataset

from homework_model_modification import LinearRegression, train_model


# --------------------------------------------------------------------------------------------------
# Задание 1
class CSVDataset(Dataset):
    def __init__(
        self,
        file_path: str,
        target_column: str,
        numeric_cols: list[str] = [],
        categorical_cols: list[str] = [],
        binary_cols: list[str] = [],
    ):
        self.data = pd.read_csv(file_path)
        self.target = target_column
        self.numeric_cols = numeric_cols
        self.categorical_cols = categorical_cols
        self.binary_cols = binary_cols
        self._preprocess_data()

    def _preprocess_data(self):
        # Разделение на признаки и таргет
        features = self.data.drop(columns=[self.target])
        self.y = self.data[self.target].values

        # Создание преобразователя для разных типов признаков
        transformers = []

        if self.numeric_cols:
            transformers.append(("num", StandardScaler(), self.numeric_cols))

        if self.categorical_cols:
            transformers.append(("cat", OneHotEncoder(), self.categorical_cols))

        if self.binary_cols:
            # Для бинарных признаков просто преобразуем в 0/1
            for col in self.binary_cols:
                features[col] = features[col].astype(int)

        # Применение преобразований
        if transformers:
            preprocessor = ColumnTransformer(transformers, remainder="passthrough")
            self.X = preprocessor.fit_transform(features)
        else:
            self.X = features.values

        # Преобразование в тензоры
        self.X = torch.FloatTensor(self.X)
        self.y = torch.FloatTensor(self.y) if self.y.dtype == float else torch.LongTensor(self.y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# --------------------------------------------------------------------------------------------------
# Задание 2


if "__main__" == __name__:
    # Обучим линейную регрессию
    dataset = CSVDataset(
        "./basics-deep-learning-and-AI/homework/task2/data/Walmart_Sales.csv",
        target_column="Unemployment",
        binary_cols=["Holiday_Flag"],
    )
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    model = LinearRegression(dataset.X.shape[1])
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    train_model(model, dataloader, criterion, optimizer)

    # Второй датасет для бинарной классификации
    dataset = CSVDataset(
        "./basics-deep-learning-and-AI/homework/task2/data/weather_forecast_data.csv",
        target_column="Rain",
    )
