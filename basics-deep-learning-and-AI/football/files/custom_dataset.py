import hashlib
import inspect
import os

import emoji
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from ...homework.task4.utils.visualization_utils import makedirs_if_not_exists
from .dataset_encode import encode_columns
from .dataset_load import load_fill_na
from .train import ROOT_DIR


class SimpleDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class FootballDataset(Dataset):
    class_hash = None

    def __init__(
        self,
        results: pd.DataFrame,
        shootouts: pd.DataFrame,
        goalscorers: pd.DataFrame,
        seq_length: int = 5,
        cache_dir: str = "./data_cache",
        force_rebuild: bool = False,
        is_test: bool = False,
    ):
        """
        Args:
            seq_length: Длина временного окна для истории
            cache_dir: Директория кеша, для избавления от повторных вычислений
            force_rebuild: необходимость пересоздания кеша
        """
        cache_filename = "dataset_train.pt" if not is_test else "dataset_test.pt"
        self.cache_filepath = os.path.join(cache_dir, cache_filename)
        self.class_cache_filepath = os.path.join(cache_dir, "dataset_class_hash.pt")
        self.seq_length = seq_length
        self.num_features = results.shape[1] - 1  # -date
        assert self.num_features > 0, "num_features должно быть положительным!"

        # Получаем хэш текущего кода класса
        self.class_hash = self.class_hash if self.class_hash else self._get_class_code_hash()

        if not force_rebuild:  # and self._cache_exists(cache_dir):
            # Загружаем из кэша
            self._load_from_cache()
        else:
            # Обработка данных
            self.results = results
            self.results_index = results.set_index(["date", "home_team", "away_team"])
            self.shootouts = shootouts.set_index(["date"])
            self.goalscorers = goalscorers.set_index(["date"])

            self.matches, self.targets = self._preprocess_matches()
            self._save_to_cache()

        self._validate()

    def _validate(self):
        """Проверяет корректность данных"""
        assert all(
            m["home_history"].shape == (self.seq_length, self.num_features) for m in self.matches
        ), f"Несоответствие размеров истории: {self.matches[0]["home_history"].shape} != {(self.seq_length, self.num_features)}"

        assert (
            self.matches[0]["home_history"].dtype == torch.float
        ), f"home_history должен быть float, а не {self.matches[0]["home_history"].dtype}"

    @classmethod
    def _get_class_code_hash(cls):
        """Возвращает хэш исходного кода ключевых методов"""
        # TODO: подобные проверки достаточно делать один раз при создании всего класса
        methods = [
            "_preprocess_matches",
            "_get_match_goals",
            "_get_team_history",
            "__len__",
            "__getitem__",
        ]

        source_lines = []
        for method_name in methods:
            method = getattr(cls, method_name)
            source_lines.append(inspect.getsource(method))

        combined_source = "\n".join(source_lines)
        return hashlib.md5(combined_source.encode("utf-8")).hexdigest()

    def _cache_exists(self, dir):
        if not os.path.exists(dir):
            return False
        if not os.path.exists(self.cache_filepath):
            return False

        # Проверяем хэш класса в кэше
        try:
            state = torch.load(self.class_cache_filepath)
            if "class_hash" not in state or state["class_hash"] != self.class_hash:
                print(emoji.emojize(":warning: Class code changed, cache invalidated"))
                return False
            return True
        except:
            return False

    def _load_from_cache(self):
        state = torch.load(self.cache_filepath)
        self.matches = state["matches"]
        self.targets = state["targets"]
        print(emoji.emojize(f"Dataset :thumbs_up: loaded from cache: {self.cache_filepath}"))

    def _save_to_cache(self):
        makedirs_if_not_exists(self.cache_filepath)
        torch.save(
            {"matches": self.matches, "targets": self.targets},
            self.cache_filepath,
        )
        torch.save(  # TODO: После создания train_dataset хэш перезаписывается
            {"class_hash": self.class_hash},
            self.class_cache_filepath,
        )
        print(emoji.emojize(f"Dataset :thumbs_up: cached to: {self.cache_filepath}"))

    def _preprocess_matches(self):
        """Создает список матчей с историей"""
        matches = []
        labels = torch.tensor(self.results[["home_score", "away_score"]].values, dtype=torch.float)

        match_features = torch.tensor(
            self.results.drop(columns=["home_score", "away_score", "date"]).values,
            dtype=torch.long,
        )

        for i, (_, match) in tqdm(
            enumerate(self.results.iterrows()),
            total=self.results.shape[0],
            desc="Preprocess Dataset",
        ):
            goals = self._get_match_goals(match)
            home, away = match["home_team"], match["away_team"]

            matches.append(
                {
                    # TODO: Можно добавить время с начала матча, неполный счет
                    # TODO: Можно добавить командную статистику до match["date"]
                    # TODO: shootouts
                    "match_features": match_features[i],
                    "home_history": self._get_team_history(home, match["date"]),
                    "away_history": self._get_team_history(away, match["date"]),
                    "goals_times": goals["times"],
                    "goals_scorers": goals["scorers"],
                }
            )

        return matches, labels

    def _get_match_goals(self, match):
        """Извлекает данные о голах в матче"""
        try:
            goals = self.goalscorers.loc[(match["date"],)]
            goals = goals[
                (goals["home_team"] == match["home_team"])
                & (goals["away_team"] == match["away_team"])
            ]
        except KeyError:
            goals = pd.DataFrame()

        # TODO: Сделать неограниченное число голов
        count = 10
        times = torch.zeros(count, dtype=torch.float)
        scorers = torch.zeros(count, dtype=torch.long)

        if not goals.empty:
            minutes = goals["minute"].values[:count] / 90.0
            scorer_ids = goals["scorer"].values[:count]
            times[: len(minutes)] = torch.tensor(minutes)
            scorers[: len(scorer_ids)] = torch.tensor(scorer_ids, dtype=torch.int32)

        # TODO: own_goal, penalty
        return {"times": times, "scorers": scorers}

    def _get_team_history(self, team, date):
        mask = (self.results_index.index.get_level_values("home_team") == team) | (
            self.results_index.index.get_level_values("away_team") == team
        )
        mask &= self.results_index.index.get_level_values("date") < date

        history = self.results_index[mask].sort_index(level="date", ascending=False)[
            : self.seq_length
        ]

        # Конвертируем в тензор
        history_tensor = torch.tensor(
            history.reset_index().drop(columns=["date"]).values, dtype=torch.float
        )

        # Паддинг до seq_length
        if len(history_tensor) < self.seq_length:
            pad_size = self.seq_length - len(history_tensor)
            padding = torch.zeros((pad_size, self.num_features), dtype=torch.float)
            history_tensor = torch.cat([history_tensor, padding])

        return history_tensor  # [seq_length, num_features]

    def __len__(self):
        return len(self.matches)

    def __getitem__(self, idx):
        return self.matches[idx], self.targets[idx]


def collate_fn(batch):
    """Кастомная функция для объединения данных разной длины"""
    batch_dict = {
        "match_features": torch.stack([item[0]["match_features"] for item in batch]),
        "home_history": torch.stack([item[0]["home_history"] for item in batch]),
        "away_history": torch.stack([item[0]["away_history"] for item in batch]),
        "goals_times": torch.stack([item[0]["goals_times"] for item in batch]),
        "goals_scorers": torch.stack([item[0]["goals_scorers"] for item in batch]),
    }
    batch_target = torch.stack([item[1] for item in batch])
    return batch_dict, batch_target


def get_train_test_datasets(test_size=0.2, random_state=42):
    data_frames = load_fill_na()
    results, shootouts, goalscorers, _ = encode_columns(*data_frames)

    X = results.drop(columns=["home_score", "away_score", "date"])
    y = results[["home_score", "away_score"]]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return SimpleDataset(X_train, y_train.values), SimpleDataset(X_test, y_test.values)


def get_advanced_datasets(test_size=0.2, seq_length=5, reset_cache=False, random_dates=False):
    data_frames = load_fill_na()
    results, shootouts, goalscorers, _ = encode_columns(*data_frames)

    # Предварительный расчет статистики команд
    # team_stats = calculate_teams_stats(results, shootouts)

    dates = sorted(results["date"].unique())

    if random_dates:
        train_dates, test_dates = train_test_split(dates, test_size=test_size, random_state=42)
    else:
        # Разделение на train/test по возрастанию даты (не случайное)
        split_idx = int(len(dates) * (1 - test_size))
        train_dates = dates[:split_idx]
        test_dates = dates[split_idx:]

    train_results = results[results["date"].isin(train_dates)]
    test_results = results[results["date"].isin(test_dates)]

    # Создание датасетов
    train_ds = FootballDataset(
        train_results,
        shootouts,
        goalscorers,
        seq_length,
        cache_dir=os.path.join(ROOT_DIR, "cache"),
        is_test=False,
        force_rebuild=reset_cache,
    )
    test_ds = FootballDataset(
        test_results,
        shootouts,
        goalscorers,
        seq_length,
        cache_dir=os.path.join(ROOT_DIR, "cache"),
        is_test=True,
        force_rebuild=reset_cache,
    )

    return train_ds, test_ds


def get_train_val_test_datasets(val_size=0.2, test_count=500, seq_length=5, reset_cache=False):
    """В качестве тестовой выборки используются последние матчи"""
    data_frames = load_fill_na()
    results, shootouts, goalscorers, _ = encode_columns(*data_frames)

    results.sort_values("date", ascending=False, inplace=True)
    test_results, results = results[:test_count], results[test_count:]
    train_results, val_results = train_test_split(
        results, test_size=val_size, random_state=42
    )

    # Создание датасетов
    cache_dir = os.path.join(ROOT_DIR, "cache")
    train_ds = FootballDataset(
        train_results,
        shootouts,
        goalscorers,
        seq_length,
        cache_dir=cache_dir,
        is_test=False,
        force_rebuild=reset_cache,
    )
    val_ds = FootballDataset(
        val_results,
        shootouts,
        goalscorers,
        seq_length,
        cache_dir=cache_dir,
        is_test=True,
        force_rebuild=reset_cache,
    )
    test_ds = FootballDataset(
        test_results,
        shootouts,
        goalscorers,
        seq_length,
        cache_dir=cache_dir,
        is_test=True,
        force_rebuild=reset_cache,
    )
    return train_ds, val_ds, test_ds
