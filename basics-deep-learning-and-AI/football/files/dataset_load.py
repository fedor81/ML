import os
import random
from collections import defaultdict

import kagglehub
import numpy as np
import pandas as pd


def get_dataset_info() -> dict:
    info = {}
    results, shootouts, goalscorers, former_names = load_fill_na()

    teams = pd.concat([results["home_team"], results["away_team"]], ignore_index=True)
    info["num_teams"] = len(teams.unique())
    info["num_tournaments"] = len(results["tournament"].unique())
    info["num_cities"] = len(results["city"].unique())
    info["num_countries"] = len(results["country"].unique())
    info["num_scorers"] = len(goalscorers["scorer"].unique())

    return info


def calculate_teams_stats(results: pd.DataFrame, shootouts: pd.DataFrame):
    """Считает среднюю статистику по командам"""
    stats = {}
    teams = pd.concat([results["home_team"], results["away_team"]]).unique()

    # Создаем индекс для быстрого поиска побед в буллитах
    shootout_wins_index = shootouts.set_index(["date", "winner"]).index

    for team in teams:
        # Фильтруем матчи команды
        mask = (results["home_team"] == team) | (results["away_team"] == team)
        team_matches = results[mask].copy()

        # Собираем статистику по буллитам
        mask = (shootouts["home_team"] == team) | (shootouts["away_team"] == team)
        team_shootouts = shootouts[mask].copy()

        # Рассчитываем голы
        team_matches["team_goals"] = np.where(
            team_matches["home_team"] == team,
            team_matches["home_score"],
            team_matches["away_score"],
        )

        # Рассчитываем победы в основное время
        regular_wins = (
            (team_matches["home_team"] == team)
            & (team_matches["home_score"] > team_matches["away_score"])
        ) | (
            (team_matches["away_team"] == team)
            & (team_matches["away_score"] > team_matches["home_score"])
        )

        # Находим матчи, закончившиеся ничьей
        draws = team_matches[team_matches["home_score"] == team_matches["away_score"]]

        # Проверяем победы в буллитах (оптимизированный способ)
        if not draws.empty:
            # Создаем MultiIndex для сравнения
            dates_teams = pd.MultiIndex.from_arrays([draws["date"], [team] * len(draws)])
            shootout_wins = dates_teams.isin(shootout_wins_index)
            shootout_wins_count = shootout_wins.sum()
        else:
            shootout_wins_count = 0

        total_wins = regular_wins.sum() + shootout_wins_count

        stats[team] = {
            "avg_goals": team_matches["team_goals"].mean(),
            "win_rate": total_wins / len(team_matches),
            "num_matches": len(team_matches),
            "num_shootouts": len(team_shootouts),
            "shootouts_win_rate": shootout_wins_count / max(len(team_shootouts), 1),
        }

    return pd.DataFrame.from_dict(stats, orient="index")


def load_fill_na():
    """Загружает датасет из Kaggle, заполнет пропуски и преобразует типы"""
    results, shootouts, goalscorers, former_names = load_dataset()
    _fill_shootouts_na(shootouts)
    _fill_goalscorers_na(goalscorers)
    _remove_typo_in_shootouts(shootouts)
    return _convert_columns(results, shootouts, goalscorers, former_names)


def load_dataset():
    """Загружает датасет из Kaggle"""
    path = kagglehub.dataset_download("martj42/international-football-results-from-1872-to-2017")

    results = pd.read_csv(os.path.join(path, "results.csv"))
    shootouts = pd.read_csv(os.path.join(path, "shootouts.csv"))
    goalscorers = pd.read_csv(os.path.join(path, "goalscorers.csv"))
    former_names = pd.read_csv(os.path.join(path, "former_names.csv"))

    return results, shootouts, goalscorers, former_names


def _remove_typo_in_shootouts(shootouts: pd.DataFrame):
    """Исправляет ошибку в датасете"""
    shootouts.replace("Åland", "Åland Islands", inplace=True)


def _fill_shootouts_na(shootouts: pd.DataFrame):
    """Заполняет пропуски в столбцах таблицы shootouts.csv"""
    # Заполнение пропусков first_shooter случайной командой
    shootouts["first_shooter"] = shootouts.apply(
        lambda row: (
            random.choice([row["home_team"], row["away_team"]])
            if pd.isna(row["first_shooter"])
            else row["first_shooter"]
        ),
        axis=1,
    )


def _fill_goalscorers_na(goalscorers: pd.DataFrame):
    """Заполнение пропусков goalscorers.csv"""

    # Заполнение пропусков scorer: случайным игроком из той же команды
    team_scorers = defaultdict(list)  # Создаем словарь {команда: [список её бомбардиров]}
    for _, row in goalscorers.dropna(subset=["scorer"]).iterrows():
        team_scorers[row["team"]].append(row["scorer"])

    def fill_scorer(row):
        """Заполняет пропуски в столбце scorer"""
        if pd.isna(row["scorer"]) and row["team"] in team_scorers:
            return np.random.choice(team_scorers[row["team"]])
        return row["scorer"]

    goalscorers["scorer"] = goalscorers.apply(fill_scorer, axis=1)

    # Заполнение пропусков minute медианным значением
    global_median = goalscorers["minute"].median()
    goalscorers["minute"] = goalscorers["minute"].fillna(global_median)


def _convert_columns(
    results: pd.DataFrame,
    shootouts: pd.DataFrame,
    goalscorers: pd.DataFrame,
    former_names: pd.DataFrame,
):
    """Делает преобразование типов для фреймов"""
    results = results.astype(
        {
            "date": "datetime64[ns]",
            "home_team": "category",
            "away_team": "category",
            "home_score": "int32",
            "away_score": "int32",
            "tournament": "category",
            "city": "category",
            "country": "category",
        }
    )

    shootouts = shootouts.astype(
        {
            "date": "datetime64[ns]",
            "home_team": "category",
            "away_team": "category",
            "first_shooter": "category",
            "winner": "category",
        }
    )

    goalscorers = goalscorers.astype(
        {
            "minute": "int32",
            "date": "datetime64[ns]",
            "home_team": "category",
            "away_team": "category",
            "team": "category",
            "scorer": "category",
        }
    )

    former_names = former_names.astype(
        {
            "start_date": "datetime64[ns]",
            "end_date": "datetime64[ns]",
        }
    )

    return results, shootouts, goalscorers, former_names
