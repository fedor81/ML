from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.calibration import LabelEncoder


min_year = 0


class LabelEncoderStartingFromOne(LabelEncoder):
    def transform(self, y):
        return super().transform(y) + 1

    def fit_transform(self, y):
        return super().fit_transform(y) + 1


def _convert_columns_to_numeric(results, shootouts, goalscorers, former_names):
    """Делает категориальные колонки числовыми"""
    results = results.astype(
        {
            "home_team": "int16",
            "away_team": "int16",
            "home_score": "int8",
            "away_score": "int8",
            "tournament": "int16",
            "city": "int16",
            "country": "int16",
            "neutral": "int8",
        }
    )

    shootouts = shootouts.astype(
        {
            "home_team": "int16",
            "away_team": "int16",
            "first_shooter": "int16",
            "winner": "int16",
        }
    )

    goalscorers = goalscorers.astype(
        {
            "home_team": "int16",
            "away_team": "int16",
            "team": "int16",
            "scorer": "int16",
            "own_goal": "int8",
            "penalty": "int8",
            "minute": "int8",
        }
    )

    return results, shootouts, goalscorers, former_names


def encode_columns(
    results: pd.DataFrame,
    shootouts: pd.DataFrame,
    goalscorers: pd.DataFrame,
    former_names: pd.DataFrame,
):
    global min_year
    min_year = results["date"].dt.year.min()

    results, shootouts, goalscorers, former_names = (
        results.copy(),
        shootouts.copy(),
        goalscorers.copy(),
        former_names.copy(),
    )

    # Обучаем LabelEncoder на таблице результатов
    team_encoder = LabelEncoderStartingFromOne()
    team_encoder.fit(
        pd.concat([results["home_team"], results["away_team"]], ignore_index=True).drop_duplicates()
    )

    _encode_columns(results, ["home_team", "away_team"], team_encoder)
    _encode_columns(goalscorers, ["home_team", "away_team", "team"], team_encoder)
    _encode_columns(shootouts, ["home_team", "away_team", "winner", "first_shooter"], team_encoder)

    preprocess_date(shootouts)
    results = _encode_results(results)
    goalscorers = _encode_goalscorers(goalscorers)

    return _convert_columns_to_numeric(results, shootouts, goalscorers, former_names)


def _encode_columns(df: pd.DataFrame, columns: list[str], encoder: LabelEncoder):
    """Кодирует команды в столбцах columns для дата-фрейма"""
    for col in columns:
        df[col] = encoder.transform(df[col])


def _encode_goalscorers(goalscorers):
    categorical_cols = (
        goalscorers.drop(columns=["home_team", "away_team", "date", "team", "scorer"])
        .select_dtypes(include=["object", "bool", "category"])
        .columns
    )

    for col in categorical_cols:
        le = LabelEncoder()
        goalscorers[col] = le.fit_transform(goalscorers[col])

    # scorer отдельно
    le = LabelEncoderStartingFromOne()
    goalscorers["scorer"] = le.fit_transform(goalscorers["scorer"])

    preprocess_date(goalscorers)
    return goalscorers


def _encode_results(results):
    categorical_cols = (
        results.drop(columns=["home_team", "away_team", "date"])
        .select_dtypes(include=["object", "bool", "category"])
        .columns
    )

    for col in categorical_cols:
        le = LabelEncoder()
        results[col] = le.fit_transform(results[col])

    preprocess_date(results)
    return results


def preprocess_date(df: pd.DataFrame):
    """Заменяет столбец date на отдельные столбцы year, month, day, нормализует year"""
    df["year"] = (df["date"].dt.year - min_year).astype("int16")
    df["month"] = df["date"].dt.month.astype("int8")
    df["day"] = df["date"].dt.day.astype("int8")
