from ..files.dataset_load import load_fill_na
from ..files.dataset_encode import encode_columns
import pandas as pd


def _create_team_map(concat_results):
    teams = (
        pd.concat(
            [
                concat_results[["home_team", "home_team_id"]].rename(
                    columns={"home_team": "team", "home_team_id": "team_id"}
                ),
                concat_results[["away_team", "away_team_id"]].rename(
                    columns={"away_team": "team", "away_team_id": "team_id"}
                ),
            ],
            ignore_index=True,
        )
        .drop_duplicates()
        .set_index("team")
    )
    return teams.to_dict()["team_id"]


def _create_id_map(concat_results, col_name: str):
    values = concat_results[[col_name, col_name + "_id"]].drop_duplicates().set_index(col_name)
    return values.to_dict()[col_name + "_id"]


def create_mappings():
    """Создает маппинги данных для UI"""
    results, shootouts, goalscorers, former_names = load_fill_na()
    encoded_results, encoded_shootouts, encoded_goalscorers, encoded_former_names = encode_columns(
        results, shootouts, goalscorers, former_names
    )

    encoded_results.rename(
        columns={
            "home_team": "home_team_id",
            "away_team": "away_team_id",
            "tournament": "tournament_id",
            "city": "city_id",
            "country": "country_id",
        },
        inplace=True,
    )

    concat_results = pd.concat(
        [
            results,
            encoded_results.drop(columns=["date", "home_score", "away_score", "neutral"]),
        ],
        axis=1,
    )

    data_mappings = {
        "team_map": _create_team_map(concat_results),
        "tournament_map": _create_id_map(concat_results, "tournament"),
        "city_map": _create_id_map(concat_results, "city"),
        "country_map": _create_id_map(concat_results, "country"),
    }
    return data_mappings
