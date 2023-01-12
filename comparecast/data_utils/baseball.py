"""
baseball data utils
"""

import os.path
from typing import Union, Iterable
from tqdm import tqdm
import numpy as np
import pandas as pd

from ..forecasters import (
    forecast_constant,
    forecast_laplace,
    forecast_k29,
    forecast_seasonal,
)

# key: fivethirtyeight's team codes
# value: a list of alternative team codes in other datasets (including self)
TEAM_CODES = {
    "ANA": ["ANA", "LAA"],
    "ARI": ["ARI"],
    "ATL": ["ATL"],
    "BAL": ["BAL"],
    "BOS": ["BOS"],
    "CHC": ["CHC", "CUB"],
    "CHW": ["CHW", "CWS"],
    "CIN": ["CIN"],
    "CLE": ["CLE"],
    "COL": ["COL"],
    "DET": ["DET"],
    "FLA": ["FLA", "MIA"],
    "HOU": ["HOU"],
    "KCR": ["KCR", "KAN"],
    "LAD": ["LAD", "LOS"],
    "MIL": ["MIL"],
    "MIN": ["MIN"],
    "NYM": ["NYM"],
    "NYY": ["NYY"],
    "OAK": ["OAK"],
    "PHI": ["PHI"],
    "PIT": ["PIT"],
    "SDP": ["SDP", "SDG"],
    "SEA": ["SEA"],
    "SFG": ["SFG", "SFO"],
    "STL": ["STL"],
    "TBD": ["TBD", "TAM"],
    "TEX": ["TEX"],
    "TOR": ["TOR"],
    "WSN": ["WSN", "WAS"],
}


def _check_valid_team_code(team_code: str):
    if not any(team_code in codes for codes in TEAM_CODES.values()):
        raise Exception(f"invalid team code: {team_code}")


def _preprocess_team_538(
        data: pd.DataFrame,
        team: str,
        forecasters: Iterable[str] = ("fivethirtyeight", "fivethirtyeight_old",
                                      "random", "laplace", "k29"),
        fillna: float = 0.5,
) -> pd.DataFrame:
    """Pre-process team-specific data and add as columns
    (time, data/win, fivethirtyeight, fivethirtyeight_old)."""
    _check_valid_team_code(team)
    data = data[(data["team1"] == team) | (data["team2"] == team)].copy()
    data["win"] = ((data["team1"] == team) &
                   (data["score1"] > data["score2"]) |
                   (data["team2"] == team) &
                   (data["score1"] < data["score2"])).astype(int)
    data["y"] = data["win"]
    date_to_index = {date: i for i, date in enumerate(data["date"], 1)}
    data["time"] = [date_to_index[date] for date in data["date"]]

    # forecasts for that team
    for forecaster in forecasters:
        column_prefix = {
            "fivethirtyeight": "rating",
            "fivethirtyeight_old": "elo",
            "random": "random",
            "laplace": "laplace",
            "k29": "k29",
        }[forecaster]
        data[forecaster] = np.nan_to_num(np.where(
            data["team1"] == team,
            data[column_prefix + "_prob1"],
            data[column_prefix + "_prob2"],
        ), nan=fillna)
        if column_prefix + "_log5_prob1" in data:
            data[forecaster + "_log5"] = np.nan_to_num(np.where(
                data["team1"] == team,
                data[column_prefix + "_log5_prob1"],
                data[column_prefix + "_log5_prob2"],
            ), nan=fillna)
    return data


def read_csv_with_date(
        data_file: str,
        start_year: int,
        end_year: int,
) -> pd.DataFrame:
    """Read pre-processed csv and process dates."""
    data = pd.read_csv(data_file)
    data["date"] = pd.to_datetime(data["date"], format="%Y-%m-%d")
    data = data[(data["date"] >= str(start_year)) &
                (data["date"] <= str(end_year + 1))].copy()
    data = data.sort_values("date", ascending=True)
    return data


def preprocess_mlb_baselines(
        start_year: int = 2010,
        end_year: int = 2019,
        baselines: Iterable[str] = ("constant", "laplace", "k29"),
        data_file: str = "data/mlb_elo_538.csv",
        save_file: str = "data/mlb_elo_538_with_baselines.csv",
) -> pd.DataFrame:
    """Pre-process baseline forecasters for all teams using the FiveThirtyEight
    dataset.

    The baseline predictions, which are team-specific win probabilities, get
    normalized in each head-to-head matchup such that the probabilities sum to 1.
    """
    assert baselines == ("constant", "laplace", "k29")
    # TODO: allow other baselines

    if os.path.exists(save_file):
        return read_csv_with_date(save_file, start_year, end_year)

    data = read_csv_with_date(data_file, start_year, end_year)

    for baseline in baselines:
        data[baseline + "_iprob1"] = np.nan
        data[baseline + "_iprob2"] = np.nan

    for team in tqdm(TEAM_CODES, desc="processing baselines for each team"):
        team_data = _preprocess_team_538(data, team, [])
        ys = team_data["y"].values
        years = sorted(team_data.season.unique())
        season_starts = np.zeros_like(years)
        season_starts[1:] = np.where(np.diff(team_data.season))[0]

        team_data["constant"] = forecast_constant(ys, 0.5)
        team_data["laplace"] = forecast_seasonal(
            ys, season_starts, forecast_laplace)
        team_data["k29"] = forecast_seasonal(
            ys, season_starts,
            # baseline c is unused; forecasts within each season
            lambda y, c: forecast_k29(y, ("rbf", 0.1), verbose=False))

        # win probabilities before rescaling
        for baseline in baselines:
            data.loc[data["team1"] == team, baseline + "_iprob1"] = \
                team_data.loc[team_data["team1"] == team, baseline]
            data.loc[data["team2"] == team, baseline + "_iprob2"] = \
                team_data.loc[team_data["team2"] == team, baseline]

    # rescale probabilities to sum to 1
    for baseline in baselines:
        # simple rescaling
        data[baseline + "_prob1"] = rescale_iprob(
            data[baseline + "_iprob1"].values,
            data[baseline + "_iprob2"].values,
        )
        data[baseline + "_prob2"] = 1 - data[baseline + "_prob1"]
        # using the log5 method
        data[baseline + "_log5_prob1"] = rescale_log5(
            data[baseline + "_iprob1"].values,
            data[baseline + "_iprob2"].values,
        )
        data[baseline + "_log5_prob2"] = 1 - data[baseline + "_log5_prob1"]

    if save_file:
        data.to_csv(save_file, index=False)
    return data


def read_538_projections(
        start_year: int = 2010,
        end_year: int = 2019,
        team: str = "MLB",
        fillna: float = 0.5,
        data_file: str = "data/mlb_elo_538_with_baselines.csv",
) -> pd.DataFrame:
    """Read FiveThirtyEight's ELO-based MLB projections data.

    The data also includes all game's scores, home-and-away,

    The `team == "MLB"` option gives the full data for the year range.
    Specifying a team gives the team's record (home or away) as well as
    an additional `win`, `game`, and the win probability forecasts
    `fivethirtyeight` (newer, preferred) and `fivethirtyeight_old`.

    Data download:
        https://projects.fivethirtyeight.com/mlb-api/mlb_elo.csv

    Further information:
        https://fivethirtyeight.com/features/how-our-mlb-predictions-work/
        https://github.com/fivethirtyeight/data/tree/master/mlb-elo
    """
    data = read_csv_with_date(data_file, start_year, end_year)
    if team == "MLB":
        return data

    # process team-specific probabilities
    return _preprocess_team_538(data, team, fillna=fillna)


def convert_odds(odds: np.ndarray) -> np.ndarray:
    """Convert American betting odds to implied probabilities.

    - Positive odds: 100 divided by (the american odds plus 100),
        multiplied by 100 to give a percentage
        e.g. american odds of 150 = (100 / (150 + 100)) * 100 = 40%.
    - Negative odds: Firstly multiply the american odds by -1 and
        use the positive value in the following formula:
        american odds divided by (the american odds plus 100),
        multiplied by 100 to give a percentage
        e.g. american odds of -300 = (300/(300+100)) * 100 = 75%.
    """
    def _convert(o):
        if o >= 0:
            return 100 / (100 + o)
        else:
            return -o / (100 - o)

    return np.array([_convert(o) for o in odds])


def rescale_iprob(
        iprob1: Union[float, np.ndarray],
        iprob2: Union[float, np.ndarray],
) -> Union[float, np.ndarray]:
    """Rescale the implied probabilities from converting odds, such that
    the resulting probabilities sum to 1.

    This is needed for betting odds, which are in general not coherent.
    (cf. vigorish & overround in bookmaking, or
     coherence & Dutch book in statistics).
    """
    return (iprob1 + 1e-8) / (iprob1 + iprob2 + 2e-8)


def rescale_log5(
        prob1: Union[float, np.ndarray],
        prob2: Union[float, np.ndarray],
) -> Union[float, np.ndarray]:
    """Compute the win probability of team 1 in a head-to-head matchup,
    given the estimated win probabilities of the two teams.

    See https://en.wikipedia.org/wiki/Log5 .
    """
    return (prob1 * (1 - prob2) + 1e-8) / (
        prob1 * (1 - prob2) + prob2 * (1 - prob1) + 2e-8)


def read_vegas_odds(
        start_year: int = 2010,
        end_year: int = 2019,
        team: str = "MLB",
        columns: Iterable[str] = ("Date", "VH", "Team", "Pitcher",
                                  "Final", "Open", "Close"),
        data_dir: str = "data",
) -> pd.DataFrame:
    """Read and pre-process the MLB historical betting odds data.

    Data download:
        https://sports-statistics.com/sports-data/mlb-historical-odds-scores-datasets/

    The resulting dataset can be merged with the the output of
    `read_538_projections`.
    """
    if team == "MLB":
        save_file = os.path.join(data_dir, f"mlb_odds.csv")
    else:
        _check_valid_team_code(team)
        save_file = os.path.join(data_dir, f"mlb_odds_{team.lower()}.csv")
    if os.path.exists(save_file):
        return read_csv_with_date(save_file, start_year, end_year)

    # read in raw data
    years = list(range(start_year, end_year + 1))
    data = pd.concat([
        pd.read_excel(os.path.join(data_dir, f"mlb odds {year}.xlsx"),
                      usecols=columns).assign(Year=year)
        for year in tqdm(years, desc="reading vegas odds dataset")
    ])

    def _make_date(row):
        return pd.to_datetime(f"{row.Year:4d}{row.Date:04d}")

    data.insert(0, "date", data.apply(_make_date, axis=1))
    data.drop(["Year", "Date"], axis=1, inplace=True)

    # combine row pairs into a game (home & away)
    home_data = data[1::2].copy().reset_index(drop=True)
    assert (home_data["VH"] != "V").all(), "odd rows must not be away data"
    home_data.drop(["VH"], axis=1, inplace=True)
    home_data.rename(lambda s: s.lower() + "1" if s in columns else s, axis=1,
                     inplace=True)
    away_data = data[0::2].copy().reset_index(drop=True)
    assert (away_data["VH"] != "H").all(), "even rows must not be home data"
    away_data.drop(["VH", "date"], axis=1, inplace=True)
    away_data.rename(lambda s: s.lower() + "2" if s in columns else s, axis=1,
                     inplace=True)
    data = pd.concat([home_data, away_data], axis=1)

    if team != "MLB":
        data = data[data["team1"].isin(TEAM_CODES[team]) |
                    data["team2"].isin(TEAM_CODES[team])]

    # implied probabilities (converted odds) & scaled win probabilities
    for s in ["open", "close"]:
        for i in [1, 2]:
            data[f"{s}_iprob{i}"] = convert_odds(
                pd.to_numeric(data[f"{s}{i}"], errors="coerce"))
        data[f"{s}_prob1"] = rescale_iprob(data[f"{s}_iprob1"].values,
                                           data[f"{s}_iprob2"].values)
        data[f"{s}_prob2"] = 1 - data[f"{s}_prob1"]
        data[f"{s}_log5_prob1"] = rescale_log5(data[f"{s}_prob1"].values,
                                               data[f"{s}_prob2"].values)
        data[f"{s}_log5_prob2"] = 1 - data[f"{s}_log5_prob1"]

    if team != "MLB":
        # win probabilities for the team
        data["vegas_open"] = np.where(data["team1"].isin(TEAM_CODES[team]),
                                      data["open_prob1"],
                                      data["open_prob2"])
        data["vegas"] = np.where(data["team1"].isin(TEAM_CODES[team]),
                                 data["close_prob1"],
                                 data["close_prob2"])
        data["vegas_open_log5"] = np.where(data["team1"].isin(TEAM_CODES[team]),
                                           data["open_log5_prob1"],
                                           data["open_log5_prob2"])
        data["vegas_log5"] = np.where(data["team1"].isin(TEAM_CODES[team]),
                                      data["close_log5_prob1"],
                                      data["close_log5_prob2"])

    os.makedirs(os.path.dirname(save_file), exist_ok=True)
    data.to_csv(save_file, index=False)
    return data
