"""
statistical postprocessing of precipitation forecasts
(Henzi et al., 2019; Henzi & Ziegel, 2021)
"""

import os.path
import pandas as pd


AIRPORTS = ["Brussels", "Frankfurt", "London", "Zurich"]
LAGS = [1, 2, 3, 4, 5]


def read_precip_fcs(
        data_dir: str = "eprob/replication_material/precip_fcs",
        pop_only: bool = False,
) -> pd.DataFrame:
    """Read in and pre-process precipitation forecasts from
    https://github.com/AlexanderHenzi/eprob/tree/master/replication_material.

    Args:
        data_dir: directory containing precipitation forecasts (csv)
        pop_only: whether to return only columns relevant to PoP forecasts

    Returns:
        A pandas dataframe containing all observations and forecasts.
    """
    dfs = []
    for airport in AIRPORTS:
        for lag in LAGS:
            df = pd.read_csv(os.path.join(data_dir, f"{airport}_{lag}.csv"))
            df["date"] = pd.to_datetime(df["date"])
            df.insert(0, "airport", airport)
            df.insert(1, "lag", lag)
            df.insert(4, "y", (df["obs"] > 0).astype(int))  # PoP
            df.insert(7, "ens", df[["p" + str(i)
                                    for i in range(1, 51)]].mean(axis=1))
            dfs.append(df)
    data = pd.concat(dfs, ignore_index=True)
    if pop_only:
        data = data[["airport", "lag", "date", "y",
                     "pop_idr", "pop_hclr", "pop_hclr_noscale"]].copy()
        data.rename(lambda s: s.replace("pop_", ""), axis=1, inplace=True)
    data = data.sort_values(["airport", "lag", "date"],
                            ascending=(True, True, True))
    return data


def read_hz_evalues(
        data_path: str = "eprob/replication_material/precip_fcs/evalues.csv",
) -> pd.DataFrame:
    """Read in and pre-process Henzi & Ziegel's e-values for the PoP forecasts,
    given by
    https://github.com/AlexanderHenzi/eprob/tree/master/replication_material .

    Args:
        data_path: directory containing e-values (csv)

    Returns:
        A pandas dataframe containing the observations, forecasts,
        comparison hypotheses, and e-values.
    """
    evalues = pd.read_csv(data_path)
    evalues.columns = ["Airport", "Lag", "Date", "Precipitation",
                       "HCLR", "HCLR_", "IDR", "PoP",
                       "Hypothesis", "E-value"]
    evalues["Date"] = pd.to_datetime(evalues["Date"])
    evalues["Hypothesis"] = evalues["Hypothesis"].map({
        "HCLR/IDR": "HCLR/IDR",
        "IDR/HCLR['-']": "IDR/HCLR_",
        "HCLR/HCLR['-']": "HCLR/HCLR_",
    })
    evalues = evalues.sort_values(["Airport", "Lag", "Date"],
                                  ascending=(True, True, True))
    return evalues

