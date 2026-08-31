import os
import argparse
import pandas as pd
import numpy as np


def build_features(demand: pd.DataFrame) -> pd.DataFrame:
    df = demand.copy()
    df = df.sort_values(["h3_cell", "hour"]).reset_index(drop=True)

    df["hour_of_day"] = df["hour"].dt.hour
    df["day_of_week"] = df["hour"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["is_lunch"] = df["hour_of_day"].between(11, 14).astype(int)
    df["is_dinner"] = df["hour_of_day"].between(19, 22).astype(int)
    df["is_peak"] = (
        (df["is_lunch"] | df["is_dinner"]) & (df["is_weekend"] == 0)
    ).astype(int)

    df["lag_1h"] = df.groupby("h3_cell")["demand"].shift(1)
    df["lag_24h"] = df.groupby("h3_cell")["demand"].shift(24)
    df["lag_168h"] = df.groupby("h3_cell")["demand"].shift(168)

    df["rolling_mean_3h"] = (
        df.groupby("h3_cell")["demand"]
        .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    )
    df["rolling_mean_7d"] = (
        df.groupby(["h3_cell", "hour_of_day"])["demand"]
        .transform(lambda x: x.shift(1).rolling(7, min_periods=1).mean())
    )

    df["h3_encoded"] = df["h3_cell"].astype("category").cat.codes
    df = df.dropna(subset=["lag_1h", "lag_24h", "lag_168h"]).reset_index(drop=True)

    return df


FEATURES = [
    "hour_of_day", "day_of_week", "is_weekend",
    "is_lunch", "is_dinner", "is_peak",
    "lag_1h", "lag_24h", "lag_168h",
    "rolling_mean_3h", "rolling_mean_7d",
    "h3_encoded",
]

TARGET = "demand"


def split(df: pd.DataFrame):
    df = df.sort_values("hour")
    n = len(df)
    return (
        df.iloc[:int(n * 0.70)],
        df.iloc[int(n * 0.70):int(n * 0.85)],
        df.iloc[int(n * 0.85):]
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir",  default="/opt/ml/processing/input")
    parser.add_argument("--output-dir", default="/opt/ml/processing/output")
    args = parser.parse_args()

    demand = pd.read_parquet(os.path.join(args.input_dir, "demand_raw.parquet"))
    features = build_features(demand)

    os.makedirs(args.output_dir, exist_ok=True)
    features.to_parquet(os.path.join(args.output_dir, "features.parquet"), index=False)
    print(f"Done — {features.shape[0]:,} rows, {features.shape[1]} columns")
