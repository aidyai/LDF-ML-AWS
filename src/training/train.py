import os
import json
import argparse
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error

FEATURES = [
    "hour_of_day", "day_of_week", "is_weekend",
    "is_lunch", "is_dinner", "is_peak",
    "lag_1h", "lag_24h", "lag_168h",
    "rolling_mean_3h", "rolling_mean_7d",
    "h3_encoded",
]

TARGET = "demand"


def split(df):
    df = df.sort_values("hour")
    n = len(df)
    return (
        df.iloc[:int(n * 0.70)],
        df.iloc[int(n * 0.70):int(n * 0.85)],
        df.iloc[int(n * 0.85):]
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir",  default="/opt/ml/input/data/train")
    parser.add_argument("--output-dir", default="/opt/ml/model")
    parser.add_argument("--num-boost-round", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--num-leaves", type=int, default=63)
    args = parser.parse_args()

    df = pd.read_parquet(os.path.join(args.input_dir, "features.parquet"))
    train_df, val_df, test_df = split(df)

    print(f"Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")

    train_data = lgb.Dataset(train_df[FEATURES], label=train_df[TARGET])
    val_data   = lgb.Dataset(val_df[FEATURES],   label=val_df[TARGET], reference=train_data)

    params = {
        "objective": "regression",
        "metric": ["rmse", "mae"],
        "learning_rate": args.learning_rate,
        "num_leaves": args.num_leaves,
        "min_child_samples": 20,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
        "seed": 42,
    }

    model = lgb.train(
        params,
        train_data,
        num_boost_round=args.num_boost_round,
        valid_sets=[train_data, val_data],
        valid_names=["train", "val"],
        callbacks=[
            lgb.early_stopping(50, verbose=False),
            lgb.log_evaluation(100),
        ],
    )

    val_preds  = model.predict(val_df[FEATURES])
    test_preds = model.predict(test_df[FEATURES])

    metrics = {
        "val_mae":   round(mean_absolute_error(val_df[TARGET], val_preds), 4),
        "val_rmse":  round(np.sqrt(mean_squared_error(val_df[TARGET], val_preds)), 4),
        "test_mae":  round(mean_absolute_error(test_df[TARGET], test_preds), 4),
        "test_rmse": round(np.sqrt(mean_squared_error(test_df[TARGET], test_preds)), 4),
        "best_iteration": model.best_iteration,
    }

    print("\n--- Metrics ---")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    os.makedirs(args.output_dir, exist_ok=True)
    model.save_model(os.path.join(args.output_dir, "model.lgb"))

    with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    importance = pd.DataFrame({
        "feature": FEATURES,
        "importance": model.feature_importance(importance_type="gain"),
    }).sort_values("importance", ascending=False)

    importance.to_csv(os.path.join(args.output_dir, "feature_importance.csv"), index=False)
    print("\nModel + artifacts saved.")
