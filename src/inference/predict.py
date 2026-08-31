import os
import json
import math
import numpy as np
import pandas as pd
import lightgbm as lgb


FEATURES = [
    "hour_of_day", "day_of_week", "is_weekend",
    "is_lunch", "is_dinner", "is_peak",
    "lag_1h", "lag_24h", "lag_168h",
    "rolling_mean_3h", "rolling_mean_7d",
    "h3_encoded",
]

AVG_ORDERS_PER_RIDER = 9
model = None


def model_fn(model_dir):
    global model
    model = lgb.Booster(model_file=os.path.join(model_dir, "model.lgb"))
    return model


def input_fn(request_body, content_type="application/json"):
    return json.loads(request_body)


def predict_fn(data, model):
    hour_of_day  = data["hour_of_day"]
    day_of_week  = data["day_of_week"]
    is_weekend   = int(day_of_week >= 5)
    is_lunch     = int(11 <= hour_of_day <= 14)
    is_dinner    = int(19 <= hour_of_day <= 22)
    is_peak      = int((is_lunch or is_dinner) and not is_weekend)

    row = pd.DataFrame([{
        "hour_of_day":       hour_of_day,
        "day_of_week":       day_of_week,
        "is_weekend":        is_weekend,
        "is_lunch":          is_lunch,
        "is_dinner":         is_dinner,
        "is_peak":           is_peak,
        "lag_1h":            data["lag_1h"],
        "lag_24h":           data["lag_24h"],
        "lag_168h":          data["lag_168h"],
        "rolling_mean_3h":   data["rolling_mean_3h"],
        "rolling_mean_7d":   data["rolling_mean_7d"],
        "h3_encoded":        data["h3_encoded"],
    }])

    raw = float(model.predict(row[FEATURES])[0])
    predicted_demand = max(0, round(raw, 1))

    return {
        "predicted_demand":    predicted_demand,
        "recommended_riders":  math.ceil(predicted_demand / AVG_ORDERS_PER_RIDER),
        "confidence":          "high" if predicted_demand >= 40 else "medium" if predicted_demand >= 10 else "low",
    }


def output_fn(prediction, accept="application/json"):
    return json.dumps(prediction), "application/json"
