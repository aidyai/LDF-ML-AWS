# Lagos Demand Forecasting Engine

Spatial-temporal demand forecasting for on-demand delivery in Lagos — predicts order volume by zone to pre-position riders before demand spikes.

---

## What it does

Given any delivery zone and the current hour, the system predicts how many orders will come in during the next hour and recommends how many riders to stage there — before the spike hits, not after.

---

## How it works

NYC TLC yellow taxi trip data is used as a spatial-temporal demand proxy. Taxi pickup demand and food delivery demand are driven by identical forces: time of day, day of week, and neighbourhood density.

Every pickup coordinate is indexed into an **H3 hexagon at resolution 8** — roughly 500m cells — giving a uniform zone grid. Demand is aggregated per cell per hour. A LightGBM regression model learns from lag features, rolling averages, and time-of-day signals to predict the next hour's demand per zone.

The prediction feeds a simple allocation formula:

```
recommended_riders = ceil(predicted_demand / avg_orders_per_rider_per_hour)
```

---

## Stack

- **Spatial indexing:** H3 (Uber Hexagonal Hierarchical Spatial Index)
- **Model:** LightGBM regression
- **Infrastructure:** AWS SageMaker — Processing Job, Training Job, Model Registry, Endpoint, Model Monitor
- **Storage:** S3

---

## Repo structure

```
lagos-demand-forecasting/
├── notebooks/
│   └── lagos_demand_forecasting.ipynb   ← run this, everything else follows
├── src/
│   ├── processing/
│   │   └── feature_engineering.py
│   ├── training/
│   │   └── train.py
│   └── inference/
│       └── predict.py
├── requirements.txt
└── README.md
```

---

## Running it

### On SageMaker (recommended)

1. Launch a SageMaker notebook instance (`ml.t3.medium`)
2. Open the terminal and clone the repo:
   ```bash
   git clone https://github.com/aidyai/lagos-demand-forecasting.git
   ```
3. Open `notebooks/lagos_demand_forecasting.ipynb`
4. Run all cells top to bottom

The notebook handles everything: data download, EDA, S3 upload, Processing Job, Training Job, Model Registry, endpoint deployment, and a live test prediction.

At the end of cell 12, you have a live endpoint URL.

### Locally / Colab

```bash
pip install -r requirements.txt
```

Run the notebook the same way — skip the SageMaker job cells and use the local training path in cell 13.

---

## Endpoint

Once deployed, call the endpoint from anywhere:

```python
import boto3, json

client = boto3.client("sagemaker-runtime", region_name="us-east-1")

response = client.invoke_endpoint(
    EndpointName="lagos-demand-forecast-v1",
    ContentType="application/json",
    Body=json.dumps({
        "h3_encoded":      42,
        "hour_of_day":     13,
        "day_of_week":     2,
        "lag_1h":          38.0,
        "lag_24h":         41.0,
        "lag_168h":        44.0,
        "rolling_mean_3h": 39.5,
        "rolling_mean_7d": 42.0,
    })
)

print(json.loads(response["Body"].read()))
# {"predicted_demand": 47.0, "recommended_riders": 6, "confidence": "high"}
```

---

## Adapting to real delivery data

| Proxy (TLC)              | Real delivery data         |
|--------------------------|----------------------------|
| `tpep_pickup_datetime`   | `order_placed_at`          |
| Pickup lat/lon           | Vendor lat/lon             |
| Trip count per zone/hr   | Order count per zone/hr    |

The feature engineering, training, and serving pipeline requires zero changes — only the data source swaps.

---

## Cost note

Delete the endpoint when not in use. The notebook instance and training jobs only bill while running. The endpoint bills continuously.

```python
predictor.delete_endpoint()
```
