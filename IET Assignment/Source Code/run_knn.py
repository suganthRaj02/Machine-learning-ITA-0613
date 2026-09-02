import pandas as pd
import numpy as np
from knn_regressor import KNNRegressor, mean_squared_error, mean_absolute_error

df = pd.read_csv("data/processed/clean_crop_yield.csv")

features = [
    "Area_ha",
    "N_req_kg_per_ha",
    "P_req_kg_per_ha",
    "K_req_kg_per_ha",
    "Temperature_C",
    "Humidity_%",
    "pH",
    "Rainfall_mm",
    "Wind_Speed_m_s",
    "Solar_Radiation_MJ_m2_day"
]

X = df[features].values
y = df["Yield_kg_per_ha"].values

# Normalize features
mean = X.mean(axis=0)
std = X.std(axis=0)
std[std == 0] = 1
X = (X - mean) / std

# Train/test split
np.random.seed(42)
indices = np.random.permutation(len(X))

split = int(0.8 * len(X))

train_idx = indices[:split]
test_idx = indices[split:]

X_train = X[train_idx]
X_test = X[test_idx]
y_train = y[train_idx]
y_test = y[test_idx]

# Compare distance metrics
results = []

for metric in ["euclidean", "mahalanobis"]:

    model = KNNRegressor(
        k=5,
        metric=metric
    )

    model.fit(X_train, y_train)

    pred = model.predict(X_test[:1000])

    mse = mean_squared_error(y_test[:1000], pred)
    mae = mean_absolute_error(y_test[:1000], pred)

    results.append([metric, mse, mae])

    print(
        f"{metric.title()} | MSE={mse:.2f} | MAE={mae:.2f}"
    )

results_df = pd.DataFrame(
    results,
    columns=["Metric", "MSE", "MAE"]
)

results_df.to_csv(
    "results/tables/knn_metric_comparison.csv",
    index=False
)

print("\nMetric comparison saved successfully.")