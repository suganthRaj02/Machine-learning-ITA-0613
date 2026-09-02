import pandas as pd
import numpy as np
from locally_weighted_regression import LocallyWeightedRegression, mean_squared_error, mean_absolute_error

df = pd.read_csv("data/processed/clean_crop_yield.csv")

features = [
    "Area_ha", "N_req_kg_per_ha", "P_req_kg_per_ha", "K_req_kg_per_ha",
    "Temperature_C", "Humidity_%", "pH", "Rainfall_mm",
    "Wind_Speed_m_s", "Solar_Radiation_MJ_m2_day"
]

X = df[features].values
y = df["Yield_kg_per_ha"].values

mean = X.mean(axis=0)
std = X.std(axis=0)
std[std == 0] = 1
X = (X - mean) / std

np.random.seed(42)
idx = np.random.permutation(len(X))[:3000]

X = X[idx]
y = y[idx]

split = int(0.8 * len(X))

X_train = X[:split]
X_test = X[split:split + 100]
y_train = y[:split]
y_test = y[split:split + 100]

results = []

for tau in [0.1, 0.5, 1.0, 2.0]:
    model = LocallyWeightedRegression(tau=tau)
    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    mse = mean_squared_error(y_test, pred)
    mae = mean_absolute_error(y_test, pred)

    print(f"tau={tau} | MSE={mse:.4f} | MAE={mae:.4f}")

    results.append([tau, mse, mae])

results_df = pd.DataFrame(results, columns=["tau", "MSE", "MAE"])
results_df.to_csv("results/tables/lwr_tau_comparison.csv", index=False)

print("\nTau comparison saved successfully.")