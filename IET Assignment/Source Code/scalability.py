import pandas as pd
import numpy as np
import time

# Load processed dataset
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

X = df[features].values.astype(float)

# Normalize features
mean = X.mean(axis=0)
std = X.std(axis=0)
std[std == 0] = 1
X = (X - mean) / std

# Benchmark sizes available in the dataset
sizes = [1000, 10000, 50000]

results = []

for size in sizes:

    X_sample = X[:size]

    # Use 10 query points
    queries = X_sample[-10:]

    start = time.time()

    for q in queries:
        distances = np.sqrt(
            np.sum((X_sample - q) ** 2, axis=1)
        )
        np.argsort(distances)[:5]

    elapsed = time.time() - start

    memory_mb = X_sample.nbytes / (1024 * 1024)

    results.append([
        size,
        elapsed,
        memory_mb
    ])

    print(
        f"Records={size} | "
        f"Time={elapsed:.4f}s | "
        f"Memory={memory_mb:.2f} MB"
    )


# Theoretical scalability
theoretical = pd.DataFrame({
    "Records": [1000, 10000, 100000, 1000000],
    "Relative_Computation": [1, 10, 100, 1000],
    "Approx_Memory_MB": [
        1000 * X.shape[1] * 8 / (1024 * 1024),
        10000 * X.shape[1] * 8 / (1024 * 1024),
        100000 * X.shape[1] * 8 / (1024 * 1024),
        1000000 * X.shape[1] * 8 / (1024 * 1024)
    ]
})

theoretical.to_csv(
    "results/tables/scalability_theoretical.csv",
    index=False
)

pd.DataFrame(
    results,
    columns=[
        "Records",
        "Prediction_Time_Seconds",
        "Memory_MB"
    ]
).to_csv(
    "results/tables/scalability_results.csv",
    index=False
)

print("\nTheoretical scalability:")
print(theoretical)

print("\nComplexity:")
print("k-NN prediction: O(n × d)")
print("Memory: O(n × d)")
print("d = number of features")

print("\nApproximation strategy:")
print("Use spatial indexing or approximate nearest-neighbor search")
print("to reduce the number of records examined per query.")

print("\nScalability analysis completed successfully.")