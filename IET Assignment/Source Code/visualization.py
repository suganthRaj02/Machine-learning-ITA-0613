import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load processed dataset
df = pd.read_csv("data/processed/clean_crop_yield.csv")

# -------------------------------
# 1. Yield Distribution
# -------------------------------
plt.figure(figsize=(8, 5))
plt.hist(df["Yield_kg_per_ha"], bins=30)
plt.xlabel("Yield (kg/ha)")
plt.ylabel("Frequency")
plt.title("Crop Yield Distribution")
plt.tight_layout()
plt.savefig("results/plots/yield_distribution.png", dpi=300)
plt.close()


# -------------------------------
# 2. Rainfall vs Yield
# -------------------------------
plt.figure(figsize=(8, 5))
plt.scatter(
    df["Rainfall_mm"],
    df["Yield_kg_per_ha"],
    s=8,
    alpha=0.4
)
plt.xlabel("Rainfall (mm)")
plt.ylabel("Yield (kg/ha)")
plt.title("Rainfall vs Crop Yield")
plt.tight_layout()
plt.savefig("results/plots/rainfall_vs_yield.png", dpi=300)
plt.close()


# -------------------------------
# 3. Temperature vs Yield
# -------------------------------
plt.figure(figsize=(8, 5))
plt.scatter(
    df["Temperature_C"],
    df["Yield_kg_per_ha"],
    s=8,
    alpha=0.4
)
plt.xlabel("Temperature (°C)")
plt.ylabel("Yield (kg/ha)")
plt.title("Temperature vs Crop Yield")
plt.tight_layout()
plt.savefig("results/plots/temperature_vs_yield.png", dpi=300)
plt.close()


# -------------------------------
# 4. Average Yield by Year
# -------------------------------
yearly = df.groupby("Year")["Yield_kg_per_ha"].mean()

plt.figure(figsize=(9, 5))
plt.plot(yearly.index, yearly.values)
plt.xlabel("Year")
plt.ylabel("Average Yield (kg/ha)")
plt.title("Average Crop Yield Over Time")
plt.grid(True)
plt.tight_layout()
plt.savefig("results/plots/yearly_yield_trend.png", dpi=300)
plt.close()


# -------------------------------
# 5. Average Yield by State
# -------------------------------
state_yield = (
    df.groupby("State Name")["Yield_kg_per_ha"]
    .mean()
    .sort_values(ascending=False)
    .head(10)
)

plt.figure(figsize=(9, 5))
plt.barh(
    state_yield.index[::-1],
    state_yield.values[::-1]
)
plt.xlabel("Average Yield (kg/ha)")
plt.ylabel("State")
plt.title("Top 10 States by Average Crop Yield")
plt.tight_layout()
plt.savefig("results/plots/state_yield_comparison.png", dpi=300)
plt.close()


# -------------------------------
# 6. Feature Correlation Matrix
# -------------------------------
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
    "Solar_Radiation_MJ_m2_day",
    "Yield_kg_per_ha"
]

corr = df[features].corr()

plt.figure(figsize=(11, 9))
plt.imshow(corr, aspect="auto")
plt.colorbar(label="Correlation")

plt.xticks(
    range(len(features)),
    features,
    rotation=90
)

plt.yticks(
    range(len(features)),
    features
)

plt.title("Feature Correlation Matrix")
plt.tight_layout()
plt.savefig("results/plots/correlation_matrix.png", dpi=300)
plt.close()


print("All visualizations created successfully.")
print("Saved in: results/plots/")