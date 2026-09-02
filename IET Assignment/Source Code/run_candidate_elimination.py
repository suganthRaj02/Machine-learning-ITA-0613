import pandas as pd
import numpy as np
from candidate_elimination import CandidateElimination, create_risk_band

# Load processed dataset
df = pd.read_csv("data/processed/clean_crop_yield.csv")

# Features used for Candidate Elimination
features = [
    "Temperature_C",
    "Humidity_%",
    "pH",
    "Rainfall_mm"
]

# Select required columns
data = df[features + ["Yield_kg_per_ha"]].copy()

# Discretize continuous features into Low / Medium / High
for col in features:
    data[col] = pd.qcut(
        data[col].rank(method="first"),
        q=3,
        labels=["Low", "Medium", "High"]
    )

# Create yield risk bands
data["Risk"] = create_risk_band(
    data["Yield_kg_per_ha"].values
)

# Keep only Low and High risk examples
data = data[data["Risk"].isin(["Low", "High"])].copy()

# Shuffle the real observations
data = data.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)

# Select balanced training examples
low = data[data["Risk"] == "Low"].head(10)
high = data[data["Risk"] == "High"].head(10)

data = pd.concat([low, high])

# Shuffle final training set
data = data.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)

# Display risk distribution
print("\nRisk Counts:")
print(data["Risk"].value_counts())

# Prepare input and target
X = data[features].astype(str).values
y = data["Risk"].values

# Define feature domains
domains = [
    ["Low", "Medium", "High"],
    ["Low", "Medium", "High"],
    ["Low", "Medium", "High"],
    ["Low", "Medium", "High"]
]

# Create Candidate Elimination model
model = CandidateElimination(
    n_features=len(features),
    domains=domains
)

# Train the model
model.fit(X, y)

# Get boundaries
S, G = model.get_boundaries()

# Display results
print("\nCandidate Elimination Results")

print("\nSpecific Boundary (S):")
print(S)

print("\nGeneral Boundary (G):")

if G:
    for g in G:
        print(g)
else:
    print("Version space is empty for this training subset.")

# Display training information
print("\nTraining examples:", len(data))

print("\nInterpretation:")

if not G:
    print("The version space is empty.")
    print("The selected climate features cannot perfectly separate")
    print("Low and High yield-risk examples under the current")
    print("Candidate Elimination hypothesis space.")
else:
    print("A consistent version space exists.")

print("\nCandidate Elimination completed successfully.")