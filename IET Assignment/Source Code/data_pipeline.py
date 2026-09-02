import pandas as pd
import numpy as np

RAW_FILE = "data/raw/Custom_Crops_yield_Historical_Dataset.csv"
PROCESSED_FILE = "data/processed/clean_crop_yield.csv"


def load_data():
    return pd.read_csv(RAW_FILE)


def clean_data(df):
    df = df.copy()

    # Remove duplicate records
    df = df.drop_duplicates()

    # Convert numeric columns safely
    numeric_cols = df.select_dtypes(include=np.number).columns
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

    # Impute missing weather values using median
    weather_cols = [
        "Temperature_C",
        "Humidity_%",
        "Rainfall_mm",
        "Wind_Speed_m_s",
        "Solar_Radiation_MJ_m2_day"
    ]

    for col in weather_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    return df


def engineer_features(df):
    df = df.copy()

    # 1. Temperature-Rainfall interaction
    df["Temp_Rainfall_Index"] = (
        df["Temperature_C"] * df["Rainfall_mm"]
    )

    # 2. Nutrient requirement index
    df["NPK_Index"] = (
        df["N_req_kg_per_ha"]
        + df["P_req_kg_per_ha"]
        + df["K_req_kg_per_ha"]
    )

    # 3. Climate stress index
    df["Climate_Stress_Index"] = (
        df["Temperature_C"] * (1 - df["Humidity_%"] / 100)
    )

    # 4. Area productivity
    df["Production_Estimate_kg"] = (
        df["Area_ha"] * df["Yield_kg_per_ha"]
    )

    return df


def save_data(df):
    df.to_csv(PROCESSED_FILE, index=False)


if __name__ == "__main__":
    data = load_data()
    data = clean_data(data)
    data = engineer_features(data)
    save_data(data)

    print("Data pipeline completed successfully.")
    print("Records:", len(data))
    print("Columns:", len(data.columns))
    print("Saved:", PROCESSED_FILE)