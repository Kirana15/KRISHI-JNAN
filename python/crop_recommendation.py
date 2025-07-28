# crop_recommendation.py

import pandas as pd

# Load crop recommendation data
df = pd.read_excel("Crop.xlsx")

def get_soil_types_for_district(district):
    district = district.strip().capitalize()
    matching_rows = df[df["District"].str.strip().str.capitalize() == district]

    if matching_rows.empty:
        return None
    else:
        return sorted(matching_rows["Soil Type"].str.strip().str.title().unique())

def recommend_crops(district, soil_type, sowing_month):
    district = district.strip().capitalize()
    soil_type = soil_type.strip().lower()
    sowing_month = sowing_month.strip().capitalize()

    filtered = df[
        (df["District"].str.strip().str.capitalize() == district) &
        (df["Soil Type"].str.strip().str.lower() == soil_type)
    ]

    recommended = filtered[
        (filtered["Sowing"].str.strip().str.capitalize() == sowing_month) |
        (filtered["Sowing"].str.strip().str.lower() == "whole year")
    ]

    if recommended.empty:
        return pd.DataFrame(columns=["Crop", "Maturity"])  # Return empty df instead of string
    else:
        result = recommended[["Crop", "Maturity"]].drop_duplicates().reset_index(drop=True)
        return result
