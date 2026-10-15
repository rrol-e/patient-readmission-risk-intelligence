"""
Feature Engineering Pipeline for Patient Readmission
Builds clinically motivated predictors:
- Healthcare Utilization History
- Comorbidity Burden & Cross-Category Interactions
- Medication Volatility & Polypharmacy Index
- In-Hospital Care Intensity Metrics
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"

MEDICATION_COLS = [
    "metformin", "repaglinide", "nateglinide", "chlorpropamide",
    "glimepiride", "acetohexamide", "glipizide", "glyburide",
    "tolbutamide", "pioglitazone", "rosiglitazone", "acarbose",
    "miglitol", "troglitazone", "tolazamide", "examide",
    "citoglipton", "insulin", "glyburide-metformin", "glipizide-metformin",
    "glimepiride-pioglitazone", "metformin-rosiglitazone", "metformin-pioglitazone"
]


def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforms preprocessed dataframe into rich feature set for machine learning.
    """
    df = df.copy()

    # 1. Healthcare Utilization Indices
    df["total_prior_visits"] = (
        df["number_outpatient"] + df["number_emergency"] + df["number_inpatient"]
    )
    df["has_prior_emergency"] = (df["number_emergency"] > 0).astype(int)
    df["has_prior_inpatient"] = (df["number_inpatient"] > 0).astype(int)
    df["frequent_inpatient"] = (df["number_inpatient"] >= 2).astype(int)
    df["utilization_severity_ratio"] = df["number_inpatient"] / (df["total_prior_visits"] + 1)

    # 2. Care Intensity & Stay Length
    df["prolonged_stay"] = (df["time_in_hospital"] >= 7).astype(int)
    df["lab_intensity"] = df["num_lab_procedures"] / (df["time_in_hospital"] + 0.1)
    df["medication_intensity"] = df["num_medications"] / (df["time_in_hospital"] + 0.1)
    df["polypharmacy_high"] = (df["num_medications"] >= 16).astype(int)
    df["has_procedures"] = (df["num_procedures"] > 0).astype(int)

    # 3. Clinical Comorbidities & Interactions
    diag_cols = ["primary_diag_category", "secondary_diag_category", "tertiary_diag_category"]
    
    # Check across any of the 3 recorded diagnoses
    df["has_circulatory"] = df[diag_cols].apply(
        lambda row: int("Circulatory" in row.values), axis=1
    )
    df["has_respiratory"] = df[diag_cols].apply(
        lambda row: int("Respiratory" in row.values), axis=1
    )
    df["has_renal"] = df[diag_cols].apply(
        lambda row: int("Genitourinary" in row.values), axis=1
    )
    df["has_diabetes_primary"] = (df["primary_diag_category"] == "Diabetes").astype(int)
    df["has_diabetes_secondary"] = df[["secondary_diag_category", "tertiary_diag_category"]].apply(
        lambda row: int("Diabetes" in row.values), axis=1
    )
    df["has_digestive"] = df[diag_cols].apply(
        lambda row: int("Digestive" in row.values), axis=1
    )
    df["has_neoplasm"] = df[diag_cols].apply(
        lambda row: int("Neoplasms" in row.values), axis=1
    )

    # Count of distinct morbid categories (multimorbidity index)
    df["distinct_disease_count"] = df[diag_cols].apply(
        lambda row: len(set(val for val in row.values if val not in ["Missing", "Other", "Supplementary/External"])),
        axis=1
    )

    # 4. Medication Volatility & Polypharmacy
    active_meds = []
    for med in MEDICATION_COLS:
        if med in df.columns:
            active_meds.append((df[med] != "No").astype(int))

    if active_meds:
        df["num_active_diabetes_meds"] = sum(active_meds)
    else:
        df["num_active_diabetes_meds"] = 0

    df["on_insulin"] = (df["insulin"] != "No").astype(int)
    df["insulin_dose_changed"] = df["insulin"].isin(["Up", "Down"]).astype(int)
    df["medication_changed"] = (df["change"] == "Ch").astype(int)
    df["on_diabetes_med"] = (df["diabetesMed"] == "Yes").astype(int)

    # Oral agents
    if "metformin" in df.columns:
        df["on_metformin"] = (df["metformin"] != "No").astype(int)
    else:
        df["on_metformin"] = 0

    sulf_meds = [m for m in ["glipizide", "glyburide", "glimepiride"] if m in df.columns]
    if sulf_meds:
        df["on_sulfonylurea"] = df[sulf_meds].apply(
            lambda row: int(any(val != "No" for val in row.values)), axis=1
        )
    else:
        df["on_sulfonylurea"] = 0

    # 5. Glycemic Control & Lab Markers
    df["a1c_tested"] = (df["A1Cresult"] != "None").astype(int)
    df["a1c_elevated"] = df["A1Cresult"].isin([">7", ">8"]).astype(int)
    df["a1c_normal"] = (df["A1Cresult"] == "Norm").astype(int)
    df["glucose_tested"] = (df["max_glu_serum"] != "None").astype(int)
    df["glucose_high"] = df["max_glu_serum"].isin([">200", ">300"]).astype(int)

    # 6. Combined Clinical Risk Multiplier
    # Known clinical triad: prior hospitalization + insulin therapy + discharged to facility
    df["high_risk_triad"] = (
        (df["has_prior_inpatient"] == 1) &
        (df["on_insulin"] == 1) &
        (df["discharged_to_snf"] == 1)
    ).astype(int)

    return df


def get_feature_columns() -> Tuple[List[str], List[str]]:
    """
    Returns numeric and categorical feature names for modeling.
    """
    numeric_features = [
        "age_num",
        "time_in_hospital",
        "num_lab_procedures",
        "num_procedures",
        "num_medications",
        "number_outpatient",
        "number_emergency",
        "number_inpatient",
        "number_diagnoses",
        "total_prior_visits",
        "has_prior_emergency",
        "has_prior_inpatient",
        "frequent_inpatient",
        "utilization_severity_ratio",
        "prolonged_stay",
        "lab_intensity",
        "medication_intensity",
        "polypharmacy_high",
        "has_procedures",
        "has_circulatory",
        "has_respiratory",
        "has_renal",
        "has_diabetes_primary",
        "has_diabetes_secondary",
        "has_digestive",
        "has_neoplasm",
        "distinct_disease_count",
        "num_active_diabetes_meds",
        "on_insulin",
        "insulin_dose_changed",
        "medication_changed",
        "on_diabetes_med",
        "on_metformin",
        "on_sulfonylurea",
        "a1c_tested",
        "a1c_elevated",
        "a1c_normal",
        "glucose_tested",
        "glucose_high",
        "discharged_to_snf",
        "discharged_to_home_health",
        "is_elderly",
        "has_weight_recorded",
        "total_recorded_encounters",
        "high_risk_triad",
    ]

    categorical_features = [
        "race",
        "gender",
        "admission_type",
        "discharge_disposition",
        "admission_source",
        "payer_grouped",
        "specialty_grouped",
        "primary_diag_category",
        "secondary_diag_category",
        "tertiary_diag_category",
        "insulin",
        "A1Cresult",
        "max_glu_serum",
    ]

    return numeric_features, categorical_features


if __name__ == "__main__":
    from src.preprocessing import clean_and_prepare_dataset
    
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    df = clean_and_prepare_dataset()
    df_features = extract_features(df)
    
    feat_output = PROCESSED_DATA_DIR / "features_data.parquet"
    df_features.to_parquet(feat_output, index=False)
    print(f"[SUCCESS] Feature extraction completed. Saved to {feat_output}")
    print(f"[INFO] Features dataframe shape: {df_features.shape}")
