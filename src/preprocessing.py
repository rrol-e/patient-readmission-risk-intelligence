"""
Clinical Data Preprocessing Module for Hospital Readmission Prediction
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, Any

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = BASE_DIR / "data" / "raw" / "diabetic_data.csv"
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"


def map_icd9_to_category(code: Any) -> str:
    """
    Categorizes ICD-9 codes into 11 clinically recognized diagnostic groups
    based on Clinical Classification Software (CCS) standards.
    """
    if pd.isna(code) or code == "?" or str(code).strip() == "":
        return "Missing"
    code_str = str(code).strip().upper()

    if code_str.startswith("V") or code_str.startswith("E"):
        return "Supplementary/External"

    # Handle float/decimal strings
    try:
        num = float(code_str.split(".")[0])
    except (ValueError, IndexError):
        return "Other"

    if 390 <= num <= 459 or num == 785:
        return "Circulatory"
    elif 460 <= num <= 519 or num == 786:
        return "Respiratory"
    elif 520 <= num <= 579 or num == 787:
        return "Digestive"
    elif num == 250:
        return "Diabetes"
    elif 800 <= num <= 999:
        return "Injury"
    elif 710 <= num <= 739:
        return "Musculoskeletal"
    elif 580 <= num <= 629 or num == 788:
        return "Genitourinary"
    elif 140 <= num <= 239:
        return "Neoplasms"
    elif 240 <= num <= 279:
        return "Endocrine/Metabolic"
    elif 680 <= num <= 709 or num == 782:
        return "Skin"
    elif 290 <= num <= 319:
        return "Mental"
    else:
        return "Other"


def map_admission_type(type_id: int) -> str:
    """Maps admission_type_id to clean clinical description."""
    mapping = {
        1: "Emergency",
        2: "Urgent",
        3: "Elective",
        4: "Newborn",
        7: "Trauma Center",
    }
    return mapping.get(type_id, "Other/Unknown")


def map_discharge_disposition(disp_id: int) -> str:
    """Maps discharge_disposition_id to actionable clinical discharge categories."""
    if disp_id in [1]:
        return "Home"
    elif disp_id in [6, 8]:
        return "Home Health / IV"
    elif disp_id in [3, 4, 24]:
        return "Skilled Nursing / Nursing Facility"
    elif disp_id in [2, 5, 22, 23, 27, 28, 29, 30]:
        return "Transferred / Other Facility"
    elif disp_id == 7:
        return "Left AMA (Against Medical Advice)"
    else:
        return "Other"


def map_admission_source(source_id: int) -> str:
    """Maps admission_source_id into consolidated categories."""
    if source_id == 7:
        return "Emergency Room"
    elif source_id in [1, 2, 3]:
        return "Referral"
    elif source_id in [4, 5, 6, 10, 18, 22]:
        return "Transfer from Facility"
    else:
        return "Other"


def clean_and_prepare_dataset(
    filepath: Path = RAW_DATA_PATH,
    deduplicate_patients: bool = True,
) -> pd.DataFrame:
    """
    Cleans raw diabetic_data.csv according to healthcare epidemiological standards:
    1. Removes deceased and hospice encounters (cannot be readmitted).
    2. Drops invalid/unknown gender.
    3. Handles missing values (? -> structured representations).
    4. Categorizes age into numerical midpoint and clinical groups.
    5. Deduplicates multiple patient encounters (keeps first encounter to avoid data leakage).
    6. Creates binary 30-day readmission target (standard CMS HRRP metric).
    """
    print(f"[INFO] Reading dataset from: {filepath}")
    df = pd.read_csv(filepath)
    initial_rows = len(df)
    print(f"[INFO] Initial rows: {initial_rows:,}, columns: {df.shape[1]}")

    # 1. Exclusion of deceased and hospice patients (IDs 11, 13, 14, 19, 20, 21)
    deceased_hospice_ids = [11, 13, 14, 19, 20, 21]
    df = df[~df["discharge_disposition_id"].isin(deceased_hospice_ids)].copy()
    print(f"[INFO] After excluding deceased/hospice patients: {len(df):,} rows")

    # 2. Exclude invalid gender
    df = df[df["gender"].isin(["Female", "Male"])].copy()

    # 3. Handle Deduplication (Keep first hospital admission per patient)
    if deduplicate_patients:
        # Before deduplication, calculate total visits recorded for each patient
        patient_encounter_counts = df.groupby("patient_nbr")["encounter_id"].count()
        df["total_recorded_encounters"] = df["patient_nbr"].map(patient_encounter_counts)
        df = df.drop_duplicates(subset=["patient_nbr"], keep="first").copy()
        print(f"[INFO] After deduplicating patients (keeping first admission): {len(df):,} unique patients")
    else:
        df["total_recorded_encounters"] = 1

    # 4. Target Variables
    # Binary target for 30-day readmission (CMS quality metric standard)
    df["readmitted_30d"] = (df["readmitted"] == "<30").astype(int)
    # Multiclass target for dashboard reporting
    df["readmission_category"] = df["readmitted"].map({
        "<30": "Readmitted <30 Days",
        ">30": "Readmitted >30 Days",
        "NO": "No Readmission",
    })

    # 5. Clean Age
    # Age format: [0-10), [10-20), ...
    age_map = {
        "[0-10)": 5,
        "[10-20)": 15,
        "[20-30)": 25,
        "[30-40)": 35,
        "[40-50)": 45,
        "[50-60)": 55,
        "[60-70)": 65,
        "[70-80)": 75,
        "[80-90)": 85,
        "[90-100)": 95,
    }
    df["age_num"] = df["age"].map(age_map)
    df["is_elderly"] = (df["age_num"] >= 65).astype(int)

    # 6. Weight missingness
    df["has_weight_recorded"] = (df["weight"] != "?").astype(int)
    df = df.drop(columns=["weight"])

    # 7. Clean Race & Gender
    df["race"] = df["race"].replace("?", "Other/Unknown")

    # 8. Clean Payer Code
    df["payer_code"] = df["payer_code"].replace("?", "Missing")
    # Top payer code grouping
    top_payers = ["MC", "MD", "HM", "UN", "BC", "SP", "CP", "Missing"]
    df["payer_grouped"] = df["payer_code"].apply(lambda x: x if x in top_payers else "Other")

    # 9. Clean Medical Specialty
    df["medical_specialty"] = df["medical_specialty"].replace("?", "Missing")
    top_specialties = [
        "InternalMedicine",
        "Emergency/Trauma",
        "Family/GeneralPractice",
        "Cardiology",
        "Surgery-General",
        "Nephrology",
        "Orthopedics",
        "Orthopedics-Reconstructive",
        "Missing",
    ]
    df["specialty_grouped"] = df["medical_specialty"].apply(
        lambda x: x if x in top_specialties else "Other"
    )

    # 10. Map IDs to clinical descriptors
    df["admission_type"] = df["admission_type_id"].map(map_admission_type)
    df["discharge_disposition"] = df["discharge_disposition_id"].map(map_discharge_disposition)
    df["admission_source"] = df["admission_source_id"].map(map_admission_source)

    # Flag for high-risk discharge (SNF or post-acute facility)
    df["discharged_to_snf"] = (df["discharge_disposition"] == "Skilled Nursing / Nursing Facility").astype(int)
    df["discharged_to_home_health"] = (df["discharge_disposition"] == "Home Health / IV").astype(int)

    # 11. Clean Diagnoses (ICD-9 mapping)
    df["primary_diag_category"] = df["diag_1"].apply(map_icd9_to_category)
    df["secondary_diag_category"] = df["diag_2"].apply(map_icd9_to_category)
    df["tertiary_diag_category"] = df["diag_3"].apply(map_icd9_to_category)

    print(f"[SUCCESS] Preprocessing completed. Prepared dataframe: {df.shape}")
    print(f"[INFO] 30-Day Readmission prevalence: {df['readmitted_30d'].mean()*100:.2f}% "
          f"({df['readmitted_30d'].sum():,} / {len(df):,})")

    return df


if __name__ == "__main__":
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    cleaned_df = clean_and_prepare_dataset()
    output_path = PROCESSED_DATA_DIR / "cleaned_data.parquet"
    cleaned_df.to_parquet(output_path, index=False)
    print(f"[SUCCESS] Saved cleaned dataset to {output_path}")
