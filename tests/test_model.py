"""
Unit tests for Machine Learning Model Artifacts & Inference Pipeline
"""

import json
import joblib
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"


@pytest.fixture(scope="module")
def artifacts():
    """Load pre-trained artifacts once for testing."""
    preprocessor = joblib.load(MODELS_DIR / "preprocessor.pkl")
    raw_model = joblib.load(MODELS_DIR / "raw_model.pkl")
    calibrated_model = joblib.load(MODELS_DIR / "calibrated_model.pkl")
    explainer = joblib.load(MODELS_DIR / "shap_explainer.pkl")
    with open(MODELS_DIR / "model_metadata.json", "r") as f:
        metadata = json.load(f)
    return preprocessor, raw_model, calibrated_model, explainer, metadata


def test_artifacts_exist():
    """Verify all critical serialized assets are present."""
    required = [
        "preprocessor.pkl",
        "raw_model.pkl",
        "calibrated_model.pkl",
        "shap_explainer.pkl",
        "model_metadata.json",
        "test_cohort_predictions.parquet",
    ]
    for filename in required:
        file_path = MODELS_DIR / filename
        assert file_path.exists(), f"Missing required asset: {filename}"
        assert file_path.stat().st_size > 0, f"Empty asset: {filename}"


def test_metadata_structure(artifacts):
    """Verify model metadata structure and champion benchmark metrics."""
    _, _, _, _, metadata = artifacts
    assert "results" in metadata
    assert "LightGBM" in metadata["results"]
    lgbm = metadata["results"]["LightGBM"]
    assert lgbm["roc_auc"] >= 0.75, f"ROC-AUC too low: {lgbm['roc_auc']}"
    assert lgbm["top_decile_lift"] > 2.0, f"Lift too low: {lgbm['top_decile_lift']}"
    assert lgbm["brier_score"] < 0.15, f"Brier score too high: {lgbm['brier_score']}"


def test_inference_pipeline(artifacts):
    """Test full inference transformation and calibrated prediction."""
    preprocessor, _, calibrated_model, _, _ = artifacts

    sample_patient = {
        "age": "[70-80)",
        "age_num": 75,
        "gender": "Female",
        "race": "Caucasian",
        "time_in_hospital": 6,
        "num_lab_procedures": 52,
        "num_procedures": 1,
        "num_medications": 22,
        "number_inpatient": 2,
        "number_emergency": 1,
        "number_outpatient": 0,
        "number_diagnoses": 9,
        "primary_diag_category": "Circulatory",
        "secondary_diag_category": "Diabetes",
        "tertiary_diag_category": "Respiratory",
        "admission_type": "Emergency",
        "discharge_disposition": "Skilled Nursing / Nursing Facility",
        "admission_source": "Emergency Room",
        "payer_grouped": "MC",
        "specialty_grouped": "InternalMedicine",
        "insulin": "Up",
        "change": "Ch",
        "diabetesMed": "Yes",
        "A1Cresult": "None",
        "max_glu_serum": "None",
        "has_weight_recorded": 0,
        "total_recorded_encounters": 3,
        "total_prior_visits": 3,
        "has_prior_emergency": 1,
        "has_prior_inpatient": 1,
        "frequent_inpatient": 1,
        "utilization_severity_ratio": 0.5,
        "prolonged_stay": 0,
        "lab_intensity": 8.66,
        "medication_intensity": 3.66,
        "polypharmacy_high": 1,
        "has_procedures": 1,
        "has_circulatory": 1,
        "has_respiratory": 1,
        "has_renal": 0,
        "has_diabetes_primary": 0,
        "has_diabetes_secondary": 1,
        "has_digestive": 0,
        "has_neoplasm": 0,
        "distinct_disease_count": 3,
        "num_active_diabetes_meds": 2,
        "on_insulin": 1,
        "insulin_dose_changed": 1,
        "medication_changed": 1,
        "on_diabetes_med": 1,
        "on_metformin": 1,
        "on_sulfonylurea": 0,
        "a1c_tested": 0,
        "a1c_elevated": 0,
        "a1c_normal": 0,
        "glucose_tested": 0,
        "glucose_high": 0,
        "discharged_to_snf": 1,
        "discharged_to_home_health": 0,
        "is_elderly": 1,
        "high_risk_triad": 1,
    }

    df = pd.DataFrame([sample_patient])
    X_trans = preprocessor.transform(df)
    assert X_trans.shape[0] == 1
    assert X_trans.shape[1] > 0

    prob = float(calibrated_model.predict_proba(X_trans)[0, 1])
    assert 0.0 <= prob <= 1.0, f"Probability out of range: {prob}"
    # High risk triad should produce elevated risk
    assert prob >= 0.15, f"Expected elevated risk for high-risk patient, got {prob}"


def test_shap_explainer(artifacts):
    """Test TreeSHAP vector calculation."""
    preprocessor, _, _, explainer, _ = artifacts

    df = pd.DataFrame([{
        "age": "[60-70)",
        "age_num": 65,
        "gender": "Male",
        "race": "Caucasian",
        "time_in_hospital": 3,
        "num_lab_procedures": 30,
        "num_procedures": 0,
        "num_medications": 10,
        "number_inpatient": 0,
        "number_emergency": 0,
        "number_outpatient": 0,
        "number_diagnoses": 5,
        "primary_diag_category": "Diabetes",
        "secondary_diag_category": "Circulatory",
        "tertiary_diag_category": "Digestive",
        "admission_type": "Elective",
        "discharge_disposition": "Home",
        "admission_source": "Physician Referral",
        "payer_grouped": "Commercial",
        "specialty_grouped": "FamilyPractice",
        "insulin": "No",
        "change": "No",
        "diabetesMed": "Yes",
        "A1Cresult": "Norm",
        "max_glu_serum": "None",
        "has_weight_recorded": 0,
        "total_recorded_encounters": 1,
        "total_prior_visits": 0,
        "has_prior_emergency": 0,
        "has_prior_inpatient": 0,
        "frequent_inpatient": 0,
        "utilization_severity_ratio": 0.0,
        "prolonged_stay": 0,
        "lab_intensity": 10.0,
        "medication_intensity": 3.33,
        "polypharmacy_high": 0,
        "has_procedures": 0,
        "has_circulatory": 1,
        "has_respiratory": 0,
        "has_renal": 0,
        "has_diabetes_primary": 1,
        "has_diabetes_secondary": 0,
        "has_digestive": 1,
        "has_neoplasm": 0,
        "distinct_disease_count": 3,
        "num_active_diabetes_meds": 1,
        "on_insulin": 0,
        "insulin_dose_changed": 0,
        "medication_changed": 0,
        "on_diabetes_med": 1,
        "on_metformin": 1,
        "on_sulfonylurea": 0,
        "a1c_tested": 1,
        "a1c_elevated": 0,
        "a1c_normal": 1,
        "glucose_tested": 0,
        "glucose_high": 0,
        "discharged_to_snf": 0,
        "discharged_to_home_health": 0,
        "is_elderly": 1,
        "high_risk_triad": 0,
    }])

    X_trans = preprocessor.transform(df)
    shap_vals = explainer(X_trans)
    assert shap_vals is not None
