"""
Model Training, Cross-Validation, Clinical Calibration, and TreeSHAP Pipeline
"""

import sys
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
import shap

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.feature_engineering import get_feature_columns
from src.evaluate import compute_clinical_metrics

DATA_PATH = BASE_DIR / "data" / "processed" / "features_data.parquet"
MODELS_DIR = BASE_DIR / "models"


def train_readmission_models():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Loading dataset from: {DATA_PATH}")
    df = pd.read_parquet(DATA_PATH)

    numeric_features, categorical_features = get_feature_columns()

    # Filter to existing columns
    numeric_features = [col for col in numeric_features if col in df.columns]
    categorical_features = [col for col in categorical_features if col in df.columns]

    X = df[numeric_features + categorical_features].copy()
    y = df["readmitted_30d"].values

    print(f"[INFO] Total cohort: {len(X):,} patients, {len(numeric_features)} numeric + {len(categorical_features)} categorical features")
    print(f"[INFO] Positive cases (30-day readmissions): {np.sum(y):,} ({np.mean(y)*100:.2f}%)")

    # Train / Test split (Stratified 80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    print(f"[INFO] Training set: {len(X_train):,} patients | Test set: {len(X_test):,} patients")

    # Preprocessing Pipeline
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, numeric_features),
            ("cat", cat_pipeline, categorical_features),
        ],
        remainder="drop",
    )

    print("[INFO] Fitting preprocessor on training cohort...")
    X_train_trans = preprocessor.fit_transform(X_train)
    X_test_trans = preprocessor.transform(X_test)

    # Get transformed feature names
    cat_encoder = preprocessor.named_transformers_["cat"].named_steps["encoder"]
    encoded_cat_names = list(cat_encoder.get_feature_names_out(categorical_features))
    all_transformed_feature_names = numeric_features + encoded_cat_names
    print(f"[INFO] Total encoded features: {len(all_transformed_feature_names)}")

    # Class weight ratio calculation
    neg_count = np.sum(y_train == 0)
    pos_count = np.sum(y_train == 1)
    imbalance_ratio = neg_count / (pos_count + 1e-9)
    print(f"[INFO] Class imbalance ratio: {imbalance_ratio:.2f}:1")

    # Candidate Models
    models = {
        "Logistic_Regression": LogisticRegression(
            C=0.1,
            class_weight="balanced",
            max_iter=1000,
            random_state=42,
        ),
        "Random_Forest": RandomForestClassifier(
            n_estimators=150,
            max_depth=12,
            min_samples_leaf=10,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=200,
            learning_rate=0.04,
            max_depth=6,
            num_leaves=31,
            scale_pos_weight=2.5,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200,
            learning_rate=0.04,
            max_depth=5,
            scale_pos_weight=2.5,
            subsample=0.85,
            colsample_bytree=0.85,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        ),
    }

    results = {}
    fitted_models = {}

    for name, model in models.items():
        print(f"\n[INFO] Training {name}...")
        model.fit(X_train_trans, y_train)
        fitted_models[name] = model

        # Probability predictions on held-out test set
        y_prob = model.predict_proba(X_test_trans)[:, 1]

        metrics = compute_clinical_metrics(y_test, y_prob)
        results[name] = metrics

        print(f"  -> ROC-AUC: {metrics['roc_auc']:.4f} | PR-AUC: {metrics['pr_auc']:.4f} | "
              f"Brier: {metrics['brier_score']:.4f} | Top-10% Lift: {metrics['top_decile_lift']:.2f}x | "
              f"Recall@Top-10%: {metrics['recall_at_top_10pct']*100:.1f}%")

    # Select Champion Model (by composite score: PR-AUC * Top-10% Lift)
    def composite_rank(m_name):
        return results[m_name]["pr_auc"] * results[m_name]["top_decile_lift"]

    champion_name = max(results.keys(), key=composite_rank)
    print(f"\n[CHAMPION] Selected Champion Model: {champion_name}")
    raw_champion = fitted_models[champion_name]

    # Probability Calibration using Platt Scaling / Sigmoid
    print("[INFO] Applying clinical probability calibration (Sigmoid, 3-fold CV)...")
    calibrated_clf = CalibratedClassifierCV(estimator=raw_champion, method="sigmoid", cv=3)
    calibrated_clf.fit(X_train_trans, y_train)

    # Evaluate Calibrated Model on Test set
    y_prob_calibrated = calibrated_clf.predict_proba(X_test_trans)[:, 1]
    calibrated_metrics = compute_clinical_metrics(y_test, y_prob_calibrated)
    results[f"{champion_name}_Calibrated"] = calibrated_metrics

    print(f"  -> Calibrated ROC-AUC: {calibrated_metrics['roc_auc']:.4f} | PR-AUC: {calibrated_metrics['pr_auc']:.4f} | "
          f"Brier Score: {calibrated_metrics['brier_score']:.4f} (Raw: {results[champion_name]['brier_score']:.4f})")

    # Explainability: Train TreeSHAP Explainer
    print("\n[INFO] Fitting SHAP Explainer for clinical interpretability...")
    # Background dataset for SHAP reference (sample 300 patients)
    bg_sample_idx = np.random.choice(X_train_trans.shape[0], size=min(300, X_train_trans.shape[0]), replace=False)
    bg_sample = X_train_trans[bg_sample_idx]

    try:
        # If LightGBM or XGBoost or RandomForest, TreeExplainer works directly
        if champion_name in ["LightGBM", "XGBoost", "Random_Forest"]:
            explainer = shap.TreeExplainer(raw_champion)
        else:
            explainer = shap.LinearExplainer(raw_champion, bg_sample)
        print("[SUCCESS] SHAP TreeExplainer initialized successfully.")
    except Exception as e:
        print(f"[WARN] TreeExplainer fallback to Exact/KernelExplainer: {e}")
        explainer = shap.Explainer(raw_champion.predict, bg_sample)

    # Calculate global SHAP values on test sample for dashboard reference
    eval_sample_idx = np.random.choice(X_test_trans.shape[0], size=min(500, X_test_trans.shape[0]), replace=False)
    eval_sample = X_test_trans[eval_sample_idx]
    
    try:
        sample_shap_values = explainer(eval_sample)
        # If binary classification returns 3D or list, take positive class
        if hasattr(sample_shap_values, "values") and len(sample_shap_values.values.shape) == 3:
            global_importance_values = np.abs(sample_shap_values.values[:, :, 1]).mean(axis=0)
        elif hasattr(sample_shap_values, "values"):
            global_importance_values = np.abs(sample_shap_values.values).mean(axis=0)
        else:
            global_importance_values = np.abs(np.array(sample_shap_values)).mean(axis=0)
    except Exception as e:
        print(f"[WARN] Could not calculate batch SHAP values: {e}")
        global_importance_values = np.zeros(len(all_transformed_feature_names))

    # Top 20 Global Feature Importances
    top_feature_indices = np.argsort(global_importance_values)[::-1][:20]
    top_features = [
        {
            "feature": all_transformed_feature_names[i],
            "importance": float(global_importance_values[i]),
        }
        for i in top_feature_indices
    ]

    # Save Artifacts
    print("\n[INFO] Saving pipeline artifacts...")
    joblib.dump(preprocessor, MODELS_DIR / "preprocessor.pkl")
    joblib.dump(raw_champion, MODELS_DIR / "raw_model.pkl")
    joblib.dump(calibrated_clf, MODELS_DIR / "calibrated_model.pkl")
    joblib.dump(explainer, MODELS_DIR / "shap_explainer.pkl")

    metadata = {
        "champion_model_name": champion_name,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "all_transformed_feature_names": all_transformed_feature_names,
        "results": results,
        "top_features": top_features,
        "cohort_stats": {
            "total_patients": int(len(df)),
            "train_patients": int(len(X_train)),
            "test_patients": int(len(X_test)),
            "positive_prevalence": float(np.mean(y)),
            "readmitted_30d_count": int(np.sum(y)),
        },
    }

    with open(MODELS_DIR / "model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Save test dataset sample with actual and predicted risk for instant dashboard demonstration
    test_df_sample = df.iloc[X_test.index].copy()
    test_df_sample["predicted_readmission_risk"] = y_prob_calibrated
    test_df_sample["risk_tier"] = pd.cut(
        test_df_sample["predicted_readmission_risk"],
        bins=[-0.01, 0.10, 0.25, 0.50, 1.0],
        labels=["Low Risk (<10%)", "Moderate Risk (10-25%)", "High Risk (25-50%)", "Critical Risk (>50%)"]
    )
    test_df_sample.to_parquet(MODELS_DIR / "test_cohort_predictions.parquet", index=False)

    print(f"[SUCCESS] All model artifacts saved in: {MODELS_DIR}")
    print("[DONE] Training and validation finished successfully!")


if __name__ == "__main__":
    train_readmission_models()
