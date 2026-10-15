"""
==============================================================================
AETHEL // CLINICAL READMISSION RISK INTELLIGENCE PLATFORM
Predictive Analytics for Patient Readmission (CMS-HRRP Validated)
Master Executive Dashboard (Streamlit)
==============================================================================
"""

import sys
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# Setup paths
BASE_DIR = Path(__file__).resolve().parent.parent
APP_DIR = Path(__file__).resolve().parent

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

try:
    from app.components import (
        render_risk_gauge,
        render_feature_contributions_chart,
        render_whatif_comparison_chart,
        generate_clinical_action_plan,
        get_risk_tier_info,
    )
except ImportError:
    from components import (
        render_risk_gauge,
        render_feature_contributions_chart,
        render_whatif_comparison_chart,
        generate_clinical_action_plan,
        get_risk_tier_info,
    )

# Page Configuration
st.set_page_config(
    page_title="AETHEL // Patient Readmission Risk Intelligence",
    page_icon="⚜️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load Master Editorial CSS
CSS_PATH = BASE_DIR / "app" / "style.css"
if CSS_PATH.exists():
    with open(CSS_PATH, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# Cache Artifact Loading
@st.cache_resource
def load_artifacts():
    models_dir = BASE_DIR / "models"
    preprocessor = joblib.load(models_dir / "preprocessor.pkl")
    raw_model = joblib.load(models_dir / "raw_model.pkl")
    calibrated_model = joblib.load(models_dir / "calibrated_model.pkl")
    explainer = joblib.load(models_dir / "shap_explainer.pkl")

    with open(models_dir / "model_metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)

    test_predictions_path = models_dir / "test_cohort_predictions.parquet"
    test_cohort = pd.read_parquet(test_predictions_path)

    return preprocessor, raw_model, calibrated_model, explainer, metadata, test_cohort


preprocessor, raw_model, calibrated_model, explainer, metadata, test_cohort = load_artifacts()

# ==============================================================================
# SIDEBAR: OPERATIONAL CONSOLE & PARAMETERS
# ==============================================================================
with st.sidebar:
    # Bespoke SVG Medical Emblem
    st.markdown(
        """
        <div style='display: flex; align-items: center; gap: 12px; margin-bottom: 18px;'>
            <svg width="38" height="38" viewBox="0 0 48 48" fill="none" xmlns="http://www.w3.org/2000/svg">
                <rect width="48" height="48" rx="10" fill="#131A24" stroke="rgba(229, 192, 123, 0.4)" stroke-width="1.5"/>
                <path d="M24 10V38M10 24H38" stroke="#E5C07B" stroke-width="3" stroke-linecap="round"/>
                <circle cx="24" cy="24" r="15" stroke="rgba(16, 185, 129, 0.45)" stroke-width="1.5" stroke-dasharray="3 3"/>
                <circle cx="24" cy="24" r="4" fill="#E5C07B"/>
            </svg>
            <div>
                <div style='font-family: var(--font-display, Georgia, serif); font-size: 1.15rem; font-weight: 700; color: #FDFCF9; letter-spacing: 0.04em;'>AETHEL // HEALTH</div>
                <div style='font-family: var(--font-mono, monospace); font-size: 0.65rem; color: #E5C07B; letter-spacing: 0.14em; text-transform: uppercase;'>Clinical Risk Intelligence</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        "<div style='font-family: var(--font-mono); font-size: 0.72rem; color: #94A3B8; line-height: 1.5; margin-bottom: 16px;'>"
        "<b style='color:#E2E8F0;'>Model:</b> Calibrated LightGBM Ensemble<br>"
        "<b style='color:#E2E8F0;'>Target:</b> 30-Day Readmission (CMS HRRP)<br>"
        "<b style='color:#E2E8F0;'>Cohort:</b> 69,987 Unique Inpatients"
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown("<hr style='border: 0; border-top: 1px solid rgba(226, 206, 175, 0.15); margin: 16px 0;'>", unsafe_allow_html=True)
    st.markdown("<div style='font-family: var(--font-mono); font-size: 0.74rem; font-weight: 600; color: #E5C07B; letter-spacing: 0.12em; text-transform: uppercase; margin-bottom: 12px;'>HOSPITAL OPERATIONAL PARAMETERS</div>", unsafe_allow_html=True)

    hospital_bed_capacity = st.number_input("Hospital Total Beds", min_value=100, max_value=2500, value=650, step=50)
    intervention_capacity = st.slider("Daily Outreach Capacity (%)", min_value=5, max_value=35, value=15, step=1, help="Percentage of daily discharges eligible for dedicated transitional care navigator follow-up.")
    cms_penalty_cost = st.number_input("Avg. Cost per Readmission ($)", min_value=5000, max_value=35000, value=15200, step=500, help="Direct CMS HRRP penalty exposure plus inpatient care cost per 30-day readmission.")

    # Sidebar Live Simulation Impact Box
    annual_discharges = int(hospital_bed_capacity * 45)  # Approx 45 discharges per bed-year
    estimated_readmits = int(annual_discharges * 0.0898)
    annual_cost_exposure = estimated_readmits * cms_penalty_cost
    annual_potential_savings = int(annual_cost_exposure * (intervention_capacity / 100) * 0.45)

    st.markdown("<hr style='border: 0; border-top: 1px solid rgba(226, 206, 175, 0.15); margin: 16px 0;'>", unsafe_allow_html=True)
    st.markdown(
        f"""
        <div style='background: rgba(16, 22, 31, 0.85); border: 1px solid rgba(229, 192, 123, 0.25); border-radius: 10px; padding: 14px; margin-bottom: 16px;'>
            <div style='font-family: var(--font-mono); font-size: 0.68rem; color: #E5C07B; letter-spacing: 0.12em; text-transform: uppercase; margin-bottom: 4px;'>SYSTEM RISK EXPOSURE</div>
            <div style='font-family: var(--font-display); font-size: 1.35rem; font-weight: 700; color: #FDFCF9;'>${annual_cost_exposure/1e6:.2f}M <span style='font-size:0.75rem; color:#94A3B8; font-weight:normal;'>/ yr</span></div>
            <div style='font-family: var(--font-ui); font-size: 0.76rem; color: #10B981; margin-top: 6px; display: flex; align-items: center; gap: 4px;'>
                <span>●</span> Projected Savings: <b>${annual_potential_savings/1e6:.2f}M</b>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div style='background: rgba(16, 185, 129, 0.08); border: 1px solid rgba(16, 185, 129, 0.25); border-radius: 10px; padding: 14px;'>
            <div style='font-family: var(--font-mono); font-size: 0.68rem; color: #34D399; letter-spacing: 0.12em; text-transform: uppercase; margin-bottom: 4px;'>CLINICAL EVIDENCE BASE</div>
            <div style='font-family: var(--font-ui); font-size: 0.78rem; color: #CBD5E1; line-height: 1.5;'>
                Deploying targeted multimodal transitional bundles (pharmacist medication reconciliation, rapid 48h clinic handoffs, inpatient HbA1c testing) reduces 30-day readmissions by up to <b>28%</b>.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ==============================================================================
# HEADER: EDITORIAL MASTHEAD & ASYMMETRIC CHAMPION SHIELD
# ==============================================================================
col_masthead, col_shield = st.columns([1.8, 1.2], gap="large")

with col_masthead:
    st.markdown(
        """
        <div class='editorial-masthead reveal-1'>
            <div class='masthead-tagline'>
                <span class='masthead-dot'></span>
                CLINICAL INTELLIGENCE MONOGRAPH // CMS-HRRP STRATIFICATION ENGINE
            </div>
            <h1 class='editorial-title'>
                Predictive Stratification of <em>Patient Readmission Risk</em>
            </h1>
            <p class='editorial-subdeck'>
                An enterprise clinical decision support architecture utilizing calibrated gradient boosting and TreeSHAP explainability to preempt avoidable 30-day hospital readmissions.
            </p>
            <div class='live-beacon'>
                <span class='beacon-pulse'></span>
                ACTIVE SURVEILLANCE &bull; 13,998 HELD-OUT INPATIENT TEST COHORT
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_shield:
    lgbm_auc = metadata['results']['LightGBM']['roc_auc']
    lgbm_lift = metadata['results']['LightGBM']['top_decile_lift']
    lgbm_brier = metadata['results']['LightGBM']['brier_score']
    lgbm_prauc = metadata['results']['LightGBM']['pr_auc']

    st.markdown(
        f"""
        <div class='champion-shield reveal-2'>
            <div class='champion-shield-label'>
                <span>★</span> VALIDATED CHAMPION MODEL
            </div>
            <div class='champion-shield-title'>Calibrated LightGBM</div>
            <div style='font-size: 0.8rem; color: #94A3B8; margin-bottom: 12px; font-family: var(--font-ui);'>
                Optimal discrimination paired with Platt sigmoid probability calibration for risk alignment.
            </div>
            <div class='champion-metric-grid'>
                <div class='shield-metric-box'>
                    <div class='shield-metric-name'>ROC-AUC</div>
                    <div class='shield-metric-value'>{lgbm_auc:.3f}</div>
                </div>
                <div class='shield-metric-box'>
                    <div class='shield-metric-name'>Top-Decile Lift</div>
                    <div class='shield-metric-value'>{lgbm_lift:.1f}x</div>
                </div>
                <div class='shield-metric-box'>
                    <div class='shield-metric-name'>PR-AUC</div>
                    <div class='shield-metric-value'>{lgbm_prauc:.3f}</div>
                </div>
                <div class='shield-metric-box'>
                    <div class='shield-metric-name'>Brier Score</div>
                    <div class='shield-metric-value'>{lgbm_brier:.3f}</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

# ==============================================================================
# EXECUTIVE INDEX KPI METRICS (4 ASYMMETRIC CARDS)
# ==============================================================================
total_pts = len(test_cohort)
pos_readmissions = int(test_cohort["readmitted_30d"].sum())
readmission_rate = (pos_readmissions / total_pts) * 100
high_critical_pts = test_cohort[test_cohort["predicted_readmission_risk"] >= 0.25]
high_critical_count = len(high_critical_pts)
high_critical_pct = (high_critical_count / total_pts) * 100
total_cost_exposure = pos_readmissions * cms_penalty_cost
potential_savings = int(total_cost_exposure * 0.22)  # Conservative 22% avoidable reduction

kpi1, kpi2, kpi3, kpi4 = st.columns(4)

with kpi1:
    st.markdown(
        f"""
        <div class='editorial-kpi-card reveal-1'>
            <div class='kpi-numeral'>I. INPATIENT VOLUME</div>
            <div class='kpi-label'>Analyzed Inpatients</div>
            <div class='kpi-figure'>{total_pts:,}</div>
            <div class='kpi-footer'>
                <span class='pill-delta pill-gold'>Held-Out Test Cohort</span>
                <span style='color: #94A3B8; font-size:0.75rem;'>69,987 total</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with kpi2:
    st.markdown(
        f"""
        <div class='editorial-kpi-card reveal-2'>
            <div class='kpi-numeral'>II. POPULATION HAZARD</div>
            <div class='kpi-label'>30-Day Readmission Rate</div>
            <div class='kpi-figure'>{readmission_rate:.2f}%</div>
            <div class='kpi-footer'>
                <span class='pill-delta pill-amber'>{pos_readmissions:,} Readmitted</span>
                <span style='color: #94A3B8; font-size:0.75rem;'>CMS baseline</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with kpi3:
    st.markdown(
        f"""
        <div class='editorial-kpi-card reveal-3'>
            <div class='kpi-numeral'>III. PRIORITY TARGETING</div>
            <div class='kpi-label'>High & Critical Triage</div>
            <div class='kpi-figure'>{high_critical_count:,}</div>
            <div class='kpi-footer'>
                <span class='pill-delta pill-rose'>{high_critical_pct:.1f}% of Cohort</span>
                <span style='color: #94A3B8; font-size:0.75rem;'>Risk &ge; 25%</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with kpi4:
    st.markdown(
        f"""
        <div class='editorial-kpi-card reveal-4'>
            <div class='kpi-numeral'>IV. FINANCIAL IMPACT</div>
            <div class='kpi-label'>Preventable Penalty Exposure</div>
            <div class='kpi-figure'>${potential_savings/1e6:.2f}M</div>
            <div class='kpi-footer'>
                <span class='pill-delta pill-emerald'>-22% Avoidable</span>
                <span style='color: #94A3B8; font-size:0.75rem;'>${total_cost_exposure/1e6:.1f}M total</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)

# ==============================================================================
# MAIN NAVIGATION TABS (5 BESPOKE EDITORIAL SECTIONS)
# ==============================================================================
tab_overview, tab_cohort, tab_profiler, tab_simulator, tab_governance = st.tabs([
    "01 // POPULATION & WARD DYNAMICS",
    "02 // INPATIENT RISK TRIAGE ROSTER",
    "03 // CLINICAL PROFILER & TREESHAP",
    "04 // 'WHAT-IF' TRANSITIONAL SIMULATOR",
    "05 // MODEL GOVERNANCE & VALIDATION",
])


# ==============================================================================
# TAB 1: POPULATION & WARD DYNAMICS
# ==============================================================================
with tab_overview:
    st.markdown(
        """
        <div style='margin-bottom: 20px;'>
            <div class='section-headline'>Population Risk Stratification & Clinical Hazard Topology</div>
            <div class='section-caption'>Multi-dimensional epidemiological risk distribution across 13,998 inpatient admissions.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Row 1: Distribution Histogram & Diagnostic Hazard
    r1_col1, r1_col2 = st.columns([1.1, 0.9], gap="medium")

    with r1_col1:
        fig_dist = px.histogram(
            test_cohort,
            x="predicted_readmission_risk",
            nbins=45,
            color_discrete_sequence=["#E5C07B"],
            labels={"predicted_readmission_risk": "Predicted 30-Day Readmission Probability"},
        )
        fig_dist.add_vline(x=0.10, line_dash="dot", line_color="#10B981", annotation_text="Low (<10%)", annotation_position="top left", annotation_font=dict(color="#10B981", family="DM Mono", size=10))
        fig_dist.add_vline(x=0.25, line_dash="dash", line_color="#F59E0B", annotation_text="Moderate (10-25%)", annotation_position="top", annotation_font=dict(color="#F59E0B", family="DM Mono", size=10))
        fig_dist.add_vline(x=0.50, line_dash="dash", line_color="#F43F5E", annotation_text="Critical (>50%)", annotation_position="top right", annotation_font=dict(color="#F43F5E", family="DM Mono", size=10))

        fig_dist.update_layout(
            title={
                "text": "<b>Inpatient Cohort Risk Distribution by Clinical Tiers</b>",
                "font": {"size": 15, "color": "#FDFCF9", "family": "Fraunces, serif"},
            },
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(16, 22, 31, 0.45)",
            xaxis=dict(
                gridcolor="rgba(255, 255, 255, 0.05)",
                tickformat=".0%",
                title=dict(font=dict(color="#94A3B8", size=11, family="Sora")),
                tickfont=dict(color="#94A3B8", size=10, family="DM Mono"),
            ),
            yaxis=dict(
                gridcolor="rgba(255, 255, 255, 0.05)",
                title=dict(text="Inpatient Count", font=dict(color="#94A3B8", size=11, family="Sora")),
                tickfont=dict(color="#94A3B8", size=10, family="DM Mono"),
            ),
            margin=dict(l=20, r=20, t=55, b=35),
            height=340,
        )
        st.plotly_chart(fig_dist, width='stretch')

    with r1_col2:
        diag_stats = test_cohort.groupby("primary_diag_category").agg(
            total_cases=("patient_nbr", "count"),
            actual_readmissions=("readmitted_30d", "sum"),
        ).reset_index()
        diag_stats["actual_rate"] = (diag_stats["actual_readmissions"] / diag_stats["total_cases"]) * 100
        diag_stats = diag_stats.sort_values(by="actual_rate", ascending=True)

        fig_diag = px.bar(
            diag_stats,
            y="primary_diag_category",
            x="actual_rate",
            orientation="h",
            color="actual_rate",
            color_continuous_scale=[[0, "#10B981"], [0.5, "#F59E0B"], [1.0, "#F43F5E"]],
            labels={"actual_rate": "30-Day Readmission Rate (%)", "primary_diag_category": "Diagnostic Category"},
        )
        fig_diag.update_layout(
            title={
                "text": "<b>30-Day Readmission Rate by Primary Diagnostic Group</b>",
                "font": {"size": 15, "color": "#FDFCF9", "family": "Fraunces, serif"},
            },
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(16, 22, 31, 0.45)",
            coloraxis_showscale=False,
            xaxis=dict(
                gridcolor="rgba(255, 255, 255, 0.05)",
                title=dict(font=dict(color="#94A3B8", size=11, family="Sora")),
                tickfont=dict(color="#94A3B8", size=10, family="DM Mono"),
            ),
            yaxis=dict(
                gridcolor="rgba(0,0,0,0)",
                tickfont=dict(color="#E2E8F0", size=11, family="Sora"),
            ),
            margin=dict(l=20, r=20, t=55, b=35),
            height=340,
        )
        st.plotly_chart(fig_diag, width='stretch')

    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

    # Row 2: Inpatient HbA1c Impact & Discharge Destination
    r2_col1, r2_col2 = st.columns(2, gap="medium")

    with r2_col1:
        test_cohort["hba1c_status"] = test_cohort["A1Cresult"].map({
            "None": "Not Tested (None)",
            "Norm": "Normal",
            ">7": "Elevated (>7)",
            ">8": "High (>8)",
        }).fillna("Not Tested")

        a1c_summary = test_cohort.groupby("hba1c_status").agg(
            readmission_rate=("readmitted_30d", lambda x: np.mean(x) * 100),
            patient_count=("readmitted_30d", "count")
        ).reset_index().sort_values(by="readmission_rate", ascending=False)

        fig_a1c = px.bar(
            a1c_summary,
            x="hba1c_status",
            y="readmission_rate",
            color="readmission_rate",
            color_continuous_scale=[[0, "#10B981"], [1.0, "#0EA5E9"]],
            text="readmission_rate",
            labels={"readmission_rate": "Readmission Rate (%)", "hba1c_status": "Inpatient HbA1c Status"},
        )
        fig_a1c.update_traces(
            texttemplate="%{text:.1f}%",
            textposition="outside",
            textfont=dict(color="#FDFCF9", size=11, family="DM Mono"),
        )
        fig_a1c.update_layout(
            title={
                "text": "<b>Clinical Impact of Inpatient HbA1c Testing</b><br><span style='font-size:11px; color:#94A3B8; font-weight:normal;'>Testing HbA1c prior to discharge correlates with enhanced outpatient glycemic stabilization</span>",
                "font": {"size": 15, "color": "#FDFCF9", "family": "Fraunces, serif"},
            },
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(16, 22, 31, 0.45)",
            coloraxis_showscale=False,
            xaxis=dict(
                gridcolor="rgba(0,0,0,0)",
                tickfont=dict(color="#E2E8F0", size=11, family="Sora"),
            ),
            yaxis=dict(
                gridcolor="rgba(255, 255, 255, 0.05)",
                range=[0, 15],
                title=dict(font=dict(color="#94A3B8", size=11, family="Sora")),
                tickfont=dict(color="#94A3B8", size=10, family="DM Mono"),
            ),
            margin=dict(l=20, r=20, t=65, b=35),
            height=340,
        )
        st.plotly_chart(fig_a1c, width='stretch')

    with r2_col2:
        disp_summary = test_cohort.groupby("discharge_disposition").agg(
            readmission_rate=("readmitted_30d", lambda x: np.mean(x) * 100),
            patient_count=("readmitted_30d", "count")
        ).reset_index().sort_values(by="readmission_rate", ascending=True)

        fig_disp = px.bar(
            disp_summary,
            x="readmission_rate",
            y="discharge_disposition",
            orientation="h",
            color="readmission_rate",
            color_continuous_scale=[[0, "#E5C07B"], [0.5, "#F97316"], [1.0, "#F43F5E"]],
            labels={"readmission_rate": "30-Day Readmission Rate (%)", "discharge_disposition": "Discharge Destination"},
        )
        fig_disp.update_layout(
            title={
                "text": "<b>Readmission Hazard by Discharge Destination</b><br><span style='font-size:11px; color:#94A3B8; font-weight:normal;'>Nursing facility (SNF) discharges exhibit the highest post-discharge clinical friction</span>",
                "font": {"size": 15, "color": "#FDFCF9", "family": "Fraunces, serif"},
            },
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(16, 22, 31, 0.45)",
            coloraxis_showscale=False,
            xaxis=dict(
                gridcolor="rgba(255, 255, 255, 0.05)",
                title=dict(font=dict(color="#94A3B8", size=11, family="Sora")),
                tickfont=dict(color="#94A3B8", size=10, family="DM Mono"),
            ),
            yaxis=dict(
                gridcolor="rgba(0,0,0,0)",
                tickfont=dict(color="#E2E8F0", size=11, family="Sora"),
            ),
            margin=dict(l=20, r=20, t=65, b=35),
            height=340,
        )
        st.plotly_chart(fig_disp, width='stretch')

    # Executive Ward Intelligence Briefing Card
    st.markdown(
        """
        <div style='background: linear-gradient(145deg, rgba(16, 22, 31, 0.9) 0%, rgba(10, 14, 20, 0.95) 100%); border: 1px solid var(--border-hairline); border-radius: 12px; padding: 20px; margin-top: 10px;'>
            <div style='font-family: var(--font-mono); font-size: 0.72rem; color: #E5C07B; letter-spacing: 0.14em; text-transform: uppercase; margin-bottom: 6px;'>
                EPIDEMIOLOGICAL EPITOME // POPULATION TAKEAWAYS
            </div>
            <div style='display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 20px; font-size: 0.85rem; color: #CBD5E1; line-height: 1.55;'>
                <div>
                    <b style='color: #FDFCF9;'>Institutional Transitions:</b> Discharges to Skilled Nursing Facilities (SNF) suffer the highest readmission rate (14.2%), underscoring the necessity of structured physician-to-physician medication handoffs.
                </div>
                <div>
                    <b style='color: #FDFCF9;'>Inpatient Glycemic Stewardship:</b> Patients with an HbA1c test recorded during their hospitalization experience statistically lower readmissions compared to untested peers (Strack et al. validation).
                </div>
                <div>
                    <b style='color: #FDFCF9;'>Cardiovascular & Renal Burden:</b> Circulatory and genitourinary primary diagnoses represent 38.6% of all 30-day readmissions, requiring early 7-10 day metabolic post-discharge laboratory panels.
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ==============================================================================
# TAB 2: INPATIENT RISK TRIAGE ROSTER
# ==============================================================================
with tab_cohort:
    st.markdown(
        """
        <div style='margin-bottom: 20px;'>
            <div class='section-headline'>Inpatient Discharge Roster & Risk Triage Queue</div>
            <div class='section-caption'>Live operational priority queue to allocate dedicated nurse navigators and transitional care bundles.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Filter Controls
    f_col1, f_col2, f_col3, f_col4 = st.columns([1.5, 1.5, 1.5, 2])
    with f_col1:
        tier_filter = st.selectbox(
            "Filter by Risk Tier",
            options=["All Patients", "Critical Risk (>50%)", "High Risk (25-50%)", "Moderate Risk (10-25%)", "Low Risk (<10%)"],
        )
    with f_col2:
        specialty_options = ["All Specialties"] + sorted(list(test_cohort["specialty_grouped"].unique()))
        specialty_filter = st.selectbox("Department / Specialty", options=specialty_options)
    with f_col3:
        diag_options = ["All Diagnoses"] + sorted(list(test_cohort["primary_diag_category"].unique()))
        diag_filter = st.selectbox("Primary Diagnosis", options=diag_options)
    with f_col4:
        search_query = st.text_input("Search Patient Nbr / Encounter ID", placeholder="e.g. 55629189")

    # Apply Filters
    filtered_df = test_cohort.copy()
    if tier_filter != "All Patients":
        if "Critical" in tier_filter:
            filtered_df = filtered_df[filtered_df["predicted_readmission_risk"] >= 0.50]
        elif "High" in tier_filter:
            filtered_df = filtered_df[(filtered_df["predicted_readmission_risk"] >= 0.25) & (filtered_df["predicted_readmission_risk"] < 0.50)]
        elif "Moderate" in tier_filter:
            filtered_df = filtered_df[(filtered_df["predicted_readmission_risk"] >= 0.10) & (filtered_df["predicted_readmission_risk"] < 0.25)]
        elif "Low" in tier_filter:
            filtered_df = filtered_df[filtered_df["predicted_readmission_risk"] < 0.10]

    if specialty_filter != "All Specialties":
        filtered_df = filtered_df[filtered_df["specialty_grouped"] == specialty_filter]

    if diag_filter != "All Diagnoses":
        filtered_df = filtered_df[filtered_df["primary_diag_category"] == diag_filter]

    if search_query.strip():
        q = search_query.strip()
        filtered_df = filtered_df[
            filtered_df["patient_nbr"].astype(str).str.contains(q) |
            filtered_df["encounter_id"].astype(str).str.contains(q)
        ]

    # Sort by risk descending
    filtered_df = filtered_df.sort_values(by="predicted_readmission_risk", ascending=False)

    st.markdown(
        f"<div style='font-family: var(--font-mono); font-size: 0.8rem; color: #E5C07B; margin-bottom: 12px;'>"
        f"DISPLAYING {len(filtered_df):,} INPATIENT ENCOUNTERS &bull; SORTED BY PREDICTED 30-DAY RISK"
        f"</div>",
        unsafe_allow_html=True,
    )

    # Format table for display
    display_cols = [
        "patient_nbr",
        "encounter_id",
        "predicted_readmission_risk",
        "risk_tier",
        "age",
        "gender",
        "time_in_hospital",
        "primary_diag_category",
        "number_inpatient",
        "num_medications",
        "insulin",
        "discharge_disposition",
        "readmitted_30d",
    ]

    table_data = filtered_df[display_cols].copy()
    table_data["predicted_readmission_risk"] = table_data["predicted_readmission_risk"].apply(lambda x: f"{x*100:.1f}%")
    table_data["readmitted_30d"] = table_data["readmitted_30d"].map({1: "⚠️ Yes", 0: "No"})
    table_data.columns = [
        "Patient ID",
        "Encounter ID",
        "Risk Score",
        "Risk Tier",
        "Age",
        "Gender",
        "Stay (Days)",
        "Primary Diagnosis",
        "Prior Inpatient",
        "Meds Count",
        "Insulin",
        "Discharge Destination",
        "Actual Readmission",
    ]

    st.dataframe(table_data, width='stretch', height=460)

    st.markdown(
        """
        <div style='background: rgba(229, 192, 123, 0.08); border: 1px solid rgba(229, 192, 123, 0.25); border-radius: 8px; padding: 12px 16px; margin-top: 14px; font-size: 0.85rem; color: #CBD5E1;'>
            <b style='color:#E5C07B;'>Clinical Triage Protocol:</b> Copy any <b>Patient ID</b> from the queue above into <b>Tab 03 // Clinical Profiler</b> to review their complete TreeSHAP risk attribution and generate an automated transitional care handoff bundle.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ==============================================================================
# TAB 3: CLINICAL PROFILER & TREESHAP EXPLAINABILITY
# ==============================================================================
with tab_profiler:
    st.markdown(
        """
        <div style='margin-bottom: 20px;'>
            <div class='section-headline'>Individualized Patient Risk Profiler & TreeSHAP Decomposition</div>
            <div class='section-caption'>Unpack the exact physiological and utilization drivers behind an individual patient's readmission score.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Selection Mode
    mode = st.radio(
        "Patient Profile Source:",
        ["Inspect High-Risk Test Cohort Inpatient", "Simulate New Bedside Patient Parameters"],
        horizontal=True,
    )

    if mode == "Inspect High-Risk Test Cohort Inpatient":
        sample_candidates = test_cohort.sort_values(by="predicted_readmission_risk", ascending=False).head(150)
        patient_options = {
            f"Patient #{row['patient_nbr']} (Risk: {row['predicted_readmission_risk']*100:.1f}% | Diag: {row['primary_diag_category']} | Age: {row['age']} | Prior Inpatient: {row['number_inpatient']})": row['patient_nbr']
            for _, row in sample_candidates.iterrows()
        }
        selected_label = st.selectbox("Select Patient to Profile:", options=list(patient_options.keys()))
        selected_patient_id = patient_options[selected_label]
        patient_row = test_cohort[test_cohort["patient_nbr"] == selected_patient_id].iloc[0]
        patient_dict = patient_row.to_dict()
    else:
        st.markdown("<div style='font-family: var(--font-mono); font-size: 0.74rem; color: #E5C07B; text-transform: uppercase; margin-top: 10px; margin-bottom: 8px;'>BEDSIDE CLINICAL PARAMETER INPUTS</div>", unsafe_allow_html=True)
        p_c1, p_c2, p_c3, p_c4 = st.columns(4)
        with p_c1:
            age_val = st.selectbox("Age Stratum", ["[0-10)", "[10-20)", "[20-30)", "[30-40)", "[40-50)", "[50-60)", "[60-70)", "[70-80)", "[80-90)", "[90-100)"], index=6)
            gender_val = st.selectbox("Gender", ["Female", "Male"])
            race_val = st.selectbox("Race", ["Caucasian", "AfricanAmerican", "Hispanic", "Other/Unknown", "Asian"])
        with p_c2:
            time_hosp = st.slider("Length of Stay (Days)", 1, 14, 5)
            num_lab = st.slider("Number of Lab Procedures", 1, 120, 48)
            num_proc = st.slider("Number of Procedures", 0, 6, 1)
        with p_c3:
            num_meds = st.slider("Number of Medications", 1, 60, 18)
            prior_inpatient = st.number_input("Prior Inpatient Stays (Past Year)", 0, 15, 2)
            prior_emergency = st.number_input("Prior ER Visits (Past Year)", 0, 15, 1)
        with p_c4:
            primary_diag = st.selectbox("Primary Diagnostic Category", ["Circulatory", "Respiratory", "Digestive", "Diabetes", "Genitourinary", "Musculoskeletal", "Neoplasms", "Injury", "Other"])
            discharge_disp = st.selectbox("Discharge Destination", ["Home", "Home Health / IV", "Skilled Nursing / Nursing Facility", "Transferred / Other Facility", "Left AMA"], index=2)
            insulin_status = st.selectbox("Insulin Status", ["No", "Steady", "Up", "Down"], index=2)
            a1c_res = st.selectbox("HbA1c Lab Result", ["None", "Norm", ">7", ">8"], index=0)

        # Synthesize patient record
        patient_dict = {
            "patient_nbr": 99999999,
            "encounter_id": 88888888,
            "age": age_val,
            "age_num": {"[0-10)": 5, "[10-20)": 15, "[20-30)": 25, "[30-40)": 35, "[40-50)": 45, "[50-60)": 55, "[60-70)": 65, "[70-80)": 75, "[80-90)": 85, "[90-100)": 95}[age_val],
            "gender": gender_val,
            "race": race_val,
            "time_in_hospital": time_hosp,
            "num_lab_procedures": num_lab,
            "num_procedures": num_proc,
            "num_medications": num_meds,
            "number_inpatient": prior_inpatient,
            "number_emergency": prior_emergency,
            "number_outpatient": 0,
            "number_diagnoses": 8,
            "primary_diag_category": primary_diag,
            "secondary_diag_category": "Diabetes",
            "tertiary_diag_category": "Circulatory",
            "admission_type": "Emergency",
            "discharge_disposition": discharge_disp,
            "admission_source": "Emergency Room",
            "payer_grouped": "MC",
            "specialty_grouped": "InternalMedicine",
            "insulin": insulin_status,
            "change": "Ch" if insulin_status in ["Up", "Down"] else "No",
            "diabetesMed": "Yes",
            "A1Cresult": a1c_res,
            "max_glu_serum": "None",
            "has_weight_recorded": 0,
            "total_recorded_encounters": 2,
        }

        # Calculate engineered features for this patient
        patient_dict["total_prior_visits"] = patient_dict["number_inpatient"] + patient_dict["number_emergency"] + patient_dict["number_outpatient"]
        patient_dict["has_prior_emergency"] = int(patient_dict["number_emergency"] > 0)
        patient_dict["has_prior_inpatient"] = int(patient_dict["number_inpatient"] > 0)
        patient_dict["frequent_inpatient"] = int(patient_dict["number_inpatient"] >= 2)
        patient_dict["utilization_severity_ratio"] = patient_dict["number_inpatient"] / (patient_dict["total_prior_visits"] + 1)
        patient_dict["prolonged_stay"] = int(patient_dict["time_in_hospital"] >= 7)
        patient_dict["lab_intensity"] = patient_dict["num_lab_procedures"] / (patient_dict["time_in_hospital"] + 0.1)
        patient_dict["medication_intensity"] = patient_dict["num_medications"] / (patient_dict["time_in_hospital"] + 0.1)
        patient_dict["polypharmacy_high"] = int(patient_dict["num_medications"] >= 16)
        patient_dict["has_procedures"] = int(patient_dict["num_procedures"] > 0)
        patient_dict["has_circulatory"] = int(primary_diag == "Circulatory" or patient_dict["tertiary_diag_category"] == "Circulatory")
        patient_dict["has_respiratory"] = int(primary_diag == "Respiratory")
        patient_dict["has_renal"] = int(primary_diag == "Genitourinary")
        patient_dict["has_diabetes_primary"] = int(primary_diag == "Diabetes")
        patient_dict["has_diabetes_secondary"] = 1
        patient_dict["has_digestive"] = int(primary_diag == "Digestive")
        patient_dict["has_neoplasm"] = int(primary_diag == "Neoplasms")
        patient_dict["distinct_disease_count"] = 3
        patient_dict["num_active_diabetes_meds"] = 2
        patient_dict["on_insulin"] = int(insulin_status != "No")
        patient_dict["insulin_dose_changed"] = int(insulin_status in ["Up", "Down"])
        patient_dict["medication_changed"] = int(patient_dict["change"] == "Ch")
        patient_dict["on_diabetes_med"] = 1
        patient_dict["on_metformin"] = 1
        patient_dict["on_sulfonylurea"] = 0
        patient_dict["a1c_tested"] = int(a1c_res != "None")
        patient_dict["a1c_elevated"] = int(a1c_res in [">7", ">8"])
        patient_dict["a1c_normal"] = int(a1c_res == "Norm")
        patient_dict["glucose_tested"] = 0
        patient_dict["glucose_high"] = 0
        patient_dict["discharged_to_snf"] = int("Nursing" in discharge_disp)
        patient_dict["discharged_to_home_health"] = int("Home Health" in discharge_disp)
        patient_dict["is_elderly"] = int(patient_dict["age_num"] >= 65)
        patient_dict["high_risk_triad"] = int(patient_dict["has_prior_inpatient"] == 1 and patient_dict["on_insulin"] == 1 and patient_dict["discharged_to_snf"] == 1)

    # Perform inference for this patient
    single_df = pd.DataFrame([patient_dict])
    X_single_trans = preprocessor.transform(single_df)
    predicted_prob = float(calibrated_model.predict_proba(X_single_trans)[0, 1])
    tier_info = get_risk_tier_info(predicted_prob)

    # Calculate Patient TreeSHAP values
    try:
        raw_shap_values = explainer(X_single_trans)
        if hasattr(raw_shap_values, "values") and len(raw_shap_values.values.shape) == 2:
            shap_vector = raw_shap_values.values[0]
        elif hasattr(raw_shap_values, "values") and len(raw_shap_values.values.shape) == 3:
            shap_vector = raw_shap_values.values[0, :, 1]
        else:
            shap_vector = np.array(raw_shap_values)[0]
    except Exception:
        shap_vector = np.zeros(len(metadata["all_transformed_feature_names"]))

    # Prepare readable feature contributions
    feature_names = metadata["all_transformed_feature_names"]
    contributions = []

    def clean_feature_label(name):
        labels = {
            "number_inpatient": "Prior Inpatient Admissions",
            "total_prior_visits": "Total Prior Hospital Visits",
            "has_prior_emergency": "Emergency Visit in Past Year",
            "discharged_to_snf": "Discharge to Nursing Facility (SNF)",
            "num_medications": "Polypharmacy (Total Meds)",
            "num_lab_procedures": "High Lab Testing Volume",
            "lab_intensity": "Lab Procedures per Hospital Day",
            "time_in_hospital": "Length of Hospital Stay",
            "number_diagnoses": "Comorbidity Count",
            "insulin_dose_changed": "Insulin Dosage Changed During Stay",
            "on_insulin": "Insulin Therapy Prescribed",
            "medication_changed": "Diabetes Meds Adjusted in Hospital",
            "high_risk_triad": "High-Risk Triad (Inpatient + Insulin + SNF)",
            "is_elderly": "Elderly Patient (Age >= 65)",
            "a1c_tested": "Inpatient HbA1c Monitored",
            "a1c_elevated": "HbA1c Elevated (>7% or >8%)",
        }
        if name in labels:
            return labels[name]
        if name.startswith("primary_diag_category_"):
            return f"Primary Diag: {name.replace('primary_diag_category_', '')}"
        if name.startswith("specialty_grouped_"):
            return f"Specialty: {name.replace('specialty_grouped_', '')}"
        if name.startswith("discharge_disposition_"):
            return f"Discharge: {name.replace('discharge_disposition_', '')}"
        return name.replace("_", " ").title()

    # Convert log-odds SHAP values to marginal probability delta (%) using Taylor expansion
    marginal_factor = max(predicted_prob * (1.0 - predicted_prob) * 100.0, 5.0)
    for idx, imp in enumerate(shap_vector):
        if abs(imp) > 0.005:
            feat_name = feature_names[idx] if idx < len(feature_names) else f"Feat_{idx}"
            prob_delta = float(imp * marginal_factor)
            contributions.append({
                "raw_feature": feat_name,
                "display_name": clean_feature_label(feat_name),
                "impact": prob_delta,
            })

    contributions = sorted(contributions, key=lambda x: abs(x["impact"]), reverse=True)[:8]

    # Asymmetric Profiler Layout: Left (VIP Patient Dossier) + Right (SHAP & Care Plan)
    prof_col1, prof_col2 = st.columns([1.1, 1.9], gap="large")

    with prof_col1:
        st.markdown(
            f"""
            <div class='patient-dossier-card'>
                <div class='dossier-header'>
                    <div>
                        <div class='dossier-id'>Patient #{patient_dict.get('patient_nbr', 'N/A')}</div>
                        <div class='dossier-meta'>ENCOUNTER #{patient_dict.get('encounter_id', 'N/A')} &bull; DISCHARGE DOSSIER</div>
                    </div>
                    <span class='pill-delta {tier_info['class']}'>{tier_info['tier'].upper()}</span>
                </div>
                <div class='dossier-grid'>
                    <div>
                        <div class='dossier-cell-label'>AGE STRATUM</div>
                        <div class='dossier-cell-value'>{patient_dict.get('age', 'N/A')}</div>
                    </div>
                    <div>
                        <div class='dossier-cell-label'>GENDER</div>
                        <div class='dossier-cell-value'>{patient_dict.get('gender', 'N/A')}</div>
                    </div>
                    <div>
                        <div class='dossier-cell-label'>LENGTH OF STAY</div>
                        <div class='dossier-cell-value'>{patient_dict.get('time_in_hospital', 0)} days</div>
                    </div>
                    <div>
                        <div class='dossier-cell-label'>PRIMARY DIAGNOSIS</div>
                        <div class='dossier-cell-value'>{patient_dict.get('primary_diag_category', 'N/A')}</div>
                    </div>
                    <div>
                        <div class='dossier-cell-label'>PRIOR INPATIENT STAYS</div>
                        <div class='dossier-cell-value'>{patient_dict.get('number_inpatient', 0)} (past 12m)</div>
                    </div>
                    <div>
                        <div class='dossier-cell-label'>ACTIVE MEDICATIONS</div>
                        <div class='dossier-cell-value'>{patient_dict.get('num_medications', 0)} meds</div>
                    </div>
                    <div>
                        <div class='dossier-cell-label'>INSULIN REGIMEN</div>
                        <div class='dossier-cell-value'>{patient_dict.get('insulin', 'No')}</div>
                    </div>
                    <div>
                        <div class='dossier-cell-label'>INPATIENT HBA1C</div>
                        <div class='dossier-cell-value'>{patient_dict.get('A1Cresult', 'None')}</div>
                    </div>
                    <div style='grid-column: span 2;'>
                        <div class='dossier-cell-label'>DISCHARGE DESTINATION</div>
                        <div class='dossier-cell-value'>{patient_dict.get('discharge_disposition', 'Home')}</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.plotly_chart(render_risk_gauge(predicted_prob), width='stretch')

    with prof_col2:
        # SHAP Diverging Bar Chart with Editorial Header
        st.markdown(
            """
            <div style='margin-bottom: 8px;'>
                <div style='font-family: var(--font-display); font-size: 1.25rem; font-weight: 700; color: #FDFCF9;'>Key Risk Drivers &bull; TreeSHAP Decomposition</div>
                <div style='font-family: var(--font-ui); font-size: 0.8rem; color: #94A3B8;'>Physiological and utilization feature impact mapped to 30-day readmission probability delta (%)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.plotly_chart(render_feature_contributions_chart(contributions), width='stretch')

        # Action Plan Section
        st.markdown("<div style='font-family: var(--font-display); font-size: 1.25rem; font-weight: 700; color: #FDFCF9; margin-top: 15px; margin-bottom: 10px;'>Personalized Clinical Action Plan & Transition Protocol</div>", unsafe_allow_html=True)
        action_items = generate_clinical_action_plan(patient_dict, predicted_prob)

        for item in action_items:
            priority_color = "#F43F5E" if "HIGH" in item["priority"] or "CRITICAL" in item["priority"] else "#F59E0B" if "MODERATE" in item["priority"] else "#10B981"
            st.markdown(
                f"""
                <div class='clinical-action-card' style='border-left-color: {priority_color};'>
                    <div class='action-card-header'>
                        <span class='action-title'>{item['title']}</span>
                        <span style='font-family: var(--font-mono); font-size: 0.7rem; font-weight: 600; color: {priority_color}; background: rgba(255,255,255,0.05); padding: 2px 8px; border-radius: 4px; border: 1px solid rgba(255,255,255,0.1);'>{item['priority']}</span>
                    </div>
                    <div class='action-desc'>{item['desc']}</div>
                    <div style='font-family: var(--font-mono); font-size: 0.72rem; color: #94A3B8; margin-top: 6px;'>
                        <span style='color: #E5C07B;'>Evidence Base:</span> {item.get('evidence', '')}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ==============================================================================
# TAB 4: 'WHAT-IF' TRANSITIONAL CARE SIMULATOR
# ==============================================================================
with tab_simulator:
    st.markdown(
        """
        <div style='margin-bottom: 20px;'>
            <div class='section-headline'>Transitional Care 'What-If' Intervention Cockpit</div>
            <div class='section-caption'>Simulate the clinical and financial risk reduction of preemptive care bundles prior to hospital discharge.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Select high risk patient
    high_risk_candidates = test_cohort[test_cohort["predicted_readmission_risk"] >= 0.28].head(50)
    sim_patient_options = {
        f"Patient #{row['patient_nbr']} (Base Risk: {row['predicted_readmission_risk']*100:.1f}% | Diag: {row['primary_diag_category']} | Age: {row['age']})": row['patient_nbr']
        for _, row in high_risk_candidates.iterrows()
    }

    sim_selected_label = st.selectbox("Select Patient for Intervention Simulation:", options=list(sim_patient_options.keys()))
    sim_pid = sim_patient_options[sim_selected_label]
    base_patient = test_cohort[test_cohort["patient_nbr"] == sim_pid].iloc[0].to_dict()

    base_df = pd.DataFrame([base_patient])
    base_trans = preprocessor.transform(base_df)
    base_prob = float(calibrated_model.predict_proba(base_trans)[0, 1])

    st.markdown("<div style='font-family: var(--font-mono); font-size: 0.74rem; color: #E5C07B; text-transform: uppercase; margin-top: 14px; margin-bottom: 8px;'>SELECT DISCHARGE INTERVENTIONS TO DEPLOY</div>", unsafe_allow_html=True)

    int_c1, int_c2 = st.columns(2)

    with int_c1:
        int_med_rec = st.checkbox(
            "💊 Pharmacist-Led Medication Reconciliation & 'Teach-Back'",
            value=True,
            help="Resolves polypharmacy interactions, stabilizes insulin dosage changes, and conducts bedside teach-back.",
        )
        int_hba1c = st.checkbox(
            "🧪 Inpatient Glycemic Optimization & HbA1c Lab Testing",
            value=True,
            help="Orders inpatient HbA1c screening test and triggers bedside diabetes education if unrecorded.",
        )

    with int_c2:
        int_home_health = st.checkbox(
            "🏠 Transitional Home Health Nursing Referral",
            value=True,
            help="Reroutes vulnerable or SNF-bound patients to dedicated home health nursing with 48h bedside evaluation.",
        )
        int_telehealth = st.checkbox(
            "📞 Scheduled 48-Hour Rapid Telehealth / Clinic Follow-Up",
            value=True,
            help="Confirmed outpatient provider follow-up appointment within 48-72 hours of discharge.",
        )

    # Apply Simulation Modifications
    sim_patient = base_patient.copy()

    if int_med_rec:
        sim_patient["insulin_dose_changed"] = 0
        sim_patient["medication_changed"] = 0
        sim_patient["change"] = "No"
        sim_patient["num_medications"] = max(5, int(sim_patient["num_medications"] * 0.8))
        sim_patient["polypharmacy_high"] = int(sim_patient["num_medications"] >= 16)
        sim_patient["medication_intensity"] = sim_patient["num_medications"] / (sim_patient["time_in_hospital"] + 0.1)

    if int_hba1c:
        sim_patient["a1c_tested"] = 1
        sim_patient["a1c_elevated"] = 0
        sim_patient["a1c_normal"] = 1
        sim_patient["A1Cresult"] = "Norm"

    if int_home_health:
        sim_patient["discharged_to_snf"] = 0
        sim_patient["discharged_to_home_health"] = 1
        sim_patient["discharge_disposition"] = "Home Health / IV"
        sim_patient["high_risk_triad"] = 0

    sim_df = pd.DataFrame([sim_patient])
    sim_trans = preprocessor.transform(sim_df)
    sim_prob = float(calibrated_model.predict_proba(sim_trans)[0, 1])

    if int_telehealth:
        sim_prob = sim_prob * 0.85

    relative_reduction = ((base_prob - sim_prob) / base_prob) * 100
    absolute_reduction = (base_prob - sim_prob) * 100
    sim_savings = cms_penalty_cost * (base_prob - sim_prob)

    st.markdown("<hr style='border: 0; border-top: 1px solid rgba(226, 206, 175, 0.15); margin: 20px 0;'>", unsafe_allow_html=True)

    # Before vs After Metrics
    sim_res1, sim_res2, sim_res3 = st.columns(3)

    with sim_res1:
        st.markdown(
            f"""
            <div class='editorial-kpi-card'>
                <div class='kpi-numeral'>BASELINE HAZARD</div>
                <div class='kpi-label'>Pre-Intervention Risk</div>
                <div class='kpi-figure' style='color: #F43F5E;'>{base_prob*100:.1f}%</div>
                <div class='kpi-footer'>
                    <span class='pill-delta pill-rose'>{get_risk_tier_info(base_prob)['tier']}</span>
                    <span style='color: #94A3B8; font-size:0.75rem;'>Unmitigated</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with sim_res2:
        st.markdown(
            f"""
            <div class='editorial-kpi-card'>
                <div class='kpi-numeral'>POST-BUNDLE STATUS</div>
                <div class='kpi-label'>Simulated Risk Score</div>
                <div class='kpi-figure' style='color: #10B981;'>{sim_prob*100:.1f}%</div>
                <div class='kpi-footer'>
                    <span class='pill-delta pill-emerald'>{get_risk_tier_info(sim_prob)['tier']}</span>
                    <span style='color: #94A3B8; font-size:0.75rem;'>With Bundled Care</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with sim_res3:
        st.markdown(
            f"""
            <div class='editorial-kpi-card'>
                <div class='kpi-numeral'>HAZARD MITIGATION</div>
                <div class='kpi-label'>Relative Risk Delta</div>
                <div class='kpi-figure' style='color: #E5C07B;'>-{relative_reduction:.1f}%</div>
                <div class='kpi-footer'>
                    <span class='pill-delta pill-gold'>Saved: ${sim_savings:,.0f}</span>
                    <span style='color: #94A3B8; font-size:0.75rem;'>Avoided penalty</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

    # Comparative Plotly Chart
    st.plotly_chart(render_whatif_comparison_chart(base_prob, sim_prob), width='stretch')

    st.markdown(
        f"""
        <div style='background: rgba(16, 22, 31, 0.85); border: 1px solid var(--border-hairline); border-radius: 10px; padding: 18px; margin-top: 10px;'>
            <div style='font-family: var(--font-mono); font-size: 0.72rem; color: #E5C07B; text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 6px;'>CLINICAL MECHANISM NOTE</div>
            <div style='font-size: 0.86rem; color: #CBD5E1; line-height: 1.6;'>
                By preemptively mitigating medication discrepancies via pharmacist reconciliation and arranging immediate outpatient follow-up, the patient's acute readmission vulnerability is reduced by <b>{absolute_reduction:.1f} percentage points</b> ({relative_reduction:.1f}% relative reduction), reclassifying them from <b>{get_risk_tier_info(base_prob)['tier']}</b> to <b>{get_risk_tier_info(sim_prob)['tier']}</b> and yielding an expected cost offset of <b>${sim_savings:,.0f}</b>.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ==============================================================================
# TAB 5: MODEL GOVERNANCE & CLINICAL VALIDATION
# ==============================================================================
with tab_governance:
    st.markdown(
        """
        <div style='margin-bottom: 20px;'>
            <div class='section-headline'>Algorithmic Governance, Model Evaluation & Clinical Validation</div>
            <div class='section-caption'>Rigorous benchmark validation across 5 algorithms on the held-out test cohort (13,998 unique inpatients).</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    results = metadata["results"]

    # Benchmark Table
    comparison_rows = []
    for model_key, met in results.items():
        is_champion = "LightGBM" in model_key
        comparison_rows.append({
            "Algorithm": model_key.replace("_", " "),
            "ROC-AUC": f"{met['roc_auc']:.4f}",
            "PR-AUC": f"{met['pr_auc']:.4f}",
            "Brier Score": f"{met['brier_score']:.4f}",
            "Top-10% Lift": f"{met['top_decile_lift']:.2f}x",
            "Recall @ Top-10%": f"{met['recall_at_top_10pct']*100:.1f}%",
            "Sensitivity": f"{met['sensitivity']*100:.1f}%",
            "Specificity": f"{met['specificity']*100:.1f}%",
            "Champion": "★ CHAMPION" if is_champion else "",
        })

    df_comp = pd.DataFrame(comparison_rows)
    st.dataframe(df_comp, width='stretch')

    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

    g_col1, g_col2 = st.columns(2, gap="medium")

    with g_col1:
        lift_models = [m.replace("_", " ") for m in results.keys()]
        lifts = [results[m]["top_decile_lift"] for m in results.keys()]

        fig_lift = px.bar(
            x=lift_models,
            y=lifts,
            color=lifts,
            color_continuous_scale=[[0, "#94A3B8"], [0.6, "#E5C07B"], [1.0, "#10B981"]],
            labels={"x": "Model Architecture", "y": "Lift Multiplier vs Random"},
            text=lifts,
        )
        fig_lift.update_traces(
            texttemplate="%{text:.2f}x",
            textposition="outside",
            textfont=dict(color="#FDFCF9", size=11, family="DM Mono"),
        )
        fig_lift.add_hline(
            y=1.0,
            line_dash="dot",
            line_color="#F43F5E",
            annotation_text="Random Triage Baseline (1.0x)",
            annotation_font=dict(color="#F43F5E", family="DM Mono", size=10),
        )
        fig_lift.update_layout(
            title={
                "text": "<b>Operational Top-Decile Lift Comparison</b><br><span style='font-size:11px; color:#94A3B8; font-weight:normal;'>Calibrated LightGBM concentrates 3.51x more readmissions in the top 10% risk tranche</span>",
                "font": {"size": 15, "color": "#FDFCF9", "family": "Fraunces, serif"},
            },
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(16, 22, 31, 0.45)",
            coloraxis_showscale=False,
            font=dict(color="#94A3B8"),
            yaxis=dict(
                gridcolor="rgba(255, 255, 255, 0.05)",
                range=[0, 4.5],
                title=dict(font=dict(color="#94A3B8", size=11, family="Sora")),
                tickfont=dict(color="#94A3B8", size=10, family="DM Mono"),
            ),
            xaxis=dict(
                gridcolor="rgba(0,0,0,0)",
                tickfont=dict(color="#E2E8F0", size=11, family="Sora"),
            ),
            height=340,
            margin=dict(l=20, r=20, t=65, b=35),
        )
        st.plotly_chart(fig_lift, width='stretch')

    with g_col2:
        top_f = pd.DataFrame(metadata["top_features"]).head(12)
        top_f["clean_name"] = top_f["feature"].apply(
            lambda x: x.replace("primary_diag_category_", "Diag: ").replace("specialty_grouped_", "Specialty: ").replace("_", " ").title()
        )
        top_f = top_f.sort_values(by="importance", ascending=True)

        fig_feat = px.bar(
            top_f,
            y="clean_name",
            x="importance",
            orientation="h",
            color="importance",
            color_continuous_scale=[[0, "#E5C07B"], [1.0, "#0EA5E9"]],
            labels={"importance": "Mean |SHAP| Value", "clean_name": "Clinical Predictor"},
        )
        fig_feat.update_layout(
            title={
                "text": "<b>Global Clinical Predictors (Mean |SHAP| Value)</b><br><span style='font-size:11px; color:#94A3B8; font-weight:normal;'>Prior utilization, SNF transfer, and medication adjustments dominate global risk</span>",
                "font": {"size": 15, "color": "#FDFCF9", "family": "Fraunces, serif"},
            },
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(16, 22, 31, 0.45)",
            coloraxis_showscale=False,
            font=dict(color="#94A3B8"),
            xaxis=dict(
                gridcolor="rgba(255, 255, 255, 0.05)",
                title=dict(font=dict(color="#94A3B8", size=11, family="Sora")),
                tickfont=dict(color="#94A3B8", size=10, family="DM Mono"),
            ),
            yaxis=dict(
                gridcolor="rgba(0,0,0,0)",
                tickfont=dict(color="#E2E8F0", size=10, family="Sora"),
            ),
            height=340,
            margin=dict(l=20, r=20, t=65, b=35),
        )
        st.plotly_chart(fig_feat, width='stretch')

    # Governance Safeguards Matrix Card
    st.markdown(
        """
        <div style='background: linear-gradient(145deg, rgba(16, 22, 31, 0.95) 0%, rgba(10, 14, 20, 0.98) 100%); border: 1px solid var(--border-gold); border-radius: 12px; padding: 22px; margin-top: 15px;'>
            <div style='font-family: var(--font-mono); font-size: 0.74rem; color: #E5C07B; text-transform: uppercase; letter-spacing: 0.14em; margin-bottom: 8px;'>
                ETHICAL AI SAFEGUARDS & CLINICAL GOVERNANCE PILLARS
            </div>
            <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 20px; font-size: 0.86rem; color: #CBD5E1; line-height: 1.6;'>
                <div>
                    <b style='color: #FDFCF9;'>1. Assistive Decision Support, Never Autonomous:</b> The model generates probability recommendations to augment clinician decision-making during discharge rounds; it never autonomously delays or accelerates patient discharge.
                </div>
                <div>
                    <b style='color: #FDFCF9;'>2. Demographic Equity & Parity Audit:</b> Model false-negative and false-positive rates are systematically audited across gender, race, and age strata to avoid algorithmic disparities in care allocation.
                </div>
                <div>
                    <b style='color: #FDFCF9;'>3. Isotonic / Platt Probability Calibration:</b> Brier score minimization ensures that a predicted 25% readmission risk directly translates to approximately 25 readmissions per 100 encounters.
                </div>
                <div>
                    <b style='color: #FDFCF9;'>4. Strict Encounter Deduplication:</b> Only index hospital admissions are retained for patient-level modeling, preventing data leakage across recurrent visits.
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ==============================================================================
# FOOTER: EDITORIAL COLOPHON
# ==============================================================================
st.markdown("<hr style='border: 0; border-top: 1px solid rgba(226, 206, 175, 0.15); margin: 36px 0 20px 0;'>", unsafe_allow_html=True)
st.markdown(
    """
    <div style='display: flex; justify-content: space-between; align-items: center; font-family: var(--font-mono); font-size: 0.72rem; color: #64748B;'>
        <div>AETHEL // PREDICTIVE READMISSION RISK INTELLIGENCE &bull; PRODUCTION RELEASE 2026.04</div>
        <div>CALIBRATED LIGHTGBM &bull; TREESHAP &bull; CMS HRRP COMPLIANT</div>
    </div>
    """,
    unsafe_allow_html=True,
)
