"""
AETHEL // CLINICAL INTELLIGENCE VISUALIZATIONS & COMPONENTS
Production-Grade Reusable Components for Patient Readmission Analytics
"""

import plotly.graph_objects as go
import pandas as pd
import numpy as np
from typing import Dict, Any, List


def get_risk_tier_info(risk_prob: float) -> Dict[str, str]:
    """Returns clinical tier label, color, badge class, and evidence-based recommendation."""
    if risk_prob < 0.10:
        return {
            "tier": "Low Risk",
            "class": "pill-emerald",
            "color": "#10B981",
            "bg_color": "rgba(16, 185, 129, 0.15)",
            "recommendation": "Standard discharge pathway with routine primary care visit within 14-21 days.",
            "urgency": "ROUTINE",
        }
    elif risk_prob < 0.25:
        return {
            "tier": "Moderate Risk",
            "class": "pill-amber",
            "color": "#F59E0B",
            "bg_color": "rgba(245, 158, 11, 0.15)",
            "recommendation": "Enhanced discharge planning: phone follow-up within 72 hours and discharge medication reconciliation.",
            "urgency": "MODERATE",
        }
    elif risk_prob < 0.50:
        return {
            "tier": "High Risk",
            "class": "pill-rose",
            "color": "#F97316",
            "bg_color": "rgba(249, 115, 22, 0.18)",
            "recommendation": "Targeted clinical intervention: dedicated nurse case navigator, 48-hour provider visit, home health referral.",
            "urgency": "HIGH PRIORITY",
        }
    else:
        return {
            "tier": "Critical Risk",
            "class": "pill-rose",
            "color": "#F43F5E",
            "bg_color": "rgba(244, 63, 94, 0.22)",
            "recommendation": "Urgent multimodal transitional care bundle: transitional care physician visit within 48h, daily tele-monitoring.",
            "urgency": "CRITICAL",
        }


def render_risk_gauge(risk_prob: float) -> go.Figure:
    """Creates a high-precision medical risk indicator dial with editorial styling."""
    pct = risk_prob * 100
    tier_info = get_risk_tier_info(risk_prob)

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=pct,
            number={
                "suffix": "%",
                "font": {"size": 44, "color": "#FDFCF9", "family": "Fraunces, serif"},
            },
            title={
                "text": f"<b>30-DAY READMISSION RISK</b><br><span style='font-size:12px; color:{tier_info['color']}; font-family:DM Mono, monospace; letter-spacing:0.12em;'>[ {tier_info['tier'].upper()} ]</span>",
                "font": {"size": 14, "color": "#CBD5E1", "family": "Sora, sans-serif"},
            },
            gauge={
                "axis": {
                    "range": [0, 100],
                    "tickwidth": 1,
                    "tickcolor": "rgba(226, 206, 175, 0.35)",
                    "ticks": "outside",
                    "tickfont": {"color": "#94A3B8", "size": 10, "family": "DM Mono, monospace"},
                    "dtick": 25,
                },
                "bar": {"color": tier_info["color"], "thickness": 0.24},
                "bgcolor": "rgba(16, 22, 31, 0.7)",
                "borderwidth": 1,
                "bordercolor": "rgba(226, 206, 175, 0.15)",
                "steps": [
                    {"range": [0, 10], "color": "rgba(16, 185, 129, 0.18)"},
                    {"range": [10, 25], "color": "rgba(245, 158, 11, 0.18)"},
                    {"range": [25, 50], "color": "rgba(249, 115, 22, 0.22)"},
                    {"range": [50, 100], "color": "rgba(244, 63, 94, 0.28)"},
                ],
                "threshold": {
                    "line": {"color": "#E5C07B", "width": 2.5},
                    "thickness": 0.75,
                    "value": 25,
                },
            },
        )
    )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=25, r=25, t=55, b=15),
        height=260,
    )
    return fig


def render_feature_contributions_chart(contributions: List[Dict[str, Any]]) -> go.Figure:
    """Renders a refined diverging horizontal bar chart showing TreeSHAP drivers."""
    if not contributions:
        fig = go.Figure()
        fig.add_annotation(
            text="No explainability data available",
            showarrow=False,
            font=dict(color="#94A3B8", family="Sora"),
        )
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        return fig

    # Sort so the highest magnitude impact is at the top
    df_c = pd.DataFrame(contributions).sort_values(by="impact", key=abs, ascending=True)

    # Red/Crimson for risk aggravators, Emerald for risk reducers
    colors = ["#F43F5E" if val > 0 else "#10B981" for val in df_c["impact"]]

    fig = go.Figure(
        go.Bar(
            x=df_c["impact"],
            y=df_c["display_name"],
            orientation="h",
            marker=dict(
                color=colors,
                line=dict(width=0),
            ),
            text=[f"{val:+.1f}%" for val in df_c["impact"]],
            textposition="auto",
            textfont=dict(color="#FDFCF9", size=11, family="DM Mono, monospace"),
            hovertemplate="<b>%{y}</b><br>SHAP Contribution: %{x:+.2f}%<extra></extra>",
        )
    )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(16, 22, 31, 0.4)",
        xaxis=dict(
            title=dict(
                text="Impact on 30-Day Readmission Probability (%)",
                font=dict(color="#94A3B8", size=11, family="Sora, sans-serif"),
            ),
            tickfont=dict(color="#94A3B8", size=10, family="DM Mono, monospace"),
            gridcolor="rgba(255, 255, 255, 0.05)",
            zerolinecolor="rgba(229, 192, 123, 0.45)",
            zerolinewidth=1.5,
        ),
        yaxis=dict(
            automargin=True,
            tickfont=dict(color="#E2E8F0", size=11, family="Sora, sans-serif"),
            gridcolor="rgba(0,0,0,0)",
        ),
        margin=dict(l=10, r=25, t=20, b=40),
        height=320,
    )
    return fig


def render_whatif_comparison_chart(base_prob: float, sim_prob: float) -> go.Figure:
    """Renders a sleek before-and-after comparative bar chart for the What-If simulator."""
    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=["Baseline Risk", "Post-Intervention Risk"],
            y=[base_prob * 100, sim_prob * 100],
            marker=dict(
                color=["#F43F5E", "#10B981"],
                line=dict(color=["rgba(244, 63, 94, 0.4)", "rgba(16, 185, 129, 0.4)"], width=1.5),
            ),
            text=[f"{base_prob*100:.1f}%", f"{sim_prob*100:.1f}%"],
            textposition="outside",
            textfont=dict(color="#FDFCF9", size=13, family="DM Mono, monospace"),
            width=0.45,
            hovertemplate="<b>%{x}</b>: %{y:.1f}%<extra></extra>",
        )
    )

    max_y = max(base_prob * 100 * 1.35, 15)
    fig.update_layout(
        title={
            "text": "<b>Direct Risk Reduction Comparison</b>",
            "font": {"size": 15, "color": "#FDFCF9", "family": "Fraunces, serif"},
        },
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(16, 22, 31, 0.4)",
        yaxis=dict(
            title=dict(text="30-Day Readmission Probability (%)", font=dict(color="#94A3B8", size=11, family="Sora")),
            range=[0, max_y],
            gridcolor="rgba(255, 255, 255, 0.05)",
            tickfont=dict(color="#94A3B8", size=10, family="DM Mono"),
        ),
        xaxis=dict(
            gridcolor="rgba(0,0,0,0)",
            tickfont=dict(color="#FDFCF9", size=12, family="Sora"),
        ),
        font=dict(color="#94A3B8"),
        height=320,
        margin=dict(l=20, r=20, t=55, b=30),
    )
    return fig


def generate_clinical_action_plan(patient_data: Dict[str, Any], risk_prob: float) -> List[Dict[str, str]]:
    """Generates personalized evidence-based clinical interventions for the discharge team."""
    actions = []

    # High Risk Universal Protocol
    if risk_prob >= 0.25:
        actions.append({
            "category": "Care Coordination",
            "priority": "HIGH PRIORITY",
            "title": "Dedicated Nurse Case Manager & 48-Hour Rapid Outpatient Visit",
            "desc": "Enroll in high-risk transition care bundle. Secure confirmed clinic or telemedicine visit within 48-72 hours of discharge with assigned nurse navigator.",
            "evidence": "Proven to reduce 30-day readmissions by 24-28% in randomized trials (Naylor et al.).",
        })

    # Prior Hospitalizations
    if patient_data.get("number_inpatient", 0) > 0:
        actions.append({
            "category": "Utilization Severity",
            "priority": "HIGH PRIORITY",
            "title": "Recurrent Admission Interdisciplinary Case Conference",
            "desc": f"Patient has {int(patient_data.get('number_inpatient', 0))} prior inpatient stay(s) in the preceding 12 months. Convene attending, social worker, and primary care before discharge.",
            "evidence": "Addresses unmanaged social determinants of health and care plan non-concordance.",
        })

    # Discharge Destination (SNF / Nursing Facility)
    disp = str(patient_data.get("discharge_disposition", ""))
    if "Nursing" in disp or "Facility" in disp or patient_data.get("discharged_to_snf", 0) == 1:
        actions.append({
            "category": "Transitional Care",
            "priority": "HIGH PRIORITY",
            "title": "Direct Physician-to-Physician Warm Handoff with SNF Team",
            "desc": "Conduct structured verbal handoff with receiving skilled nursing facility physician. Verify 14-day supply of all maintenance and diabetes medications is dispatched.",
            "evidence": "Prevents early medication errors and diagnostic ambiguities during institutional transfer.",
        })

    # Diabetes & Glycemic Management
    a1c_tested = patient_data.get("a1c_tested", 0)
    a1c_elevated = patient_data.get("a1c_elevated", 0)
    if not a1c_tested or a1c_elevated:
        actions.append({
            "category": "Endocrine Protocol",
            "priority": "MODERATE PRIORITY",
            "title": "Inpatient HbA1c Lab & Certified Diabetes Educator (CDCES) Referral",
            "desc": "Perform inpatient HbA1c testing prior to discharge if unrecorded. If >8.0%, trigger bedside diabetes self-management education and outpatient endocrinology follow-up.",
            "evidence": "Inpatient HbA1c testing is associated with statistically significant readmission reduction (Strack et al.).",
        })

    # Insulin & Medication Changes
    insulin_val = str(patient_data.get("insulin", "No"))
    med_changed = patient_data.get("medication_changed", 0) == 1 or patient_data.get("change") == "Ch"
    if insulin_val != "No" or med_changed:
        actions.append({
            "category": "Clinical Pharmacy",
            "priority": "HIGH PRIORITY",
            "title": "Clinical Pharmacist Bedside Medication Reconciliation & 'Teach-Back'",
            "desc": "Pharmacist conducts full reconciliation, resolving formulary conflicts, demonstrating insulin pen/syringe technique, and ensuring patient can physically obtain discharge meds.",
            "evidence": "Medication discrepancies account for up to 50% of post-discharge hospital readmissions.",
        })

    # Polypharmacy
    if patient_data.get("num_medications", 0) >= 15:
        actions.append({
            "category": "Pharmacotherapy",
            "priority": "MODERATE PRIORITY",
            "title": "Polypharmacy Regimen Deprescribing & Simplification Review",
            "desc": f"Patient is prescribed {int(patient_data.get('num_medications', 0))} concurrent medications. Screen for anticholinergic burden, hypoglycemic risk, and simplify dosing schedule.",
            "evidence": "Regimens exceeding 15 medications carry a 3.2x relative hazard for adverse drug events.",
        })

    # Comorbidity Cardiorenal
    if patient_data.get("has_circulatory", 0) == 1 or patient_data.get("has_renal", 0) == 1:
        actions.append({
            "category": "Cardiorenal",
            "priority": "MODERATE PRIORITY",
            "title": "Standing Post-Discharge Lab Requisition (BMP / Renal Profile)",
            "desc": "Issue standing outpatient lab requisition for Basic Metabolic Panel (electrolytes, BUN, creatinine) within 7-10 days to monitor fluid balance and renal safety.",
            "evidence": "Early lab checks identify acute kidney injury and electrolyte shifts before clinical decompensation.",
        })

    if not actions:
        actions.append({
            "category": "Standard Care",
            "priority": "ROUTINE",
            "title": "Routine Patient-Centered Discharge Instructions",
            "desc": "Provide clear, linguistically concordant written discharge instructions, red-flag symptom guidelines, and confirm scheduled primary care visit in 14-21 days.",
            "evidence": "Standard low-risk clinical pathway.",
        })

    return actions
