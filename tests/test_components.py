"""
Unit tests for Clinical Components, Risk Gauges & Action Plan Generators
"""

import pytest
import plotly.graph_objects as go
from app.components import (
    get_risk_tier_info,
    render_risk_gauge,
    render_feature_contributions_chart,
    render_whatif_comparison_chart,
    generate_clinical_action_plan,
)


def test_get_risk_tier_info():
    """Verify tier assignment cutoffs and associated metadata."""
    low = get_risk_tier_info(0.05)
    assert low["tier"] == "Low Risk"
    assert low["class"] == "pill-emerald"

    mod = get_risk_tier_info(0.18)
    assert mod["tier"] == "Moderate Risk"
    assert mod["class"] == "pill-amber"

    high = get_risk_tier_info(0.35)
    assert high["tier"] == "High Risk"
    assert high["class"] == "pill-rose"

    crit = get_risk_tier_info(0.65)
    assert crit["tier"] == "Critical Risk"
    assert crit["class"] == "pill-rose"


def test_render_risk_gauge():
    """Verify risk gauge generates a valid Plotly Figure."""
    fig = render_risk_gauge(0.32)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) > 0
    assert fig.data[0].type == "indicator"


def test_render_feature_contributions_chart():
    """Verify TreeSHAP bar chart generates without errors."""
    contributions = [
        {"raw_feature": "number_inpatient", "display_name": "Prior Inpatient Admissions", "impact": 14.5},
        {"raw_feature": "on_insulin", "display_name": "Insulin Therapy Prescribed", "impact": 8.2},
        {"raw_feature": "a1c_tested", "display_name": "Inpatient HbA1c Monitored", "impact": -5.1},
    ]
    fig = render_feature_contributions_chart(contributions)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) > 0
    assert fig.data[0].type == "bar"

    # Empty test
    empty_fig = render_feature_contributions_chart([])
    assert isinstance(empty_fig, go.Figure)


def test_render_whatif_comparison_chart():
    """Verify What-If comparative bar chart renders properly."""
    fig = render_whatif_comparison_chart(0.38, 0.19)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) > 0


def test_generate_clinical_action_plan():
    """Verify evidence-based clinical rules are triggered appropriately."""
    high_risk_pt = {
        "number_inpatient": 2,
        "discharged_to_snf": 1,
        "discharge_disposition": "Skilled Nursing / Nursing Facility",
        "num_medications": 18,
        "insulin": "Up",
        "change": "Ch",
        "a1c_tested": 0,
        "a1c_elevated": 0,
        "has_circulatory": 1,
    }
    actions = generate_clinical_action_plan(high_risk_pt, risk_prob=0.38)
    assert len(actions) >= 4

    categories = [a["category"] for a in actions]
    assert "Care Coordination" in categories
    assert "Utilization Severity" in categories
    assert "Transitional Care" in categories
    assert "Clinical Pharmacy" in categories
    assert "Pharmacotherapy" in categories

    for action in actions:
        assert "evidence" in action
        assert len(action["evidence"]) > 10
