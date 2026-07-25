import html
import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import xgboost as xgb

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="SepsisGuard AI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# STYLE
# ============================================================

st.markdown(
    """
<style>
    :root {
        --navy: #17354a;
        --blue: #2478a5;
        --blue-soft: #eef6fb;
        --border: #d8e0e6;
        --surface: #ffffff;
        --background: #f4f7f9;
        --success: #26734d;
        --warning: #a86100;
        --danger: #b42318;
        --muted: #64748b;
    }

    .stApp {
        background: var(--background);
        font-size: 17px;
    }

    html, body, [class*="css"] {
        font-size: 17px;
    }

    p, li, label, .stCaption {
        font-size: 16px !important;
        line-height: 1.5;
    }

    h1 { font-size: 2.25rem !important; }
    h2 { font-size: 1.75rem !important; }
    h3 { font-size: 1.35rem !important; }

    .block-container {
        max-width: 1320px;
        padding-top: 1rem;
        padding-bottom: 2rem;
    }

    [data-testid="stSidebar"] {
        background: var(--navy);
        border-right: 1px solid #0e2838;
    }

    [data-testid="stSidebar"] * {
        color: white;
    }

    .app-header {
        background: white;
        border: 1px solid var(--border);
        border-left: 6px solid var(--blue);
        border-radius: 9px;
        padding: 16px 20px;
        margin-bottom: 16px;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
    }

    .app-title {
        color: var(--navy);
        font-size: 29px;
        font-weight: 800;
        margin: 0;
    }

    .app-subtitle {
        color: var(--muted);
        font-size: 14px;
        margin-top: 3px;
    }

    .selection-card,
    .content-card {
        background: white;
        border: 1px solid var(--border);
        border-radius: 9px;
        padding: 17px 18px;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
        margin-bottom: 12px;
    }

    .patient-banner {
        background: var(--blue-soft);
        border: 1px solid #c8ddea;
        border-radius: 8px;
        padding: 11px 14px;
        margin-bottom: 12px;
        color: #23465d;
        display: flex;
        justify-content: space-between;
        gap: 16px;
        font-size: 14px;
    }

    .risk-card {
        background: white;
        border: 1px solid var(--border);
        border-left: 7px solid var(--status-color);
        border-radius: 9px;
        padding: 19px;
        margin-bottom: 12px;
    }

    .risk-score {
        color: var(--status-color);
        font-size: 48px;
        font-weight: 850;
        line-height: 1;
    }

    .risk-label {
        color: var(--status-color);
        font-size: 21px;
        font-weight: 750;
        margin-top: 7px;
    }

    .factor-card {
        background: white;
        border: 1px solid var(--border);
        border-left: 5px solid var(--factor-color);
        border-radius: 8px;
        padding: 13px 15px;
        margin-bottom: 9px;
    }

    .factor-title {
        color: var(--navy);
        font-weight: 750;
        font-size: 15px;
    }

    .factor-value {
        color: #334155;
        font-size: 14px;
        margin-top: 3px;
    }

    .factor-explanation {
        color: var(--muted);
        font-size: 13px;
        margin-top: 4px;
    }

    .patient-overview-grid {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 12px;
        margin-top: 12px;
        align-items: stretch;
    }

    .overview-item {
        background: #ffffff;
        border: 1px solid var(--border);
        border-radius: 9px;
        padding: 14px 14px;
        min-height: 112px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        box-sizing: border-box;
    }

    .overview-label {
        color: var(--muted);
        font-size: 14px;
        line-height: 1.25;
        min-height: 36px;
        margin-bottom: 8px;
        font-weight: 650;
    }

    .overview-value {
        color: var(--navy);
        font-size: 22px;
        line-height: 1.1;
        font-weight: 800;
    }

    .status-badge {
        display: inline-block;
        padding: 3px 9px;
        border-radius: 999px;
        font-size: 12px;
        font-weight: 750;
        margin-left: 6px;
    }

    .status-high {
        background: #fde8e7;
        color: #9f1c14;
    }

    .status-mid {
        background: #fff1cf;
        color: #8a5200;
    }

    .status-low {
        background: #e2f3e9;
        color: #1f6a46;
    }

    .risk-meta {
        display: flex;
        gap: 10px;
        flex-wrap: wrap;
        margin-top: 10px;
    }

    .risk-chip {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 999px;
        background: #edf2f7;
        color: #334155;
        font-size: 12px;
        font-weight: 650;
    }

    .plain-summary {
        background: #eef6fb;
        border: 1px solid #c8ddea;
        border-left: 5px solid var(--blue);
        border-radius: 8px;
        padding: 14px 16px;
        color: #23465d;
        margin-top: 10px;
    }

    .safety-note {
        background: #fff8e8;
        border: 1px solid #f2d59b;
        border-left: 5px solid #b7791f;
        border-radius: 8px;
        padding: 12px 14px;
        color: #5b4311;
        font-size: 13px;
        margin-top: 12px;
    }

    div[data-testid="stMetric"] {
        background: white;
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 12px;
    }

    .stButton > button,
    .stFormSubmitButton > button {
        border-radius: 7px;
        font-weight: 700;
    }

    .st-key-floating_chat_launcher {
        position: fixed;
        right: 28px;
        bottom: 24px;
        z-index: 9999;
        width: 250px;
    }

    .st-key-floating_chat_launcher button {
        background: #17354a !important;
        color: white !important;
        border: none !important;
        border-radius: 999px !important;
        padding: 0.75rem 1.15rem !important;
        font-size: 16px !important;
        font-weight: 750 !important;
        box-shadow: 0 8px 24px rgba(15, 23, 42, 0.25) !important;
    }

    .st-key-floating_chat_launcher button:hover {
        background: #2478a5 !important;
        transform: translateY(-1px);
    }


    .login-shell {
        max-width: 460px;
        margin: 6vh auto 0 auto;
        background: white;
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 28px;
        box-shadow: 0 12px 30px rgba(15, 23, 42, 0.08);
    }

    .login-logo {
        width: 54px;
        height: 54px;
        border-radius: 14px;
        background: var(--navy);
        color: white;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 28px;
        margin-bottom: 14px;
    }

    .dashboard-hero {
        background: linear-gradient(135deg, #17354a, #2478a5);
        color: white;
        border-radius: 12px;
        padding: 22px 24px;
        margin-bottom: 16px;
        box-shadow: 0 8px 22px rgba(23, 53, 74, 0.15);
    }

    .dashboard-hero h2 {
        margin: 0 0 4px 0;
        color: white;
    }

    .dashboard-hero p {
        margin: 0;
        color: #dbeaf3;
    }

    .step-row {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 10px;
        margin: 12px 0 16px 0;
    }

    .step-card {
        background: white;
        border: 1px solid var(--border);
        border-radius: 9px;
        padding: 12px 14px;
    }

    .step-number {
        width: 28px;
        height: 28px;
        border-radius: 50%;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        background: var(--blue-soft);
        color: var(--blue);
        font-weight: 800;
        margin-right: 7px;
    }

    .queue-row {
        background: white;
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 10px 12px;
        margin-bottom: 7px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }

    .queue-risk-high { color: var(--danger); font-weight: 750; }
    .queue-risk-mid { color: var(--warning); font-weight: 750; }
    .queue-risk-low { color: var(--success); font-weight: 750; }

    .chat-intro {
        background: var(--blue-soft);
        border: 1px solid #c8ddea;
        border-radius: 9px;
        padding: 12px 14px;
        margin-bottom: 12px;
        color: #23465d;
    }

</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# PATHS AND CONFIG
# ============================================================

BASE = Path(__file__).parent
DATA_PATH = BASE / "held_out_test_patients.csv"

MODEL_DIR = BASE / "model_artifacts"
MODEL_PATH = MODEL_DIR / "tuned_xgboost_native_missing_model.pkl"
COLS_PATH = MODEL_DIR / "tuned_xgboost_native_missing_feature_columns.pkl"

THRESHOLD = 0.35

BACKEND_URL = os.getenv(
    "SEPSIS_BACKEND_URL",
    "http://localhost:8000/explain-sepsis-risk",
)


BASE_NAMES = {
    "Age": "Age",
    "Gender": "Gender",
    "HR": "Heart rate",
    "O2Sat": "Oxygen saturation",
    "Temp": "Temperature",
    "SBP": "Systolic blood pressure",
    "MAP": "Mean arterial pressure",
    "DBP": "Diastolic blood pressure",
    "Resp": "Respiratory rate",
    "BUN": "BUN",
    "Creatinine": "Creatinine",
    "Glucose": "Glucose",
    "Lactate": "Lactate",
    "WBC": "White blood cell count",
    "Platelets": "Platelets",
    "Hct": "Hematocrit",
    "Hgb": "Hemoglobin",
    "Potassium": "Potassium",
}

SUMMARY_NAMES = {
    "mean": "Average",
    "min": "Minimum",
    "max": "Maximum",
    "last": "Latest",
}

UNITS = {
    "HR": "bpm",
    "O2Sat": "%",
    "Temp": "°C",
    "SBP": "mmHg",
    "MAP": "mmHg",
    "DBP": "mmHg",
    "Resp": "/min",
    "Lactate": "mmol/L",
    "Creatinine": "mg/dL",
    "BUN": "mg/dL",
    "Glucose": "mg/dL",
    "WBC": "×10⁹/L",
    "Platelets": "×10⁹/L",
    "Hct": "%",
    "Hgb": "g/dL",
    "Potassium": "mmol/L",
}


# ============================================================
# LOADERS
# ============================================================


@st.cache_resource
def load_artifacts():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    if not COLS_PATH.exists():
        raise FileNotFoundError(f"Feature-column file not found: {COLS_PATH}")

    model = joblib.load(MODEL_PATH)
    columns = list(joblib.load(COLS_PATH))
    return model, columns


@st.cache_data
def load_patients():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "held_out_test_patients.csv was not found. "
            "Run prepare_test_patients.py first."
        )

    df = pd.read_csv(DATA_PATH)
    df["patient_id"] = df["patient_id"].astype(str)
    return df


# ============================================================
# FEATURE HELPERS
# ============================================================


def split_feature(feature: str) -> tuple[str, str | None]:
    for suffix in ("mean", "min", "max", "last"):
        token = "_" + suffix
        if feature.endswith(token):
            return feature[: -len(token)], suffix
    return feature, None


def display_name(feature: str) -> str:
    base, suffix = split_feature(feature)
    base_name = BASE_NAMES.get(base, base.replace("_", " "))

    if suffix is None:
        return base_name

    return f"{SUMMARY_NAMES[suffix]} {base_name.lower()}"


def format_value(feature: str, value: Any) -> str:
    if pd.isna(value):
        return "Unavailable"

    base, _ = split_feature(feature)

    if base == "Gender":
        return "Male" if float(value) == 1 else "Female"

    number = f"{float(value):.2f}".rstrip("0").rstrip(".")
    unit = UNITS.get(base, "")

    return f"{number} {unit}".strip()


def classify_finding(feature: str, value: float) -> tuple[str, str]:
    """Return a clinician-friendly status label and CSS class."""
    base, _ = split_feature(feature)

    if base == "Lactate":
        if value > 2:
            return "Elevated", "status-high"
        return "Within expected range", "status-low"

    if base == "Temp":
        if value > 38 or value < 36:
            return "Abnormal", "status-high"
        return "Within expected range", "status-low"

    if base == "MAP":
        if value < 65:
            return "Low", "status-high"
        if value < 70:
            return "Borderline", "status-mid"
        return "Within expected range", "status-low"

    if base == "SBP":
        if value < 90:
            return "Low", "status-high"
        if value < 100:
            return "Borderline", "status-mid"
        return "Within expected range", "status-low"

    if base == "Resp":
        if value > 22:
            return "Elevated", "status-high"
        return "Within expected range", "status-low"

    if base == "HR":
        if value > 100 or value < 50:
            return "Abnormal", "status-high"
        return "Within expected range", "status-low"

    if base == "O2Sat":
        if value < 92:
            return "Low", "status-high"
        if value < 95:
            return "Borderline", "status-mid"
        return "Within expected range", "status-low"

    if base == "Creatinine":
        if value > 1.3:
            return "Elevated", "status-high"
        return "Within expected range", "status-low"

    if base == "WBC":
        if value < 4 or value > 12:
            return "Outside expected range", "status-high"
        return "Within expected range", "status-low"

    if base == "Platelets":
        if value < 150:
            return "Low", "status-high"
        return "Within expected range", "status-low"

    if base == "Potassium":
        if value < 3.5 or value > 5.0:
            return "Abnormal", "status-high"
        return "Within expected range", "status-low"

    return "Contributing finding", "status-mid"


def latest_value(patient: pd.Series, column: str) -> str:
    value = patient.get(column, np.nan)
    return format_value(column, value)


def clinical_interpretation(feature: str, value: float, direction: str) -> str:
    """
    Convert the engineered feature into concise clinician-facing wording.
    """

    base, suffix = split_feature(feature)
    period = SUMMARY_NAMES.get(suffix, "")
    prefix = f"The {period.lower()} " if period else "The "

    if base == "Lactate":
        if suffix == "min" and value > 2:
            return (
                f"{prefix}lactate was {format_value(feature, value)}, "
                "meaning lactate remained elevated throughout the recorded period."
            )
        if value > 2:
            return f"{prefix}lactate was elevated at {format_value(feature, value)}."

    if base == "Temp":
        if value > 38:
            return (
                f"{prefix}temperature was elevated at {format_value(feature, value)}."
            )
        if value < 36:
            return f"{prefix}temperature was low at {format_value(feature, value)}."

    if base == "MAP" and value < 65:
        return (
            f"{prefix}mean arterial pressure was low at {format_value(feature, value)}."
        )

    if base == "SBP" and value < 90:
        return (
            f"{prefix}systolic blood pressure was low at "
            f"{format_value(feature, value)}."
        )

    if base == "Resp" and value > 22:
        return (
            f"{prefix}respiratory rate was elevated at {format_value(feature, value)}."
        )

    if base == "HR" and value > 100:
        return f"{prefix}heart rate was elevated at {format_value(feature, value)}."

    if base == "O2Sat" and value < 94:
        return (
            f"{prefix}oxygen saturation was reduced at {format_value(feature, value)}."
        )

    if base == "Creatinine" and value > 1.3:
        return f"{prefix}creatinine was elevated at {format_value(feature, value)}."

    if base == "WBC" and (value < 4 or value > 12):
        return (
            f"{prefix}white blood cell count was outside the usual range "
            f"at {format_value(feature, value)}."
        )

    if base == "Platelets" and value < 150:
        return f"{prefix}platelet count was reduced at {format_value(feature, value)}."

    if direction == "increased risk":
        return (
            f"{display_name(feature)} was {format_value(feature, value)} "
            "and increased the AI-generated sepsis risk."
        )

    return (
        f"{display_name(feature)} was {format_value(feature, value)} "
        "and reduced the AI-generated sepsis risk."
    )


# ============================================================
# MODEL HELPERS
# ============================================================


def risk_category(risk: float) -> dict[str, str]:
    if risk >= THRESHOLD:
        return {
            "label": "High Risk",
            "color": "#b42318",
            "message": (
                "The patient's recorded ICU findings were strongly associated "
                "with a higher sepsis risk."
            ),
        }

    if risk >= max(0.15, THRESHOLD * 0.60):
        return {
            "label": "Moderate Risk",
            "color": "#a86100",
            "message": (
                "The patient's recorded ICU findings were associated with a "
                "moderate sepsis risk."
            ),
        }

    return {
        "label": "Low Risk",
        "color": "#26734d",
        "message": (
            "The patient's recorded ICU findings were associated with a lower "
            "sepsis risk."
        ),
    }


def contribution_table(model, model_input, feature_columns):
    dmatrix = xgb.DMatrix(
        model_input,
        feature_names=feature_columns,
    )

    values = model.get_booster().predict(
        dmatrix,
        pred_contribs=True,
        validate_features=False,
    )[0][:-1]

    result = pd.DataFrame(
        {
            "feature": feature_columns,
            "contribution": values,
        }
    )

    result["absolute_contribution"] = result["contribution"].abs()

    return result.sort_values(
        "absolute_contribution",
        ascending=False,
    )


def top_factors(patient, contributions, n=5):
    factors = []

    for _, row in contributions.iterrows():
        feature = row["feature"]
        value = patient.get(feature, np.nan)

        if pd.isna(value):
            continue

        contribution = float(row["contribution"])

        direction = (
            "increased risk"
            if contribution > 0
            else "decreased risk"
            if contribution < 0
            else "neutral"
        )

        factors.append(
            {
                "feature": feature,
                "display_name": display_name(feature),
                "value": float(value),
                "formatted_value": format_value(feature, value),
                "contribution": contribution,
                "direction": direction,
                "clinical_text": clinical_interpretation(
                    feature,
                    float(value),
                    direction,
                ),
                "status_label": classify_finding(
                    feature,
                    float(value),
                )[0],
                "status_class": classify_finding(
                    feature,
                    float(value),
                )[1],
            }
        )

        if len(factors) == n:
            break

    return factors


def create_gauge(risk: float):
    category = risk_category(risk)
    monitor_cutoff = max(0.15, THRESHOLD * 0.60)

    figure = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=risk * 100,
            number={
                "suffix": "%",
                "font": {"size": 40},
            },
            title={"text": "Sepsis risk"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": category["color"]},
                "steps": [
                    {
                        "range": [0, monitor_cutoff * 100],
                        "color": "#dff1e7",
                    },
                    {
                        "range": [
                            monitor_cutoff * 100,
                            THRESHOLD * 100,
                        ],
                        "color": "#fff1cf",
                    },
                    {
                        "range": [THRESHOLD * 100, 100],
                        "color": "#f9dddd",
                    },
                ],
            },
        )
    )

    figure.update_layout(
        height=270,
        margin={"l": 20, "r": 20, "t": 45, "b": 15},
        paper_bgcolor="#ffffff",
    )

    return figure


def create_contribution_chart(factors):
    names = [factor["display_name"] for factor in factors][::-1]
    values = [factor["contribution"] for factor in factors][::-1]

    maximum = max(
        [abs(value) for value in values],
        default=1.0,
    )

    normalized = [value / maximum for value in values]

    colors = [
        "#b42318" if value > 0 else "#26734d" if value < 0 else "#64748b"
        for value in normalized
    ]

    figure = go.Figure(
        go.Bar(
            x=normalized,
            y=names,
            orientation="h",
            marker_color=colors,
            hovertemplate="%{y}<extra></extra>",
        )
    )

    figure.update_layout(
        height=330,
        margin={"l": 10, "r": 10, "t": 10, "b": 35},
        xaxis={
            "title": "Relative contribution",
            "range": [-1.05, 1.05],
            "zeroline": True,
            "zerolinecolor": "#334155",
            "showgrid": True,
            "gridcolor": "#e5e7eb",
        },
        yaxis={"title": ""},
        showlegend=False,
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
    )

    return figure


def create_factor_pie_chart(factors):
    """Show the relative share of the displayed patient-specific contributors."""
    labels = [factor["display_name"] for factor in factors]
    values = [abs(float(factor["contribution"])) for factor in factors]

    if not values or sum(values) == 0:
        values = [1 for _ in labels]

    figure = go.Figure(
        go.Pie(
            labels=labels,
            values=values,
            hole=0.52,
            textinfo="percent",
            hovertemplate="<b>%{label}</b><br>Share: %{percent}<extra></extra>",
            sort=False,
        )
    )

    figure.update_layout(
        title={
            "text": "Share of displayed clinical contributors",
            "x": 0.5,
            "xanchor": "center",
            "font": {"size": 17},
        },
        height=330,
        margin={"l": 15, "r": 15, "t": 55, "b": 15},
        showlegend=True,
        legend={
            "orientation": "h",
            "yanchor": "top",
            "y": -0.08,
            "xanchor": "center",
            "x": 0.5,
            "font": {"size": 12},
        },
        paper_bgcolor="#ffffff",
    )

    return figure


def plain_language_summary(factors, category):
    increased = [
        factor["clinical_text"]
        for factor in factors
        if factor["direction"] == "increased risk"
    ]

    decreased = [
        factor["clinical_text"]
        for factor in factors
        if factor["direction"] == "decreased risk"
    ]

    parts = []

    if increased:
        parts.append(
            "The strongest findings increasing the sepsis propensity were: "
            + " ".join(increased[:3])
        )

    if decreased:
        parts.append(
            "Factors reducing the sepsis propensity included: "
            + " ".join(decreased[:2])
        )

    if not parts:
        parts.append("No single measured feature clearly dominated the estimate.")

    return " ".join(parts)


# ============================================================
# BACKEND PAYLOAD
# ============================================================


def get_numeric(patient, column):
    value = patient.get(column, np.nan)
    return None if pd.isna(value) else float(value)


def explanation_payload(
    patient_id,
    patient,
    risk,
    factors,
    question,
    chat_history=None,
):
    chat_history = chat_history or []

    age_value = patient.get("Age", np.nan)

    # The backend requires an integer age.
    age = 0 if pd.isna(age_value) else round(float(age_value))

    gender_value = patient.get("Gender", np.nan)

    gender = (
        "Unknown"
        if pd.isna(gender_value)
        else "Male"
        if float(gender_value) == 1
        else "Female"
    )

    category = risk_category(risk)

    clinical_relevance = " ".join(factor["clinical_text"] for factor in factors[:5])  # noqa: F841

    return {
        "patient_id": str(patient_id),
        "demographics": {
            "age": age,
            "gender": gender,
        },
        "vitals": {
            "heart_rate": get_numeric(patient, "HR_last"),
            "oxygen_saturation": get_numeric(patient, "O2Sat_last"),
            "temperature_c": get_numeric(patient, "Temp_last"),
            "systolic_blood_pressure": get_numeric(patient, "SBP_last"),
            "mean_arterial_pressure": get_numeric(patient, "MAP_last"),
            "diastolic_blood_pressure": get_numeric(patient, "DBP_last"),
            "respiratory_rate": get_numeric(patient, "Resp_last"),
        },
        "labs": {
            "bun": get_numeric(patient, "BUN_last"),
            "creatinine": get_numeric(patient, "Creatinine_last"),
            "glucose": get_numeric(patient, "Glucose_last"),
            "lactate": get_numeric(patient, "Lactate_last"),
            "wbc": get_numeric(patient, "WBC_last"),
            "platelets": get_numeric(patient, "Platelets_last"),
            "hematocrit": get_numeric(patient, "Hct_last"),
            "hemoglobin": get_numeric(patient, "Hgb_last"),
            "potassium": get_numeric(patient, "Potassium_last"),
        },
        "model_output": {
            "sepsis_risk_score": float(risk),
            "operating_threshold": float(THRESHOLD),
            "risk_category": category["label"],
            "top_contributors": [
                {
                    "feature": factor["feature"],
                    "display_name": factor["display_name"],
                    "value": float(factor["value"]),
                    "contribution": float(factor["contribution"]),
                    "direction": factor["direction"],
                }
                for factor in factors
            ],
        },
        "question": question,
        "chat_history": chat_history[-10:],
        "clinician_context": (
            "This is a patient-level sepsis risk based on aggregated ICU "
            "measurements. Use the recorded patient values and contributing findings "
            "as the source of truth. Synthesize related findings rather than listing "
            "each one independently. Directly answer the clinician's current question "
            "and use earlier conversation turns when relevant. Do not diagnose sepsis, "
            "invent missing information, or recommend treatment."
        ),
        "model_name": "llama-3.3-70b-versatile",
        "temperature": 0.35,
        "max_tokens": 500,
    }


def request_explanation(payload):
    response = requests.post(
        BACKEND_URL,
        json=payload,
        timeout=60,
    )

    if response.status_code != 200:
        raise RuntimeError(f"Backend returned {response.status_code}: {response.text}")

    data = response.json()
    explanation = data.get("explanation")

    if not explanation:
        raise RuntimeError("Backend response did not include an explanation.")

    return explanation


# ============================================================
# DASHBOARD AND CHAT HELPERS
# ============================================================


def build_prediction_table(
    model, patients: pd.DataFrame, feature_columns: list[str]
) -> pd.DataFrame:
    """Generate one cached sepsis risk per held-out patient for the demo dashboard."""
    matrix = patients.reindex(columns=feature_columns).astype("float32")
    risks = model.predict_proba(matrix)[:, 1]

    result = patients[["patient_id"]].copy()
    result["risk"] = risks
    result["risk_percent"] = (result["risk"] * 100).round(1)
    result["risk_category"] = result["risk"].apply(
        lambda value: risk_category(float(value))["label"]
    )
    return result


def get_demo_panel(predictions: pd.DataFrame, panel_size: int = 24) -> pd.DataFrame:
    """Create a stable demo patient panel with a mix of risk categories."""
    high = predictions[predictions["risk_category"] == "High Risk"].nlargest(8, "risk")
    moderate = predictions[predictions["risk_category"] == "Moderate Risk"].nlargest(
        8, "risk"
    )
    low = predictions[predictions["risk_category"] == "Low Risk"].nsmallest(8, "risk")

    panel = pd.concat([high, moderate, low], ignore_index=True)
    if len(panel) < panel_size:
        remainder = predictions[~predictions["patient_id"].isin(panel["patient_id"])]
        panel = pd.concat(
            [panel, remainder.head(panel_size - len(panel))], ignore_index=True
        )

    return panel.head(panel_size)


def load_assessment(selected_id, patients, model, feature_columns):
    matching = patients[patients["patient_id"] == str(selected_id)]
    if matching.empty:
        raise ValueError("Patient ID not found.")

    patient = matching.iloc[0]
    model_input = pd.DataFrame(
        [{feature: patient.get(feature, np.nan) for feature in feature_columns}],
        columns=feature_columns,
    ).astype("float32")

    risk = float(model.predict_proba(model_input)[0, 1])
    contributions = contribution_table(model, model_input, feature_columns)
    factors = top_factors(patient, contributions, n=5)

    assessment = {
        "patient_id": str(selected_id),
        "patient": patient.to_dict(),
        "risk": risk,
        "factors": factors,
    }

    st.session_state["assessment"] = assessment
    st.session_state["chat_messages"] = []
    st.session_state.pop("explanation", None)
    return assessment


def render_patient_snapshot(patient: pd.Series):
    st.markdown(
        f"""
<div class="content-card">
    <b>Patient snapshot</b>
    <div class="patient-overview-grid">
        <div class="overview-item"><div class="overview-label">Latest heart rate</div><div class="overview-value">{html.escape(latest_value(patient, "HR_last"))}</div></div>
        <div class="overview-item"><div class="overview-label">Latest temperature</div><div class="overview-value">{html.escape(latest_value(patient, "Temp_last"))}</div></div>
        <div class="overview-item"><div class="overview-label">Latest MAP</div><div class="overview-value">{html.escape(latest_value(patient, "MAP_last"))}</div></div>
        <div class="overview-item"><div class="overview-label">Latest oxygen saturation</div><div class="overview-value">{html.escape(latest_value(patient, "O2Sat_last"))}</div></div>
        <div class="overview-item"><div class="overview-label">Latest respiratory rate</div><div class="overview-value">{html.escape(latest_value(patient, "Resp_last"))}</div></div>
        <div class="overview-item"><div class="overview-label">Latest lactate</div><div class="overview-value">{html.escape(latest_value(patient, "Lactate_last"))}</div></div>
    </div>
</div>
""",
        unsafe_allow_html=True,
    )


def ask_assistant(question: str, assessment: dict, chat_history=None):
    patient = pd.Series(assessment["patient"])
    risk = float(assessment["risk"])
    factors = assessment["factors"]

    payload = explanation_payload(
        assessment["patient_id"],
        patient,
        risk,
        factors,
        question,
        chat_history=chat_history,
    )

    with st.spinner("Reviewing the patient context..."):
        return request_explanation(payload)


def render_chat_interface(assessment: dict):
    st.caption(
        "Ask free-text questions about the selected patient. Responses are grounded "
        "in the displayed patient values, sepsis risk, and contributing findings."
    )

    quick_prompts = [
        "Summarize why this patient received this sepsis risk.",
        "Which abnormal findings matter most and why?",
        "Which findings increased the sepsis risk?",
        "Which findings reduced the sepsis risk?",
    ]

    prompt_cols = st.columns(2)
    selected_quick_prompt = None

    for index, prompt in enumerate(quick_prompts):
        if prompt_cols[index % 2].button(
            prompt,
            key=f"dialog_quick_prompt_{index}",
            use_container_width=True,
        ):
            selected_quick_prompt = prompt

    for message in st.session_state.get("chat_messages", []):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    typed_question = st.chat_input(
        "Ask a question about this patient's sepsis risk",
        key="dialog_chat_input",
    )

    active_question = typed_question or selected_quick_prompt

    if active_question:
        previous_history = st.session_state.get("chat_messages", []).copy()

        st.session_state["chat_messages"].append(
            {"role": "user", "content": active_question}
        )

        with st.chat_message("user"):
            st.markdown(active_question)

        try:
            answer = ask_assistant(
                active_question,
                assessment,
                chat_history=previous_history[-10:],
            )
        except Exception as error:  # noqa: BLE001
            answer = f"The explanation service could not respond.\\n\\n{error}"

        st.session_state["chat_messages"].append(
            {"role": "assistant", "content": answer}
        )

        with st.chat_message("assistant"):
            st.markdown(answer)

    if st.session_state.get("chat_messages"):  # noqa: SIM102
        if st.button("Clear conversation", key="dialog_clear_chat"):
            st.session_state["chat_messages"] = []
            st.rerun()

    st.markdown(
        """
<div class="safety-note">
    <b>Research prototype:</b> The AI model generates the sepsis risk.
    The AI assistant explains the result and does not diagnose sepsis or recommend treatment.
</div>
""",
        unsafe_allow_html=True,
    )


if hasattr(st, "dialog"):

    @st.dialog("Ask SepsisGuard AI", width="large")
    def open_chat_dialog():
        current = st.session_state.get("assessment")
        if current is None:
            st.info("Select a patient and generate an assessment first.")
            return
        render_chat_interface(current)


# ============================================================
# APP
# ============================================================


# Initialize session state.
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False
if "active_page" not in st.session_state:
    st.session_state["active_page"] = "Dashboard"
if "chat_messages" not in st.session_state:
    st.session_state["chat_messages"] = []


# ------------------------------------------------------------
# DEMO LOGIN
# ------------------------------------------------------------
if not st.session_state["authenticated"]:
    st.markdown(
        """
<div class="login-shell">
    <div class="login-logo">🩺</div>
    <div class="app-title">SepsisGuard AI</div>
    <div class="app-subtitle">Clinical risk review prototype</div>
</div>
""",
        unsafe_allow_html=True,
    )

    left, center, right = st.columns([1, 1.05, 1])
    with center:
        with st.form("login_form"):
            st.subheader("Clinician sign in")
            username = st.text_input("Username", placeholder="doctor")
            password = st.text_input(
                "Password", type="password", placeholder="sepsisguard"
            )
            submitted = st.form_submit_button(
                "Sign in", type="primary", use_container_width=True
            )

        st.caption("Demo credentials: doctor / sepsisguard")

        if submitted:
            if username.strip().lower() == "doctor" and password == "sepsisguard":
                st.session_state["authenticated"] = True
                st.session_state["clinician_name"] = "Dr. Maya Patel"
                st.rerun()
            else:
                st.error("Incorrect demo username or password.")

    st.stop()


# ------------------------------------------------------------
# LOAD DATA AND MODEL
# ------------------------------------------------------------
try:
    model, feature_columns = load_artifacts()
    patients = load_patients()
except Exception as error:  # noqa: BLE001
    st.error("The model or patient data could not be loaded.")
    st.exception(error)
    st.stop()

if "patient_predictions" not in st.session_state:
    st.session_state["patient_predictions"] = build_prediction_table(
        model,
        patients,
        feature_columns,
    )

predictions = st.session_state["patient_predictions"]
demo_panel = get_demo_panel(predictions)
patient_ids = sorted(patients["patient_id"].astype(str).unique().tolist())


# ------------------------------------------------------------
# SIDEBAR NAVIGATION
# ------------------------------------------------------------
st.sidebar.markdown("## 🩺 SepsisGuard AI")
st.sidebar.caption("Clinical review prototype")
st.sidebar.markdown("---")
st.sidebar.markdown(f"**{st.session_state.get('clinician_name', 'Demo Clinician')}**")
st.sidebar.caption("Critical Care Service")

page = st.sidebar.radio(
    "Navigation",
    ["Dashboard", "Patient Assessment"],
    index=0 if st.session_state["active_page"] == "Dashboard" else 1,
)
st.session_state["active_page"] = page

st.sidebar.markdown("---")
if st.sidebar.button("Sign out"):
    st.session_state.clear()
    st.rerun()


# ------------------------------------------------------------
# DASHBOARD PAGE
# ------------------------------------------------------------
if page == "Dashboard":
    st.markdown(
        f"""
<div class="dashboard-hero">
    <h2>Welcome, {html.escape(st.session_state.get("clinician_name", "Clinician"))}</h2>
    <p>Review the demo patient panel and open a patient assessment.</p>
</div>
""",
        unsafe_allow_html=True,
    )

    high_count = int((demo_panel["risk_category"] == "High Risk").sum())
    moderate_count = int((demo_panel["risk_category"] == "Moderate Risk").sum())
    low_count = int((demo_panel["risk_category"] == "Low Risk").sum())

    metric_1, metric_2, metric_3, metric_4 = st.columns(4)
    metric_1.metric("Patients in demo panel", len(demo_panel))
    metric_2.metric("High risk", high_count)
    metric_3.metric("Moderate risk", moderate_count)
    metric_4.metric("Low risk", low_count)

    st.markdown("### Patient queue")
    st.caption(
        "This is a demonstration queue created from held-out evaluation patients."
    )

    queue_left, queue_right = st.columns([1.3, 0.7], gap="large")

    with queue_left:
        queue_table = demo_panel.copy()
        queue_table["Sepsis risk"] = queue_table["risk_percent"].map(
            lambda value: f"{value:.1f}%"
        )
        queue_table = queue_table.rename(
            columns={
                "patient_id": "Patient ID",
                "risk_category": "Risk category",
            }
        )[["Patient ID", "Sepsis risk", "Risk category"]]
        st.dataframe(queue_table, hide_index=True, use_container_width=True)

    with queue_right:
        st.markdown('<div class="selection-card">', unsafe_allow_html=True)
        st.subheader("Open patient")
        dashboard_patient = st.selectbox(
            "Patient ID",
            demo_panel["patient_id"].astype(str).tolist(),
            key="dashboard_patient",
        )
        if st.button("Open assessment", type="primary", use_container_width=True):
            load_assessment(dashboard_patient, patients, model, feature_columns)
            st.session_state["active_page"] = "Patient Assessment"
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)


# ------------------------------------------------------------
# PATIENT ASSESSMENT PAGE
# ------------------------------------------------------------
else:
    st.markdown(
        """
<div class="app-header">
    <div class="app-title">Patient Assessment</div>
    <div class="app-subtitle">Review risk, recorded findings, and a grounded AI explanation.</div>
</div>
""",
        unsafe_allow_html=True,
    )

    top_left, top_right = st.columns([0.82, 1.18], gap="large")

    with top_left:
        st.markdown('<div class="selection-card">', unsafe_allow_html=True)
        st.subheader("Select patient")

        current_id = None
        if st.session_state.get("assessment"):
            current_id = st.session_state["assessment"]["patient_id"]

        default_index = (
            patient_ids.index(current_id) if current_id in patient_ids else None
        )
        selected_id = st.selectbox(
            "Patient ID",
            patient_ids,
            index=default_index,
            placeholder="Type to search for a patient ID",
            key="assessment_patient",
        )

        if st.button(
            "Generate assessment",
            type="primary",
            disabled=selected_id is None,
            use_container_width=True,
        ):
            try:
                load_assessment(selected_id, patients, model, feature_columns)
                st.rerun()
            except Exception as error:  # noqa: BLE001
                st.error(str(error))

        st.caption(
            f"{len(patient_ids):,} held-out patient records are available. Type to search."
        )
        st.markdown("</div>", unsafe_allow_html=True)

        if st.session_state.get("assessment"):
            render_patient_snapshot(
                pd.Series(st.session_state["assessment"]["patient"])
            )

    assessment = st.session_state.get("assessment")

    with top_right:
        if assessment is None:
            st.markdown(
                """
<div class="content-card">
    <b>No assessment generated</b>
    <p style="color:#64748b;">Search for a valid patient ID and generate the assessment.</p>
</div>
""",
                unsafe_allow_html=True,
            )
        else:
            patient = pd.Series(assessment["patient"])
            risk = float(assessment["risk"])
            category = risk_category(risk)

            age = patient.get("Age", np.nan)
            gender_value = patient.get("Gender", np.nan)
            age_text = "Unknown" if pd.isna(age) else str(round(float(age)))
            gender_text = (
                "Unknown"
                if pd.isna(gender_value)
                else "Male"
                if float(gender_value) == 1
                else "Female"
            )

            st.markdown(
                f"""
<div class="patient-banner">
    <span><b>{html.escape(assessment["patient_id"])}</b></span>
    <span>Age {age_text} · {html.escape(gender_text)}</span>
</div>
""",
                unsafe_allow_html=True,
            )

            st.markdown(
                f"""
<div class="risk-card" style="--status-color:{category["color"]};">
    <div style="color:#64748b;font-size:15px;">Sepsis risk</div>
    <div class="risk-score">{risk * 100:.1f}%</div>
    <div class="risk-label">{category["label"]}</div>
    <p>{html.escape(category["message"])}</p>
    <div class="risk-meta">
        <span class="risk-chip">
            Review boundary: {THRESHOLD * 100:.0f}%
        </span>
        <span class="risk-chip">
            {"Meets the review boundary" if risk >= THRESHOLD else "Below the review boundary"}
        </span>
    </div>
    <p style="color:#64748b;font-size:14px;margin-top:10px;">
        The 35% review boundary is the point at which this prototype displays a patient
        as High Risk. It is intentionally lower than 50% so that more potentially
        concerning patterns are brought forward for clinical review.
    </p>
</div>
""",
                unsafe_allow_html=True,
            )

            st.plotly_chart(
                create_gauge(risk),
                use_container_width=True,
                config={"displayModeBar": False},
            )

    if assessment is not None:
        patient = pd.Series(assessment["patient"])
        risk = float(assessment["risk"])
        factors = assessment["factors"]
        category = risk_category(risk)

        st.markdown("---")
        st.subheader("Clinical Findings Influencing Sepsis Propensity")

        factors_left, factors_right = st.columns([1.05, 0.95], gap="large")

        with factors_left:
            for factor in factors:
                color = (
                    "#b42318"
                    if factor["contribution"] > 0
                    else "#26734d"
                    if factor["contribution"] < 0
                    else "#64748b"
                )

                st.markdown(
                    f"""
<div class="factor-card" style="--factor-color:{color};">
    <div class="factor-title">
        {html.escape(factor["display_name"])}
        <span class="status-badge {factor["status_class"]}">{html.escape(factor["status_label"])}</span>
    </div>
    <div class="factor-value">
        <b>{html.escape(factor["formatted_value"])}</b>
    </div>
</div>
""",
                    unsafe_allow_html=True,
                )

            st.markdown(
                f"""
<div class="plain-summary">
    <b>Key Clinical Findings</b><br>
    {html.escape(plain_language_summary(factors, category))}
</div>
""",
                unsafe_allow_html=True,
            )

        with factors_right:
            st.plotly_chart(
                create_contribution_chart(factors),
                use_container_width=True,
                config={"displayModeBar": False},
            )

            st.plotly_chart(
                create_factor_pie_chart(factors),
                use_container_width=True,
                config={"displayModeBar": False},
            )

        st.markdown("---")

        st.markdown(
            """
<div class="safety-note">
    <b>Research prototype:</b> The sepsis risk is generated by the trained AI model.
    The AI assistant only explains the result and does not diagnose sepsis or recommend treatment.
</div>
""",
            unsafe_allow_html=True,
        )


# Floating AI launcher.
with st.container(key="floating_chat_launcher"):
    if st.button(
        "💬 Ask SepsisGuard AI", key="open_floating_chat", use_container_width=True
    ):
        if st.session_state.get("assessment") is None:
            st.warning("Select a patient and generate an assessment first.")
        elif hasattr(st, "dialog"):
            open_chat_dialog()
        else:
            st.session_state["show_inline_chat_fallback"] = True

if st.session_state.get("show_inline_chat_fallback") and st.session_state.get(
    "assessment"
):
    st.markdown("---")
    st.subheader("Ask SepsisGuard AI")
    render_chat_interface(st.session_state["assessment"])
