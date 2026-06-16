# Section 1  Imports & Configuration
# Section 2  Load Model & Reports
# Section 3  WOE Transformation Engine
# Section 4  Applicant Input Form
# Section 5  Prediction Engine
# Section 6  Score & Decision
# Section 7  AI Explainability 
# Section 8  Dashboard Visualizations
# Section 9  Model Monitoring
# Section 10 Model Information
# Section 11 Download Report

#Section 1  Imports & Configuration
import asyncio
import sys

if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(
        asyncio.WindowsSelectorEventLoopPolicy()
    )

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import plotly.graph_objects as go
import re
import matplotlib.pyplot as plt
from datetime import datetime
import statsmodels.api as sm

from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
from io import BytesIO

# SHAP will be added later
# import shap

# ---------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------

st.set_page_config(
    page_title="Credit Risk Scorecard",
    page_icon="💳",
    layout="wide"
)

st.title("💳 AI Powered Credit Risk Scorecard")

st.markdown(
"""
This application predicts the **Probability of Default (PD)**,
calculates the **Credit Score**,
assigns a **Risk Band**,
and recommends a lending decision.
"""
)

#Section 2  Load Model & Reports
# ---------------------------------------------------
# LOAD MODEL FILES
# ---------------------------------------------------

@st.cache_resource
def load_model():
    # 1. Get the directory of the current app.py file
    current_dir = Path(__file__).parent

    # 2. Resolve the absolute paths to all your required files
    model_path = current_dir.parent / "Models" / "final_logit.pkl"
    features_path = current_dir.parent / "Models" / "final_features.pkl"  # Adjust filename if different
    woe_path = current_dir.parent / "Models" / "woe_bins.pkl"  # Adjust filename if different

    # 3. Load each file safely using joblib
    model = joblib.load(model_path)
    feature_list = joblib.load(features_path)
    woe_bins = joblib.load(woe_path)

    # 4. Return them all together
    return model, feature_list, woe_bins

global model, feature_list, woe_bins

model, feature_list, woe_bins = load_model()



# ---------------------------------------------------
# LOAD REPORTS
# ---------------------------------------------------

@st.cache_data
def load_reports():
    # 1. Get the directory of the current app.py file
    current_dir = Path(__file__).parent

    # 2. Resolve the absolute paths to your report files
    metrics_path = current_dir.parent / "Reports" / "final_metrics.csv"
    approval_path = current_dir.parent / "Reports" / "approval_strategy.csv"
    risk_path = current_dir.parent / "Reports" / "risk_band_summary.csv"

    # 3. Read the CSVs using the absolute path objects
    metrics = pd.read_csv(metrics_path)
    approval_strategy = pd.read_csv(approval_path)
    risk_band_summary = pd.read_csv(risk_path)

    return metrics, risk_band_summary, approval_strategy

metrics, risk_band_summary, approval_strategy = load_reports()

current_dir = Path(__file__).parent

decision_policy = pd.read_csv(
    current_dir.parent / "Reports" / "decision_policy.csv"
)

risk_band_lookup = pd.read_csv(
    current_dir.parent / "Reports" / "risk_band_lookup.csv"
)

# Section 3  SIDEBAR

st.sidebar.image(
    "https://img.icons8.com/color/96/bank-card-back-side.png",
    width=80
)

st.sidebar.title("Page Navigation")

page = st.sidebar.radio(
    "Select Module",
    [
        "Prediction",
        "Model Monitoring",
        "Model Information"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### AI Powered Credit Risk Scorecard")
st.sidebar.caption("Version 1.0")
st.sidebar.caption("Developed by Shrabani Das")

#Section 4 WOE Transformation Engine
# ---------------------------------------------------
# WOE LOOKUP FUNCTION
# ---------------------------------------------------

import numpy as np

def lookup_woe(value, bin_table):

    for _, row in bin_table.iterrows():

        interval = row["bin"]

        interval = (
            interval.replace("[", "")
                    .replace(")", "")
                    .replace("]", "")
        )

        lower, upper = interval.split(",")

        lower = lower.strip()
        upper = upper.strip()

        lower = -np.inf if lower == "-inf" else float(lower)
        upper = np.inf if upper == "inf" else float(upper)

        if lower <= value < upper:
            return row["woe"]

    return np.nan

# ---------------------------------------------------
# RAW TO WOE TRANSFORMATION
# ---------------------------------------------------

def transform_to_woe(raw_input, woe_bins):

    transformed = {}

    for variable, value in raw_input.items():

        if variable in woe_bins:

            transformed[variable+"_woe"] = lookup_woe(
                value,
                woe_bins[variable]
            )

    return pd.DataFrame([transformed])

# ---------------------------------------------------
# DECISION POLICY
# ---------------------------------------------------

def assign_decision(score):

    for _, row in decision_policy.iterrows():

        if row["Min Score"] <= score <= row["Max Score"]:

            return row["Decision"]

    return "UNKNOWN"


# ---------------------------------------------------
# RISK BAND LOOKUP
# ---------------------------------------------------

def assign_risk_band(score):

    # print("Score =", score)

    # print(risk_band_lookup)

    for _, row in risk_band_lookup.iterrows():

        # print(row)

        if row["Min Score"] <= score <= row["Max Score"]:

            # print("Matched!")

            return (
                row["Risk Band"],
                row["Expected Bad Rate"]
            )

    return (
        "Unknown",
        np.nan
    )


# =====================================================
# SECTION 5 & 6 : PREDICTION PAGE & Applicant Input Form
# =====================================================

def display_gauge(score):

    fig = go.Figure(

        go.Indicator(

            mode="gauge+number",

            value=score,

            title={"text":"Credit Score"},

            gauge={

                "axis":{"range":[300,850]},

                "bar":{"color":"darkblue"},

                "steps":[

                    {"range":[300,580],"color":"#ffcccc"},

                    {"range":[580,620],"color":"#fff3cd"},

                    {"range":[620,680],"color":"#d4edda"},

                    {"range":[680,850],"color":"#b7eb8f"}

                ]

            }

        )

    )

    st.plotly_chart(
        fig,
        width="stretch"
    )
def display_explanation(input_df, model):

    st.divider()

    st.header("🧠 AI Explainability")

    # Logistic Regression Coefficients
    coefficients = model.params.drop("const")

    # Applicant WOE Values
    applicant = input_df.drop(columns="const").iloc[0]

    # Feature Contributions
    contribution = coefficients * applicant

    explanation = pd.DataFrame({

        "Feature": contribution.index,
        "Contribution": contribution.values

    })

    # Business-friendly feature names
    feature_labels = {

        "age_years_woe": "Applicant Age",
        "num_approved_woe": "Approved Loans",
        "num_refused_woe": "Refused Loans",
        "avg_credit_age_years_woe": "Average Credit Age",
        "avg_credit_grant_ratio_woe": "Credit Grant Ratio",
        "avg_instalment_remaining_woe": "Average EMI Remaining"

    }

    explanation["Feature"] = explanation["Feature"].map(feature_labels)

    # Sort by importance
    explanation["Absolute"] = explanation["Contribution"].abs()

    explanation = explanation.sort_values(
        "Absolute",
        ascending=False
    )

    # Risk Direction
    explanation["Impact"] = np.where(

        explanation["Contribution"] > 0,

        "🔺 Increased Risk",

        "🟢 Reduced Risk"

    )

    # Round values
    explanation["Contribution"] = explanation["Contribution"].round(3)

    st.subheader("Top Risk Drivers")

    st.table(

        explanation[
            [
                "Feature",
                "Impact",
                "Contribution"
            ]
        ]

    )

    return explanation


def display_recommendation(

        decision,

        probability,

        score,

        risk_band,

        expected_bad_rate,

        explanation

):

    st.divider()

    st.header("🤖 AI Recommendation")

    if decision=="APPROVE":

        st.success(f"""

### Recommended Action

✅ APPROVE

• Probability of Default: **{probability:2%}**

• Credit Score: **{score}**

• Risk Band: **{risk_band}**

• Expected Bad Rate: **{expected_bad_rate}**

Applicant satisfies current lending policy.

""")

    elif decision=="UNDERWRITER REVIEW":

        st.warning(f"""

### Recommended Action

🟠 MANUAL UNDERWRITER REVIEW

• Probability of Default: **{probability:2%}**

• Credit Score: **{score}**

• Risk Band: **{risk_band}**

• Verify income

• Verify bureau history

• Review repayment capacity

""")

    else:

        st.error(f"""

### Recommended Action

❌ REJECT APPLICATION

• Probability of Default: **{probability:2%}**

• Credit Score: **{score}**

• Risk Band: **{risk_band}**

• Expected Bad Rate: **{expected_bad_rate}**

Application exceeds acceptable credit risk.

""")

    st.subheader("📌 Key Drivers")

    top3 = explanation.head(3)

    for _, row in top3.iterrows():

        if row["Contribution"] > 0:

            st.write(

                f"🔺 **{row['Feature']}** increased predicted default risk."

            )

        else:

            st.write(

                f"🟢 **{row['Feature']}** reduced predicted default risk."

            )

# =====================================================
# PDF REPORT
# =====================================================

def generate_pdf(
    raw_input,
    probability,
    score,
    risk_band,
    decision,
    expected_bad_rate,
    explanation
):

    buffer = BytesIO()

    doc = SimpleDocTemplate(buffer)

    styles = getSampleStyleSheet()

    story = []

    story.append(
        Paragraph("<b>AI Powered Credit Risk Assessment Report</b>", styles["Title"])
    )

    story.append(
        Paragraph(
            f"Generated On : {datetime.now().strftime('%d-%b-%Y %H:%M')}",
            styles["Normal"]
        )
    )

    story.append(Paragraph("<br/>", styles["Normal"]))

    # --------------------------------------------------
    # Applicant Details
    # --------------------------------------------------

    story.append(
        Paragraph("<b>Applicant Information</b>", styles["Heading2"])
    )

    for key, value in raw_input.items():

        label = key.replace("_", " ").title()

        story.append(
            Paragraph(
                f"<b>{label}</b> : {value}",
                styles["Normal"]
            )
        )

    story.append(Paragraph("<br/>", styles["Normal"]))

    # --------------------------------------------------
    # Prediction
    # --------------------------------------------------

    story.append(
        Paragraph("<b>Prediction Results</b>", styles["Heading2"])
    )

    story.append(
        Paragraph(
            f"Probability of Default : {probability:.2%}",
            styles["Normal"]
        )
    )

    story.append(
        Paragraph(
            f"Credit Score : {score}",
            styles["Normal"]
        )
    )

    story.append(
        Paragraph(
            f"Risk Band : {risk_band}",
            styles["Normal"]
        )
    )

    story.append(
        Paragraph(
            f"Decision : {decision}",
            styles["Normal"]
        )
    )

    story.append(
        Paragraph(
            f"Expected Bad Rate : {expected_bad_rate}",
            styles["Normal"]
        )
    )

    story.append(Paragraph("<br/>", styles["Normal"]))

    # --------------------------------------------------
    # Top Drivers
    # --------------------------------------------------

    story.append(
        Paragraph("<b>Top Risk Drivers</b>", styles["Heading2"])
    )

    for _, row in explanation.head(5).iterrows():

        story.append(

            Paragraph(

                f"{row['Feature']} : {row['Impact']} ({row['Contribution']})",

                styles["Normal"]

            )

        )

    story.append(Paragraph("<br/>", styles["Normal"]))

    # --------------------------------------------------
    # Recommendation
    # --------------------------------------------------

    story.append(
        Paragraph("<b>AI Recommendation</b>", styles["Heading2"])
    )

    story.append(
        Paragraph(
            f"<b>{decision}</b>",
            styles["Normal"]
        )
    )

    doc.build(story)

    buffer.seek(0)

    return buffer

if page == "Prediction":

    st.header("📝 Applicant Information")
    st.caption("Enter applicant information to estimate Probability of Default.")

    col1, col2 = st.columns(2)

    with col1:

        age_years = st.number_input(
            "Applicant Age (Years)",
            min_value=18,
            max_value=80,
            value=30
        )

        num_approved = st.number_input(
            "Number of Approved Loans",
            min_value=0,
            max_value=20,
            value=1
        )

        num_refused = st.number_input(
            "Number of Refused Loans",
            min_value=0,
            max_value=20,
            value=0
        )

    with col2:

        avg_credit_age_years = st.number_input(
            "Average Credit Age (Years)",
            min_value=0.0,
            value=5.0
        )

        avg_credit_grant_ratio = st.number_input(
            "Average Credit Grant Ratio",
            min_value=0.0,
            value=0.80,
            step=0.01
        )

        avg_instalment_remaining = st.number_input(
            "Average EMI Remaining (₹)",
            min_value=0.0,
            value=5000.0,
            step=500.0
        )

    # -----------------------------------------
    # Create Raw Input Dictionary
    # -----------------------------------------

    raw_input = {

        "age_years": age_years,
        "num_approved": num_approved,
        "num_refused": num_refused,
        "avg_credit_age_years": avg_credit_age_years,
        "avg_credit_grant_ratio": avg_credit_grant_ratio,
        "avg_instalment_remaining": avg_instalment_remaining

    }

    st.divider()

    predict = st.button(
        "🚀 Predict Credit Risk",
        use_container_width=True
    )

    if predict:

        # -----------------------------------------
        # Convert Raw Values to WOE
        # -----------------------------------------
        model, feature_list, woe_bins = load_model()

        input_df = transform_to_woe(raw_input, woe_bins)
        # print(raw_input)
        # print(input_df)

        # Add constant for Statsmodels Logistic Regression

        input_df = sm.add_constant(
            input_df,
            has_constant="add"
        )

        # Arrange columns exactly as model expects

        input_df = input_df[
            model.model.exog_names
        ]

        # st.write("### WOE Input")
        # st.dataframe(input_df)

        # -----------------------------------------
        # Predict Probability of Default
        # -----------------------------------------

        probability = float(
            model.predict(input_df).iloc[0]
        )

        # st.write(input_df)

        # -----------------------------------------
        # Credit Score
        # -----------------------------------------

        BASE_SCORE = 600
        PDO = 20
        BASE_ODDS = 50

        factor = PDO / np.log(2)

        offset = (
            BASE_SCORE
            - factor * np.log(BASE_ODDS)
        )

        odds = (
            (1 - probability)
            /
            probability
        )

        score = round(

            offset
            +
            factor * np.log(odds)

        )

        # st.write("PD =", probability)
        # st.write("Score =", score)
        # st.write(input_df)

        # -----------------------------------------
        # Decision & Risk Band
        # -----------------------------------------

        decision = assign_decision(score)

        risk_band_name, expected_bad_rate = assign_risk_band(score)

        # -----------------------------------------
        # Results
        # -----------------------------------------

        st.subheader("📊 Prediction Results")

        c1, c2, c3, c4 = st.columns(4)

        with c1:

            st.subheader("Risk Meter")

            risk = int((1 - probability) * 100)

            st.progress(risk)

            st.write(f"Risk Score : {risk}/100")

        with c2:

            st.metric(
                "Credit Score",
                score
            )

        with c3:

            st.metric(
                "Risk Band",
                risk_band_name,
                delta=f"Expected BR {expected_bad_rate}"
            )

        with c4:

            st.write("### Decision")

            if decision == "APPROVE":

                st.success("APPROVE")

            elif decision == "UNDERWRITER REVIEW":

                st.warning("UNDERWRITER REVIEW")

            else:

                st.error("REJECT")

        st.divider()

        st.subheader("📋 Decision Summary")

        if decision == "APPROVE":

            st.success(
                """
                ✅ Excellent Credit Score

                ✅ Low Probability of Default

                ✅ Low Expected Bad Rate

                ✅ Auto Approval Recommended
                """
            )

        elif decision == "UNDERWRITER REVIEW":

            st.warning(
                """
                🟠 Medium Risk Applicant

                🟠 Borderline Credit Score

                🟠 Manual Verification Recommended

                🟠 Review Income & Bureau History
                """
            )

        else:

            st.error(
                """
                ❌ High Probability of Default

                ❌ Score Below Approval Threshold

                ❌ High Expected Bad Rate

                ❌ Reject Application
                """
            )

        display_gauge(score)

        explanation = display_explanation(

            input_df,

            model

        )

        display_recommendation(

            decision,

            probability,

            score,

            risk_band_name,

            expected_bad_rate,

            explanation

        )

        st.divider()

        st.header("📄 Download Report")

        pdf = generate_pdf(
            raw_input,
            probability,
            score,
            risk_band_name,
            decision,
            expected_bad_rate,
            explanation
        )

        st.download_button(
            label="📥 Download Credit Assessment Report",
            data=pdf,
            file_name="Credit_Risk_Assessment_Report.pdf",
            mime="application/pdf",
            width="stretch"
        )

# =====================================================
# MODEL MONITORING
# =====================================================

if page == "Model Monitoring":

    st.header("📈 Model Monitoring Dashboard")

    # Load Monitoring Reports
    current_dir = Path(__file__).parent

    psi = pd.read_csv(
        current_dir.parent / "Reports" / "psi_summary.csv"
    )

    csi = pd.read_csv(
        current_dir.parent / "Reports" / "csi_summary.csv"
    )

    calibration = pd.read_csv(
        current_dir.parent / "Reports" / "calibration_summary.csv"
    )

    # Convert Metric-Value table into dictionary
    metric_dict = dict(zip(metrics["Metric"], metrics["Value"]))

    # =====================================================
    # PERFORMANCE METRICS
    # =====================================================

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            "Test AUC",
            f"{metric_dict['Test_AUC']}"
        )

    with c2:
        st.metric(
            "KS Statistic",
            f"{metric_dict['KS']}"
        )

    with c3:
        st.metric(
            "Test Gini",
            f"{metric_dict['Test_Gini']}"
        )

    with c4:
        st.metric(
            "Portfolio Bad Rate",
            f"{metric_dict['Portfolio_Bad_Rate']}"
        )

    st.divider()

    # =====================================================
    # SCORE RANGE & LIFT
    # =====================================================

    col1, col2 = st.columns(2)

    with col1:

        st.info(f"""
### Score Range

- Minimum Score : **{metric_dict['Min_Score']}**
- Maximum Score : **{metric_dict['Max_Score']}**
""")

    with col2:

        st.info(f"""
### Model Lift

- Lift : **{metric_dict['Lift']}x**

Higher lift indicates better separation of good and bad customers.
""")

    st.divider()

    # =====================================================
    # MONITORING INDICATORS
    # =====================================================

    st.subheader("📊 Monitoring Indicators")

    max_error = (
        calibration["avg_predicted_pd"] -
        calibration["observed_bad_rate"]
    ).abs().max()

    if max_error < 0.02:
        calibration_status = "Good"

    elif max_error < 0.05:
        calibration_status = "Acceptable"

    else:
        calibration_status = "Needs Review"

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric(
            "PSI",
            round(psi.loc[0, "PSI"], 4)
        )

    with c2:
        st.metric(
            "CSI",
            round(csi.loc[0, "CSI"], 4)
        )

    with c3:
        st.metric(
            "Calibration",
            calibration_status
        )

    st.divider()

    # =====================================================
    # CALIBRATION PLOT
    # =====================================================

    st.subheader("Calibration Assessment")

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=calibration["avg_predicted_pd"],
            y=calibration["observed_bad_rate"],
            mode="lines+markers",
            name="Observed"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=calibration["avg_predicted_pd"],
            y=calibration["avg_predicted_pd"],
            mode="lines",
            name="Perfect Calibration",
            line=dict(dash="dash")
        )
    )

    fig.update_layout(
        title="Predicted PD vs Observed Bad Rate",
        xaxis_title="Average Predicted PD",
        yaxis_title="Observed Bad Rate",
        height=450
    )

    st.plotly_chart(
        fig,
        width="stretch"
    )

    # =====================================================
    # CALIBRATION TABLE
    # =====================================================

    st.subheader("Calibration Summary")

    calibration_display = calibration[
        [
            "pd_bucket",
            "avg_predicted_pd",
            "observed_bad_rate"
        ]
    ].copy()

    calibration_display = calibration_display.rename(
        columns={
            "pd_bucket": "PD Bucket",
            "avg_predicted_pd": "Average Predicted PD",
            "observed_bad_rate": "Observed Bad Rate"
        }
    )

    st.dataframe(
        calibration_display.round(4),
        width="stretch",
        hide_index=True
    )

    st.divider()

    # =====================================================
    # MODEL HEALTH
    # =====================================================

    st.success(f"""
### ✅ Model Health Summary

- Test AUC : **{metric_dict['Test_AUC']}**
- KS Statistic : **{metric_dict['KS']}**
- Test Gini : **{metric_dict['Test_Gini']}**
- PSI indicates stable population.
- CSI indicates stable characteristics.
- Calibration Status : **{calibration_status}**

The scorecard demonstrates stable discrimination, calibration and monitoring performance.
""")

if page=="Model Information":

    st.header("ℹ Model Information")

    info=pd.DataFrame({

        "Property":[

            "Model Type",

            "Algorithm",

            "Target",

            "Variables",

            "WOE Transformation",

            "Scorecard",

            "Version"

        ],

        "Value":[

            "Application PD Model",

            "Logistic Regression",

            "Probability of Default",

            len(feature_list),

            "Yes",

            "Yes",

            "1.0"

        ]

    })

    info["Value"] = info["Value"].astype(str)

    st.table(info)    
