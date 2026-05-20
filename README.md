# AI-Augmented Credit Risk Scorecard & PD Modeling Platform
End-to-end retail credit risk analytics platform combining traditional scorecard modeling, machine learning, explainable AI, and interactive Streamlit deployment.

## Objective

To build a scalable and interpretable AI-augmented credit risk decisioning framework for retail lending by leveraging scorecard modeling, Probability of Default (PD) estimation, machine learning, and explainable AI techniques. The platform is designed to emulate real-world banking risk analytics workflows, enabling automated borrower risk segmentation, credit scoring, and intelligent underwriting support through interactive analytics and deployment-ready dashboards.

## Business Problem

Financial institutions must accurately assess borrower default risk to minimize credit losses and optimize lending decisions. Traditional underwriting approaches often struggle with scalability, interpretability, and dynamic risk assessment.

This project aims to simulate a real-world retail lending risk workflow by building a Probability of Default (PD) modeling and scorecard framework capable of:
- Predicting borrower default probability
- Segmenting customers by risk profile
- Supporting underwriting decisions
- Improving portfolio monitoring
- Providing explainable AI-driven recommendations

## Dataset

Dataset: Home Credit Default Risk

Source:
[https://www.kaggle.com/competitions/home-credit-default-risk/data](https://www.kaggle.com/competitions/home-credit-default-risk/data)

Key datasets used:
- application_train.csv
- bureau.csv
- previous_application.csv
  
## Project Workflow

1. Business Understanding
2. Data Cleaning & Preprocessing
3. Exploratory Data Analysis (EDA)
4. Feature Engineering
5. WOE & IV Analysis
6. Scorecard Development
7. Probability of Default (PD) Modeling
8. Machine Learning Model Benchmarking
9. Explainable AI using SHAP
10. AI-Based Underwriting Recommendation Engine
11. Streamlit Dashboard Deployment


## Key Features

- End-to-end retail credit risk pipeline
- Traditional scorecard development using WOE & IV
- Probability of Default (PD) estimation
- Logistic Regression scorecard model
- Machine Learning risk modeling
- Explainable AI using SHAP
- AI-driven underwriting recommendations
- Interactive Streamlit dashboard
- Risk segmentation & score banding
- Portfolio-level analytics

## Tech Stack

| Category | Tools & Libraries |
|---|---|
| Programming | Python |
| Data Processing | Pandas, NumPy |
| Visualization | Plotly, Matplotlib |
| Modeling | Scikit-learn |
| Scorecard Development | scorecardpy |
| Machine Learning | XGBoost, LightGBM |
| Explainability | SHAP |
| Dashboard | Streamlit |
| Version Control | Git & GitHub |

## Modeling Techniques

### Traditional Risk Modeling
- Weight of Evidence (WOE)
- Information Value (IV)
- Logistic Regression Scorecard
- Score Scaling & Risk Banding

### Machine Learning Models
- Random Forest
- XGBoost
- LightGBM

### Evaluation Metrics
- ROC-AUC
- KS Statistic
- Gini Coefficient
- Precision & Recall

## Explainable AI

The project integrates SHAP-based explainability techniques to interpret model predictions and identify key risk drivers influencing borrower default probability.

## Streamlit Dashboard Features

- Applicant Risk Scoring
- Probability of Default Prediction
- Credit Score Generation
- SHAP-Based Risk Explanation
- Risk Segment Visualization
- AI Recommendation Engine
- Portfolio Risk Monitoring

## Future Enhancements

- IFRS9 Expected Credit Loss (ECL) Modeling
- Real-time API Deployment
- Drift Monitoring
- MLOps Integration
- Alternative Data Integration
- Generative AI Risk Assistant

## Results

- Achieved robust PD prediction performance
- Built interpretable scorecard-based risk framework
- Successfully deployed interactive Streamlit dashboard
- Enabled AI-driven underwriting recommendations

## Author

<h3><i>Shrabani Das</i></h3>

<p style="color:lightgrey;">
<i>
Data Scientist • Credit Risk Analytics • Machine Learning • AI in Finance
</i>
</p>

<!-- <p style="color:lightgrey;">
<i>
Building intelligent, interpretable, and industry-oriented risk analytics solutions at the intersection of banking, quantitative modeling, and AI-driven decision systems.
</i>
</p> -->

---

<p style="color:lightgrey;">
<i>
© 2026 Shrabani Das. All Rights Reserved.
<br><br>
This project is developed for educational, research, and portfolio purposes only. Unauthorized reproduction, redistribution, or commercial usage of the project code, architecture, or documentation without prior permission is prohibited.
</i>
</p>
