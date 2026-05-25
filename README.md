<div align="center">

# 🤖 AI-Augmented Credit Risk Scorecard & PD Modeling Platform

### <i>End-to-end retail credit risk analytics platform combining traditional scorecard modeling, machine learning, explainable AI, and interactive Streamlit deployment.</i>

</div>

---

## 🎯 Objective

To build a scalable and interpretable AI-augmented credit risk decisioning framework for retail lending by leveraging scorecard modeling, Probability of Default (PD) estimation, machine learning, and explainable AI techniques.

---

## 🏦 Business Problem

Financial institutions must accurately assess borrower default risk to minimize credit losses and optimize lending decisions.

This project simulates a real-world retail lending risk workflow capable of:

- Predicting borrower default probability
- Segmenting customers by risk profile
- Supporting underwriting decisions
- Improving portfolio monitoring
- Providing explainable AI-driven recommendations

---

## 📊 Dataset

**Dataset:** Home Credit Default Risk

🔗 Source:  
https://www.kaggle.com/competitions/home-credit-default-risk/data

### Key datasets used:
- application_train.csv
- bureau.csv
- previous_application.csv

---

## ⚙️ Project Workflow

```text
Obejective & Business Understanding
        ↓
Data Cleaning & Preprocessing
        ↓
Exploratory Data Analysis
        ↓
Roll Rate and Vinatge Analysis for Performance window and target definition slection
        ↓
Feature Engineering
        ↓
Initial Characteristic Analysis 
        ↓
Scorecard Development
        ↓
PD Modeling
        ↓
Machine Learning Benchmarking
        ↓
Explainable AI (SHAP)
        ↓
AI Recommendation Engine
        ↓
Streamlit Dashboard Deployment
```

---

## 🚀 Key Features

✔ End-to-end retail credit risk pipeline  
✔ Traditional scorecard development using WOE & IV  
✔ Probability of Default (PD) estimation  
✔ Logistic Regression scorecard model  
✔ Machine Learning risk modeling  
✔ Explainable AI using SHAP  
✔ AI-driven underwriting recommendations  
✔ Interactive Streamlit dashboard  
✔ Risk segmentation & score banding  
✔ Portfolio-level analytics  

---

## 🛠️ Tech Stack

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

---

## 📈 Modeling Techniques

### 📌 Traditional Risk Modeling
- Weight of Evidence (WOE)
- Information Value (IV)
- Logistic Regression Scorecard
- Score Scaling & Risk Banding

### 🤖 Machine Learning Models
- Random Forest
- XGBoost

### 📉 Evaluation Metrics
- ROC-AUC
- KS Statistic
- Gini Coefficient
- Precision & Recall

---

## 🧠 Explainable AI

The project integrates SHAP-based explainability techniques to interpret model predictions and identify key risk drivers influencing borrower default probability.

---

## 📊 Streamlit Dashboard Features

- Applicant Risk Scoring
- Probability of Default Prediction
- Credit Score Generation
- SHAP-Based Risk Explanation
- Risk Segment Visualization
- AI Recommendation Engine
- Portfolio Risk Monitoring

---

## 🔮 Future Enhancements

- IFRS9 Expected Credit Loss (ECL) Modeling
- Real-time API Deployment
- Drift Monitoring
- MLOps Integration
- Alternative Data Integration
- Generative AI Risk Assistant

---

## 📌 Results

- Achieved robust PD prediction performance
- Built interpretable scorecard-based risk framework
- Successfully deployed interactive Streamlit dashboard
- Enabled AI-driven underwriting recommendations

---

## 💡 Insights

### 1. Roll Rate + Vinatge Analysis Insights

<img width="569" height="119" alt="image" src="https://github.com/user-attachments/assets/60244a7b-5cb5-4847-b75d-aa3a6ef66438" />

<img width="570" height="200" alt="image" src="https://github.com/user-attachments/assets/b9971cae-990f-4c06-949d-c8523ce30f03" />

- The vintage curve exhibits fluctuations across Months on Book (MOB), likely due to the presence of multiple loan product types with heterogeneous repayment structures and risk behaviors.

- Additionally, the Home Credit dataset represents a competition-oriented dataset rather than a fully production-grade banking portfolio, resulting in approximate account aging and noisy delinquency emergence patterns.

- Despite the observed fluctuations, most meaningful delinquency emergence appears within the earlier repayment periods, supporting the use of a medium-term performance window such as 12 months for application scorecard development.


## 👨‍💻 Author

### *Shrabani Das*

> Data Scientist • Credit Risk Analytics • Machine Learning • AI in Finance

---

<div align="center">

<i>

© 2026 Shrabani Das. All Rights Reserved.

This project is developed for educational, research, and portfolio purposes only.

</i>

</div>
