# Task 1: Data Analysis & Preprocessing

## Overview

In Task 1, I performed an in-depth exploratory data analysis (EDA) and comprehensive preprocessing on two datasets: the e-commerce transaction dataset (`Fraud_Data.csv`) and the bank credit card transactions dataset (`creditcard.csv`). The goal was to prepare clean, feature-rich, and balanced datasets suitable for fraud detection modeling.

---

## Key Steps and Decisions

### 1. Data Loading and Inspection

- Loaded the raw CSV files into Pandas DataFrames.
- Verified dataset structure, column data types, and basic statistics.
- Checked for missing values; **no missing data was found** in either dataset, simplifying preprocessing.

### 2. Data Cleaning

- Removed duplicate rows to avoid biased learning.
- Converted timestamp columns (`signup_time`, `purchase_time`) into `datetime` objects to enable time-based feature engineering.

### 3. Exploratory Data Analysis (EDA)

- Analyzed target variable distribution and confirmed significant **class imbalance** with very few fraud cases.
- Visualized feature distributions (e.g., purchase values, transaction times) to identify patterns and potential fraud indicators.

### 4. Geolocation Mapping

- Converted IP addresses to integer format.
- Mapped transaction IP addresses to countries using provided IP-to-country range data.
- Added a `country` feature to enrich the dataset for geographic fraud pattern recognition.

### 5. Feature Engineering

- Created **transaction frequency** features (`user_transaction_count`) indicating user activity level.
- Extracted **time-based features**:
  - `hour_of_day`: Hour when the transaction occurred.
  - `day_of_week`: Weekday number of the transaction.
  - `time_since_signup`: Time elapsed between user signup and transaction.
- Calculated **transaction velocity** (optional): Time difference between consecutive user transactions.

### 6. Encoding Categorical Variables

- One-hot encoded categorical features (`browser`, `source`, `sex`) to convert them into numerical format for machine learning models.

### 7. Handling Class Imbalance

- Recognized the significant class imbalance challenge typical of fraud detection.
- Planned to apply **SMOTE (Synthetic Minority Over-sampling Technique)** on training data to balance classes during model training.

### 8. Data Scaling

- Used **StandardScaler** to normalize numerical features, preparing the dataset for models sensitive to feature scales, such as Logistic Regression.

---

## Summary

Task 1 established a clean, enriched, and balanced foundation for modeling by:

- Ensuring data integrity and quality
- Adding meaningful behavioral and temporal features
- Encoding categorical data properly
- Addressing class imbalance thoughtfully

This structured preprocessing pipeline improves the model's ability to detect fraud accurately while reducing false positives.


# 🧠 Task 2: Model Building and Training

🎯 Objective: Train and evaluate models on two fraud-related datasets — Fraud_Data and creditcard.csv — to identify the most effective algorithm for fraud detection.

🧩 Workflow Overview

    Data Loading
    Load preprocessed train/test datasets for both:

        Fraud_Data → X_train_final, X_test_final, y_train_final, y_test_final

        creditcard.csv → X_train_final_credit, X_test_final_credit, y_train_final_credit, y_test_final_credit

    Model Selection

        Logistic Regression: Simple, interpretable baseline.

        Random Forest: Powerful ensemble model for capturing complex patterns.

    Model Training & Evaluation

        Metrics used:

            F1 Score: Balance between precision and recall.

            AUC-PR (Area Under Precision-Recall Curve): Robust to class imbalance.

            Confusion Matrix: Breakdown of true/false predictions.

        Models were trained and evaluated separately on both datasets.

📊 Results Summary
Fraud_Data
Model	AUC-PR	F1 Score	TP	FP	FN	TN
Logistic Regression	0.2554	0.2753	1772	8271	1058	19122
Random Forest	0.6161	0.5785	1523	912	1307	26481

✅ Best Model: Random Forest
→ Superior AUC-PR, F1 Score, and major reduction in false positives.
Credit_Card
Model	AUC-PR	F1 Score	TP	FP	FN	TN
Logistic Regression	0.7150	0.1002	83	1479	12	55172
Random Forest	—	0.8276	72	7	23	56644

✅ Best Model: Random Forest
→ Near-perfect classification with minimal error.
🏁 Conclusion

Across both datasets, Random Forest outperformed Logistic Regression in all major metrics, especially under class imbalance. It is recommended as the primary model for deployment due to its strong performance and generalization capabilities.

## 🧠 Task 3 – Model Explainability with SHAP

### 📌 Objective
The goal of this task is to interpret and visualize the behavior of the best-performing model using SHAP (SHapley Additive exPlanations). SHAP helps uncover which features drive the model’s predictions and how they contribute to fraud detection decisions — both globally and for individual transactions.

---

### 🔍 Key Steps Performed

1. **Model Selection for Explainability**
   - The best-performing model from Task 2 (e.g., `RandomForestClassifier`) was selected based on AUC-PR and F1-score.
   - The model was trained on the processed training data and evaluated on the test set.

2. **SHAP Explainer Initialization**
   - `TreeExplainer` was used because the selected model is tree-based (Random Forest or XGBoost).
   - A subset of the test data (e.g., 1000 samples) was used to reduce computation time.

3. **SHAP Value Computation**
   - SHAP values were computed for the sampled test data.
   - Both global (feature importance) and local (single prediction) explanations were generated.

4. **Visualizations**
   - `summary_plot`: Shows top features influencing fraud predictions globally.
   - `force_plot`: Displays how individual feature values push a prediction toward fraud or not.

---

### 📊 Key Insights

- **Top Predictors Identified:**
  - For the credit card dataset: Features such as `V17`, `V14`, `V12`, and `Amount` had strong impact on the fraud prediction.
  - For the e-commerce dataset: Engineered features like `time_since_signup`, `hour_of_day`, and certain browser or country indicators showed high importance.

- **Interpretable Decisions:**
  - The force plot for a specific transaction showed how abnormal values (e.g., very high `purchase_value` or late-night hours) significantly increased fraud likelihood.

---

### ⚠️ Notes

- SHAP was run on **unscaled, original feature names** to preserve interpretability.
- Due to performance constraints, SHAP was applied on a **subset of the test set**.
- Force plots require a browser or Jupyter-compatible viewer (`shap.initjs()` included).

---
