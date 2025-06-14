# BURNOUT MotoGP Datathon – Regression Solution

This repository contains the solution developed for the **BURNOUT MotoGP Datathon**, which focuses on predicting the target variable using regression techniques. The objective was to minimize the RMSE (Root Mean Squared Error) using the provided train, validation, and test datasets.

---

## 📁 Project Structure

```
BURNOUT-DATATHON/
├── dataset/
│   ├── train.csv
│   ├── val.csv
│   ├── test.csv
│   └── sample_submission.csv
├── catboost_info/
│   └── [CatBoost model logs]
├── notebooks/
│   ├── eda.ipynb
│   ├── model_xgboost.ipynb
│   ├── model_catboost.ipynb
│   └── stacking_regressor.ipynb
├── models/
│   └── [Pickled models if any]
├── solution.csv
└── README.md
```

---

## Solution Overview

The solution employs a combination of ensemble regression models, primarily **XGBoost** and **CatBoost**, followed by **model stacking** for performance enhancement. Here's the summary of steps:

1. **Exploratory Data Analysis (EDA)**:
   Identified missing values, feature distributions, and correlations. Applied transformations and encoding where necessary.

2. **Preprocessing**:

   * Handled categorical columns using `LabelEncoder`.
   * Managed missing values and inconsistencies.
   * Standardized and aligned feature formats across datasets.

3. **Modeling**:

   * **XGBoost**: Fine-tuned with `n_estimators`, `learning_rate`, and `max_depth`.
   * **CatBoost**: Used its in-built categorical feature handling.
   * **Stacking Regressor**: Combined predictions from XGBoost and CatBoost using a Ridge Regressor as the meta-model.

4. **Evaluation**:

   * Used **RMSE** on the validation set as the key metric.
   * CatBoost and XGBoost both performed well individually, with stacking offering marginal improvement.

5. **Submission**:

   * Final predictions generated on `test.csv`.
   * Output saved to `solution.csv` in the required format.

---

## Performance

| Model    | RMSE (Validation) |
| -------- | ----------------- |
| XGBoost  | *10.635147372997457*      |
| CatBoost | *11.268709307109962*      |
| Linear Regression | *11.538985186856433*      |
| Random Forest Regression | *11.248978232453*      |

---

## Requirements

* Python 3.8+
* pandas, numpy
* scikit-learn
* xgboost
* catboost
* matplotlib / seaborn

Install all requirements with:

```bash
pip install -r requirements.txt
```

---

## Notes

* CatBoost logs are saved under `catboost_info/` for inspection.
* All model outputs and experiments are version-controlled for reproducibility.
* The solution avoids LightGBM as per project constraints.

---
