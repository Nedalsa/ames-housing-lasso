# 🏠 A Proposed Statistical Model to Estimate The Real Value of Real Estates

**A Practical Study in The City of Damascus**

This repository contains the code developed as part of a Master's thesis submitted to the **Department of Statistics and Information Systems, Faculty of Economics, Damascus University**, for the degree of Master's in Quantitative Methods.


The thesis proposes a LASSO regression model for estimating real estate prices, validated first on the internationally recognized Ames Housing Dataset (Kaggle), then applied to field data collected from the Damascus real estate market in 2024.

---

## 📌 Project Overview

The study is structured in two stages:

**Stage 1 — Ames, Iowa (USA):** The Ames Housing Dataset was used as a controlled, well-documented benchmark to test the LASSO model's ability to perform variable selection and accurate price prediction across a large set of real estate features.

**Stage 2 — Damascus, Syria:** The validated methodology was then applied to 250 real estate observations collected directly from the Damascus market (November–December 2024), covering both quantitative and qualitative property characteristics.

---

## 🗂️ Repository Structure

```
ames-housing-lasso/
│
├── lasso_model.ipynb     # Main notebook: preprocessing → modeling → evaluation
├── submission.csv        # Kaggle submission file (generated after running the notebook)
└── README.md             # This file
```

> **Note:** `train.csv` and `test.csv` are not included.  
> Download them from [Kaggle](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data) and place them in the same folder as the notebook before running.

---

## ⚙️ Modeling Workflow

| Step | Description |
|------|-------------|
| 1. Load Data | Combine train and Kaggle test sets for consistent preprocessing |
| 2. Missing Values | Domain-aware imputation (neighborhood median for LotFrontage, absence encoding for Garage/Basement/Pool) |
| 3. Ordinal Encoding | Quality/condition features mapped to ordered integers (e.g. Po=0 → Ex=4) |
| 4. Outlier Removal | Extreme values removed from training set only, based on visual inspection |
| 5. Target Transform | `log1p(SalePrice)` to normalize right-skewed price distribution |
| 6. Feature Encoding | One-hot encoding for remaining categoricals (`drop_first=True`) |
| 7. Modeling | `StandardScaler` + `Lasso` in a `Pipeline` to prevent data leakage |
| 8. Tuning | `GridSearchCV` with 5-fold cross-validation over 50 log-spaced alpha values |
| 9. Evaluation | RMSE and R² reported on held-out validation set (original dollar scale) |
| 10. Submission | Kaggle predictions exported to `submission.csv` |

---

## 📊 Results — Ames Dataset

Results are reported on log-transformed SalePrice (as used in model training), from **Table 4** of the thesis:

| Metric | Training Set | Test Set |
|--------|-------------|----------|
| RMSE (log scale) | 0.1035 | 0.1153 |
| Features retained by LASSO | 114 out of 240 | — |

LASSO reduced the feature space from **240 variables** down to **114**, while maintaining strong predictive accuracy. This demonstrates LASSO's core advantage: automatic variable selection without sacrificing model quality.

---

## 🔑 Top Features Selected by LASSO (Ames)

The 10 most influential features identified by the model (standardized coefficients):

| Feature | Coefficient | Description |
|---------|------------|-------------|
| GrLivArea | 0.1280 | Above-ground living area (sq ft) |
| OverallQual | 0.0786 | Overall material and finish quality |
| YearBuilt | 0.0633 | Original construction year |
| OverallCond | 0.0450 | Overall condition rating |
| BsmtFinSF1 | 0.0288 | Type 1 finished basement area |
| TotalBsmtSF | 0.0261 | Total basement area |
| GarageCars | 0.0234 | Garage capacity (number of cars) |
| LotArea | 0.0212 | Lot size (sq ft) |
| Neighborhood_Crawfor | 0.0160 | Located in Crawford neighborhood |
| ExterQual | 0.0155 | Exterior material quality |

---

## 🧰 Requirements

```bash
pip install pandas numpy matplotlib scikit-learn
```

Python version: **3.8+**

---

## 🚀 How to Run

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/ames-housing-lasso.git
cd ames-housing-lasso

# 2. Place train.csv and test.csv in this folder (download from Kaggle link above)

# 3. Install dependencies
pip install pandas numpy matplotlib scikit-learn

# 4. Launch the notebook
jupyter notebook lasso_model.ipynb
```

---

## 📚 Dataset

- **Source:** [Kaggle – House Prices: Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)
- **Size:** 1,460 training samples, 1,459 test samples
- **Features:** 79 explanatory variables (area, quality, location, age, condition, etc.)

---

## 🔍 Model Highlights

- **Pipeline architecture** ensures `StandardScaler` is fit only on training data during cross-validation — no data leakage
- **LASSO** performs automatic variable selection by shrinking irrelevant coefficients to exactly zero
- **Log-transform** on the target stabilizes variance and improves linear model fit
- **Neighborhood-aware imputation** for `LotFrontage` instead of global median
- **Coordinate descent algorithm** used for efficient LASSO parameter estimation
- The Ames model served as a validated baseline before applying the methodology to the Damascus market

---

## 👤 Author

**Nidal Al-Saqqa**
