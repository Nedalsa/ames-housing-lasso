# 🏠 Ames Housing Price Prediction – LASSO Regression

Master's thesis project using the [Ames Housing Dataset](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques) to predict residential property sale prices with regularized regression.

---

## 📌 Project Overview

This project applies a full supervised machine learning pipeline to predict house prices in Ames, Iowa. The focus is on interpretable modeling using **LASSO regression**, which simultaneously performs feature selection and regularization.

---

## 🗂️ Repository Structure

```
ames-housing-lasso/
│
├── lasso_model.ipynb     # Main notebook: preprocessing → modeling → evaluation
├── submission.csv        # Kaggle submission file (generated after running the notebook)
└── README.md             # This file
```

> **Note:** `train.csv` and `test.csv` are not included in this repository.  
> Download them from [Kaggle](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data) and place them in the same folder as the notebook.

---

## ⚙️ Workflow

| Step | Description |
|------|-------------|
| 1. Load Data | Combine train and test sets for consistent preprocessing |
| 2. Missing Values | Domain-aware imputation (neighborhood median, absence encoding) |
| 3. Ordinal Encoding | Quality/condition features mapped to ordered integers |
| 4. Outlier Removal | Extreme values removed from training set only |
| 5. Target Transform | `log1p(SalePrice)` to normalize right-skewed distribution |
| 6. Feature Encoding | One-hot encoding for remaining categoricals (`drop_first=True`) |
| 7. Modeling | `StandardScaler` + `Lasso` in a `Pipeline` to prevent data leakage |
| 8. Tuning | `GridSearchCV` with 5-fold cross-validation over 50 alpha values |
| 9. Evaluation | RMSE and R² on held-out validation set (original dollar scale) |
| 10. Submission | Predictions exported to `submission.csv` for Kaggle |

---

## 📊 Key Results

| Metric | Value |
|--------|-------|
| Best alpha (λ) | Tuned via cross-validation |
| Validation RMSE | ~$X,XXX *(run notebook to see)* |
| Validation R² | ~0.XX *(run notebook to see)* |
| Features selected | X out of ~200+ *(LASSO auto-selection)* |

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

# 2. Place train.csv and test.csv in this folder (from Kaggle)

# 3. Install dependencies
pip install pandas numpy matplotlib scikit-learn

# 4. Launch the notebook
jupyter notebook lasso_model.ipynb
```

---

## 📚 Dataset

- **Source:** [Kaggle – House Prices: Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)
- **Size:** 1,460 training samples, 1,459 test samples
- **Features:** 79 explanatory variables (area, quality, location, age, etc.)

---

## 🔍 Model Highlights

- **Pipeline architecture** ensures the scaler is fit only on training data during cross-validation — no data leakage
- **LASSO** performs automatic feature selection by shrinking irrelevant coefficients to exactly zero
- **Log-transform** on the target variable stabilizes variance and improves linear model fit
- **Neighborhood-aware imputation** for `LotFrontage` instead of global median

---

## 👤 Author

Master's thesis project — feel free to open an issue or reach out with questions.
