"""
train.py — House Price Prediction Training Script
==================================================
• Works with OR without XGBoost installed.
  - If xgboost is available  → uses XGBRegressor (best performance)
  - Otherwise                → falls back to HistGradientBoostingRegressor
• Feature set is EXACTLY what app.py expects (no engineered features fed
  into the model; app.py passes raw features directly).
• Saves model to:  models/streamlit_model.pkl
"""

import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor

# Try to import XGBoost; fall back gracefully
try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
    print("✓ XGBoost is available — will use XGBRegressor")
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠  XGBoost not installed — falling back to HistGradientBoostingRegressor")
    print("   Install with: pip install xgboost")

# ── Directory Setup ────────────────────────────────────────────────────────────
os.makedirs("models", exist_ok=True)
os.makedirs("reports/figures", exist_ok=True)

# ── 1. Feature List (MUST match app.py FEATURES exactly) ──────────────────────
# These are the raw columns fed to the model — no engineered features.
FEATURES = [
    "OverallQual",
    "GrLivArea",
    "GarageCars",
    "GarageArea",
    "TotalBsmtSF",
    "1stFlrSF",
    "FullBath",
    "TotRmsAbvGrd",
    "YearBuilt",
    "YearRemodAdd",
    "MasVnrArea",
    "Fireplaces",
    "BsmtFinSF1",
    "LotArea",
    "OpenPorchSF",
    "WoodDeckSF",
    "2ndFlrSF",
    "HalfBath",
    "BedroomAbvGr",
    "KitchenAbvGr",
]

# ── 2. Load Data ───────────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("STEP 1: Loading Data")
print("=" * 55)

# Support both flat and data/ subdirectory layouts
def _find_csv(name):
    for p in [f"data/{name}", name]:
        if os.path.exists(p):
            return pd.read_csv(p)
    raise FileNotFoundError(
        f"Cannot find '{name}'. Place it in the same folder as train.py "
        f"or in a 'data/' subdirectory."
    )

train = _find_csv("train.csv")
test  = _find_csv("test.csv")

print(f"Train shape : {train.shape}")
print(f"Test  shape : {test.shape}")

# ── 3. Basic EDA ───────────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("STEP 2: Basic EDA")
print("=" * 55)
print(train[FEATURES + ["SalePrice"]].describe().T[["mean","min","50%","max"]])

# ── 4. SalePrice distribution plots ───────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
sns.histplot(train["SalePrice"], kde=True, color="steelblue", ax=axes[0])
axes[0].set_title("SalePrice (raw)")
axes[0].set_xlabel("SalePrice ($)")

sns.histplot(np.log1p(train["SalePrice"]), kde=True, color="darkorange", ax=axes[1])
axes[1].set_title("SalePrice (log-transformed)")
axes[1].set_xlabel("log(SalePrice + 1)")

plt.tight_layout()
plt.savefig("reports/figures/saleprice_distribution.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: reports/figures/saleprice_distribution.png")

# ── 5. Missing Value Summary ───────────────────────────────────────────────────
print("\n" + "=" * 55)
print("STEP 3: Missing Values in Selected Features")
print("=" * 55)
missing = train[FEATURES].isnull().sum()
missing = missing[missing > 0].sort_values(ascending=False)
print(missing if len(missing) else "No missing values in selected features ✓")

# ── 6. Prepare X / y ──────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("STEP 4: Preparing Features & Target")
print("=" * 55)

# Ensure all FEATURES exist (fill missing test cols with 0)
for col in FEATURES:
    if col not in train.columns:
        raise KeyError(f"Feature '{col}' not found in train.csv")
    if col not in test.columns:
        print(f"  ⚠  '{col}' missing from test.csv — filling with 0")
        test[col] = 0

y      = np.log1p(train["SalePrice"])
X      = train[FEATURES].copy()
X_test = test[FEATURES].copy()

print(f"X shape : {X.shape}  |  features: {len(FEATURES)}")

# ── 7. Train / Validation Split ────────────────────────────────────────────────
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.20, random_state=42
)
print(f"X_train : {X_train.shape}  |  X_valid : {X_valid.shape}")

# ── 8. Preprocessing Pipeline (median imputation for numeric cols) ─────────────
preprocessor = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

# ── 9. Random Forest ───────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("STEP 5: Training Random Forest")
print("=" * 55)

rf_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(
        n_estimators=500,
        max_depth=20,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1,
    )),
])

rf_pipeline.fit(X_train, y_train)
rf_score = r2_score(y_valid, rf_pipeline.predict(X_valid))
print(f"Random Forest  Validation R² : {rf_score:.4f}")

# ── 10. Gradient Boosting (XGBoost or HGB fallback) ───────────────────────────
print("\n" + "=" * 55)
print("STEP 6: Training Gradient Boosting")
print("=" * 55)

if XGBOOST_AVAILABLE:
    gb_model = XGBRegressor(
        n_estimators=1000,
        learning_rate=0.03,
        max_depth=3,
        min_child_weight=5,
        subsample=0.75,
        colsample_bytree=0.75,
        gamma=0.1,
        reg_alpha=0.5,
        reg_lambda=2.0,
        random_state=42,
        objective="reg:squarederror",
        n_jobs=-1,
    )
    algo_name = "XGBoost"
else:
    gb_model = HistGradientBoostingRegressor(
        max_iter=1000,
        learning_rate=0.03,
        max_depth=3,
        min_samples_leaf=20,
        l2_regularization=2.0,
        random_state=42,
    )
    algo_name = "HistGradientBoosting (XGBoost fallback)"

gb_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", gb_model),
])

gb_pipeline.fit(X_train, y_train)
gb_score = r2_score(y_valid, gb_pipeline.predict(X_valid))
print(f"{algo_name}  Validation R² : {gb_score:.4f}")

# ── 11. Model Comparison ───────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("STEP 7: Model Comparison")
print("=" * 55)

comparison_df = pd.DataFrame({
    "Model":    ["Random Forest", algo_name],
    "Val R²":   [rf_score, gb_score],
})
print(comparison_df.to_string(index=False))

fig, ax = plt.subplots(figsize=(7, 4))
colors = ["#4C72B0", "#DD8452"]
ax.barh(comparison_df["Model"], comparison_df["Val R²"], color=colors)
ax.set_xlim(0.80, 1.0)
ax.set_xlabel("Validation R²")
ax.set_title("Model Comparison")
for i, v in enumerate(comparison_df["Val R²"]):
    ax.text(v + 0.001, i, f"{v:.4f}", va="center", fontsize=10)
plt.tight_layout()
plt.savefig("reports/figures/model_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: reports/figures/model_comparison.png")

# ── 12. Select Best Model ──────────────────────────────────────────────────────
if gb_score >= rf_score:
    best_pipeline  = gb_pipeline
    best_name      = algo_name
    best_val_score = gb_score
else:
    best_pipeline  = rf_pipeline
    best_name      = "Random Forest"
    best_val_score = rf_score

print(f"\n✓ Best Model : {best_name}  (Val R² = {best_val_score:.4f})")

# ── 13. Retrain on Full Training Set ──────────────────────────────────────────
print("\n" + "=" * 55)
print("STEP 8: Retraining Best Model on Full Dataset")
print("=" * 55)

best_pipeline.fit(X, y)
full_score = r2_score(y, best_pipeline.predict(X))
print(f"Full-data R² (train) : {full_score:.4f}")

# ── 14. Feature Importance ────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("STEP 9: Feature Importance")
print("=" * 55)

inner = best_pipeline.named_steps["model"]

if hasattr(inner, "feature_importances_"):
    importance = inner.feature_importances_
else:
    importance = np.ones(len(FEATURES)) / len(FEATURES)  # uniform fallback

feat_df = pd.DataFrame({"Feature": FEATURES, "Importance": importance})
feat_df = feat_df.sort_values("Importance", ascending=False)
print(feat_df.to_string(index=False))

fig, ax = plt.subplots(figsize=(8, 6))
colors = [f"rgba(102,126,234,{0.4 + 0.06*i})" for i in range(len(FEATURES))]
ax.barh(feat_df["Feature"][::-1], feat_df["Importance"][::-1], color="#667eea")
ax.set_xlabel("Importance")
ax.set_title(f"Feature Importance — {best_name}")
plt.tight_layout()
plt.savefig("reports/figures/feature_importance.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: reports/figures/feature_importance.png")

# ── 15. Test Predictions / Submission ─────────────────────────────────────────
print("\n" + "=" * 55)
print("STEP 10: Generating Test Predictions")
print("=" * 55)

test_log_preds = best_pipeline.predict(X_test)
test_preds     = np.expm1(test_log_preds)

submission = pd.DataFrame({"Id": test["Id"], "SalePrice": test_preds})
submission.to_csv("models/submission.csv", index=False)
print("Saved: models/submission.csv")
print(submission.head())

# ── 16. Save Model ─────────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("STEP 11: Saving Model")
print("=" * 55)

joblib.dump(best_pipeline, "models/streamlit_model.pkl")
print("Saved: models/streamlit_model.pkl")

# ── 17. Summary ───────────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("TRAINING COMPLETE")
print("=" * 55)
print(f"Algorithm      : {best_name}")
print(f"Features       : {len(FEATURES)}")
print(f"Validation R²  : {best_val_score:.4f}")
print(f"Full-data R²   : {full_score:.4f}")
print(f"Model saved to : models/streamlit_model.pkl")
print("=" * 55)