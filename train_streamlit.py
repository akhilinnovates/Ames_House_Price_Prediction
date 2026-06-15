import os
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor

# ── Setup ──────────────────────────────────────────────────────────────────────
os.makedirs("models", exist_ok=True)

# ── Load Data ──────────────────────────────────────────────────────────────────
train = pd.read_csv("data/train.csv")

# ── Features (must match app.py exactly) ──────────────────────────────────────
FEATURES = [
    "OverallQual", "GrLivArea", "GarageCars", "GarageArea",
    "TotalBsmtSF", "1stFlrSF", "FullBath", "TotRmsAbvGrd",
    "YearBuilt", "YearRemodAdd", "MasVnrArea", "Fireplaces",
    "BsmtFinSF1", "LotArea", "OpenPorchSF", "WoodDeckSF",
    "2ndFlrSF", "HalfBath", "BedroomAbvGr", "KitchenAbvGr",
]

X = train[FEATURES]
y = np.log1p(train["SalePrice"])  # log-transform target, NOT the dataframe column

# ── Train/Validation Split ─────────────────────────────────────────────────────
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ── Pipeline (imputer + model) ─────────────────────────────────────────────────
pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),  # handles MasVnrArea nulls
    ("model", XGBRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
    )),
])

# ── Validate ───────────────────────────────────────────────────────────────────
pipeline.fit(X_train, y_train)
preds = pipeline.predict(X_valid)
print(f"Validation R² Score: {r2_score(y_valid, preds):.4f}")

# ── Retrain on Full Data ───────────────────────────────────────────────────────
pipeline.fit(X, y)

# ── Save ───────────────────────────────────────────────────────────────────────
joblib.dump(pipeline, "models/streamlit_model.pkl")
print("Model saved to models/streamlit_model.pkl")