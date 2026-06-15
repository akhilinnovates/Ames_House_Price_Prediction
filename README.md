# 🏡 HomePriceIQ — House Price Prediction App

A Machine Learning web application that predicts residential property prices
using the Ames Housing Dataset, deployed with Streamlit.

---

## 📌 Project Overview

HomePriceIQ is a Machine Learning project that predicts the selling price of
residential properties using the Ames Housing Dataset.

The project follows a complete Machine Learning pipeline, including:

- Data Cleaning & Missing Value Handling
- Exploratory Data Analysis (EDA)
- Model Training & Evaluation
- Model Deployment using Streamlit

The final deployed model uses **XGBoost Regressor**, which achieved the best
performance among all tested models.

---

## 🎯 Objective

Accurately predict house prices based on various property characteristics such as:

- Living Area & Floor Space
- Overall Quality Rating
- Garage Capacity & Area
- Basement Features
- Property Age & Remodel Year
- Outdoor Features (Porch, Deck)

---

## 📊 Dataset Used

**House Prices: Advanced Regression Techniques** (Kaggle Competition Dataset)

| Attribute | Value |
|---|---|
| Total Records | 1,460 |
| Total Features | 81 |
| Target Variable | SalePrice |
| Problem Type | Regression |
| Price Range | $35k – $755k |
| Median Price | $163,000 |

---

## 🧹 Data Preprocessing

- **Missing Value Handling** — Median imputation for numerical features
- **Log Transformation** — Target variable (SalePrice) log-transformed using
  `log1p()` to reduce skewness
- **Feature Selection** — 20 most impactful features selected for the model

---

## 📈 Exploratory Data Analysis (EDA)

- SalePrice Distribution Analysis
- Log Transformation Analysis
- Correlation Analysis
- Feature Importance Analysis
- Neighborhood Price Analysis

### Key Insights
- Quality 10 homes sell for **8.7× more** than Quality 1
- NridgHt median price ($315k) is **3.6× MeadowV** ($88k)
- Post-2000 builds command **$60k+ premium** over 1980s homes
- Each garage car spot adds **~$18k** in value
- Overall Quality explains **~38%** of price variance

---

## 🤖 Models Implemented

| Model | Description |
|---|---|
| Random Forest Regressor | Captures complex non-linear relationships |
| XGBoost Regressor | Final selected model — best performance |

---

## 📈 Model Evaluation

Models evaluated using:
- R² Score
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)

### Final Selected Model
✅ **XGBoost Regressor**

| Parameter | Value |
|---|---|
| n_estimators | 1,000 |
| learning_rate | 0.05 |
| max_depth | 4 |
| subsample | 0.8 |
| colsample_bytree | 0.8 |
| Validation R² | ~0.91 |

---

## 🔮 Features Used for Prediction

| # | Feature | Description |
|---|---|---|
| 1 | OverallQual | Overall material and finish quality |
| 2 | GrLivArea | Above ground living area (sq ft) |
| 3 | GarageCars | Garage capacity (cars) |
| 4 | GarageArea | Garage area (sq ft) |
| 5 | TotalBsmtSF | Total basement area (sq ft) |
| 6 | 1stFlrSF | First floor area (sq ft) |
| 7 | FullBath | Full bathrooms above grade |
| 8 | TotRmsAbvGrd | Total rooms above ground |
| 9 | YearBuilt | Year house was built |
| 10 | YearRemodAdd | Year of remodel |
| 11 | MasVnrArea | Masonry veneer area (sq ft) |
| 12 | Fireplaces | Number of fireplaces |
| 13 | BsmtFinSF1 | Finished basement area (sq ft) |
| 14 | LotArea | Lot size (sq ft) |
| 15 | OpenPorchSF | Open porch area (sq ft) |
| 16 | WoodDeckSF | Wood deck area (sq ft) |
| 17 | 2ndFlrSF | Second floor area (sq ft) |
| 18 | HalfBath | Half bathrooms |
| 19 | BedroomAbvGr | Bedrooms above ground |
| 20 | KitchenAbvGr | Kitchens above ground |

---

## 🚀 Streamlit Web Application

A user-friendly dashboard built with Streamlit featuring:

- **Dashboard** — Key market stats, charts, and insights
- **Price Predictor** — Interactive sliders to predict any home's value
- **Market Analytics** — Neighborhood prices, quality trends, era analysis
- **Batch Predict** — Upload CSV and predict multiple homes at once
- **About** — Model architecture and technical details

---

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core language |
| Pandas & NumPy | Data processing |
| Scikit-Learn | ML pipeline & preprocessing |
| XGBoost | Prediction model |
| Streamlit | Web application |
| Plotly | Interactive charts |
| Matplotlib & Seaborn | EDA visualizations |
| Joblib | Model serialization |

---

## 📂 Project Structure


---

## ▶️ Installation & Running Locally

**1. Clone the repository:**
```bash
git clone https://github.com/yourusername/ML_house_Price_Prediction.git
cd ML_house_Price_Prediction
```

**2. Install dependencies:**
```bash
pip install -r requirements.txt
```

**3. Train the model:**
```bash
python train_streamlit.py
```

**4. Run the app:**
```bash
streamlit run app.py
```

---

## 📚 Dataset Source

**House Prices: Advanced Regression Techniques** — Kaggle

- [Kaggle Competition](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)
- Original Author: Dean De Cock, Professor of Statistics, Iowa State University

---

## ⭐ Acknowledgements

Special thanks to Kaggle, Dean De Cock, and the Open Source
Machine Learning Community for providing the dataset and resources
that made this project possible.
