# Uber Ride Price Prediction — Data Analysis & Regression Modelling

An end-to-end data analysis and machine learning project that predicts Uber ride fares from pickup and drop-off coordinates, temporal features, and passenger count using **Linear Regression** and **Random Forest Regression**.

---

## Project Overview

Predicting accurate ride fares requires cleaning noisy GPS/fare records, engineering spatial and temporal features, handling statistical outliers, and comparing linear vs. tree-based regression models.

### Key Objectives
1. **Data Preprocessing & Cleaning:** Remove redundant columns, parse UTC timestamps, drop missing/duplicate rows, and filter invalid fares, passenger counts, and out-of-bounds NYC coordinates.
2. **Feature Engineering:** Compute great-circle trip distance using the **Haversine formula** and extract temporal attributes (`hour`, `day`, `month`, `year`, `day_of_week`).
3. **Outlier Detection & Removal:** Identify and filter extreme values in `fare_amount` and `distance` using the **Interquartile Range (IQR)** method.
4. **Correlation Analysis:** Analyze feature relationships with `fare_amount` via a correlation matrix and Seaborn heatmap.
5. **Model Implementation & Evaluation:** Train and compare **Linear Regression** and **Random Forest Regressor** models using **$R^2$ Score**, **MAE**, and **RMSE**.
6. **Interactive Fare Predictor:** Accept custom ride inputs (coordinates, timestamp, passenger count) to generate real-time fare estimates.

---

## Dataset & Preprocessing Pipeline

- **Raw Dataset (`uber.csv`):** 200,000 records and 9 columns (`Unnamed: 0`, `key`, `fare_amount`, `pickup_datetime`, `pickup_longitude`, `pickup_latitude`, `dropoff_longitude`, `dropoff_latitude`, `passenger_count`).
- **Data Cleaning (`195,096` rows retained):**
  - Dropped `Unnamed: 0` and `key`.
  - Converted `pickup_datetime` to UTC datetime and removed null/duplicate entries.
  - Filtered valid fares (`$0 < fare_amount <= $200`), passenger counts (`1` to `6`), and NYC bounding-box coordinates (`latitude` 40°–42°, `longitude` -75° to -72.5°).
- **Feature Engineering & Distance Filtering:**
  - Calculated `distance` (in km) via the Haversine formula and restricted rides to `0 < distance <= 100 km`.
  - Extracted `hour`, `day`, `month`, `year`, and `day_of_week`.
- **IQR Outlier Removal (`168,801` rows final):**
  - `fare_amount` bounds: `[-3.75, 22.25]` (16,512 outliers identified).
  - `distance` bounds: `[-2.71, 7.93]` km (16,028 outliers identified).

---

## Key Insights & Model Evaluation

- **Strongest Predictor:** Trip `distance` shows the highest positive correlation with `fare_amount` (**$r = 0.80$**), followed by `year` (**$r = 0.17$**).
- **Train-Test Split:** 80% training (`135,040` rides) / 20% testing (`33,761` rides) with `random_state=42`.

| Model | $R^2$ Score | MAE ($) | RMSE ($) |
| :--- | :---: | :---: | :---: |
| **Linear Regression** | 0.6695 | 1.52 | 2.13 |
| **Random Forest Regression** (`n_estimators=100`) | **0.7440** | **1.30** | **1.87** |

> **Conclusion:** **Random Forest Regression** outperforms Linear Regression across every metric, explaining **74.40%** of variance in ride fares and reducing RMSE from **$2.13** to **$1.87** by capturing non-linear spatial and temporal pricing patterns.

---

## Tech Stack

- **Language:** Python 3
- **Data Manipulation:** `pandas`, `NumPy`
- **Visualization:** `Matplotlib`, `Seaborn`
- **Machine Learning:** `scikit-learn` (`train_test_split`, `LinearRegression`, `RandomForestRegressor`, evaluation metrics)

---

## Repository Structure

```text
Uber-Price-Prediction-Data-Analysis/
├── Uber_Price_Prediction.ipynb   # Jupyter Notebook with EDA, visualizations, models & predictor
├── uber.csv                      # Dataset
└── README.md                     # Project documentation
