# ⚾ MLB Salary Prediction with LightGBM

Regression model predicting baseball player salaries using the 1986-87 Hitters dataset with LightGBM and feature engineering.

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![LightGBM](https://img.shields.io/badge/LightGBM-3.3+-brightgreen)
![License](https://img.shields.io/badge/License-MIT-green)

## Problem Statement

Predict the salary of Major League Baseball players based on their 1986-87 season statistics. The dataset contains 322 observations with 20 features covering batting, fielding, and career statistics.

## Dataset Features

| Feature | Description |
|---------|-------------|
| AtBat | Number of at-bats in 1986 |
| Hits | Number of hits in 1986 |
| HmRun | Home runs in 1986 |
| Runs | Runs scored in 1986 |
| RBI | Runs batted in during 1986 |
| Walks | Number of walks in 1986 |
| Years | Years in the major leagues |
| CAtBat | Career at-bats |
| CHits | Career hits |
| CHmRun | Career home runs |
| CRuns | Career runs |
| CRBI | Career RBIs |
| CWalks | Career walks |
| League | Player's league (A/N) |
| Division | Division (E/W) |
| PutOuts | Number of put outs in 1986 |
| Assists | Number of assists in 1986 |
| Errors | Number of errors in 1986 |
| NewLeague | League at the start of 1987 |
| **Salary** | **Target: 1987 annual salary (thousands of dollars)** |

## Approach

1. **EDA** — Column classification, distribution analysis, missing value inspection
2. **Missing Value Handling** — Rows with missing salary dropped
3. **Feature Engineering** — Label encoding, rare category encoding, one-hot encoding
4. **Scaling** — RobustScaler (handles salary outliers well)
5. **Model** — LightGBM Regressor with GridSearchCV hyperparameter tuning

## Results

| Model | RMSE |
|-------|------|
| LightGBM (default) | Baseline RMSE |
| LightGBM (tuned via GridSearchCV) | **77.56** |

## Tech Stack

- **Python 3.8+** — Core language
- **LightGBM** — Gradient boosting regressor
- **Scikit-learn** — Preprocessing, GridSearchCV, metrics
- **Pandas / NumPy** — Data manipulation
- **Seaborn / Matplotlib** — Visualization

## Project Structure

```
Hitters_ML_Light_GBM/
├── helpers/
│   ├── __init__.py
│   ├── data_prep.py          # Outlier handling, imputation, encoding utilities
│   └── eda.py                # EDA summary and visualization functions
├── hitters_light_gbm.py     # Main ML pipeline
├── requirements.txt
└── README.md
```

## Getting Started

```bash
git clone https://github.com/eboekenh/Hitters_ML_Light_GBM.git
cd Hitters_ML_Light_GBM
pip install -r requirements.txt
```

The Hitters dataset is available from the ISLR R package or [Kaggle](https://www.kaggle.com/datasets/floser/hitters). Place `hitters.csv` in the project root.

```bash
python hitters_light_gbm.py
```

## License

MIT
