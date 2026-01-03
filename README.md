# diamond-price-analysis
Diamond price prediction and deal detection (Python, scikit-learn)
# Diamond Price Prediction & Deal Detection

Machine Learning project for predicting diamond prices and identifying potentially undervalued diamonds.

## Problem
Diamond prices depend on multiple categorical and numerical characteristics.
The goal is to:
- predict a fair market price
- identify diamonds that appear undervalued compared to model predictions

## Data
Tabular dataset with diamond characteristics:
carat weight, cut, color, clarity, polish, symmetry, report, and price.

## Approach
- Data cleaning and preprocessing
- One-hot encoding for categorical features
- RandomForest regression
- Cross-validation with out-of-fold predictions
- Deal detection using predicted price vs actual price ratio

## Metrics
- Mean Absolute Error (MAE)
- R² score

## Tech stack
Python, pandas, scikit-learn, matplotlib, seaborn

## Project structure
- `train_model.py` — model training and evaluation
- `notebooks/` — exploratory data analysis (EDA)

## Notes
Dataset is not included due to size/licensing limitations.
