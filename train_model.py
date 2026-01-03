import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

import matplotlib.pyplot as plt
import seaborn as sns


def main():
    file_path = "data/data.xlsx"  # <-- файл лежит в папке data/
    data = pd.read_excel(file_path)

    # Normalize column names
    data.columns = data.columns.str.strip().str.lower()

    print("Columns:", list(data.columns))
    print("Missing values:\n", data.isnull().sum())

    data = data.dropna().copy()

    target_col = "price"
    drop_cols = ["id", target_col]

    X = data.drop(columns=drop_cols, errors="ignore")
    y = data[target_col]

    categorical_cols = [c for c in X.columns if X[c].dtype == "object"]
    numerical_cols = [c for c in X.columns if c not in categorical_cols]

    print("Categorical:", categorical_cols)
    print("Numerical:", numerical_cols)

    preprocess = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("num", "passthrough", numerical_cols),
        ]
    )

    model = RandomForestRegressor(
        n_estimators=400,
        random_state=42,
        n_jobs=-1
    )

    pipe = Pipeline(steps=[("prep", preprocess), ("model", model)])

    # Honest hold-out metrics
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    print("MAE:", mean_absolute_error(y_test, y_pred))
    print("R2 :", r2_score(y_test, y_pred))

    # Out-of-fold predictions for deal ranking (less leakage)
    oof_pred = cross_val_predict(pipe, X, y, cv=5, n_jobs=-1)
    data["predicted_price_oof"] = oof_pred

    # Deal score: predicted / actual (bigger => model thinks undervalued)
    data["deal_ratio"] = data["predicted_price_oof"] / data["price"]
    data["price_deviation"] = data["predicted_price_oof"] - data["price"]

    top_deals = data.sort_values("deal_ratio", ascending=False).head(10)
    print("\nTop 10 deals by deal_ratio:")
    cols_to_show = [c for c in ["id", "price", "predicted_price_oof", "deal_ratio"] if c in data.columns]
    print(top_deals[cols_to_show])

    # Plots
    sns.histplot(data["price_deviation"], bins=30, kde=True)
    plt.title("Predicted - Actual (OOF) deviation distribution")
    plt.xlabel("Deviation")
    plt.ylabel("Frequency")
    plt.show()

    if "carat weight" in data.columns:
        sns.scatterplot(x="carat weight", y="price", data=data)
        plt.title("Carat Weight vs Price")
        plt.xlabel("Carat Weight")
        plt.ylabel("Price")
        plt.show()

    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.title("Predicted vs Actual (hold-out test)")
    plt.show()

    plt.figure(figsize=(10, 6))
    sns.histplot(data["deal_ratio"], bins=30, kde=True)
    plt.title("Deal ratio distribution (predicted / actual)")
    plt.xlabel("Deal ratio")
    plt.ylabel("Frequency")
    plt.axvline(data["deal_ratio"].mean(), linestyle="--", label="Mean")
    plt.legend()
    plt.show()

    output_path = "diamond_analysis_results.xlsx"
    data.to_excel(output_path, index=False)
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
