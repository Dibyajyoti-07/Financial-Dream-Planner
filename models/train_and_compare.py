import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

DATA_PATH = Path(__file__).parent.parent / "data" / "salary_data.csv"
MODEL_PATH = Path(__file__).parent / "salary_model.pkl"
METADATA_PATH = Path(__file__).parent / "model_metadata.json"

CATEGORICAL_COLS = ["City", "Education", "Job_Role"]
NUMERIC_COLS = ["Age"]
TARGET_COL = "Monthly_Salary"
RANDOM_STATE = 42


def load_salary_data():
    df = pd.read_csv(DATA_PATH)
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    assert "Experience" not in df.columns, "Experience must never be a feature (fresher rule)"
    return df


def build_pipeline(model):
    preprocessor = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_COLS),
        ("num", StandardScaler(), NUMERIC_COLS),
    ])
    pipeline = Pipeline([("prep", preprocessor), ("model", model)])
    return TransformedTargetRegressor(regressor=pipeline, transformer=StandardScaler())


def get_models():
    return {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0, random_state=RANDOM_STATE),
        "Lasso Regression": Lasso(alpha=1.0, random_state=RANDOM_STATE),
        "ElasticNet": ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeRegressor(random_state=RANDOM_STATE),
        "Random Forest": RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE),
        "Extra Trees": ExtraTreesRegressor(n_estimators=200, random_state=RANDOM_STATE),
        "AdaBoost": AdaBoostRegressor(random_state=RANDOM_STATE),
        "Gradient Boosting": GradientBoostingRegressor(random_state=RANDOM_STATE),
        "Hist Gradient Boosting": HistGradientBoostingRegressor(random_state=RANDOM_STATE),
        "K-Nearest Neighbors": KNeighborsRegressor(n_neighbors=5),
        "Support Vector Regressor": SVR(kernel="rbf"),
        "Neural Network (MLP)": MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=2000, random_state=RANDOM_STATE),
    }


def evaluate_all(df):
    X = df[CATEGORICAL_COLS + NUMERIC_COLS]
    y = df[TARGET_COL]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    results = {}
    for name, model in get_models().items():
        estimator = build_pipeline(model)
        cv_scores = cross_val_score(estimator, X_train, y_train, cv=cv, scoring="r2")
        estimator.fit(X_train, y_train)
        preds = estimator.predict(X_test)
        results[name] = {
            "CV_R2_Mean": float(cv_scores.mean()),
            "Test_R2": float(r2_score(y_test, preds)),
            "Test_MAE": float(mean_absolute_error(y_test, preds)),
            "Test_RMSE": float(np.sqrt(mean_squared_error(y_test, preds))),
        }
        print(name, results[name])
    return results


def select_best(results):
    return sorted(results.items(), key=lambda kv: (kv[1]["Test_MAE"], -kv[1]["Test_R2"]))[0][0]


def main():
    df = load_salary_data()
    results = evaluate_all(df)
    best_name = select_best(results)
    print(f"\nSelected: {best_name} (lowest Test_MAE, ties broken by higher Test_R2)")

    best_model = get_models()[best_name]
    best_estimator = build_pipeline(best_model)
    X = df[CATEGORICAL_COLS + NUMERIC_COLS]
    y = df[TARGET_COL]
    best_estimator.fit(X, y)

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(best_estimator, MODEL_PATH)

    metadata = {
        "selected_model": best_name,
        "selection_rule": "lowest Test_MAE on 80/20 held-out split; ties broken by higher Test_R2",
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "random_state": RANDOM_STATE,
        "test_size": 0.2,
        "cv_folds": 5,
        "features": CATEGORICAL_COLS + NUMERIC_COLS,
        "target": TARGET_COL,
        "refit_on_full_data": True,
        "training_salary_range": {"min": float(y.min()), "max": float(y.max())},
        "metrics": results,
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved {MODEL_PATH} and {METADATA_PATH}")


if __name__ == "__main__":
    main()
