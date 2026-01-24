import pandas as pd
import os
import yaml
import pickle

from sklearn.metrics import confusion_matrix, classification_report
from mlflow.models import infer_signature
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from xgboost import XGBClassifier
from urllib.parse import urlparse
import mlflow
from sklearn.preprocessing import StandardScaler

os.environ["MLFLOW_TRACKING_URI"] = "https://dagshub.com/Mohit-Bhoir/AlgoTradingBacktest.mlflow"
os.environ["MLFLOW_TRACKING_USERNAME"] = "Mohit-Bhoir"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "8407fd9e577fa22b1f5f3df39f7a37252fd1ded3"

def hyperparameter_tuning(X_train, y_train, param_grid):
    xgb = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
    )
    tscv = TimeSeriesSplit(n_splits=3)
    grid_search = GridSearchCV(
        estimator=xgb,
        param_grid=param_grid,
        scoring="accuracy",
        cv=tscv,
        n_jobs=-1,
        verbose=2,
    )
    grid_search.fit(X_train, y_train)
    return grid_search

# Load all the params from params.yaml
params = yaml.safe_load(open("params.yaml"))["train"]

def train(data_path, model_path, random_state, n_estimators, max_depth):
    data = pd.read_csv(data_path)

    if "direction" not in data.columns:
        raise ValueError("Expected target column 'direction' not found in training data.")

    X = data.drop("direction", axis=1)

    # Drop leakage columns if present
    drop_cols = [c for c in ["time", "returns", "price"] if c in X.columns]
    if drop_cols:
        X = X.drop(columns=drop_cols)

    y = data["direction"]

    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])

    with mlflow.start_run():
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=params["test_size"], random_state=random_state, shuffle=False
        )
        signature = infer_signature(X_train, y_train)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Valid XGBoost hyperparameters
        param_grid = {
            "n_estimators": [n_estimators] if n_estimators is not None else [100, 200],
            "max_depth": [max_depth] if max_depth is not None else [3, 5, 8],
            "learning_rate": [0.05, 0.1],
            "subsample": [0.8, 1.0],
            "colsample_bytree": [0.8, 1.0],
        }

        grid_search = hyperparameter_tuning(X_train, y_train, param_grid)
        best_model = grid_search.best_estimator_

        y_pred = best_model.predict(X_test)

        print(confusion_matrix(y_test, y_pred))
        print(classification_report(y_test, y_pred))

        mlflow.log_params(grid_search.best_params_)
        mlflow.log_metric("accuracy", best_model.score(X_test, y_test))

        cm = confusion_matrix(y_test, y_pred)
        cr = classification_report(y_test, y_pred, output_dict=True)
        mlflow.log_text(str(cm), "confusion_matrix.txt")
        mlflow.log_dict(cr, "classification_report.json")

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        if tracking_url_type_store != "file":
            mlflow.sklearn.log_model(
                best_model,
                "model",
                signature=signature,
                registered_model_name="XGBClassifier_AlgoTrading",
            )
        else:
            mlflow.sklearn.log_model(best_model, "model", signature=signature)

        # Default model path if not provided in params.yaml
        if not model_path:
            model_path = os.path.join("models", "xgb.pkl")

        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        with open(model_path, "wb") as f:
            pickle.dump(best_model, f)

        print(f"Model saved to {model_path}")
        print("Training completed.")

if __name__ == "__main__":
    train(
        params["data_path"],
        params.get("model_path", os.path.join("models", "xgb.pkl")),
        params["random_state"],
        params.get("n_estimators"),
        params.get("max_depth"),
    )
