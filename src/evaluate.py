import pandas as pd
import pickle
import yaml
import os
from sklearn.metrics import accuracy_score, classification_report

# Load params
params = yaml.safe_load(open("params.yaml"))["evaluate"]

def evaluate(data_path, model_path):
    # Load data
    data = pd.read_csv(data_path)
    
    # Load model
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # Ensure target exists
    if "direction" not in data.columns:
        raise ValueError("Target column 'direction' missing")

    y = data["direction"]
    X = data.drop("direction", axis=1)

    # --- FIX: DROP NON-NUMERIC & LEAKAGE COLUMNS ---
    # This must match exactly what was done in train.py
    cols_to_drop = [c for c in ["time", "price", "returns"] if c in X.columns]
    X = X.drop(columns=cols_to_drop)
    # -----------------------------------------------

    # Predict
    predictions = model.predict(X)

    # Log metrics
    acc = accuracy_score(y, predictions)
    print(f"Accuracy: {acc}")
    print("Classification Report:")
    print(classification_report(y, predictions))

if __name__ == "__main__":
    evaluate(params["data_path"], params["model_path"])



