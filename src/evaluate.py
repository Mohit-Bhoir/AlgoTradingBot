import pandas as pd
import pickle
import yaml
import os
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import mlflow

# --- CREDENTIALS ---
os.environ["MLFLOW_TRACKING_URI"] = "https://dagshub.com/Mohit-Bhoir/AlgoTradingBacktest.mlflow"
os.environ["MLFLOW_TRACKING_USERNAME"] = "Mohit-Bhoir"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "8407fd9e577fa22b1f5f3df39f7a37252fd1ded3"

# Load params
# Note: We need loading 'train' params too to get the random_state/test_size
params = yaml.safe_load(open("params.yaml"))
eval_params = params["evaluate"]
train_params = params["train"] 

def evaluate(data_path, model_path):
    # Load data
    data = pd.read_csv(data_path)
    
    # Ensure target exists
    if "direction" not in data.columns:
        raise ValueError("Target column 'direction' missing")

    y = data["direction"]
    X = data.drop("direction", axis=1)

    # --- DROP LEAKAGE VOLUMNS ---
    cols_to_drop = [c for c in ["time", "price", "returns"] if c in X.columns]
    X = X.drop(columns=cols_to_drop)

    # --- CRITICAL FIX: RE-SPLIT DATA ---
    # We must use the EXACT same random_state and test_size as train.py
    # If using Time Series, ensure shuffle matches train.py (usually False for time series)
    _, X_test, _, y_test = train_test_split(
        X, y, 
        test_size=train_params["test_size"], 
        random_state=train_params["random_state"],
        shuffle=False # KPI: Must match train.py logic
    )
    # -----------------------------------

    # Load model
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # Predict ONLY on Test Data
    predictions = model.predict(X_test)

    # Log metrics
    acc = accuracy_score(y_test, predictions)
    
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    with mlflow.start_run():
        mlflow.log_metric("test_accuracy", acc) # Rename to test_accuracy
        print(f"Test Set Accuracy: {acc}")
        
        cm = confusion_matrix(y_test, predictions)
        cr = classification_report(y_test, predictions, output_dict=True)
        
        mlflow.log_text(str(cm), "confusion_matrix.txt")
        mlflow.log_dict(cr, "classification_report.json")

if __name__ == "__main__":
    evaluate(eval_params["data_path"], eval_params["model_path"])



