import os
import json
import argparse
import pandas as pd
import numpy as np
import tarfile
import xgboost as xgb
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

def evaluate(model_path, test_path, output_path):
    print("Loading model...")
    model_file = os.path.join(model_path, "model.tar.gz")
    with tarfile.open(model_file) as tar:
        tar.extractall(path=model_path)

    model = xgb.Booster()
    model.load_model(os.path.join(model_path, "xgboost-model"))

    print("Loading test data...")
    test_df = pd.read_csv(os.path.join(test_path, "test.csv"), header=None)

    X_test = test_df.iloc[:, 1:].values
    y_test = test_df.iloc[:, 0].values

    dtest = xgb.DMatrix(X_test)
    predictions_prob = model.predict(dtest)
    predictions = (predictions_prob > 0.5).astype(int)

    accuracy = accuracy_score(y_test, predictions)
    auc = roc_auc_score(y_test, predictions_prob)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"AUC: {auc:.4f}")
    print(classification_report(y_test, predictions, target_names=["No Churn", "Churn"]))

    # Write metrics JSON - SageMaker Pipeline reads this for the Condition step
    metrics = {
        "classification_metrics": {
            "accuracy": {"value": accuracy},
            "auc": {"value": auc}
        }
    }

    os.makedirs(output_path, exist_ok=True)
    output_file = os.path.join(output_path, "evaluation.json")
    with open(output_file, "w") as f:
        json.dump(metrics, f)

    print(f"Metrics written to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="/opt/ml/processing/model")
    parser.add_argument("--test-path", type=str, default="/opt/ml/processing/test")
    parser.add_argument("--output-path", type=str, default="/opt/ml/processing/evaluation")
    args = parser.parse_args()

    evaluate(args.model_path, args.test_path, args.output_path)