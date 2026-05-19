import boto3
import os
import tarfile
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, roc_auc_score
import json

ACCOUNT_ID = os.environ.get("AWS_ACCOUNT_ID", "YOUR_ACCOUNT_ID")
BUCKET = f"aws-mlops-churn-{ACCOUNT_ID}"
REGION = "ap-south-1"

print("Loading processed data...")
train_df = pd.read_csv("data/output/train.csv", header=None)
test_df = pd.read_csv("data/output/test.csv", header=None)

X_train = train_df.iloc[:, 1:].values
y_train = train_df.iloc[:, 0].values
X_test = test_df.iloc[:, 1:].values
y_test = test_df.iloc[:, 0].values

print("Training XGBoost model...")
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

params = {
    "objective": "binary:logistic",
    "max_depth": 5,
    "eta": 0.2,
    "subsample": 0.8,
    "eval_metric": "auc"
}

model = xgb.train(
    params,
    dtrain,
    num_boost_round=100,
    evals=[(dtest, "test")],
    verbose_eval=10
)

# Evaluate
preds_prob = model.predict(dtest)
preds = (preds_prob > 0.5).astype(int)
accuracy = accuracy_score(y_test, preds)
auc = roc_auc_score(y_test, preds_prob)
print(f"\nAccuracy: {accuracy:.4f}")
print(f"AUC: {auc:.4f}")

# Save metrics
os.makedirs("data/evaluation", exist_ok=True)
metrics = {
    "classification_metrics": {
        "accuracy": {"value": accuracy},
        "auc": {"value": auc}
    }
}
with open("data/evaluation/evaluation.json", "w") as f:
    json.dump(metrics, f)
print("Metrics saved to data/evaluation/evaluation.json")

# Save model
os.makedirs("data/model", exist_ok=True)
model.save_model("data/model/xgboost-model")

# Package as tar.gz (SageMaker format)
with tarfile.open("data/model/model.tar.gz", "w:gz") as tar:
    tar.add("data/model/xgboost-model", arcname="xgboost-model")
print("Model packaged as model.tar.gz")

# Upload to S3
s3 = boto3.client("s3", region_name=REGION)
s3_key = "model-artifacts/local-training/model.tar.gz"
s3.upload_file("data/model/model.tar.gz", BUCKET, s3_key)
model_s3_uri = f"s3://{BUCKET}/{s3_key}"
print(f"Model uploaded to {model_s3_uri}")

# Register model in SageMaker Model Registry
sagemaker = boto3.client("sagemaker", region_name=REGION)

# Get XGBoost image URI
import sagemaker as sm
from sagemaker import image_uris
image_uri = image_uris.retrieve(
    framework="xgboost",
    region=REGION,
    version="1.7-1",
    image_scope="training"
)

try:
    sagemaker.create_model_package_group(
        ModelPackageGroupName="ChurnModelPackageGroup",
        ModelPackageGroupDescription="Churn prediction models"
    )
    print("Model package group created")
except sagemaker.exceptions.from_code('ValidationException'):
    print("Model package group already exists")

response = sagemaker.create_model_package(
    ModelPackageGroupName="ChurnModelPackageGroup",
    ModelPackageDescription=f"XGBoost churn model - accuracy={accuracy:.4f} auc={auc:.4f}",
    InferenceSpecification={
        "Containers": [{
            "Image": image_uri,
            "ModelDataUrl": model_s3_uri
        }],
        "SupportedContentTypes": ["text/csv"],
        "SupportedResponseMIMETypes": ["text/csv"],
        "SupportedTransformInstanceTypes": ["ml.m5.xlarge"],
        "SupportedRealtimeInferenceInstanceTypes": ["ml.t2.medium"]
    },
    ModelApprovalStatus="PendingManualApproval"
)

print(f"Model registered: {response['ModelPackageArn']}")
print("\nDone! Go to SageMaker Model Registry and approve the model to trigger auto-deployment.")