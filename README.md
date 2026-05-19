# AWS MLOps Churn Prediction Pipeline

An end-to-end MLOps pipeline that trains, evaluates, registers, and auto-deploys a customer churn prediction model on AWS — with drift monitoring, automated deployment triggers, and full inference logging.

---

## Architecture

![Architecture](docs/architecture.png)

### Flow

```
S3 (raw data)
    ↓
SageMaker Pipeline
    ├── Step 1: Processing Job     → feature engineering (SKLearn)
    ├── Step 2: Training Job       → XGBoost model training
    ├── Step 3: Evaluation Job     → accuracy + AUC metrics
    └── Step 4: Condition Step     → register only if accuracy ≥ 75%
                    ↓
            Model Registry (PendingManualApproval)
                    ↓ (on Approved)
            EventBridge → Lambda (auto-deploy)
                    ↓
            SageMaker Endpoint (ml.t2.medium)
                    ↓
            API Gateway (REST, IAM auth)
                    ↓
            DynamoDB (inference logs) + CloudWatch (drift alarms)
```

---

## Results

| Metric | Value |
|---|---|
| Model | XGBoost (built-in SageMaker algorithm) |
| Dataset | IBM Telco Customer Churn (7,043 customers) |
| Train / Test split | 80% / 20% |
| Accuracy | **80.24%** |
| AUC | **0.8541** |
| Endpoint instance | ml.t2.medium |
| Inference latency | < 100ms |

---

## Services Used

| Service | Purpose |
|---|---|
| S3 | Raw data, processed features, model artifacts |
| SageMaker Pipelines | Orchestrated train → evaluate → register workflow |
| SageMaker Model Registry | Model versioning and approval gate |
| EventBridge | Triggers Lambda on model approval event |
| Lambda | Auto-deploys approved model to endpoint |
| SageMaker Endpoint | Real-time inference (REST) |
| DynamoDB | Inference logs — prediction ID, timestamp, probability |
| CloudWatch | 3 alarms: invocation rate, errors, latency |
| SNS | Email alerts on alarm breach |
| CloudFormation | Infrastructure as Code for all resources |

---

## Project Structure

```
aws-mlops-churn/
├── README.md
├── buildspec.yml                  ← CodeBuild CI/CD spec
├── requirements.txt
├── data/
│   └── telco_churn.csv            ← IBM Telco dataset
├── notebooks/
│   └── exploration.ipynb
├── src/
│   ├── preprocessing.py           ← SKLearn feature engineering
│   ├── train_local.py             ← Local training + S3 upload + Model Registry
│   ├── evaluate.py                ← Metrics generation
│   ├── pipeline.py                ← SageMaker Pipeline definition
│   ├── inference_logger.py        ← Prediction logging to DynamoDB
│   └── setup_monitoring.py        ← CloudWatch alarms + SNS alerts
├── lambda/
│   └── auto_deploy/
│       └── handler.py             ← Auto-deployment Lambda function
├── infra/
│   └── template.yaml              ← CloudFormation: IAM, Lambda, EventBridge
└── docs/
    └── architecture.png
```

---

## Setup & Deployment

### Prerequisites

- AWS CLI configured (`aws configure`)
- Python 3.11
- Virtual environment activated

### Install dependencies

```bash
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1       # Windows
pip install -r requirements.txt
```

### 1. Deploy infrastructure

```bash
aws cloudformation deploy \
  --template-file infra/template.yaml \
  --stack-name mlops-churn-iam \
  --capabilities CAPABILITY_NAMED_IAM \
  --region ap-south-1
```

### 2. Upload dataset to S3

```bash
aws s3 cp data/telco_churn.csv s3://aws-mlops-churn-{ACCOUNT_ID}/raw/telco_churn.csv
```

### 3. Run preprocessing

```bash
python src/preprocessing.py \
  --input-path data \
  --output-path data/output
```

### 4. Train and register model

```bash
python src/train_local.py
```

### 5. Approve model to trigger auto-deployment

```bash
aws sagemaker update-model-package \
  --model-package-arn arn:aws:sagemaker:ap-south-1:{ACCOUNT_ID}:model-package/ChurnModelPackageGroup/1 \
  --model-approval-status Approved \
  --region ap-south-1
```

EventBridge detects the approval and Lambda automatically deploys the endpoint.

### 6. Set up monitoring

```bash
python src/setup_monitoring.py
```

### 7. Run inference with logging

```bash
python src/inference_logger.py
```

---

## Cost Estimate

| Service | Estimated Cost |
|---|---|
| SageMaker Training (ml.m5.xlarge) | ~$0.10 / run |
| SageMaker Endpoint (ml.t2.medium) | ~$0.05 / hr |
| Lambda + EventBridge | Free tier |
| S3 | < $0.01 |
| DynamoDB | Free tier |
| CloudWatch (3 alarms) | Free tier |
| **Total for dev/demo** | **< $5** |

> **Note:** Delete the endpoint when not in use to avoid charges.
> ```bash
> aws sagemaker delete-endpoint --endpoint-name churn-prediction-endpoint --region ap-south-1
> ```

---

## MLOps Design Decisions

**Why a Condition step?** Prevents degraded models from being deployed automatically. Only models exceeding the accuracy threshold reach the registry.

**Why local training instead of SageMaker training jobs?** Free tier accounts have a default quota of 0 for SageMaker training instances in ap-south-1. The "Bring Your Own Model" pattern is a valid production approach — training happens wherever compute is available, and the model artifact is registered centrally.

**Why EventBridge + Lambda for deployment?** Decouples the approval event from the deployment action. Any system (console, CLI, CI/CD) can approve a model and deployment happens automatically without manual steps.

**Why DynamoDB for inference logs?** Provides a queryable audit trail for every prediction — useful for detecting data drift over time by comparing live input distributions against training data.

---

## What I'd Improve Next

- Add A/B endpoint routing to compare model versions in production
- Replace local training with SageMaker Autopilot for automated model selection
- Add SageMaker Model Monitor for automated drift detection (requires ml.m5.xlarge quota)
- Wire CodePipeline to retrigger training automatically on new S3 uploads
- Add a lightweight FastAPI wrapper around the endpoint for easier consumption

---

## Dataset

[IBM Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) — 7,043 customers, 33 features, binary churn label. Publicly available on Kaggle.

---

## License

MIT
