import boto3
import sagemaker
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import ProcessingStep, TrainingStep
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.functions import JsonGet
from sagemaker.workflow.properties import PropertyFile
from sagemaker.processing import ScriptProcessor, ProcessingInput, ProcessingOutput
from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput
from sagemaker import image_uris

# ── Config ──────────────────────────────────────────────────────────────────
REGION          = "ap-south-1"
ACCOUNT_ID      = "916554063443"
BUCKET          = f"aws-mlops-churn-{ACCOUNT_ID}"
ROLE_ARN        = f"arn:aws:iam::{ACCOUNT_ID}:role/SageMakerChurnExecutionRole"
PIPELINE_NAME   = "ChurnPredictionPipeline"

# S3 paths
RAW_DATA_URI    = f"s3://{BUCKET}/raw"
PROCESSED_URI   = f"s3://{BUCKET}/processed"
ARTIFACTS_URI   = f"s3://{BUCKET}/model-artifacts"

# ── Session ──────────────────────────────────────────────────────────────────
boto_session    = boto3.Session(region_name=REGION)
sagemaker_session = sagemaker.Session(boto_session=boto_session)

# ── Step 1: Processing ───────────────────────────────────────────────────────
sklearn_image = image_uris.retrieve(
    framework="sklearn",
    region=REGION,
    version="1.2-1",
    image_scope="training"
)

processor = ScriptProcessor(
    image_uri=sklearn_image,
    command=["python3"],
    instance_type="ml.t3.medium",
    instance_count=1,
    role=ROLE_ARN,
    sagemaker_session=sagemaker_session,
)

processing_step = ProcessingStep(
    name="PreprocessChurnData",
    processor=processor,
    code="src/preprocessing.py",
    inputs=[
        ProcessingInput(
            source=RAW_DATA_URI,
            destination="/opt/ml/processing/input"
        )
    ],
    outputs=[
        ProcessingOutput(
            output_name="train",
            source="/opt/ml/processing/output",
            destination=PROCESSED_URI
        )
    ]
)

# ── Step 2: Training ─────────────────────────────────────────────────────────
xgboost_image = image_uris.retrieve(
    framework="xgboost",
    region=REGION,
    version="1.7-1",
    image_scope="training"
)

estimator = Estimator(
    image_uri=xgboost_image,
    instance_type="ml.t3.medium",
    instance_count=1,
    output_path=ARTIFACTS_URI,
    role=ROLE_ARN,
    sagemaker_session=sagemaker_session,
    hyperparameters={
        "objective": "binary:logistic",
        "num_round": 100,
        "max_depth": 5,
        "eta": 0.2,
        "subsample": 0.8,
        "eval_metric": "auc"
    }
)

training_step = TrainingStep(
    name="TrainChurnModel",
    estimator=estimator,
    inputs={
        "train": TrainingInput(
            s3_data=processing_step.properties.ProcessingOutputConfig.Outputs[
                "train"
            ].S3Output.S3Uri,
            content_type="text/csv"
        )
    }
)

# ── Step 3: Evaluation ───────────────────────────────────────────────────────
evaluation_processor = ScriptProcessor(
    image_uri=sklearn_image,
    command=["python3"],
    instance_type="ml.t3.medium",
    instance_count=1,
    role=ROLE_ARN,
    sagemaker_session=sagemaker_session,
)

evaluation_report = PropertyFile(
    name="EvaluationReport",
    output_name="evaluation",
    path="evaluation.json"
)

evaluation_step = ProcessingStep(
    name="EvaluateChurnModel",
    processor=evaluation_processor,
    code="src/evaluate.py",
    inputs=[
        ProcessingInput(
            source=training_step.properties.ModelArtifacts.S3ModelArtifacts,
            destination="/opt/ml/processing/model"
        ),
        ProcessingInput(
            source=PROCESSED_URI,
            destination="/opt/ml/processing/test"
        )
    ],
    outputs=[
        ProcessingOutput(
            output_name="evaluation",
            source="/opt/ml/processing/evaluation"
        )
    ],
    property_files=[evaluation_report]
)

# ── Step 4: Register model (on pass) ────────────────────────────────────────
register_step = RegisterModel(
    name="RegisterChurnModel",
    estimator=estimator,
    model_data=training_step.properties.ModelArtifacts.S3ModelArtifacts,
    content_types=["text/csv"],
    response_types=["text/csv"],
    inference_instances=["ml.t2.medium"],
    transform_instances=["ml.m5.xlarge"],
    model_package_group_name="ChurnModelPackageGroup",
    approval_status="PendingManualApproval",
)

# ── Step 5: Condition ────────────────────────────────────────────────────────
condition = ConditionGreaterThanOrEqualTo(
    left=JsonGet(
        step_name=evaluation_step.name,
        property_file=evaluation_report,
        json_path="classification_metrics.accuracy.value"
    ),
    right=0.75
)

condition_step = ConditionStep(
    name="CheckAccuracyThreshold",
    conditions=[condition],
    if_steps=[register_step],
    else_steps=[]
)

# ── Build and upsert pipeline ────────────────────────────────────────────────
pipeline = Pipeline(
    name=PIPELINE_NAME,
    steps=[processing_step, training_step, evaluation_step, condition_step],
    sagemaker_session=sagemaker_session,
)

if __name__ == "__main__":
    print("Upserting pipeline...")
    pipeline.upsert(role_arn=ROLE_ARN)
    print(f"Pipeline '{PIPELINE_NAME}' created/updated successfully.")
    print("Starting pipeline execution...")
    execution = pipeline.start()
    print(f"Execution ARN: {execution.arn}")
    print("Pipeline running! Monitor at:")
    print(f"https://ap-south-1.console.aws.amazon.com/sagemaker/home?region=ap-south-1#/pipelines")