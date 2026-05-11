import boto3
import os
import json

sagemaker = boto3.client('sagemaker')

ENDPOINT_NAME = os.environ.get('ENDPOINT_NAME', 'churn-prediction-endpoint')
ENDPOINT_CONFIG_NAME = os.environ.get('ENDPOINT_CONFIG_NAME', 'churn-prediction-config')
MODEL_PACKAGE_GROUP = os.environ.get('MODEL_PACKAGE_GROUP', 'ChurnModelPackageGroup')

def lambda_handler(event, context):
    print("Event received:", json.dumps(event))

    # Get the latest approved model package
    response = sagemaker.list_model_packages(
        ModelPackageGroupName=MODEL_PACKAGE_GROUP,
        ModelApprovalStatus='Approved',
        SortBy='CreationTime',
        SortOrder='Descending',
        MaxResults=1
    )

    packages = response.get('ModelPackageSummaryList', [])
    if not packages:
        print("No approved model packages found.")
        return {'statusCode': 404, 'body': 'No approved model found'}

    model_package_arn = packages[0]['ModelPackageArn']
    print(f"Deploying model package: {model_package_arn}")

    # Create model
    model_name = f"churn-model-{context.aws_request_id[:8]}"
    sagemaker.create_model(
        ModelName=model_name,
        PrimaryContainer={
            'ModelPackageName': model_package_arn
        },
        ExecutionRoleArn=os.environ['SAGEMAKER_ROLE_ARN']
    )
    print(f"Model created: {model_name}")

    # Create endpoint config
    sagemaker.create_endpoint_config(
        EndpointConfigName=ENDPOINT_CONFIG_NAME,
        ProductionVariants=[{
            'VariantName': 'primary',
            'ModelName': model_name,
            'InitialInstanceCount': 1,
            'InstanceType': 'ml.t2.medium'
        }]
    )
    print(f"Endpoint config created: {ENDPOINT_CONFIG_NAME}")

    # Create or update endpoint
    try:
        sagemaker.describe_endpoint(EndpointName=ENDPOINT_NAME)
        # Endpoint exists - update it
        sagemaker.update_endpoint(
            EndpointName=ENDPOINT_NAME,
            EndpointConfigName=ENDPOINT_CONFIG_NAME
        )
        print(f"Endpoint updated: {ENDPOINT_NAME}")
    except sagemaker.exceptions.ClientError:
        # Endpoint doesn't exist - create it
        sagemaker.create_endpoint(
            EndpointName=ENDPOINT_NAME,
            EndpointConfigName=ENDPOINT_CONFIG_NAME
        )
        print(f"Endpoint created: {ENDPOINT_NAME}")

    return {
        'statusCode': 200,
        'body': json.dumps({
            'endpoint': ENDPOINT_NAME,
            'model_package': model_package_arn
        })
    }