import boto3
import uuid
import json
from datetime import datetime

dynamodb = boto3.resource('dynamodb', region_name='ap-south-1')
table = dynamodb.Table('churn-predictions')
runtime = boto3.client('sagemaker-runtime', region_name='ap-south-1')

def predict_and_log(features: list):
    payload = ','.join(map(str, features))

    # Get prediction
    response = runtime.invoke_endpoint(
        EndpointName='churn-prediction-endpoint',
        ContentType='text/csv',
        Body=payload
    )
    probability = float(response['Body'].read().decode('utf-8'))
    prediction = 1 if probability > 0.5 else 0

    # Log to DynamoDB
    item = {
        'prediction_id': str(uuid.uuid4()),
        'timestamp': datetime.utcnow().isoformat(),
        'features': json.dumps(features),
        'probability': str(round(probability, 4)),
        'prediction': prediction,
        'endpoint': 'churn-prediction-endpoint'
    }
    table.put_item(Item=item)
    print(f"Prediction: {prediction} (prob: {probability:.4f}) — logged to DynamoDB")
    return item

if __name__ == "__main__":
    import pandas as pd

    df = pd.read_csv('data/output/test.csv', header=None)

    print("Running predictions on 10 test samples...")
    for i in range(10):
        features = df.iloc[i, 1:].tolist()
        predict_and_log(features)

    print("\nVerifying DynamoDB logs...")
    dynamodb_client = boto3.client('dynamodb', region_name='ap-south-1')
    result = dynamodb_client.scan(TableName='churn-predictions', Limit=5)
    print(f"Total items logged: {result['Count']}")