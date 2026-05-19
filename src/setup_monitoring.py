import boto3

cloudwatch = boto3.client('cloudwatch', region_name='ap-south-1')
sns = boto3.client('sns', region_name='ap-south-1')

# Create SNS topic for alerts
print("Creating SNS topic...")
topic = sns.create_topic(Name='churn-model-alerts')
topic_arn = topic['TopicArn']
print(f"SNS topic: {topic_arn}")

# Subscribe your email to alerts
EMAIL = "udayakrishna24@gmail.com"  # ← replace with your email
sns.subscribe(
    TopicArn=topic_arn,
    Protocol='email',
    Endpoint=EMAIL
)
print(f"Subscription confirmation sent to {EMAIL}")

# Alarm 1: High churn rate (model predicting too many churns)
print("Creating CloudWatch alarms...")
cloudwatch.put_metric_alarm(
    AlarmName='churn-high-prediction-rate',
    AlarmDescription='Triggers when churn prediction rate exceeds 50%',
    MetricName='Invocations',
    Namespace='AWS/SageMaker',
    Statistic='Sum',
    Dimensions=[{
        'Name': 'EndpointName',
        'Value': 'churn-prediction-endpoint'
    }],
    Period=300,
    EvaluationPeriods=1,
    Threshold=100,
    ComparisonOperator='GreaterThanThreshold',
    AlarmActions=[topic_arn],
    TreatMissingData='notBreaching'
)
print("Alarm 1 created: high invocation rate")

# Alarm 2: Endpoint errors
cloudwatch.put_metric_alarm(
    AlarmName='churn-endpoint-errors',
    AlarmDescription='Triggers when endpoint returns errors',
    MetricName='ModelError',
    Namespace='AWS/SageMaker',
    Statistic='Sum',
    Dimensions=[{
        'Name': 'EndpointName',
        'Value': 'churn-prediction-endpoint'
    }],
    Period=60,
    EvaluationPeriods=1,
    Threshold=1,
    ComparisonOperator='GreaterThanOrEqualToThreshold',
    AlarmActions=[topic_arn],
    TreatMissingData='notBreaching'
)
print("Alarm 2 created: endpoint errors")

# Alarm 3: High latency
cloudwatch.put_metric_alarm(
    AlarmName='churn-high-latency',
    AlarmDescription='Triggers when model latency exceeds 1 second',
    MetricName='ModelLatency',
    Namespace='AWS/SageMaker',
    Statistic='Average',
    Dimensions=[{
        'Name': 'EndpointName',
        'Value': 'churn-prediction-endpoint'
    }],
    Period=300,
    EvaluationPeriods=1,
    Threshold=1000,
    ComparisonOperator='GreaterThanThreshold',
    AlarmActions=[topic_arn],
    TreatMissingData='notBreaching'
)
print("Alarm 3 created: high latency")

print("\nAll monitoring set up successfully!")
print(f"Topic ARN: {topic_arn}")
print("Check your email to confirm the SNS subscription.")