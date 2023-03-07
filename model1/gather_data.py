import sagemaker
import boto3
import pandas as pd
import os
import io

BUCKET = "circleci-sagemaker"

role = os.environ['SAGEMAKER_EXECUTION_ROLE']
aws_access_key_id = os.environ['AWS_ACCESS_KEY_ID']
aws_secret_access_key = os.environ['AWS_SECRET_ACCESS_KEY']
region_name = "us-east-1"
model_name = "xgboost-churn"


boto_session = boto3.Session(
    aws_access_key_id = aws_access_key_id,
    aws_secret_access_key = aws_secret_access_key,
    region_name = region_name
)
s3_client = boto_session.client(service_name="s3")

s3_client.download_file(f"sagemaker-sample-files", "datasets/tabular/synthetic/churn.txt", "churn.txt")
churn = pd.read_csv("./churn.txt")



csv_buffer = io.BytesIO()
churn.to_csv(csv_buffer, index=False)

s3_client.put_object(Bucket=BUCKET, Body=csv_buffer.getvalue(), Key=f"{model_name}/data/train.csv")

