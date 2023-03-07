import sagemaker
import boto3
import pandas as pd
import os

BUCKET = "circleci-sagemaker"

role = os.environ['SAGEMAKER_EXECUTION_ROLE']
aws_access_key_id = os.environ['AWS_ACCESS_KEY_ID']
print(aws_access_key_id)
aws_secret_access_key = os.environ['AWS_SECRET_ACCESS_KEY']
region_name = "us-east-1"
prefix = "xgboost-churn"


boto_session = boto3.Session(
    aws_access_key_id = aws_access_key_id,
    aws_secret_access_key = aws_secret_access_key,
    region_name = region_name
)
s3_client = boto_session.client(service_name="s3")

s3_client.download_file(f"sagemaker-sample-files", "datasets/tabular/synthetic/churn.txt", "churn.txt")
churn = pd.read_csv("./churn.txt")

churn.to_csv("train.csv", header=False, index=False)


s3_client.Bucket(BUCKET).Object(
    f"{prefix}/data/train.csv"
).upload_file("train.csv")

# boto3.Session().resource("s3").Bucket(bucket).Object(
#     os.path.join(prefix, "validation/validation.csv")
# ).upload_file("validation.csv")
