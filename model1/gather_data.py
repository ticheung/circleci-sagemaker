import sagemaker
import boto3
import pandas as pd
import os

BUCKET = "circleci-sagemaker"

role = sagemaker.get_execution_role()
sess = sagemaker.Session()
prefix = "xgboost-churn"


boto_session = boto3.Session()
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
