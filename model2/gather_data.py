import boto3
import pandas as pd
import numpy as np
import os
import io
from zipfile import ZipFile

bucket = os.environ["AWS_BUCKET"]
region_name = os.environ["AWS_REGION"]
model_name = os.environ["MODEL_NAME"]


# Set up the session and client we will need for this step
boto_session = boto3.Session(region_name=region_name)
s3_client = boto_session.client(service_name="s3")


# Data retrieval and processing taken from
# https://github.com/aws/amazon-sagemaker-examples/blob/main/introduction_to_applying_machine_learning/credit_card_fraud_detector/credit_card_fraud_detector.ipynb
# You would likely replace this part for your own use case, such as querying from Snowflake or Redshift
s3_client.download_file(f"sagemaker-solutions-prod", "Fraud-detection-using-machine-learning/3.4.1/data/creditcardfraud.zip", "creditcardfraud.zip")
with ZipFile("creditcardfraud.zip", "r") as zf:
    zf.extractall()
data = pd.read_csv("creditcard.csv", delimiter=",")

feature_columns = data.columns[:-1]
label_column = data.columns[-1]
features = data[feature_columns].values.astype("float32")
labels = (data[label_column].values).astype("float32")

model_data = pd.concat(
    [labels, features], axis=1
)

train_data, validation_data, test_data = np.split(
    model_data.sample(frac=1, random_state=1729),
    [int(0.7 * len(model_data)), int(0.9 * len(model_data))],
)


# Upload training and validation data to S3
csv_buffer = io.BytesIO()
train_data.to_csv(csv_buffer, index=False)
s3_client.put_object(Bucket=bucket, Body=csv_buffer.getvalue(), Key=f"{model_name}/train/train.csv")

csv_buffer = io.BytesIO()
validation_data.to_csv(csv_buffer, index=False)
s3_client.put_object(Bucket=bucket, Body=csv_buffer.getvalue(), Key=f"{model_name}/validation/validation.csv")
