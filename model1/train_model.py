import sagemaker
from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput
import boto3
import pandas as pd
import numpy as np
import os
import io

bucket = os.environ["AWS_BUCKET"]
region_name = os.environ["AWS_REGION"]
model_name = os.environ["MODEL_NAME"]
role_arn = os.environ["SAGEMAKER_EXECUTION_ROLE_ARN"]


# Set up the session and client we will need for this step
boto_session = boto3.Session(region_name=region_name)
sagemaker_client = boto_session.client(service_name="sagemaker")
sagemaker_runtime_client = boto_session.client(service_name="sagemaker-runtime")
sagemaker_session = sagemaker.Session(
    boto_session = boto_session,
    sagemaker_client = sagemaker_client,
    sagemaker_runtime_client = sagemaker_runtime_client,
    default_bucket = bucket
)
# s3_client = boto_session.client(service_name="s3")

# Set up dataset locations
train_set_location = f"s3://{bucket}/{model_name}/train/"
validation_set_location = f"s3://{bucket}/{model_name}/validation/"
model_location = f"s3://{bucket}/{model_name}/model/"

train_set_pointer = TrainingInput(s3_data=train_set_location, content_type='csv')
validation_set_pointer = TrainingInput(s3_data=validation_set_location, content_type='csv')

# Retrieve xgboost image
image_uri = sagemaker.image_uris.retrieve(
    framework = "xgboost",
    region = region_name,
    version = "1.5-1"
)

# Configure training estimator
xgb_estimator = Estimator(
    base_job_name = model_name,
    image_uri = image_uri,
    instance_type = "ml.m5.large",
    instance_count = 1,
    output_path = model_location,
    sagemaker_session = sagemaker_session,
    role = role_arn,
    hyperparameters = {
        "objective": "binary:logistic",
        "max_depth": 5,
        "eta": 0.2,
        "gamma": 4,
        "min_child_weight": 6,
        "subsample": 0.8,
        "verbosity": 0,
        "num_round": 100,
    }
)

xgb_estimator.fit({"train": train_set_pointer, "validation": validation_set_pointer})

training_job_name = xgb_estimator.latest_training_job.job_name
print("training_job_name:", training_job_name)