"""
autotrain_and_version.py

Auto-retraining + model registry + canary deployment pipeline template for AWS SageMaker.

Pre-reqs:
- AWS credentials configured (role with SageMaker, S3, SNS privileges)
- sagemaker SDK installed: pip install sagemaker boto3
- A training script / container that saves model to S3 (model_dir) and writes evaluation metrics to a JSON in S3
"""

import os
import time
import json
import boto3
import sagemaker
from sagemaker.estimator import Estimator 
from sagemaker.model import Model
from sagemaker import Session
from datetime import datetime
from typing import Dict, Any

# ----------------------------
# CONFIG - customize these
# ----------------------------
AWS_REGION = "us-east-1"
S3_BUCKET = "manas-bucket100"
TRAINING_DATA_S3 = f"s3://{S3_BUCKET}/training_data/"
OUTPUT_S3 = f"s3://{S3_BUCKET}/model-artifacts/"
MODEL_PACKAGE_GROUP_NAME = "loan-fraud-models"        # registry group name
ROLE_ARN = "<YOUR_SAGEMAKER_EXECUTION_ROLE_ARN>"      # SageMaker execution role

# Training container / script config (fill in)
TRAINING_IMAGE = "<ECR_IMAGE_URI_OR_USE_SAGEMAKER_BUILTIN>"  # e.g., your custom container or builtin framework URI
ENTRY_POINT = "train.py"         # if using ScriptMode (container should support it)
HYPERPARAMS = {"epochs": "10", "batch_size": "64"}  # example

# SNS / Notification config
SNS_TOPIC_ARN = "arn:aws:sns:us-east-1:491085388405:ModelApprovalTopic"
APPROVAL_WAIT_SECONDS = 60 * 60 * 24  # how long to wait for human approval (24h)

# Validation thresholds (example)
MIN_AUC = 0.78              # candidate must meet/exceed this
MAX_FD_FEATURE_DRIFT = 0.4  # example custom threshold for feature drift (if you calculate)

# Endpoint config
PROD_ENDPOINT_NAME = "fraud-ml-endpoint-prod"
STAGING_ENDPOINT_NAME = "fraud-ml-endpoint-staging"

# ----------------------------
# clients & session
# ----------------------------
boto_sess = boto3.session.Session(region_name=AWS_REGION)
sm_client = boto_sess.client("sagemaker")
s3_client = boto_sess.client("s3")
sns_client = boto_sess.client("sns")
sagemaker_session = Session(boto_session=boto_sess)
sm_role = ROLE_ARN

# ----------------------------
# Utility functions
# ----------------------------
def timestamp():
    return datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")

def upload_json_to_s3(obj: Dict, s3_uri: str):
    """
    s3_uri: s3://bucket/path/file.json
    """
    assert s3_uri.startswith("s3://")
    parts = s3_uri[5:].split("/", 1)
    bucket = parts[0]
    key = parts[1]
    s3_client.put_object(Bucket=bucket, Key=key, Body=json.dumps(obj).encode("utf-8"))
    print(f"Uploaded JSON to {s3_uri}")

def download_json_from_s3(s3_uri: str) -> Dict:
    parts = s3_uri[5:].split("/", 1)
    bucket = parts[0]
    key = parts[1]
    resp = s3_client.get_object(Bucket=bucket, Key=key)
    return json.loads(resp["Body"].read().decode("utf-8"))

# ----------------------------
# Step 1: Train candidate model
# ----------------------------
def train_candidate_model(training_job_name: str, hyperparameters: Dict[str,str]=None) -> Dict[str, Any]:
    """
    Launches a SageMaker TrainingJob and returns training job description.
    This example uses Estimator with a custom image. Adjust as needed for built-in frameworks.
    """
    print("Starting training job:", training_job_name)
    estimator = Estimator(
        image_uri=TRAINING_IMAGE,
        role=sm_role,
        instance_count=1,
        instance_type="ml.m5.xlarge",   # change as needed
        volume_size=50,
        output_path=OUTPUT_S3,
        sagemaker_session=sagemaker_session,
        hyperparameters=hyperparameters or HYPERPARAMS,
    )

    # If you have a ScriptMode entry point, use estimator.fit({"train": TRAINING_DATA_S3}, job_name=training_job_name)
    estimator.fit({"train": TRAINING_DATA_S3}, job_name=training_job_name)
    # Wait for training to complete is handled by fit blocking call
    training_desc = sm_client.describe_training_job(TrainingJobName=training_job_name)
    print("Training job finished, model artifacts at:", training_desc["ModelArtifacts"]["S3ModelArtifacts"])
    return training_desc

# ----------------------------
# Step 2: Evaluate candidate
# ----------------------------
def evaluate_candidate(evaluation_s3_uri: str) -> Dict[str, Any]:
    """
    Expects your training job or downstream script to write eval metrics to a JSON file in S3.
    e.g., s3://.../model-artifacts/<training_job>/evaluation.json
    Structure example: {"auc":0.81, "precision":0.75, "recall":0.60}
    """
    metrics = download_json_from_s3(evaluation_s3_uri)
    print("Loaded evaluation metrics:", metrics)
    return metrics

def validation_suite(metrics: Dict[str, Any]) -> bool:
    """
    Simple validation: check primary metric meets threshold.
    Extend with fairness, explainability, resource tests, etc.
    """
    auc = metrics.get("auc") or metrics.get("AUC") or 0.0
    if auc >= MIN_AUC:
        print(f"Candidate AUC {auc} >= threshold {MIN_AUC} -> PASS")
        return True
    else:
        print(f"Candidate AUC {auc} < threshold {MIN_AUC} -> FAIL")
        return False

# ----------------------------
# Step 3: Register candidate in Model Registry
# ----------------------------
def ensure_model_package_group(group_name: str):
    try:
        sm_client.create_model_package_group(ModelPackageGroupName=group_name,
                                             ModelPackageGroupDescription="Loan fraud model group")
        print("Created model package group:", group_name)
    except sm_client.exceptions.ResourceInUse:
        print("Model package group already exists:", group_name)

def register_model_package(model_artifact_s3: str, inference_image: str, model_metrics: Dict[str,Any], group_name: str) -> str:
    """
    Creates a model package (unapproved) and returns model_package_arn.
    """
    timestamp_str = timestamp()
    package_name = f"{group_name}-{timestamp_str}"
    model_package_input = {
        "ModelPackageGroupName": group_name,
        "ModelPackageDescription": f"Candidate model created at {timestamp_str}",
        "InferenceSpecification": {
            "Containers": [
                {
                    "Image": inference_image,
                    "ModelDataUrl": model_artifact_s3
                }
            ],
            "SupportedContentTypes": ["application/json"],
            "SupportedResponseMIMETypes": ["application/json"]
        },
        "ModelMetrics": {
            "ModelQuality": {
                "Statistics": {
                    "ContentType": "application/json",
                    "S3Uri": ""  # optional: you may upload a metrics JSON and provide S3Uri
                }
            }
        },
        # Leave ModelApprovalStatus as 'PendingManualApproval' to require human approval
        "ModelApprovalStatus": "PendingManualApproval"
    }

    resp = sm_client.create_model_package(**model_package_input)
    model_package_arn = resp["ModelPackageArn"]
    print("Created model package:", model_package_arn)
    # Optionally tag the package with metrics
    sm_client.add_tags(ResourceArn=model_package_arn, Tags=[{"Key":k,"Value":str(v)} for k,v in model_metrics.items()])
    return model_package_arn

# ----------------------------
# Step 4: Notify approver (SNS)
# ----------------------------
def notify_human_for_approval(model_package_arn: str, evaluation_metrics: Dict[str,Any], s3_report_uri: str):
    """
    Publishes a message to SNS with links to artifacts and model_package_arn.
    Configure an email subscription to the SNS topic so approver receives mail with links.
    """
    subject = "Model Approval Request: Candidate ready for review"
    body = {
        "message": "A new candidate model is ready for manual approval.",
        "model_package_arn": model_package_arn,
        "evaluation_metrics": evaluation_metrics,
        "report_s3_uri": s3_report_uri
    }
    resp = sns_client.publish(TopicArn=SNS_TOPIC_ARN, Subject=subject, Message=json.dumps(body))
    print("Published SNS approval message:", resp.get("MessageId"))

# ----------------------------
# Step 5: Deploy candidate to staging (shadow/canary)
# ----------------------------
def deploy_candidate_to_staging(model_package_arn: str, staging_endpoint_name: str):
    """
    Creates a SageMaker Model from the model package and deploys to an endpoint (All traffic).
    For canary style you could create two variants in EndpointConfig with different weights.
    """
    # Create model from package
    model_name = f"candidate-model-{timestamp()}"
    create_resp = sm_client.create_model(
        ModelName=model_name,
        PrimaryContainer={"ModelPackageName": model_package_arn},
        ExecutionRoleArn=sm_role
    )
    print("Created model:", create_resp)

    # Create endpoint config
    endpoint_config_name = f"cfg-{model_name}"
    sm_client.create_endpoint_config(
        EndpointConfigName=endpoint_config_name,
        ProductionVariants=[
            {
                "VariantName": "AllTraffic",
                "ModelName": model_name,
                "InitialInstanceCount": 1,
                "InstanceType": "ml.m5.xlarge",
                "InitialVariantWeight": 1.0
            }
        ]
    )
    # Create or update endpoint
    try:
        sm_client.create_endpoint(EndpointName=staging_endpoint_name, EndpointConfigName=endpoint_config_name)
        sm_client.get_waiter("endpoint_in_service").wait(EndpointName=staging_endpoint_name)
        print("Staging endpoint created:", staging_endpoint_name)
    except sm_client.exceptions.ResourceLimitExceeded:
        print("Endpoint exists or resources limit. Consider updating existing endpoint config.")
    except sm_client.exceptions.ResourceInUse:
        print("Endpoint already exists. You may update it instead.")

# ----------------------------
# Step 6: Canary promotion
# ----------------------------
def promote_model_to_production(new_model_name: str, prod_endpoint_name: str, canary_percentage: float = 0.01):
    """
    Example: create an EndpointConfig with two variants: champion and challenger. Adjust weights to shift traffic gradually.
    This assumes the 'champion' model already exists as a model name; for brevity this is a template.
    """
    # Implementation details depend on how your production endpoint is configured (single model or variant)
    raise NotImplementedError("Implement promotion logic tailored to your endpoint strategy (variant weights).")

# ----------------------------
# Example orchestrator / main flow
# ----------------------------
def run_pipeline():
    # 1. Train
    training_job_name = f"loan-fraud-train-{timestamp()}"
    training_desc = train_candidate_model(training_job_name, hyperparameters=HYPERPARAMS)

    # 2. Post-train: assume training wrote evaluation to S3 in known location
    # Example evaluation file path convention:
    model_artifact_s3 = training_desc["ModelArtifacts"]["S3ModelArtifacts"]  # s3://bucket/path/to/artifact.tar.gz
    # assume evaluation metrics are written by your training script to: <artifact prefix>/evaluation.json
    artifact_prefix = model_artifact_s3.rsplit("/", 1)[0]
    evaluation_s3_uri = artifact_prefix + "/evaluation.json"

    # Wait a small period for eventual consistency (only if needed)
    time.sleep(5)

    # 3. Load evaluation (your training script must write it)
    try:
        eval_metrics = evaluate_candidate(evaluation_s3_uri)
    except Exception as e:
        print("Could not load evaluation metrics:", e)
        eval_metrics = {}

    # 4. Validation suite
    passed = validation_suite(eval_metrics)
    # Also add other checks here: fairness, latency test, resource checks, etc.

    if not passed:
        print("Candidate failed validation. Aborting registration.")
        # Optionally save report to S3 and notify team
        return

    # 5. Register in Model Registry (pending manual approval)
    ensure_model_package_group(MODEL_PACKAGE_GROUP_NAME)
    model_package_arn = register_model_package(model_artifact_s3, inference_image=TRAINING_IMAGE, model_metrics=eval_metrics, group_name=MODEL_PACKAGE_GROUP_NAME)

    # 6. Save a quick summary report to S3 for human inspection
    report_obj = {
        "training_job": training_job_name,
        "artifact": model_artifact_s3,
        "evaluation": eval_metrics,
        "model_package_arn": model_package_arn,
        "timestamp": timestamp()
    }
    report_s3_uri = f"{OUTPUT_S3}{training_job_name}/candidate_report.json"
    upload_json_to_s3(report_obj, report_s3_uri)

    # 7. Notify approver (SNS)
    notify_human_for_approval(model_package_arn, eval_metrics, report_s3_uri)

    print("Pipeline finished. Waiting for human approval to promote model to production.")

    # 8. Optionally implement a loop to poll for approval status (or use event-driven webhook)
    # Example polling (demo only; you may prefer manual UI to approve in the SageMaker Console)
    approved = wait_for_approval(model_package_arn, timeout_seconds=APPROVAL_WAIT_SECONDS)
    if approved:
        print("Model approved. You may now deploy candidate to staging or promote to production.")
        # optionally deploy_candidate_to_staging(model_package_arn, STAGING_ENDPOINT_NAME)
    else:
        print("Model not approved or timeout. Exiting.")

# ----------------------------
# Helper: wait for manual approval
# ----------------------------
def wait_for_approval(model_package_arn: str, timeout_seconds: int = 86400, poll_interval: int = 60) -> bool:
    """
    Polls the model package status in the registry until ModelApprovalStatus becomes 'Approved' or timeout.
    """
    start = time.time()
    while time.time() - start < timeout_seconds:
        resp = sm_client.describe_model_package(ModelPackageName=model_package_arn)
        status = resp.get("ModelApprovalStatus")
        print("Current approval status:", status)
        if status == "Approved":
            return True
        if status == "Rejected":
            return False
        time.sleep(poll_interval)
    return False

# ----------------------------
# If run as script
# ----------------------------
if __name__ == "__main__":
    run_pipeline()
