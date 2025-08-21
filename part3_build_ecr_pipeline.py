# part3_build_ecr_pipeline.py
Part 3 — Build & Push Docker to ECR + Final Pipeline 
import subprocess

# Docker/ECR config
account_id = boto3.client("sts").get_caller_identity()["Account"]
ecr_repo = "custom-model-repo"
ecr_uri = f"{account_id}.dkr.ecr.{region}.amazonaws.com/{ecr_repo}:latest"

# Build and push to ECR
subprocess.run(["aws", "ecr", "create-repository", "--repository-name", ecr_repo, "--region", region])
subprocess.run(["docker", "build", "-t", ecr_uri, "."])
subprocess.run(["aws", "ecr", "get-login-password", "--region", region, "|", "docker", "login", "--username", "AWS", "--password-stdin", f"{account_id}.dkr.ecr.{region}.amazonaws.com"])
subprocess.run(["docker", "push", ecr_uri]) 

# Final pipeline
pipeline = Pipeline(
    name="DriftDetectionMultiAlgoPipeline",
    steps=[drift_step, cond_step]
)

pipeline.upsert(role_arn=role)
execution = pipeline.start(parameters={
    "InputData": "s3://your-bucket/data/train/",
    "BaselineData": "s3://your-bucket/data/baseline/"
})
