# CICD-ECR-Sagemaker

This repo contains a machine learning model and Docker setup for deployment using AWS CodePipeline and CodeBuild.

## Structure

- `model/` : Contains trained model files
- `inference.py` : Script for making predictions
- `Dockerfile` : For building the container
- `buildspec.yml` : For AWS CodeBuild
- `requirements.txt` : Python dependencies
