from sagemaker.workflow.model_step import ModelStep
from sagemaker.model import Model
from sagemaker.workflow.steps import RegisterModel 

best_model = Model(
    image_uri=ecr_uri,
    model_data="s3://your-bucket/best-model/model.tar.gz",
    role=role
)

register_step = RegisterModel(
    name="RegisterBestModel",
    estimator=None,  # If using BYO container, pass Model instead 
    model=best_model,
    content_types=["text/csv"],
    response_types=["text/csv"],
    inference_instances=["ml.m5.large"],
    transform_instances=["ml.m5.large"],
    model_package_group_name="MultiAlgoModels",
    approval_status="PendingManualApproval"
)
