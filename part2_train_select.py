# part2_train_select.py
#Part 2 — Multi-Algorithm Training & Best Model Selection 
from sagemaker.estimator import Estimator
from sagemaker.workflow.steps import TrainingStep, ProcessingStep 
from sagemaker.processing import ScriptProcessor
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.conditions import ConditionGreaterThan
from sagemaker.workflow.functions import JsonGet

# Define algorithms (example: XGBoost, Linear Learner, Random Forest in sklearn)
xgb_estimator = Estimator(
    image_uri=sagemaker.image_uris.retrieve("xgboost", region, "1.5-1"),
    role=role,
    instance_count=1,
    instance_type="ml.m5.2xlarge",
    output_path=f"s3://{sess.default_bucket()}/output/xgb"
)

rf_estimator = Estimator(
    image_uri=sagemaker.image_uris.retrieve("sklearn", region),
    role=role,
    instance_count=1,
    instance_type="ml.m5.2xlarge",
    output_path=f"s3://{sess.default_bucket()}/output/rf"
)

ll_estimator = Estimator(
    image_uri=sagemaker.image_uris.retrieve("linear-learner", region),
    role=role,
    instance_count=1,
    instance_type="ml.m5.2xlarge",
    output_path=f"s3://{sess.default_bucket()}/output/ll"
)

# Training steps
xgb_train_step = TrainingStep(name="XGBTrainStep", estimator=xgb_estimator, inputs={"train": input_data})
rf_train_step = TrainingStep(name="RFTrainStep", estimator=rf_estimator, inputs={"train": input_data})
ll_train_step = TrainingStep(name="LLTrainStep", estimator=ll_estimator, inputs={"train": input_data})

# Evaluation step
eval_processor = ScriptProcessor(
    image_uri=sagemaker.image_uris.retrieve("sklearn", region),
    command=["python3"],
    role=role,
    instance_count=1,
    instance_type="ml.m5.xlarge"
)

eval_step = ProcessingStep(
    name="EvaluateModels",
    processor=eval_processor,
    inputs=[
        sagemaker.processing.ProcessingInput(source=xgb_train_step.properties.ModelArtifacts.S3ModelArtifacts, destination="/opt/ml/processing/xgb"),
        sagemaker.processing.ProcessingInput(source=rf_train_step.properties.ModelArtifacts.S3ModelArtifacts, destination="/opt/ml/processing/rf"),
        sagemaker.processing.ProcessingInput(source=ll_train_step.properties.ModelArtifacts.S3ModelArtifacts, destination="/opt/ml/processing/ll")
    ],
    outputs=[
        sagemaker.processing.ProcessingOutput(output_name="best_model", source="/opt/ml/processing/output")
    ],
    code="evaluate_models.py"
)

# Condition to check if retrain is required
cond_step = ConditionStep(
    name="CheckDriftCondition",
    conditions=[
        ConditionGreaterThan(
            left=JsonGet(step_name="DriftDetectionStep", property_file="drift_results", json_path="drift_score"),
            right=0.1
        )
    ],
    if_steps=[xgb_train_step, rf_train_step, ll_train_step, eval_step],
    else_steps=[]
)
