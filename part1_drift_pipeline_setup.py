# part1_drift_pipeline_setup.py
#Part 1 — Pipeline Setup & Drift Detection
import sagemaker 
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.parameters import ParameterString 
from sagemaker.workflow.steps import ProcessingStep
from sagemaker.processing import ScriptProcessor
from sagemaker.session import Session

region = sagemaker.Session().boto_region_name 
role = "<SAGEMAKER_ROLE_ARN>"
sess = sagemaker.Session() 

# Pipeline parameters
input_data = ParameterString(name="InputData", default_value="s3://your-bucket/data/train/")
baseline_data = ParameterString(name="BaselineData", default_value="s3://your-bucket/data/baseline/")

# Drift detection step
drift_processor = ScriptProcessor(
    image_uri=sagemaker.image_uris.retrieve("sklearn", region),
    command=["python3"],
    role=role,
    instance_count=1,
    instance_type="ml.m5.xlarge"
)

drift_step = ProcessingStep(
    name="DriftDetectionStep",
    processor=drift_processor,
    inputs=[
        sagemaker.processing.ProcessingInput(source=baseline_data, destination="/opt/ml/processing/baseline"),
        sagemaker.processing.ProcessingInput(source=input_data, destination="/opt/ml/processing/input")
    ],
    outputs=[
        sagemaker.processing.ProcessingOutput(output_name="drift_results", source="/opt/ml/processing/output")
    ],
    code="detect_drift.py"
)
