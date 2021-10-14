"""Example workflow pipeline script for abalone pipeline.
                                               . -RegisterModel
                                              .
    Process-> Train -> Evaluate -> Condition .
                                              .
                                               . -(stop)
Implements a get_pipeline(**kwargs) method.
"""
import os

import boto3
import sagemaker
import sagemaker.session

from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput
from sagemaker.model_metrics import (
    MetricsSource,
    ModelMetrics,
)
from sagemaker.processing import (
    ProcessingInput,
    ProcessingOutput,
    ScriptProcessor,
)
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.huggingface import HuggingFace
from sagemaker.workflow.conditions import ConditionLessThanOrEqualTo, ConditionGreaterThanOrEqualTo
from sagemaker.workflow.condition_step import (
    ConditionStep,
    JsonGet,
)
from sagemaker.workflow.parameters import (
    ParameterInteger,
    ParameterString,
)
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.steps import (
    ProcessingStep,
    TrainingStep,
)
from sagemaker.workflow.step_collections import RegisterModel


BASE_DIR = os.path.dirname(os.path.realpath(__file__))

def get_sagemaker_client(region):
     """Gets the sagemaker client.
        Args:
            region: the aws region to start the session
            default_bucket: the bucket to use for storing the artifacts
        Returns:
            `sagemaker.session.Session instance
        """
    boto_session = boto3.Session(region_name=region)
    sagemaker_client = boto_session.client("sagemaker")
    return sagemaker_client


def get_session(region, default_bucket):
    """Gets the sagemaker session based on the region.
    Args:
        region: the aws region to start the session
        default_bucket: the bucket to use for storing the artifacts
    Returns:
        `sagemaker.session.Session instance
    """

    boto_session = boto3.Session(region_name=region)

    sagemaker_client = boto_session.client("sagemaker")
    runtime_client = boto_session.client("sagemaker-runtime")
    return sagemaker.session.Session(
        boto_session=boto_session,
        sagemaker_client=sagemaker_client,
        sagemaker_runtime_client=runtime_client,
        default_bucket=default_bucket,
    )

def get_pipeline_custom_tags(new_tags, region, sagemaker_project_arn=None):
    try:
        sm_client = get_sagemaker_client(region)
        response = sm_client.list_tags(
            ResourceArn=sagemaker_project_arn)
        project_tags = response["Tags"]
        for project_tag in project_tags:
            new_tags.append(project_tag)
    except Exception as e:
        print(f"Error getting project tags: {e}")
    return new_tags


def get_pipeline(
    region,
    sagemaker_project_arn=None,
    role=None,
    default_bucket=None,
    model_package_group_name="ADEPackageGroup",
    pipeline_name="ADEPipeline",
    base_job_prefix="ADE",
):
    """Gets a SageMaker ML Pipeline instance working with on abalone data.
    Args:
        region: AWS region to create and run the pipeline.
        role: IAM role to create and run steps and pipeline.
        default_bucket: the bucket to use for storing the artifacts
    Returns:
        an instance of a pipeline
    """
    sagemaker_session = get_session(region, default_bucket)
    if role is None:
        role = sagemaker.session.get_execution_role(sagemaker_session)

    # parameters for pipeline execution
    processing_instance_count = ParameterInteger(name="ProcessingInstanceCount", default_value=1)
    processing_instance_type = ParameterString(
        name="ProcessingInstanceType", default_value="ml.m5.xlarge"
    )
    training_instance_type = ParameterString(
        name="TrainingInstanceType", default_value="ml.p3.2xlarge"
    )
    model_approval_status = ParameterString(
        name="ModelApprovalStatus", default_value="PendingManualApproval"
    )
    
    training_image_uri = sagemaker.image_uris.retrieve(
        framework="huggingface",
        region=region,
        version="4.11.0",
        instance_type=training_instance_type,
        image_scope='training',
        base_framework_version='pytorch1.9.0'
    )
    
    
 # ================================= Pre-Processing DataSet =================================

    script_preprocess = ScriptProcessor(
        image_uri=training_image_uri,
        command=["python3"],
        instance_type=training_instance_type,
        instance_count=processing_instance_count,
        base_job_name=f"{base_job_prefix}/script-ade-preprocess",
        sagemaker_session=sagemaker_session,
        role=role,
    )
    
    
    step_process = ProcessingStep(
        name="PreprocessADEData",
        processor=script_preprocess,
        outputs=[
            ProcessingOutput(output_name="train", source="/opt/ml/processing/train"),
            ProcessingOutput(output_name="validation", source="/opt/ml/processing/validation"),
            ProcessingOutput(output_name="test", source="/opt/ml/processing/test"),
        ],
        code=os.path.join(BASE_DIR, "preprocess.py"),
        job_arguments = ["--dataset-name", "ade_corpus_v2",
                         "--datasubset-name", "Ade_corpus_v2_classification",
                         "--model-name", "distilbert-base-uncased",
                         "--train-ratio", "0.7",
                         "--val-ratio", "0.15"],

    )
    

    
 # ================================= Training the model =================================

    hyperparameters={'epochs': 1,
                     'train_batch_size': 32,
                     'model_name':'distilbert-base-uncased'
                     }
    
    hf_estimator = HuggingFace(
        entry_point='train.py',
        source_dir=BASE_DIR,
        instance_type='ml.p3.2xlarge',
        instance_count=1,
        role=role,
        transformers_version='4.11.0',
        pytorch_version='1.9.0',
        py_version='py38',
        output_path=f's3://{default_bucket}/{base_job_prefix}/training_output/',
        base_job_name="az-ade-training",
        hyperparameters=hyperparameters,
        disable_profiler=True,
)

    step_train = TrainingStep(
        name="TrainADEModel",
        estimator=hf_estimator,
        inputs={
            "train": TrainingInput(
                s3_data=step_process.properties.ProcessingOutputConfig.Outputs[
                    "train"
                ].S3Output.S3Uri
            ),
            "validation": TrainingInput(
                s3_data=step_process.properties.ProcessingOutputConfig.Outputs[
                    "validation"
                ].S3Output.S3Uri
            ),
        },
    )

    
    
 # ================================= Evaluating the model on the test set =================================
    
    
    script_eval = ScriptProcessor(
        image_uri=training_image_uri,
        command=["python3"],
        instance_type=training_instance_type,
        instance_count=1,
        base_job_name=f"{default_bucket}/script-ade-eval",
        sagemaker_session=sagemaker_session,
        role=role,
    )

    evaluation_report = PropertyFile(
        name="ADEEvaluationReport",
        output_name="evaluation",
        path="evaluation.json",
    )

    step_eval = ProcessingStep(
        name="EvaluateADEModel",
        processor=script_eval,
            inputs=[
                ProcessingInput(
                    source=step_train.properties.ModelArtifacts.S3ModelArtifacts,
                    destination="/opt/ml/processing/model",
                ),
                ProcessingInput(
                    source=step_process.properties.ProcessingOutputConfig.Outputs[
                        "test"
                    ].S3Output.S3Uri,
                    destination="/opt/ml/processing/test",
                ),
            ],
            outputs=[
                ProcessingOutput(output_name="evaluation", source="/opt/ml/processing/evaluation"),
            ],

        code=os.path.join(BASE_DIR, "evaluate.py"),
        property_files=[evaluation_report],
    )
    
    
 # ================================= Registering the Model =================================

    inference_image_uri = sagemaker.image_uris.retrieve(
        framework="huggingface",
        region=region,
        version="4.11.0",
        instance_type='ml.m5.xlarge',
        image_scope='inference',
        base_framework_version='pytorch1.9.0'
    )

    model_metrics = ModelMetrics(
        model_statistics=MetricsSource(
            s3_uri="{}/evaluation.json".format(
                step_eval.arguments["ProcessingOutputConfig"]["Outputs"][0]["S3Output"]["S3Uri"]
            ),
            content_type="application/json"
        )
    )
    
    step_register = RegisterModel(
        name="RegisterADEModel",
        estimator=hf_estimator,
        image_uri=inference_image_uri,
        model_data=step_train.properties.ModelArtifacts.S3ModelArtifacts,
        content_types=["text/csv"],
        response_types=["text/csv"],
        inference_instances=["ml.m5.xlarge","ml.m5.2xlarge"],
        transform_instances=["ml.m5.xlarge"],
        model_package_group_name=model_package_group_name,
        approval_status=model_approval_status,
        model_metrics=model_metrics,
    )

    # condition step for evaluating model quality and branching execution
    cond_gte = ConditionGreaterThanOrEqualTo(
        left=JsonGet(
            step=step_eval,
            property_file=evaluation_report,
            json_path="eval_f1"
        ),
        right=0.6,
    )

    step_cond = ConditionStep(
        name="CheckF1ADEEvaluation",
        conditions=[cond_gte],
        if_steps=[step_register],
        else_steps=[],
    )
    
    # pipeline instance
    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[
            processing_instance_type,
            processing_instance_count,
            training_instance_type,
            model_approval_status,
        ],
        steps=[step_process, step_train, step_eval, step_cond],
#         steps=[step_process,step_train, step_eval],
#         steps = [step_eval],
        sagemaker_session=sagemaker_session,
    )
    
    
    return pipeline
