"""Build and optionally upsert the SageMaker training pipeline."""

from __future__ import annotations

import argparse
from typing import Any

import boto3
from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput
from sagemaker.model import Model
from sagemaker.model_metrics import MetricsSource, ModelMetrics
from sagemaker.processing import ProcessingInput, ProcessingOutput, Processor
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.conditions import (
    ConditionGreaterThanOrEqualTo,
)
from sagemaker.workflow.fail_step import FailStep
from sagemaker.workflow.functions import JsonGet, Join
from sagemaker.workflow.parameters import (
    ParameterFloat,
    ParameterInteger,
    ParameterString,
)
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.model_step import ModelStep
from sagemaker.workflow.steps import ProcessingStep, TrainingStep

from gene_edit_ranking.config import load_config


def build_pipeline(config: dict[str, Any]) -> Pipeline:
    """Build a SageMaker Pipeline with training and evaluation steps."""

    region = config["aws"]["region"]
    bucket = config["aws"]["s3_bucket"]

    pipeline_config = config["sagemaker_pipeline"]
    training_config = pipeline_config["training"]
    evaluation_config = pipeline_config["evaluation"]
    inference_config = pipeline_config["inference"]
    gold_config = config["gold_dataset"]

    boto_session = boto3.Session(region_name=region)

    pipeline_session = PipelineSession(
        boto_session=boto_session,
        default_bucket=bucket,
    )

    input_data_uri = ParameterString(
        name="GoldDatasetS3Uri",
        default_value=(
            f"s3://{bucket}/gold/training/"
            f"dataset_version={gold_config['dataset_version']}/"
            f"processing_date={gold_config['processing_date']}/"
            "training_dataset.parquet"
        ),
    )

    training_instance_type = ParameterString(
        name="TrainingInstanceType",
        default_value=training_config["instance_type"],
    )

    training_instance_count = ParameterInteger(
        name="TrainingInstanceCount",
        default_value=training_config["instance_count"],
    )

    evaluation_instance_type = ParameterString(
        name="EvaluationInstanceType",
        default_value=evaluation_config["instance_type"],
    )

    model_approval_status = ParameterString(
        name="ModelApprovalStatus",
        default_value=pipeline_config["default_approval_status"],
    )

    minimum_test_pr_auc = ParameterFloat(
        name="MinimumTestPRAUC",
        default_value=pipeline_config["quality_gate"][
            "minimum_test_pr_auc"
        ],
    )

    estimator = Estimator(
        image_uri=training_config["image_uri"],
        role=pipeline_config["role_arn"],
        instance_count=training_instance_count,
        instance_type=training_instance_type,
        volume_size=training_config["volume_size_gb"],
        max_run=training_config["max_runtime_seconds"],
        output_path=(
            f"s3://{bucket}/models/sagemaker-pipeline-output/"
        ),
        sagemaker_session=pipeline_session,
        base_job_name="gene-edit-ranking-pipeline-train",
        metric_definitions=[
            {
                "Name": "validation:pr_auc",
                "Regex": r"validation_pr_auc=([0-9\\.]+)",
            },
            {
                "Name": "test:pr_auc",
                "Regex": r"test_pr_auc=([0-9\\.]+)",
            },
            {
                "Name": "test:roc_auc",
                "Regex": r"test_roc_auc=([0-9\\.]+)",
            },
            {
                "Name": "test:f1",
                "Regex": r"test_f1=([0-9\\.]+)",
            },
        ],
        disable_profiler=True,
        debugger_hook_config=False,
        enable_sagemaker_metrics=True,
    )

    train_args = estimator.fit(
        inputs={
            "training": TrainingInput(
                s3_data=input_data_uri,
                content_type="application/vnd.apache.parquet",
                input_mode="File",
            )
        }
    )

    training_step = TrainingStep(
        name="TrainEditSuccessModel",
        step_args=train_args,
    )

    evaluation_processor = Processor(
        image_uri=evaluation_config["image_uri"],
        role=pipeline_config["role_arn"],
        instance_count=evaluation_config["instance_count"],
        instance_type=evaluation_instance_type,
        volume_size_in_gb=evaluation_config["volume_size_gb"],
        max_runtime_in_seconds=evaluation_config[
            "max_runtime_seconds"
        ],
        sagemaker_session=pipeline_session,
        base_job_name="gene-edit-ranking-evaluate",
    )

    evaluation_args = evaluation_processor.run(
        inputs=[
            ProcessingInput(
                source=training_step.properties.ModelArtifacts.S3ModelArtifacts,
                destination="/opt/ml/processing/model",
                input_name="model",
            ),
            ProcessingInput(
                source=input_data_uri,
                destination="/opt/ml/processing/data",
                input_name="data",
            ),
        ],
        outputs=[
            ProcessingOutput(
                output_name="evaluation",
                source="/opt/ml/processing/evaluation",
                destination=(
                    f"s3://{bucket}/monitoring/model-evaluation/"
                ),
            )
        ],
    )

    evaluation_report = PropertyFile(
        name="EvaluationReport",
        output_name="evaluation",
        path="evaluation.json",
    )

    evaluation_step = ProcessingStep(
        name="EvaluateEditSuccessModel",
        step_args=evaluation_args,
        property_files=[evaluation_report],
    )

    evaluation_s3_uri = Join(
        on="",
        values=[
            evaluation_step.properties.ProcessingOutputConfig.Outputs[
                "evaluation"
            ].S3Output.S3Uri,
            "evaluation.json",
        ],
    )

    model_metrics = ModelMetrics(
        model_statistics=MetricsSource(
            s3_uri=evaluation_s3_uri,
            content_type="application/json",
        )
    )

    inference_model = Model(
        image_uri=inference_config["image_uri"],
        model_data=(
            training_step.properties.ModelArtifacts.S3ModelArtifacts
        ),
        role=pipeline_config["role_arn"],
        sagemaker_session=pipeline_session,
        name="gene-edit-ranking-edit-success",
    )

    register_args = inference_model.register(
        content_types=inference_config["content_types"],
        response_types=inference_config["response_types"],
        inference_instances=inference_config[
            "inference_instances"
        ],
        transform_instances=inference_config[
            "transform_instances"
        ],
        model_package_group_name=pipeline_config[
            "model_package_group_name"
        ],
        approval_status=model_approval_status,
        model_metrics=model_metrics,
        description=(
            "Synthetic gene-edit success model. "
            "Portfolio demonstration only."
        ),
    )

    register_model_step = ModelStep(
        name="RegisterEditSuccessModel",
        step_args=register_args,
    )

    pr_auc_condition = ConditionGreaterThanOrEqualTo(
        left=JsonGet(
            step_name=evaluation_step.name,
            property_file=evaluation_report,
            json_path="quality_gate_metric.value",
        ),
        right=minimum_test_pr_auc,
    )

    quality_gate_failure = FailStep(
        name="FailModelQualityGate",
        error_message=Join(
            on="",
            values=[
                "Model failed quality gate. Required test PR-AUC: ",
                minimum_test_pr_auc,
            ],
        ),
    )

    quality_gate_step = ConditionStep(
        name="CheckModelQuality",
        conditions=[pr_auc_condition],
        if_steps=[register_model_step],
        else_steps=[quality_gate_failure],
    )

    return Pipeline(
        name=pipeline_config["name"],
        parameters=[
            input_data_uri,
            training_instance_type,
            training_instance_count,
            evaluation_instance_type,
            model_approval_status,
            minimum_test_pr_auc,
        ],
        steps=[
            training_step,
            evaluation_step,
            quality_gate_step,
        ],
        sagemaker_session=pipeline_session,
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--upsert",
        action="store_true",
        help="Create or update the SageMaker Pipeline.",
    )

    parser.add_argument(
        "--print-definition",
        action="store_true",
        help="Print the generated pipeline JSON definition.",
    )

    return parser.parse_args()


def main() -> None:
    """Build and optionally upsert the pipeline."""

    args = parse_args()
    config = load_config()
    pipeline = build_pipeline(config)

    if args.print_definition:
        print(pipeline.definition())

    if args.upsert:
        response = pipeline.upsert(
            role_arn=config["sagemaker_pipeline"]["role_arn"],
            description=(
                "Synthetic gene-edit ranking training and "
                "evaluation pipeline."
            ),
        )

        print("SageMaker Pipeline upsert: SUCCESS")
        print(f"pipeline_name={pipeline.name}")
        print(f"pipeline_arn={response['PipelineArn']}")


if __name__ == "__main__":
    main()
