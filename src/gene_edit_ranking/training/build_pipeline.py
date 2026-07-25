"""Build and optionally upsert the SageMaker training pipeline."""

from __future__ import annotations

import argparse
from typing import Any

import boto3
from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput
from sagemaker.workflow.parameters import (
    ParameterInteger,
    ParameterString,
)
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.steps import TrainingStep

from gene_edit_ranking.config import load_config


def build_pipeline(config: dict[str, Any]) -> Pipeline:
    """Build the training-only SageMaker Pipeline."""

    region = config["aws"]["region"]
    bucket = config["aws"]["s3_bucket"]

    pipeline_config = config["sagemaker_pipeline"]
    training_config = pipeline_config["training"]
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

    model_approval_status = ParameterString(
        name="ModelApprovalStatus",
        default_value=pipeline_config["default_approval_status"],
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
        tags=[
            {
                "Key": "Project",
                "Value": "GeneEditCandidateRanking",
            },
            {
                "Key": "Environment",
                "Value": "dev",
            },
            {
                "Key": "SyntheticData",
                "Value": "true",
            },
        ],
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

    return Pipeline(
        name=pipeline_config["name"],
        parameters=[
            input_data_uri,
            training_instance_type,
            training_instance_count,
            model_approval_status,
        ],
        steps=[training_step],
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
                "Synthetic gene-edit ranking training pipeline."
            ),
        )

        print("SageMaker Pipeline upsert: SUCCESS")
        print(f"pipeline_name={pipeline.name}")
        print(f"pipeline_arn={response['PipelineArn']}")


if __name__ == "__main__":
    main()
