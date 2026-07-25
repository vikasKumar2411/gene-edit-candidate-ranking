"""Start a SageMaker Pipeline execution."""

from __future__ import annotations

import boto3

from gene_edit_ranking.config import load_config


def main() -> None:
    config = load_config()
    region = config["aws"]["region"]
    pipeline_name = config["sagemaker_pipeline"]["name"]

    client = boto3.client("sagemaker", region_name=region)

    response = client.start_pipeline_execution(
        PipelineName=pipeline_name,
        PipelineExecutionDisplayName="gene-edit-ranking-training-v1",
        PipelineParameters=[
            {
                "Name": "TrainingInstanceType",
                "Value": "ml.m5.large",
            },
            {
                "Name": "TrainingInstanceCount",
                "Value": "1",
            },
            {
                "Name": "ModelApprovalStatus",
                "Value": "PendingManualApproval",
            },
        ],
    )

    print("Pipeline execution started")
    print(
        "pipeline_execution_arn="
        f"{response['PipelineExecutionArn']}"
    )


if __name__ == "__main__":
    main()
