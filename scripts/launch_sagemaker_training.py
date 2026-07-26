"""Launch the custom training container as a SageMaker Training Job."""

from __future__ import annotations

from datetime import UTC, datetime

import boto3

from gene_edit_ranking.config import load_config


def main() -> None:
    """Create one SageMaker Training Job."""

    config = load_config()

    region = config["aws"]["region"]
    bucket = config["aws"]["s3_bucket"]
    training_config = config["training"]
    gold_config = config["gold_dataset"]

    role_arn = (
        "arn:aws:iam::975050327570:"
        "role/AmazonSageMakerExecutionRole-GeneEditRanking"
    )

    image_uri = (
        "975050327570.dkr.ecr.us-east-1.amazonaws.com/"
        "gene-edit-ranking-training@"
        "sha256:f924d42357a512dbcc36f1b36c2ebea"
        "63c5a16f778e630082c569af912cb34ed"
    )

    input_s3_uri = (
        f"s3://{bucket}/gold/training/"
        f"dataset_version={gold_config['dataset_version']}/"
        f"processing_date={gold_config['processing_date']}/"
        "training_dataset.parquet"
    )

    output_s3_uri = (
        f"s3://{bucket}/models/sagemaker-training-output/"
    )

    timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")

    training_job_name = (
        f"gene-edit-ranking-{timestamp}"
    )

    client = boto3.client("sagemaker", region_name=region)

    response = client.create_training_job(
        TrainingJobName=training_job_name,
        RoleArn=role_arn,
        AlgorithmSpecification={
            "TrainingImage": image_uri,
            "TrainingInputMode": "File",
            "MetricDefinitions": [
                {
                    "Name": "validation:pr_auc",
                    "Regex": (
                        r"validation_pr_auc=([0-9\\.]+)"
                    ),
                },
                {
                    "Name": "test:pr_auc",
                    "Regex": (
                        r"test_pr_auc=([0-9\\.]+)"
                    ),
                },
                {
                    "Name": "test:roc_auc",
                    "Regex": (
                        r"test_roc_auc=([0-9\\.]+)"
                    ),
                },
                {
                    "Name": "test:f1",
                    "Regex": (
                        r"test_f1=([0-9\\.]+)"
                    ),
                },
            ],
        },
        InputDataConfig=[
            {
                "ChannelName": "training",
                "DataSource": {
                    "S3DataSource": {
                        "S3DataType": "S3Prefix",
                        "S3Uri": input_s3_uri,
                        "S3DataDistributionType": "FullyReplicated",
                    }
                },
                "ContentType": "application/vnd.apache.parquet",
                "InputMode": "File",
            }
        ],
        OutputDataConfig={
            "S3OutputPath": output_s3_uri,
        },
        ResourceConfig={
            "InstanceType": "ml.m5.large",
            "InstanceCount": 1,
            "VolumeSizeInGB": 10,
        },
        StoppingCondition={
            "MaxRuntimeInSeconds": 1800,
        },
        EnableNetworkIsolation=False,
        EnableManagedSpotTraining=False,
        Tags=[
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
            {
                "Key": "ModelName",
                "Value": training_config["model_name"],
            },
        ],
    )

    print("SageMaker training job submitted")
    print(f"training_job_name={training_job_name}")
    print(f"training_job_arn={response['TrainingJobArn']}")
    print(f"input={input_s3_uri}")
    print(f"output_prefix={output_s3_uri}")


if __name__ == "__main__":
    main()
