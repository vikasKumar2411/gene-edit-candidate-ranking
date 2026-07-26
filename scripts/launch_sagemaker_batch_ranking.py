"""Launch the batch-ranking container as a SageMaker Processing job."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime

import boto3
from sagemaker.processing import Processor

from gene_edit_ranking.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input-s3-uri",
        required=True,
    )

    parser.add_argument(
        "--scoring-date",
        default=datetime.now(UTC).date().isoformat(),
    )

    return parser.parse_args()


def resolve_batch_image(
    *,
    repository_name: str,
    tag: str,
    account_id: str,
    region: str,
) -> str:
    """Resolve an ECR tag to an immutable digest URI."""

    client = boto3.client("ecr", region_name=region)

    response = client.describe_images(
        repositoryName=repository_name,
        imageIds=[{"imageTag": tag}],
    )

    digest = response["imageDetails"][0]["imageDigest"]

    return (
        f"{account_id}.dkr.ecr.{region}.amazonaws.com/"
        f"{repository_name}@{digest}"
    )


def main() -> None:
    args = parse_args()
    config = load_config()

    region = config["aws"]["region"]
    bucket = config["aws"]["s3_bucket"]

    pipeline_config = config["sagemaker_pipeline"]
    role_arn = pipeline_config["role_arn"]

    account_id = boto3.client(
        "sts",
        region_name=region,
    ).get_caller_identity()["Account"]

    image_uri = resolve_batch_image(
        repository_name="gene-edit-ranking-batch",
        tag="v1",
        account_id=account_id,
        region=region,
    )

    processor = Processor(
        image_uri=image_uri,
        role=role_arn,
        instance_count=1,
        instance_type="ml.m5.large",
        volume_size_in_gb=10,
        max_runtime_in_seconds=1800,
        base_job_name="gene-edit-ranking-batch",
    )

    processor.run(
        arguments=[
            "--input-s3-uri",
            args.input_s3_uri,
            "--scoring-date",
            args.scoring_date,
        ],
        wait=False,
        logs=False,
    )

    job_name = processor.latest_job.name

    print("SageMaker batch-ranking job started")
    print(f"job_name={job_name}")
    print(f"image_uri={image_uri}")
    print(f"input_s3_uri={args.input_s3_uri}")
    print(f"scoring_date={args.scoring_date}")
    print(
        "expected_output="
        f"s3://{bucket}/gold/predictions/"
        f"model_version=2/"
        f"scoring_date={args.scoring_date}/"
    )


if __name__ == "__main__":
    main()
