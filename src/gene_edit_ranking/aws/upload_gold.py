"""Upload the validated Gold training dataset to Amazon S3."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import boto3
from botocore.exceptions import BotoCoreError, ClientError

from gene_edit_ranking.config import PROJECT_ROOT, load_config
from gene_edit_ranking.validation.validate_gold import run_validation


def calculate_sha256(file_path: Path) -> str:
    """Calculate the SHA-256 checksum of a file."""

    digest = hashlib.sha256()

    with file_path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def build_local_path(
    dataset_version: str,
    processing_date: str,
) -> Path:
    """Build the local Gold dataset path."""

    return (
        PROJECT_ROOT
        / "data"
        / "gold"
        / "training"
        / f"dataset_version={dataset_version}"
        / f"processing_date={processing_date}"
        / "training_dataset.parquet"
    )


def build_s3_key(
    gold_prefix: str,
    dataset_version: str,
    processing_date: str,
) -> str:
    """Build the partitioned Gold S3 key."""

    return (
        f"{gold_prefix}/training/"
        f"dataset_version={dataset_version}/"
        f"processing_date={processing_date}/"
        "training_dataset.parquet"
    )


def validate_before_upload() -> None:
    """Cancel upload when Gold validation fails."""

    results = run_validation()
    failures = [result for result in results if not result.passed]

    if failures:
        details = "\n".join(
            f"- {result.check_name}: {result.details}"
            for result in failures
        )
        raise RuntimeError(
            "Gold validation failed. Upload cancelled:\n"
            f"{details}"
        )

    print(
        f"Pre-upload Gold validation: SUCCESS "
        f"({len(results)} checks passed)"
    )


def upload_gold_dataset(
    config: dict[str, Any],
) -> str:
    """Upload the Gold training dataset to S3."""

    region = config["aws"]["region"]
    bucket = config["aws"]["s3_bucket"]
    gold_prefix = config["data"]["s3_prefixes"]["gold"]

    gold_config = config["gold_dataset"]
    dataset_version = gold_config["dataset_version"]
    processing_date = gold_config["processing_date"]

    source_version = config["data"]["source_version"]
    feature_version = config["feature_generation"]["feature_version"]

    local_path = build_local_path(
        dataset_version=dataset_version,
        processing_date=processing_date,
    )

    if not local_path.exists():
        raise FileNotFoundError(
            f"Local Gold dataset does not exist: {local_path}"
        )

    s3_key = build_s3_key(
        gold_prefix=gold_prefix,
        dataset_version=dataset_version,
        processing_date=processing_date,
    )

    checksum = calculate_sha256(local_path)

    session = boto3.Session(region_name=region)
    s3_client = session.client("s3")

    try:
        s3_client.upload_file(
            Filename=str(local_path),
            Bucket=bucket,
            Key=s3_key,
            ExtraArgs={
                "ContentType": "application/vnd.apache.parquet",
                "Metadata": {
                    "dataset-name": "training_dataset",
                    "dataset-version": dataset_version,
                    "source-version": source_version,
                    "feature-version": feature_version,
                    "processing-date": processing_date,
                    "data-layer": "gold",
                    "grain": "experiment",
                    "is-synthetic": "true",
                    "sha256": checksum,
                },
            },
        )
    except (ClientError, BotoCoreError) as exc:
        raise RuntimeError(
            f"Failed to upload {local_path} "
            f"to s3://{bucket}/{s3_key}"
        ) from exc

    s3_uri = f"s3://{bucket}/{s3_key}"

    print(
        f"Uploaded Gold dataset "
        f"size={local_path.stat().st_size} bytes "
        f"sha256={checksum[:12]}... "
        f"destination={s3_uri}"
    )

    return s3_uri


def main() -> None:
    """Validate and upload the Gold training dataset."""

    config = load_config()

    validate_before_upload()
    upload_gold_dataset(config)

    print()
    print("Gold S3 upload: SUCCESS")


if __name__ == "__main__":
    main()
