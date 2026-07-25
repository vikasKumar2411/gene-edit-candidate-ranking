"""Upload validated Silver Parquet datasets to Amazon S3."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import boto3
from botocore.exceptions import BotoCoreError, ClientError

from gene_edit_ranking.config import PROJECT_ROOT, load_config
from gene_edit_ranking.validation.silver_schemas import SILVER_SCHEMAS
from gene_edit_ranking.validation.validate_silver import run_validation


def calculate_sha256(file_path: Path) -> str:
    """Calculate the SHA-256 checksum of a file."""

    digest = hashlib.sha256()

    with file_path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def build_local_path(
    dataset_name: str,
    processing_date: str,
    source_version: str,
) -> Path:
    """Build the local Silver Parquet path."""

    return (
        PROJECT_ROOT
        / "data"
        / "silver"
        / dataset_name
        / f"processing_date={processing_date}"
        / f"source_version={source_version}"
        / f"{dataset_name}.parquet"
    )


def build_s3_key(
    silver_prefix: str,
    dataset_name: str,
    processing_date: str,
    source_version: str,
) -> str:
    """Build the Silver S3 object key."""

    return (
        f"{silver_prefix}/{dataset_name}/"
        f"processing_date={processing_date}/"
        f"source_version={source_version}/"
        f"{dataset_name}.parquet"
    )


def validate_before_upload() -> None:
    """Stop the upload when Silver validation fails."""

    results = run_validation()
    failures = [result for result in results if not result.passed]

    if failures:
        details = "\n".join(
            f"- {result.check_name}: {result.details}"
            for result in failures
        )
        raise RuntimeError(
            "Silver validation failed. Upload cancelled:\n"
            f"{details}"
        )

    print(
        f"Pre-upload Silver validation: SUCCESS "
        f"({len(results)} checks passed)"
    )


def upload_silver_datasets(
    config: dict[str, Any],
) -> list[str]:
    """Upload all Silver Parquet datasets to S3."""

    region = config["aws"]["region"]
    bucket = config["aws"]["s3_bucket"]
    silver_prefix = config["data"]["s3_prefixes"]["silver"]
    source_version = config["data"]["source_version"]
    processing_date = config["silver_processing"]["processing_date"]

    session = boto3.Session(region_name=region)
    s3_client = session.client("s3")

    uploaded_uris: list[str] = []

    for dataset_name in SILVER_SCHEMAS:
        local_path = build_local_path(
            dataset_name=dataset_name,
            processing_date=processing_date,
            source_version=source_version,
        )

        if not local_path.exists():
            raise FileNotFoundError(
                f"Local Silver file does not exist: {local_path}"
            )

        checksum = calculate_sha256(local_path)

        s3_key = build_s3_key(
            silver_prefix=silver_prefix,
            dataset_name=dataset_name,
            processing_date=processing_date,
            source_version=source_version,
        )

        try:
            s3_client.upload_file(
                Filename=str(local_path),
                Bucket=bucket,
                Key=s3_key,
                ExtraArgs={
                    "ContentType": "application/vnd.apache.parquet",
                    "Metadata": {
                        "dataset-name": dataset_name,
                        "source-version": source_version,
                        "processing-date": processing_date,
                        "data-layer": "silver",
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
        uploaded_uris.append(s3_uri)

        print(
            f"Uploaded {dataset_name:<12} "
            f"size={local_path.stat().st_size:>8} bytes "
            f"sha256={checksum[:12]}... "
            f"destination={s3_uri}"
        )

    return uploaded_uris


def main() -> None:
    """Validate and upload Silver datasets."""

    config = load_config()

    validate_before_upload()
    uploaded_uris = upload_silver_datasets(config)

    print()
    print(
        f"Silver S3 upload: SUCCESS "
        f"({len(uploaded_uris)} files uploaded)"
    )


if __name__ == "__main__":
    main()
