"""Upload validated reusable feature tables to Amazon S3."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import boto3
from botocore.exceptions import BotoCoreError, ClientError

from gene_edit_ranking.config import PROJECT_ROOT, load_config
from gene_edit_ranking.validation.feature_schemas import build_feature_schemas
from gene_edit_ranking.validation.validate_features import run_validation


def calculate_sha256(file_path: Path) -> str:
    """Calculate a SHA-256 checksum."""

    digest = hashlib.sha256()

    with file_path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def build_local_path(
    feature_table: str,
    feature_version: str,
    processing_date: str,
) -> Path:
    """Build the local feature-table path."""

    return (
        PROJECT_ROOT
        / "data"
        / "features"
        / feature_table
        / f"feature_version={feature_version}"
        / f"processing_date={processing_date}"
        / f"{feature_table}.parquet"
    )


def build_s3_key(
    features_prefix: str,
    feature_table: str,
    feature_version: str,
    processing_date: str,
) -> str:
    """Build the partitioned feature-table S3 key."""

    return (
        f"{features_prefix}/{feature_table}/"
        f"feature_version={feature_version}/"
        f"processing_date={processing_date}/"
        f"{feature_table}.parquet"
    )


def validate_before_upload() -> None:
    """Cancel upload when reusable-feature validation fails."""

    results = run_validation()
    failures = [result for result in results if not result.passed]

    if failures:
        details = "\n".join(
            f"- {result.check_name}: {result.details}"
            for result in failures
        )
        raise RuntimeError(
            "Feature validation failed. Upload cancelled:\n"
            f"{details}"
        )

    print(
        f"Pre-upload feature validation: SUCCESS "
        f"({len(results)} checks passed)"
    )


def upload_feature_tables(
    config: dict[str, Any],
) -> list[str]:
    """Upload all configured feature tables to S3."""

    region = config["aws"]["region"]
    bucket = config["aws"]["s3_bucket"]
    features_prefix = config["data"]["s3_prefixes"]["features"]

    feature_config = config["feature_generation"]
    feature_version = feature_config["feature_version"]
    processing_date = feature_config["processing_date"]
    dimensions = feature_config["embedding_dimensions"]

    schemas = build_feature_schemas(
        gene_embedding_dimension=dimensions["gene"],
        crop_line_embedding_dimension=dimensions["crop_line"],
        environment_embedding_dimension=dimensions["environment"],
    )

    session = boto3.Session(region_name=region)
    s3_client = session.client("s3")

    uploaded_uris: list[str] = []

    for feature_table, schema in schemas.items():
        local_path = build_local_path(
            feature_table=feature_table,
            feature_version=feature_version,
            processing_date=processing_date,
        )

        if not local_path.exists():
            raise FileNotFoundError(
                f"Local feature table does not exist: {local_path}"
            )

        checksum = calculate_sha256(local_path)

        s3_key = build_s3_key(
            features_prefix=features_prefix,
            feature_table=feature_table,
            feature_version=feature_version,
            processing_date=processing_date,
        )

        try:
            s3_client.upload_file(
                Filename=str(local_path),
                Bucket=bucket,
                Key=s3_key,
                ExtraArgs={
                    "ContentType": "application/vnd.apache.parquet",
                    "Metadata": {
                        "feature-table": feature_table,
                        "feature-version": feature_version,
                        "processing-date": processing_date,
                        "entity-key": schema.entity_key,
                        "embedding-dimension": str(
                            schema.embedding_dimension
                        ),
                        "data-layer": "features",
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
            f"Uploaded {feature_table:<24} "
            f"size={local_path.stat().st_size:>8} bytes "
            f"sha256={checksum[:12]}... "
            f"destination={s3_uri}"
        )

    return uploaded_uris


def main() -> None:
    """Validate and upload reusable feature tables."""

    config = load_config()

    validate_before_upload()
    uploaded_uris = upload_feature_tables(config)

    print()
    print(
        f"Feature S3 upload: SUCCESS "
        f"({len(uploaded_uris)} files uploaded)"
    )


if __name__ == "__main__":
    main()
