"""Upload validated synthetic Raw datasets to partitioned Amazon S3 paths."""

from __future__ import annotations

import argparse
import hashlib
from datetime import date
from pathlib import Path
from typing import Any

import boto3
from botocore.exceptions import BotoCoreError, ClientError

from gene_edit_ranking.config import PROJECT_ROOT, load_config
from gene_edit_ranking.validation.validate_raw import run_validation


DATASET_FILES = {
    "genes": "genes.csv",
    "crop_lines": "crop_lines.csv",
    "environments": "environments.csv",
    "candidates": "candidate_edits.csv",
    "experiments": "experiments.csv",
}


def calculate_sha256(file_path: Path) -> str:
    """Calculate the SHA-256 checksum of a local file."""

    digest = hashlib.sha256()

    with file_path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def build_s3_key(
    raw_prefix: str,
    dataset_name: str,
    ingestion_date: str,
    source_version: str,
    filename: str,
) -> str:
    """Build the partitioned S3 object key."""

    return (
        f"{raw_prefix}/{dataset_name}/"
        f"ingestion_date={ingestion_date}/"
        f"source_version={source_version}/"
        f"{filename}"
    )


def validate_before_upload() -> None:
    """Stop the upload if any local Raw data validation fails."""

    results = run_validation()
    failures = [result for result in results if not result.passed]

    if failures:
        failure_messages = "\n".join(
            f"- {result.check_name}: {result.details}"
            for result in failures
        )
        raise RuntimeError(
            "Raw data validation failed. Upload cancelled:\n"
            f"{failure_messages}"
        )

    print(f"Pre-upload validation: SUCCESS ({len(results)} checks passed)")


def upload_raw_datasets(
    config: dict[str, Any],
    ingestion_date: str,
) -> list[str]:
    """Upload all local Raw datasets and return their S3 URIs."""

    region = config["aws"]["region"]
    bucket = config["aws"]["s3_bucket"]
    raw_prefix = config["data"]["s3_prefixes"]["raw"]
    source_version = config["data"]["source_version"]
    raw_dir = PROJECT_ROOT / config["data"]["local_raw_dir"]

    session = boto3.Session(region_name=region)
    s3_client = session.client("s3")

    uploaded_uris: list[str] = []

    for dataset_name, filename in DATASET_FILES.items():
        local_path = raw_dir / filename

        if not local_path.exists():
            raise FileNotFoundError(
                f"Local Raw file does not exist: {local_path}"
            )

        checksum = calculate_sha256(local_path)

        s3_key = build_s3_key(
            raw_prefix=raw_prefix,
            dataset_name=dataset_name,
            ingestion_date=ingestion_date,
            source_version=source_version,
            filename=filename,
        )

        try:
            s3_client.upload_file(
                Filename=str(local_path),
                Bucket=bucket,
                Key=s3_key,
                ExtraArgs={
                    "ContentType": "text/csv",
                    "Metadata": {
                        "dataset-name": dataset_name,
                        "source-version": source_version,
                        "ingestion-date": ingestion_date,
                        "is-synthetic": "true",
                        "sha256": checksum,
                    },
                },
            )
        except (ClientError, BotoCoreError) as exc:
            raise RuntimeError(
                f"Failed to upload {local_path} to s3://{bucket}/{s3_key}"
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


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Upload validated synthetic Raw datasets to S3."
    )

    parser.add_argument(
        "--ingestion-date",
        default=date.today().isoformat(),
        help="Partition date in YYYY-MM-DD format. Defaults to today.",
    )

    return parser.parse_args()


def main() -> None:
    """Validate and upload all synthetic Raw datasets."""

    args = parse_args()

    try:
        date.fromisoformat(args.ingestion_date)
    except ValueError as exc:
        raise ValueError(
            "--ingestion-date must use YYYY-MM-DD format."
        ) from exc

    config = load_config()

    validate_before_upload()

    uploaded_uris = upload_raw_datasets(
        config=config,
        ingestion_date=args.ingestion_date,
    )

    print()
    print(f"Raw S3 upload: SUCCESS ({len(uploaded_uris)} files uploaded)")


if __name__ == "__main__":
    main()
