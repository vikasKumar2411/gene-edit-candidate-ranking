"""Resolve and download the latest approved SageMaker model package."""

from __future__ import annotations

import tarfile
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import boto3


@dataclass(frozen=True)
class ApprovedModel:
    """Metadata for an approved SageMaker model package."""

    model_package_arn: str
    model_package_version: int
    model_data_uri: str
    image_uri: str


def get_latest_approved_model(
    *,
    model_package_group_name: str,
    region: str,
) -> ApprovedModel:
    """Return the newest completed and approved model package."""

    client = boto3.client("sagemaker", region_name=region)

    paginator = client.get_paginator("list_model_packages")

    for page in paginator.paginate(
        ModelPackageGroupName=model_package_group_name,
        ModelApprovalStatus="Approved",
        SortBy="CreationTime",
        SortOrder="Descending",
    ):
        for package in page["ModelPackageSummaryList"]:
            if package["ModelPackageStatus"] != "Completed":
                continue

            details = client.describe_model_package(
                ModelPackageName=package["ModelPackageArn"]
            )

            container = details["InferenceSpecification"][
                "Containers"
            ][0]

            return ApprovedModel(
                model_package_arn=package["ModelPackageArn"],
                model_package_version=package[
                    "ModelPackageVersion"
                ],
                model_data_uri=container["ModelDataUrl"],
                image_uri=container["Image"],
            )

    raise RuntimeError(
        "No completed and approved model package found in "
        f"{model_package_group_name}."
    )


def parse_s3_uri(uri: str) -> tuple[str, str]:
    """Split an S3 URI into bucket and object key."""

    parsed = urlparse(uri)

    if parsed.scheme != "s3" or not parsed.netloc:
        raise ValueError(f"Invalid S3 URI: {uri}")

    return parsed.netloc, parsed.path.lstrip("/")


def download_and_extract_model(
    *,
    model_data_uri: str,
    destination: Path,
    region: str,
) -> Path:
    """Download and extract a SageMaker model archive."""

    bucket, key = parse_s3_uri(model_data_uri)

    destination.mkdir(parents=True, exist_ok=True)

    archive_path = destination / "model.tar.gz"

    s3 = boto3.client("s3", region_name=region)
    s3.download_file(bucket, key, str(archive_path))

    extracted_path = destination / "extracted"
    extracted_path.mkdir(parents=True, exist_ok=True)

    with tarfile.open(archive_path, mode="r:gz") as archive:
        archive.extractall(
            extracted_path,
            filter="data",
        )

    return extracted_path
