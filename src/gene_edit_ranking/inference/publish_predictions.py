"""Publish batch-ranking outputs to versioned S3 locations."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import boto3


def sha256_file(path: Path) -> str:
    """Calculate the SHA-256 checksum of a file."""

    digest = hashlib.sha256()

    with path.open("rb") as file_handle:
        for chunk in iter(
            lambda: file_handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def sha256_run_metadata(path: Path) -> str:
    """Checksum stable run metadata fields.

    The generation timestamp is intentionally excluded because it changes
    on a safe retry even when the model, input, and output are identical.
    """

    metadata = json.loads(
        path.read_text(encoding="utf-8")
    )

    metadata.pop("generated_at_utc", None)

    canonical = json.dumps(
        metadata,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")

    return hashlib.sha256(canonical).hexdigest()


def parse_s3_uri(uri: str) -> tuple[str, str]:
    """Split an S3 URI into bucket and prefix."""

    parsed = urlparse(uri)

    if parsed.scheme != "s3" or not parsed.netloc:
        raise ValueError(f"Invalid S3 URI: {uri}")

    return parsed.netloc, parsed.path.lstrip("/").rstrip("/")


def get_existing_checksum(
    *,
    s3: Any,
    bucket: str,
    key: str,
) -> str | None:
    """Return an existing object's stored SHA-256 metadata."""

    try:
        response = s3.head_object(
            Bucket=bucket,
            Key=key,
        )
    except s3.exceptions.ClientError as exc:
        error_code = exc.response["Error"]["Code"]

        if error_code in {"404", "NoSuchKey", "NotFound"}:
            return None

        raise

    return response.get("Metadata", {}).get("sha256")


def upload_if_unchanged_or_missing(
    *,
    s3: Any,
    local_path: Path,
    bucket: str,
    key: str,
    content_type: str,
    checksum: str,
) -> str:
    """Upload a file or confirm an identical object already exists."""

    existing_checksum = get_existing_checksum(
        s3=s3,
        bucket=bucket,
        key=key,
    )

    if existing_checksum is not None:
        if existing_checksum != checksum:
            raise RuntimeError(
                "Refusing to overwrite an existing S3 object with "
                f"different content: s3://{bucket}/{key}"
            )

        return "already_exists_identical"

    s3.upload_file(
        str(local_path),
        bucket,
        key,
        ExtraArgs={
            "ContentType": content_type,
            "Metadata": {
                "sha256": checksum,
            },
        },
    )

    return "uploaded"


def publish_batch_outputs(
    *,
    output_directory: Path,
    destination_uri: str,
    region: str,
) -> dict[str, Any]:
    """Upload ranking outputs and return publication metadata."""

    predictions_path = (
        output_directory
        / "ranked_candidates.parquet"
    )

    metadata_path = (
        output_directory
        / "run_metadata.json"
    )

    if not predictions_path.exists():
        raise FileNotFoundError(
            f"Missing predictions file: {predictions_path}"
        )

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Missing metadata file: {metadata_path}"
        )

    bucket, prefix = parse_s3_uri(destination_uri)

    predictions_key = (
        f"{prefix}/ranked_candidates.parquet"
    )

    metadata_key = (
        f"{prefix}/run_metadata.json"
    )

    manifest_key = (
        f"{prefix}/publication_manifest.json"
    )

    predictions_checksum = sha256_file(
        predictions_path
    )

    metadata_checksum = sha256_run_metadata(
        metadata_path
    )

    s3 = boto3.client(
        "s3",
        region_name=region,
    )

    predictions_status = upload_if_unchanged_or_missing(
        s3=s3,
        local_path=predictions_path,
        bucket=bucket,
        key=predictions_key,
        content_type="application/vnd.apache.parquet",
        checksum=predictions_checksum,
    )

    metadata_status = upload_if_unchanged_or_missing(
        s3=s3,
        local_path=metadata_path,
        bucket=bucket,
        key=metadata_key,
        content_type="application/json",
        checksum=metadata_checksum,
    )

    publication_manifest = {
        "destination_uri": destination_uri,
        "objects": [
            {
                "name": "ranked_candidates",
                "s3_uri": (
                    f"s3://{bucket}/{predictions_key}"
                ),
                "sha256": predictions_checksum,
                "publication_status": predictions_status,
            },
            {
                "name": "run_metadata",
                "s3_uri": (
                    f"s3://{bucket}/{metadata_key}"
                ),
                "sha256": metadata_checksum,
                "publication_status": metadata_status,
            },
        ],
    }

    s3.put_object(
        Bucket=bucket,
        Key=manifest_key,
        Body=json.dumps(
            publication_manifest,
            indent=2,
        ).encode("utf-8"),
        ContentType="application/json",
    )

    publication_manifest["manifest_s3_uri"] = (
        f"s3://{bucket}/{manifest_key}"
    )

    return publication_manifest
