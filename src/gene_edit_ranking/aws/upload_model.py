"""Upload a packaged model artifact and metadata to Amazon S3."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import boto3
from botocore.exceptions import BotoCoreError, ClientError

from gene_edit_ranking.config import PROJECT_ROOT, load_config


def calculate_sha256(path: Path) -> str:
    """Calculate a SHA-256 checksum."""

    digest = hashlib.sha256()

    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    """Load a JSON document."""

    with path.open("r", encoding="utf-8") as file_handle:
        return json.load(file_handle)


def upload_model(config: dict[str, Any]) -> list[str]:
    """Upload model bundle, checksum, and metadata to S3."""

    training = config["training"]

    model_name = training["model_name"]
    model_version = training["model_version"]
    bucket = config["aws"]["s3_bucket"]
    region = config["aws"]["region"]
    models_prefix = config["data"]["s3_prefixes"]["models"]

    training_dir = PROJECT_ROOT / training["output_dir"]

    bundle_dir = (
        training_dir
        / model_name
        / f"model_version={model_version}"
    )

    bundle_path = bundle_dir / "model.tar.gz"
    checksum_path = bundle_dir / "model.tar.gz.sha256"
    metadata_path = training_dir / "model_metadata.json"

    required_paths = (
        bundle_path,
        checksum_path,
        metadata_path,
    )

    missing_paths = [
        str(path)
        for path in required_paths
        if not path.exists()
    ]

    if missing_paths:
        raise FileNotFoundError(
            f"Missing model artifacts: {missing_paths}"
        )

    checksum = calculate_sha256(bundle_path)
    metadata = load_json(metadata_path)

    expected_checksum = checksum_path.read_text(
        encoding="utf-8"
    ).split()[0]

    if checksum != expected_checksum:
        raise RuntimeError(
            "Model bundle checksum does not match the checksum file."
        )

    s3_base_key = (
        f"{models_prefix}/{model_name}/"
        f"model_version={model_version}"
    )

    uploads = {
        bundle_path: f"{s3_base_key}/model.tar.gz",
        checksum_path: f"{s3_base_key}/model.tar.gz.sha256",
        metadata_path: f"{s3_base_key}/model_metadata.json",
    }

    s3_client = boto3.Session(
        region_name=region
    ).client("s3")

    uploaded_uris: list[str] = []

    for local_path, s3_key in uploads.items():
        if local_path.suffix == ".json":
            content_type = "application/json"
        elif local_path.name.endswith(".sha256"):
            content_type = "text/plain"
        else:
            content_type = "application/gzip"

        extra_args: dict[str, Any] = {
            "ContentType": content_type,
            "Metadata": {
                "model-name": model_name,
                "model-version": model_version,
                "task-type": metadata["task_type"],
                "target-column": metadata["target_column"],
                "gold-dataset-version": metadata[
                    "gold_dataset_version"
                ],
                "feature-version": metadata["feature_version"],
                "selected-threshold": str(
                    metadata["selected_threshold"]
                ),
                "is-synthetic": "true",
            },
        }

        if local_path == bundle_path:
            extra_args["Metadata"]["sha256"] = checksum

        try:
            s3_client.upload_file(
                Filename=str(local_path),
                Bucket=bucket,
                Key=s3_key,
                ExtraArgs=extra_args,
            )
        except (ClientError, BotoCoreError) as exc:
            raise RuntimeError(
                f"Failed to upload {local_path} "
                f"to s3://{bucket}/{s3_key}"
            ) from exc

        s3_uri = f"s3://{bucket}/{s3_key}"
        uploaded_uris.append(s3_uri)

        print(
            f"Uploaded {local_path.name:<24} "
            f"size={local_path.stat().st_size:>7} bytes "
            f"destination={s3_uri}"
        )

    return uploaded_uris


def main() -> None:
    """Upload packaged model artifacts."""

    config = load_config()
    uploaded_uris = upload_model(config)

    print()
    print(
        f"Model S3 upload: SUCCESS "
        f"({len(uploaded_uris)} files uploaded)"
    )


if __name__ == "__main__":
    main()
