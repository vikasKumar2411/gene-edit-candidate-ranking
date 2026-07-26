"""Container entry point for batch candidate ranking."""

from __future__ import annotations

import argparse
import json
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlparse

import boto3

from gene_edit_ranking.config import load_config
from gene_edit_ranking.inference.batch_rank import run_batch_ranking
from gene_edit_ranking.inference.model_registry import (
    download_and_extract_model,
    get_latest_approved_model,
)
from gene_edit_ranking.inference.publish_predictions import (
    publish_batch_outputs,
)


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


def download_s3_file(
    *,
    uri: str,
    destination: Path,
    region: str,
) -> Path:
    parsed = urlparse(uri)

    if parsed.scheme != "s3" or not parsed.netloc:
        raise ValueError(f"Invalid S3 URI: {uri}")

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    boto3.client(
        "s3",
        region_name=region,
    ).download_file(
        parsed.netloc,
        parsed.path.lstrip("/"),
        str(destination),
    )

    return destination


def main() -> None:
    args = parse_args()
    config = load_config()

    region = config["aws"]["region"]
    bucket = config["aws"]["s3_bucket"]
    pipeline_config = config["sagemaker_pipeline"]

    approved_model = get_latest_approved_model(
        model_package_group_name=pipeline_config[
            "model_package_group_name"
        ],
        region=region,
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        workspace = Path(temp_dir)

        input_path = download_s3_file(
            uri=args.input_s3_uri,
            destination=workspace / "input.parquet",
            region=region,
        )

        model_directory = download_and_extract_model(
            model_data_uri=approved_model.model_data_uri,
            destination=workspace / "model",
            region=region,
        )

        output_directory = workspace / "output"

        _, summary = run_batch_ranking(
            input_path=input_path,
            model_directory=model_directory,
            output_directory=output_directory,
            model_package_version=(
                approved_model.model_package_version
            ),
            model_package_arn=(
                approved_model.model_package_arn
            ),
            model_data_uri=approved_model.model_data_uri,
            input_data_uri=args.input_s3_uri,
            ranking_columns=[
                "crop_line_id",
                "environment_id",
            ],
        )

        destination_uri = (
            f"s3://{bucket}/gold/predictions/"
            f"model_version={approved_model.model_package_version}/"
            f"scoring_date={args.scoring_date}"
        )

        manifest = publish_batch_outputs(
            output_directory=output_directory,
            destination_uri=destination_uri,
            region=region,
        )

        print("Container batch ranking: SUCCESS")
        print(f"input_rows={summary.input_rows}")
        print(f"output_rows={summary.output_rows}")
        print(
            f"model_version="
            f"{approved_model.model_package_version}"
        )
        print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
