"""Resolve and download the latest approved model."""

from __future__ import annotations

from pathlib import Path

from gene_edit_ranking.config import load_config
from gene_edit_ranking.inference.model_registry import (
    download_and_extract_model,
    get_latest_approved_model,
)


def main() -> None:
    config = load_config()

    region = config["aws"]["region"]
    pipeline_config = config["sagemaker_pipeline"]

    model = get_latest_approved_model(
        model_package_group_name=pipeline_config[
            "model_package_group_name"
        ],
        region=region,
    )

    destination = Path(
        "artifacts/approved-model"
    ) / f"version={model.model_package_version}"

    extracted_path = download_and_extract_model(
        model_data_uri=model.model_data_uri,
        destination=destination,
        region=region,
    )

    print("Approved model resolution: SUCCESS")
    print(f"version={model.model_package_version}")
    print(f"package_arn={model.model_package_arn}")
    print(f"model_data_uri={model.model_data_uri}")
    print(f"image_uri={model.image_uri}")
    print(f"extracted_path={extracted_path}")

    print(
        "files="
        f"{sorted(path.name for path in extracted_path.iterdir())}"
    )


if __name__ == "__main__":
    main()
