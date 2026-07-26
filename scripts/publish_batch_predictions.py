"""Publish local batch-ranking outputs to S3."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from gene_edit_ranking.config import load_config
from gene_edit_ranking.inference.model_registry import (
    get_latest_approved_model,
)
from gene_edit_ranking.inference.publish_predictions import (
    publish_batch_outputs,
)


def main() -> None:
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

    scoring_date = datetime.now(
        UTC
    ).date().isoformat()

    output_directory = Path(
        "data/gold/predictions/"
        f"model_version={approved_model.model_package_version}/"
        f"scoring_date={scoring_date}"
    )

    destination_uri = (
        f"s3://{bucket}/gold/predictions/"
        f"model_version={approved_model.model_package_version}/"
        f"scoring_date={scoring_date}"
    )

    manifest = publish_batch_outputs(
        output_directory=output_directory,
        destination_uri=destination_uri,
        region=region,
    )

    print("Batch prediction publishing: SUCCESS")
    print(
        json.dumps(
            manifest,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
