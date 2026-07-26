"""Resolve the approved model and run local batch ranking."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from gene_edit_ranking.config import load_config
from gene_edit_ranking.inference.batch_rank import (
    run_batch_ranking,
)
from gene_edit_ranking.inference.model_registry import (
    download_and_extract_model,
    get_latest_approved_model,
)


def main() -> None:
    config = load_config()

    region = config["aws"]["region"]
    pipeline_config = config["sagemaker_pipeline"]

    approved_model = get_latest_approved_model(
        model_package_group_name=pipeline_config[
            "model_package_group_name"
        ],
        region=region,
    )

    model_directory = download_and_extract_model(
        model_data_uri=approved_model.model_data_uri,
        destination=(
            Path("artifacts/approved-model")
            / f"version={approved_model.model_package_version}"
        ),
        region=region,
    )

    scoring_date = datetime.now(UTC).date().isoformat()

    input_path = Path(
        "data/gold/training/"
        "dataset_version=v1/"
        "processing_date=2026-07-25/"
        "training_dataset.parquet"
    )

    input_data_uri = (
        f"file://{input_path.resolve()}"
    )

    output_directory = Path(
        "data/gold/predictions/"
        f"model_version={approved_model.model_package_version}/"
        f"scoring_date={scoring_date}"
    )

    scored, summary = run_batch_ranking(
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
        input_data_uri=input_data_uri,
        ranking_columns=[
            "crop_line_id",
            "environment_id",
        ],
    )

    print("Local batch ranking: SUCCESS")
    print(f"input_rows={summary.input_rows}")
    print(f"output_rows={summary.output_rows}")
    print(
        "model_package_version="
        f"{summary.model_package_version}"
    )
    print(
        "selected_threshold="
        f"{summary.selected_threshold}"
    )
    print(f"output_directory={output_directory}")

    print("top_ranked_candidates=")
    print(
        scored[
            [
                "candidate_id",
                "crop_line_id",
                "environment_id",
                "edit_success_probability",
                "predicted_edit_success",
                "experiment_count",
                "rank_within_group",
            ]
        ]
        .head(10)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
