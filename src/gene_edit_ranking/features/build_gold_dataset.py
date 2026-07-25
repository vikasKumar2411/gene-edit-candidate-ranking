"""Build the model-ready Gold training dataset."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from gene_edit_ranking.config import PROJECT_ROOT, load_config


@dataclass(frozen=True)
class GoldBuildSummary:
    """Summary of the Gold dataset build."""

    rows: int
    columns: int
    output_path: Path


def load_silver_dataset(
    dataset_name: str,
    processing_date: str,
    source_version: str,
) -> pd.DataFrame:
    """Load one local Silver dataset."""

    path = (
        PROJECT_ROOT
        / "data"
        / "silver"
        / dataset_name
        / f"processing_date={processing_date}"
        / f"source_version={source_version}"
        / f"{dataset_name}.parquet"
    )

    if not path.exists():
        raise FileNotFoundError(f"Missing Silver dataset: {path}")

    return pd.read_parquet(path)


def load_feature_table(
    feature_table: str,
    feature_version: str,
    processing_date: str,
) -> pd.DataFrame:
    """Load one reusable local feature table."""

    path = (
        PROJECT_ROOT
        / "data"
        / "features"
        / feature_table
        / f"feature_version={feature_version}"
        / f"processing_date={processing_date}"
        / f"{feature_table}.parquet"
    )

    if not path.exists():
        raise FileNotFoundError(f"Missing feature table: {path}")

    return pd.read_parquet(path)


def drop_operational_columns(
    dataframe: pd.DataFrame,
    configured_drop_columns: list[str],
) -> pd.DataFrame:
    """Remove operational metadata that should not become model inputs."""

    columns_to_drop = [
        column
        for column in configured_drop_columns
        if column in dataframe.columns
    ]

    return dataframe.drop(columns=columns_to_drop)


def build_gold_dataframe(
    experiments: pd.DataFrame,
    candidates: pd.DataFrame,
    gene_features: pd.DataFrame,
    crop_line_features: pd.DataFrame,
    environment_features: pd.DataFrame,
) -> pd.DataFrame:
    """Join Silver entities and reusable features at experiment grain."""

    operational_columns = {
        "is_synthetic",
        "_source_file",
        "_source_version",
        "_processing_date",
        "_processed_at_utc",
        "_feature_version",
        "_feature_generated_at_utc",
    }

    def remove_operational_columns(
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """Remove per-source lineage before combining datasets."""

        columns_to_drop = [
            column
            for column in operational_columns
            if column in dataframe.columns
        ]

        return dataframe.drop(columns=columns_to_drop).copy()

    experiments_clean = remove_operational_columns(experiments)
    candidates_clean = remove_operational_columns(candidates)
    gene_features_clean = remove_operational_columns(gene_features)
    crop_line_features_clean = remove_operational_columns(
        crop_line_features
    )
    environment_features_clean = remove_operational_columns(
        environment_features
    )

    candidate_columns = [
        "candidate_id",
        "gene_id",
        "crop_line_id",
        "editing_method",
        "target_position",
        "predicted_edit_efficiency",
        "predicted_off_target_risk",
        "design_batch",
        "created_date",
    ]

    gold = experiments_clean.merge(
        candidates_clean[candidate_columns],
        on="candidate_id",
        how="left",
        validate="many_to_one",
    )

    gold = gold.merge(
        gene_features_clean,
        on="gene_id",
        how="left",
        validate="many_to_one",
    )

    gold = gold.merge(
        crop_line_features_clean,
        on="crop_line_id",
        how="left",
        validate="many_to_one",
    )

    gold = gold.merge(
        environment_features_clean,
        on="environment_id",
        how="left",
        validate="many_to_one",
    )

    return gold


def add_gold_metadata(
    dataframe: pd.DataFrame,
    dataset_version: str,
    processing_date: str,
    source_version: str,
    feature_version: str,
) -> pd.DataFrame:
    """Add Gold dataset lineage metadata."""

    result = dataframe.copy()

    result["_gold_dataset_version"] = dataset_version
    result["_source_version"] = source_version
    result["_feature_version"] = feature_version
    result["_processing_date"] = pd.to_datetime(
        processing_date,
        utc=True,
    )
    result["_gold_generated_at_utc"] = pd.Timestamp.now(tz="UTC")

    return result


def write_gold_dataset(
    dataframe: pd.DataFrame,
    dataset_version: str,
    processing_date: str,
    compression: str,
) -> Path:
    """Write the Gold dataset to partitioned local Parquet storage."""

    output_path = (
        PROJECT_ROOT
        / "data"
        / "gold"
        / "training"
        / f"dataset_version={dataset_version}"
        / f"processing_date={processing_date}"
        / "training_dataset.parquet"
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    dataframe.to_parquet(
        output_path,
        index=False,
        engine="pyarrow",
        compression=compression,
    )

    return output_path


def build_all(config: dict[str, Any]) -> GoldBuildSummary:
    """Build and persist the configured Gold training dataset."""

    source_version = config["data"]["source_version"]
    silver_processing_date = config["silver_processing"]["processing_date"]

    feature_config = config["feature_generation"]
    feature_version = feature_config["feature_version"]
    feature_processing_date = feature_config["processing_date"]

    gold_config = config["gold_dataset"]
    dataset_version = gold_config["dataset_version"]
    processing_date = gold_config["processing_date"]
    compression = gold_config["compression"]

    experiments = load_silver_dataset(
        "experiments",
        silver_processing_date,
        source_version,
    )

    candidates = load_silver_dataset(
        "candidates",
        silver_processing_date,
        source_version,
    )

    gene_features = load_feature_table(
        "gene_features",
        feature_version,
        feature_processing_date,
    )

    crop_line_features = load_feature_table(
        "crop_line_features",
        feature_version,
        feature_processing_date,
    )

    environment_features = load_feature_table(
        "environment_features",
        feature_version,
        feature_processing_date,
    )

    gold = build_gold_dataframe(
        experiments=experiments,
        candidates=candidates,
        gene_features=gene_features,
        crop_line_features=crop_line_features,
        environment_features=environment_features,
    )

    gold = drop_operational_columns(
        dataframe=gold,
        configured_drop_columns=gold_config["drop_columns"],
    )

    gold = add_gold_metadata(
        dataframe=gold,
        dataset_version=dataset_version,
        processing_date=processing_date,
        source_version=source_version,
        feature_version=feature_version,
    )

    output_path = write_gold_dataset(
        dataframe=gold,
        dataset_version=dataset_version,
        processing_date=processing_date,
        compression=compression,
    )

    return GoldBuildSummary(
        rows=len(gold),
        columns=len(gold.columns),
        output_path=output_path,
    )


def main() -> None:
    """Build the local Gold training dataset."""

    config = load_config()
    summary = build_all(config)

    print("Gold training dataset build: SUCCESS")
    print(f"rows={summary.rows}")
    print(f"columns={summary.columns}")
    print(f"file={summary.output_path}")


if __name__ == "__main__":
    main()
