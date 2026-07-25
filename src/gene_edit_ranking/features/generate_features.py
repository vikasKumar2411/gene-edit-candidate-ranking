"""Generate deterministic reusable offline feature tables from Silver data."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gene_edit_ranking.config import PROJECT_ROOT, load_config
from gene_edit_ranking.validation.feature_schemas import (
    FeatureSchema,
    build_feature_schemas,
)


@dataclass(frozen=True)
class FeatureGenerationSummary:
    """Summary of one generated feature table."""

    feature_table: str
    rows: int
    columns: int
    output_path: Path


def stable_seed(value: str) -> int:
    """Create a stable 32-bit seed from a string identifier."""

    digest = hashlib.sha256(value.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], byteorder="big", signed=False)


def deterministic_embedding(
    entity_id: str,
    dimension: int,
) -> np.ndarray:
    """Generate a deterministic normalized synthetic embedding."""

    rng = np.random.default_rng(stable_seed(entity_id))
    vector = rng.normal(loc=0.0, scale=1.0, size=dimension)

    norm = np.linalg.norm(vector)

    if norm == 0:
        return vector

    return vector / norm


def add_embedding_columns(
    dataframe: pd.DataFrame,
    entity_key: str,
    prefix: str,
    dimension: int,
) -> pd.DataFrame:
    """Append deterministic embedding dimensions to a dataframe."""

    result = dataframe.copy()

    embedding_matrix = np.vstack(
        [
            deterministic_embedding(str(entity_id), dimension)
            for entity_id in result[entity_key]
        ]
    )

    for index in range(dimension):
        result[f"{prefix}_{index:03d}"] = embedding_matrix[:, index]

    return result


def load_silver_dataset(
    dataset_name: str,
    processing_date: str,
    source_version: str,
) -> pd.DataFrame:
    """Load one local Silver Parquet dataset."""

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


def generate_gene_features(
    genes: pd.DataFrame,
    schema: FeatureSchema,
) -> pd.DataFrame:
    """Generate reusable synthetic gene-level features."""

    features = genes[
        [
            "gene_id",
            "start_position",
            "end_position",
            "synthetic_conservation_score",
        ]
    ].copy()

    features["gene_length"] = (
        features["end_position"] - features["start_position"]
    ).astype("int64")

    max_position = max(float(features["start_position"].max()), 1.0)

    features["normalized_gene_position"] = (
        features["start_position"] / max_position
    ).astype("float64")

    features = features.drop(
        columns=["start_position", "end_position"]
    )

    return add_embedding_columns(
        dataframe=features,
        entity_key=schema.entity_key,
        prefix=schema.embedding_prefix,
        dimension=schema.embedding_dimension,
    )


def generate_crop_line_features(
    crop_lines: pd.DataFrame,
    schema: FeatureSchema,
) -> pd.DataFrame:
    """Generate reusable synthetic crop-line features."""

    features = crop_lines[
        [
            "crop_line_id",
            "baseline_yield_index",
            "baseline_drought_tolerance",
        ]
    ].copy()

    features["yield_drought_interaction"] = (
        features["baseline_yield_index"]
        * features["baseline_drought_tolerance"]
    ).astype("float64")

    return add_embedding_columns(
        dataframe=features,
        entity_key=schema.entity_key,
        prefix=schema.embedding_prefix,
        dimension=schema.embedding_dimension,
    )


def generate_environment_features(
    environments: pd.DataFrame,
    schema: FeatureSchema,
) -> pd.DataFrame:
    """Generate reusable synthetic environment-level features."""

    features = environments[
        [
            "environment_id",
            "rainfall_mm",
            "average_temperature_c",
            "synthetic_drought_index",
        ]
    ].copy()

    rainfall_component = 1 - (
        features["rainfall_mm"]
        / max(float(features["rainfall_mm"].max()), 1.0)
    )

    temperature_component = (
        features["average_temperature_c"] - 20
    ).clip(lower=0) / 20

    features["environment_stress_score"] = (
        0.6 * features["synthetic_drought_index"]
        + 0.25 * rainfall_component
        + 0.15 * temperature_component
    ).clip(0, 1)

    return add_embedding_columns(
        dataframe=features,
        entity_key=schema.entity_key,
        prefix=schema.embedding_prefix,
        dimension=schema.embedding_dimension,
    )


def add_feature_metadata(
    dataframe: pd.DataFrame,
    feature_version: str,
    processing_date: str,
) -> pd.DataFrame:
    """Add feature version and lineage metadata."""

    result = dataframe.copy()

    result["_feature_version"] = feature_version
    result["_processing_date"] = pd.to_datetime(
        processing_date,
        utc=True,
    )
    result["_feature_generated_at_utc"] = pd.Timestamp.now(tz="UTC")

    return result


def write_feature_table(
    dataframe: pd.DataFrame,
    feature_table: str,
    feature_version: str,
    processing_date: str,
    compression: str,
) -> Path:
    """Write one feature table to partitioned local Parquet storage."""

    output_path = (
        PROJECT_ROOT
        / "data"
        / "features"
        / feature_table
        / f"feature_version={feature_version}"
        / f"processing_date={processing_date}"
        / f"{feature_table}.parquet"
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    dataframe.to_parquet(
        output_path,
        index=False,
        engine="pyarrow",
        compression=compression,
    )

    return output_path


def generate_all(
    config: dict[str, Any],
) -> list[FeatureGenerationSummary]:
    """Generate all reusable feature tables."""

    source_version = config["data"]["source_version"]
    silver_processing_date = config["silver_processing"]["processing_date"]

    feature_config = config["feature_generation"]
    feature_version = feature_config["feature_version"]
    processing_date = feature_config["processing_date"]
    compression = feature_config["compression"]
    dimensions = feature_config["embedding_dimensions"]

    schemas = build_feature_schemas(
        gene_embedding_dimension=dimensions["gene"],
        crop_line_embedding_dimension=dimensions["crop_line"],
        environment_embedding_dimension=dimensions["environment"],
    )

    genes = load_silver_dataset(
        dataset_name="genes",
        processing_date=silver_processing_date,
        source_version=source_version,
    )
    crop_lines = load_silver_dataset(
        dataset_name="crop_lines",
        processing_date=silver_processing_date,
        source_version=source_version,
    )
    environments = load_silver_dataset(
        dataset_name="environments",
        processing_date=silver_processing_date,
        source_version=source_version,
    )

    feature_tables = {
        "gene_features": generate_gene_features(
            genes,
            schemas["gene_features"],
        ),
        "crop_line_features": generate_crop_line_features(
            crop_lines,
            schemas["crop_line_features"],
        ),
        "environment_features": generate_environment_features(
            environments,
            schemas["environment_features"],
        ),
    }

    summaries: list[FeatureGenerationSummary] = []

    for feature_table, dataframe in feature_tables.items():
        dataframe = add_feature_metadata(
            dataframe=dataframe,
            feature_version=feature_version,
            processing_date=processing_date,
        )

        output_path = write_feature_table(
            dataframe=dataframe,
            feature_table=feature_table,
            feature_version=feature_version,
            processing_date=processing_date,
            compression=compression,
        )

        summaries.append(
            FeatureGenerationSummary(
                feature_table=feature_table,
                rows=len(dataframe),
                columns=len(dataframe.columns),
                output_path=output_path,
            )
        )

    return summaries


def main() -> None:
    """Generate all configured reusable feature tables."""

    config = load_config()
    summaries = generate_all(config)

    print("Reusable feature generation: SUCCESS")
    print()

    for summary in summaries:
        print(
            f"{summary.feature_table:<24} "
            f"rows={summary.rows:>5} "
            f"columns={summary.columns:>3} "
            f"file={summary.output_path}"
        )


if __name__ == "__main__":
    main()
