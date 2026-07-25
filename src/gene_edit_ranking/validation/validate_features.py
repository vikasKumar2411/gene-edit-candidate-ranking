"""Validation checks for reusable offline feature tables."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from gene_edit_ranking.config import PROJECT_ROOT, load_config
from gene_edit_ranking.validation.feature_schemas import (
    FeatureSchema,
    build_feature_schemas,
)


@dataclass(frozen=True)
class ValidationResult:
    """Result of one feature validation check."""

    check_name: str
    passed: bool
    details: str


def record(
    results: list[ValidationResult],
    check_name: str,
    condition: bool,
    success_details: str,
    failure_details: str,
) -> None:
    """Append one validation result."""

    results.append(
        ValidationResult(
            check_name=check_name,
            passed=bool(condition),
            details=success_details if condition else failure_details,
        )
    )


def feature_path(
    feature_table: str,
    feature_version: str,
    processing_date: str,
) -> Path:
    """Build the local feature-table path."""

    return (
        PROJECT_ROOT
        / "data"
        / "features"
        / feature_table
        / f"feature_version={feature_version}"
        / f"processing_date={processing_date}"
        / f"{feature_table}.parquet"
    )


def silver_path(
    dataset_name: str,
    source_version: str,
    processing_date: str,
) -> Path:
    """Build the local Silver dataset path."""

    return (
        PROJECT_ROOT
        / "data"
        / "silver"
        / dataset_name
        / f"processing_date={processing_date}"
        / f"source_version={source_version}"
        / f"{dataset_name}.parquet"
    )


def load_feature_tables(
    schemas: dict[str, FeatureSchema],
    feature_version: str,
    processing_date: str,
) -> dict[str, pd.DataFrame]:
    """Load all reusable feature tables."""

    tables: dict[str, pd.DataFrame] = {}

    for feature_table in schemas:
        path = feature_path(
            feature_table=feature_table,
            feature_version=feature_version,
            processing_date=processing_date,
        )

        if not path.exists():
            raise FileNotFoundError(f"Missing feature table: {path}")

        tables[feature_table] = pd.read_parquet(path)

    return tables


def validate_embedding(
    results: list[ValidationResult],
    table_name: str,
    dataframe: pd.DataFrame,
    schema: FeatureSchema,
) -> None:
    """Validate embedding columns, values, and vector norms."""

    expected_embedding_columns = [
        f"{schema.embedding_prefix}_{index:03d}"
        for index in range(schema.embedding_dimension)
    ]

    missing_columns = set(expected_embedding_columns) - set(dataframe.columns)

    record(
        results,
        f"{table_name}: embedding columns",
        not missing_columns,
        (
            f"Found all {schema.embedding_dimension} expected "
            "embedding columns."
        ),
        f"Missing embedding columns: {sorted(missing_columns)}",
    )

    if missing_columns:
        return

    embedding_matrix = dataframe[expected_embedding_columns].to_numpy(
        dtype=float
    )

    finite_values = bool(np.isfinite(embedding_matrix).all())

    record(
        results,
        f"{table_name}: finite embedding values",
        finite_values,
        "All embedding values are finite.",
        "Embedding values contain NaN or infinity.",
    )

    norms = np.linalg.norm(embedding_matrix, axis=1)
    normalized = bool(
        np.allclose(
            norms,
            np.ones(len(norms)),
            atol=1e-6,
        )
    )

    record(
        results,
        f"{table_name}: normalized embeddings",
        normalized,
        "All embedding vectors have approximately unit norm.",
        (
            f"Embedding norms range from "
            f"{float(norms.min()):.8f} to {float(norms.max()):.8f}."
        ),
    )


def validate_entity_coverage(
    results: list[ValidationResult],
    feature_table_name: str,
    feature_dataframe: pd.DataFrame,
    feature_key: str,
    silver_dataset_name: str,
    silver_dataframe: pd.DataFrame,
) -> None:
    """Validate one-to-one feature coverage for Silver entities."""

    feature_ids = set(feature_dataframe[feature_key])
    silver_ids = set(silver_dataframe[feature_key])

    missing_feature_ids = silver_ids - feature_ids
    unexpected_feature_ids = feature_ids - silver_ids

    record(
        results,
        f"{feature_table_name}: Silver entity coverage",
        not missing_feature_ids and not unexpected_feature_ids,
        "Feature table exactly covers the corresponding Silver entities.",
        (
            f"Missing feature IDs: {sorted(missing_feature_ids)[:10]}; "
            f"unexpected feature IDs: {sorted(unexpected_feature_ids)[:10]}"
        ),
    )


def run_validation() -> list[ValidationResult]:
    """Run all reusable-feature validation checks."""

    config = load_config()

    source_version = config["data"]["source_version"]
    silver_processing_date = config["silver_processing"]["processing_date"]

    feature_config = config["feature_generation"]
    feature_version = feature_config["feature_version"]
    feature_processing_date = feature_config["processing_date"]
    dimensions = feature_config["embedding_dimensions"]

    schemas = build_feature_schemas(
        gene_embedding_dimension=dimensions["gene"],
        crop_line_embedding_dimension=dimensions["crop_line"],
        environment_embedding_dimension=dimensions["environment"],
    )

    feature_tables = load_feature_tables(
        schemas=schemas,
        feature_version=feature_version,
        processing_date=feature_processing_date,
    )

    silver_mapping = {
        "gene_features": "genes",
        "crop_line_features": "crop_lines",
        "environment_features": "environments",
    }

    results: list[ValidationResult] = []

    for table_name, dataframe in feature_tables.items():
        schema = schemas[table_name]
        entity_key = schema.entity_key

        missing_required_columns = (
            set(schema.required_columns) - set(dataframe.columns)
        )

        record(
            results,
            f"{table_name}: required columns",
            not missing_required_columns,
            "All required feature columns are present.",
            f"Missing columns: {sorted(missing_required_columns)}",
        )

        record(
            results,
            f"{table_name}: unique entity key",
            dataframe[entity_key].is_unique,
            f"{entity_key} is unique.",
            f"{entity_key} contains duplicate values.",
        )

        record(
            results,
            f"{table_name}: populated entity key",
            not dataframe[entity_key].isna().any(),
            f"{entity_key} contains no null values.",
            f"{entity_key} contains null values.",
        )

        required_metadata = {
            "_feature_version",
            "_processing_date",
            "_feature_generated_at_utc",
        }
        missing_metadata = required_metadata - set(dataframe.columns)

        record(
            results,
            f"{table_name}: metadata columns",
            not missing_metadata,
            "All feature metadata columns are present.",
            f"Missing metadata columns: {sorted(missing_metadata)}",
        )

        version_matches = bool(
            (dataframe["_feature_version"] == feature_version).all()
        )

        record(
            results,
            f"{table_name}: feature version",
            version_matches,
            f"Every row uses feature version {feature_version}.",
            "One or more rows have an unexpected feature version.",
        )

        numeric_columns = [
            column
            for column in dataframe.select_dtypes(include=["number"]).columns
        ]

        numeric_values = dataframe[numeric_columns].to_numpy(dtype=float)
        finite_numeric_values = bool(np.isfinite(numeric_values).all())

        record(
            results,
            f"{table_name}: finite numeric features",
            finite_numeric_values,
            "All numeric feature values are finite.",
            "Numeric features contain NaN or infinity.",
        )

        validate_embedding(
            results=results,
            table_name=table_name,
            dataframe=dataframe,
            schema=schema,
        )

        silver_dataset_name = silver_mapping[table_name]
        corresponding_silver = pd.read_parquet(
            silver_path(
                dataset_name=silver_dataset_name,
                source_version=source_version,
                processing_date=silver_processing_date,
            )
        )

        record(
            results,
            f"{table_name}: expected row count",
            len(dataframe) == len(corresponding_silver),
            (
                f"Feature rows match Silver entity count: "
                f"{len(dataframe)}."
            ),
            (
                f"Feature rows={len(dataframe)}, "
                f"Silver rows={len(corresponding_silver)}."
            ),
        )

        validate_entity_coverage(
            results=results,
            feature_table_name=table_name,
            feature_dataframe=dataframe,
            feature_key=entity_key,
            silver_dataset_name=silver_dataset_name,
            silver_dataframe=corresponding_silver,
        )

    return results


def main() -> None:
    """Print validation results and exit nonzero on failure."""

    results = run_validation()
    failures = [result for result in results if not result.passed]

    print("Reusable feature validation")
    print("=" * 72)

    for result in results:
        status = "PASS" if result.passed else "FAIL"
        print(f"[{status}] {result.check_name}")

        if not result.passed:
            print(f"       {result.details}")

    print("=" * 72)
    print(f"Checks run: {len(results)}")
    print(f"Passed:     {len(results) - len(failures)}")
    print(f"Failed:     {len(failures)}")

    if failures:
        raise SystemExit(1)

    print("Validation result: SUCCESS")


if __name__ == "__main__":
    main()
