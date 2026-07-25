"""Validation checks for the Gold training dataset."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from gene_edit_ranking.config import PROJECT_ROOT, load_config
from gene_edit_ranking.validation.gold_schema import GOLD_TRAINING_SCHEMA


@dataclass(frozen=True)
class ValidationResult:
    """Result of one Gold dataset validation check."""

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


def gold_path(
    dataset_version: str,
    processing_date: str,
) -> Path:
    """Build the local Gold dataset path."""

    return (
        PROJECT_ROOT
        / "data"
        / "gold"
        / "training"
        / f"dataset_version={dataset_version}"
        / f"processing_date={processing_date}"
        / "training_dataset.parquet"
    )


def load_gold_dataset() -> pd.DataFrame:
    """Load the configured local Gold training dataset."""

    config = load_config()
    gold_config = config["gold_dataset"]

    path = gold_path(
        dataset_version=gold_config["dataset_version"],
        processing_date=gold_config["processing_date"],
    )

    if not path.exists():
        raise FileNotFoundError(f"Missing Gold dataset: {path}")

    return pd.read_parquet(path)


def run_validation() -> list[ValidationResult]:
    """Run all Gold training dataset validation checks."""

    config = load_config()
    dataframe = load_gold_dataset()

    gold_config = config["gold_dataset"]
    expected_rows = config["synthetic_data"]["row_counts"]["experiments"]

    results: list[ValidationResult] = []

    required_columns = set(
        GOLD_TRAINING_SCHEMA.entity_columns
        + GOLD_TRAINING_SCHEMA.target_columns
        + GOLD_TRAINING_SCHEMA.required_numeric_columns
        + GOLD_TRAINING_SCHEMA.required_categorical_columns
    )

    missing_required = required_columns - set(dataframe.columns)

    record(
        results,
        "gold: required columns",
        not missing_required,
        "All required Gold columns are present.",
        f"Missing columns: {sorted(missing_required)}",
    )

    record(
        results,
        "gold: expected row count",
        len(dataframe) == expected_rows,
        f"Found expected {len(dataframe)} rows.",
        f"Expected {expected_rows} rows, found {len(dataframe)}.",
    )

    primary_key = GOLD_TRAINING_SCHEMA.primary_key

    record(
        results,
        "gold: unique primary key",
        dataframe[primary_key].is_unique,
        f"{primary_key} is unique.",
        f"{primary_key} contains duplicate values.",
    )

    record(
        results,
        "gold: populated primary key",
        not dataframe[primary_key].isna().any(),
        f"{primary_key} contains no null values.",
        f"{primary_key} contains null values.",
    )

    entity_nulls = dataframe[
        list(GOLD_TRAINING_SCHEMA.entity_columns)
    ].isna().sum()

    failing_entities = {
        column: int(count)
        for column, count in entity_nulls.items()
        if count > 0
    }

    record(
        results,
        "gold: complete entity joins",
        not failing_entities,
        "All entity joins are complete.",
        f"Null entity references: {failing_entities}",
    )

    numeric_columns = list(
        GOLD_TRAINING_SCHEMA.required_numeric_columns
        + GOLD_TRAINING_SCHEMA.target_columns
    )

    numeric_matrix = dataframe[numeric_columns].to_numpy(dtype=float)

    record(
        results,
        "gold: finite numeric values",
        bool(np.isfinite(numeric_matrix).all()),
        "All required numeric and target values are finite.",
        "Numeric columns contain NaN or infinity.",
    )

    categorical_nulls = dataframe[
        list(GOLD_TRAINING_SCHEMA.required_categorical_columns)
    ].isna().sum()

    failing_categories = {
        column: int(count)
        for column, count in categorical_nulls.items()
        if count > 0
    }

    record(
        results,
        "gold: populated categorical features",
        not failing_categories,
        "All required categorical features are populated.",
        f"Null categorical values: {failing_categories}",
    )

    observed_values = set(
        dataframe["observed_edit_success"].dropna().unique().tolist()
    )

    record(
        results,
        "gold: binary classification target",
        observed_values.issubset({0, 1}),
        "observed_edit_success contains only 0 and 1.",
        f"Unexpected values: {sorted(observed_values)}",
    )

    record(
        results,
        "gold: measurement quality range",
        bool(
            dataframe["measurement_quality_score"]
            .between(0, 1, inclusive="both")
            .all()
        ),
        "measurement_quality_score is within [0, 1].",
        "measurement_quality_score contains values outside [0, 1].",
    )

    embedding_expectations = {
        "gene_embedding_": config["feature_generation"][
            "embedding_dimensions"
        ]["gene"],
        "crop_line_embedding_": config["feature_generation"][
            "embedding_dimensions"
        ]["crop_line"],
        "environment_embedding_": config["feature_generation"][
            "embedding_dimensions"
        ]["environment"],
    }

    for prefix, expected_dimension in embedding_expectations.items():
        embedding_columns = [
            column
            for column in dataframe.columns
            if column.startswith(prefix)
        ]

        record(
            results,
            f"gold: {prefix} dimension",
            len(embedding_columns) == expected_dimension,
            f"Found expected {expected_dimension} columns.",
            (
                f"Expected {expected_dimension} columns, "
                f"found {len(embedding_columns)}."
            ),
        )

    required_metadata = {
        "_gold_dataset_version",
        "_source_version",
        "_feature_version",
        "_processing_date",
        "_gold_generated_at_utc",
    }

    missing_metadata = required_metadata - set(dataframe.columns)

    record(
        results,
        "gold: lineage metadata",
        not missing_metadata,
        "All Gold lineage metadata columns are present.",
        f"Missing metadata columns: {sorted(missing_metadata)}",
    )

    record(
        results,
        "gold: dataset version",
        bool(
            (
                dataframe["_gold_dataset_version"]
                == gold_config["dataset_version"]
            ).all()
        ),
        (
            "Every row uses Gold dataset version "
            f"{gold_config['dataset_version']}."
        ),
        "One or more rows have an unexpected Gold dataset version.",
    )

    prohibited_source_columns = {
        "_source_file",
        "_processed_at_utc",
        "_feature_generated_at_utc",
        "is_synthetic",
    }

    leaked_columns = prohibited_source_columns & set(dataframe.columns)

    record(
        results,
        "gold: no source operational leakage",
        not leaked_columns,
        "Source operational columns were removed before the Gold join.",
        f"Unexpected columns present: {sorted(leaked_columns)}",
    )

    return results


def main() -> None:
    """Print Gold validation results and fail on any invalid condition."""

    results = run_validation()
    failures = [result for result in results if not result.passed]

    print("Gold training dataset validation")
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
