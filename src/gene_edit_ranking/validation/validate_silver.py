"""Validation checks for local Silver Parquet datasets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from gene_edit_ranking.config import PROJECT_ROOT, load_config
from gene_edit_ranking.validation.silver_schemas import SILVER_SCHEMAS


@dataclass(frozen=True)
class ValidationResult:
    """Result of one Silver validation check."""

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
    """Add a validation result."""

    results.append(
        ValidationResult(
            check_name=check_name,
            passed=bool(condition),
            details=success_details if condition else failure_details,
        )
    )


def build_silver_path(
    dataset_name: str,
    processing_date: str,
    source_version: str,
) -> Path:
    """Build a local Silver Parquet path."""

    return (
        PROJECT_ROOT
        / "data"
        / "silver"
        / dataset_name
        / f"processing_date={processing_date}"
        / f"source_version={source_version}"
        / f"{dataset_name}.parquet"
    )


def load_silver_datasets() -> dict[str, pd.DataFrame]:
    """Load all configured Silver datasets."""

    config = load_config()
    processing_date = config["silver_processing"]["processing_date"]
    source_version = config["data"]["source_version"]

    datasets: dict[str, pd.DataFrame] = {}

    for dataset_name in SILVER_SCHEMAS:
        path = build_silver_path(
            dataset_name=dataset_name,
            processing_date=processing_date,
            source_version=source_version,
        )

        if not path.exists():
            raise FileNotFoundError(f"Missing Silver file: {path}")

        datasets[dataset_name] = pd.read_parquet(path)

    return datasets


def validate_foreign_key(
    results: list[ValidationResult],
    child_name: str,
    child: pd.DataFrame,
    child_column: str,
    parent_name: str,
    parent: pd.DataFrame,
    parent_column: str,
) -> None:
    """Validate a Silver foreign-key relationship."""

    unmatched = set(child[child_column]) - set(parent[parent_column])

    record(
        results,
        f"{child_name}.{child_column} -> {parent_name}.{parent_column}",
        not unmatched,
        "All foreign keys reference existing parent rows.",
        f"Unmatched values: {sorted(unmatched)[:10]}",
    )


def run_validation() -> list[ValidationResult]:
    """Run all Silver validation checks."""

    config = load_config()
    datasets = load_silver_datasets()
    expected_counts = config["synthetic_data"]["row_counts"]

    results: list[ValidationResult] = []

    lineage_columns = {
        "_source_file",
        "_source_version",
        "_processing_date",
        "_processed_at_utc",
    }

    for dataset_name, dataframe in datasets.items():
        schema = SILVER_SCHEMAS[dataset_name]
        primary_key = schema.primary_key

        record(
            results,
            f"{dataset_name}: expected row count",
            len(dataframe) == expected_counts[dataset_name],
            f"Found expected {len(dataframe)} rows.",
            (
                f"Expected {expected_counts[dataset_name]} rows, "
                f"found {len(dataframe)}."
            ),
        )

        record(
            results,
            f"{dataset_name}: unique primary key",
            dataframe[primary_key].is_unique,
            f"{primary_key} is unique.",
            f"{primary_key} contains duplicate values.",
        )

        record(
            results,
            f"{dataset_name}: populated primary key",
            not dataframe[primary_key].isna().any(),
            f"{primary_key} contains no null values.",
            f"{primary_key} contains null values.",
        )

        missing_lineage = lineage_columns - set(dataframe.columns)

        record(
            results,
            f"{dataset_name}: lineage columns",
            not missing_lineage,
            "All lineage columns are present.",
            f"Missing lineage columns: {sorted(missing_lineage)}",
        )

        for column in schema.string_columns:
            record(
                results,
                f"{dataset_name}: {column} string type",
                isinstance(dataframe[column].dtype, pd.StringDtype),
                f"{column} uses pandas string dtype.",
                f"{column} has dtype {dataframe[column].dtype}.",
            )

        for column in schema.integer_columns:
            record(
                results,
                f"{dataset_name}: {column} integer type",
                pd.api.types.is_integer_dtype(dataframe[column]),
                f"{column} is an integer.",
                f"{column} has dtype {dataframe[column].dtype}.",
            )

        for column in schema.float_columns:
            record(
                results,
                f"{dataset_name}: {column} float type",
                pd.api.types.is_float_dtype(dataframe[column]),
                f"{column} is a floating-point value.",
                f"{column} has dtype {dataframe[column].dtype}.",
            )

        for column in schema.boolean_columns:
            record(
                results,
                f"{dataset_name}: {column} boolean type",
                pd.api.types.is_bool_dtype(dataframe[column]),
                f"{column} is boolean.",
                f"{column} has dtype {dataframe[column].dtype}.",
            )

        for column in schema.date_columns:
            record(
                results,
                f"{dataset_name}: {column} timestamp type",
                pd.api.types.is_datetime64_any_dtype(dataframe[column]),
                f"{column} is a timestamp.",
                f"{column} has dtype {dataframe[column].dtype}.",
            )

    genes = datasets["genes"]
    crop_lines = datasets["crop_lines"]
    environments = datasets["environments"]
    candidates = datasets["candidates"]
    experiments = datasets["experiments"]

    validate_foreign_key(
        results,
        "candidates",
        candidates,
        "gene_id",
        "genes",
        genes,
        "gene_id",
    )
    validate_foreign_key(
        results,
        "candidates",
        candidates,
        "crop_line_id",
        "crop_lines",
        crop_lines,
        "crop_line_id",
    )
    validate_foreign_key(
        results,
        "experiments",
        experiments,
        "candidate_id",
        "candidates",
        candidates,
        "candidate_id",
    )
    validate_foreign_key(
        results,
        "experiments",
        experiments,
        "environment_id",
        "environments",
        environments,
        "environment_id",
    )

    return results


def main() -> None:
    """Print validation results and exit nonzero on failure."""

    results = run_validation()
    failures = [result for result in results if not result.passed]

    print("Silver data validation")
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
