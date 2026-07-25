"""Validation checks for locally generated synthetic Raw datasets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from gene_edit_ranking.config import PROJECT_ROOT, load_config


@dataclass
class ValidationResult:
    """Represents the result of a single validation check."""

    check_name: str
    passed: bool
    details: str


def _record(
    results: list[ValidationResult],
    check_name: str,
    condition: bool,
    success_details: str,
    failure_details: str,
) -> None:
    """Append a validation result."""

    results.append(
        ValidationResult(
            check_name=check_name,
            passed=bool(condition),
            details=success_details if condition else failure_details,
        )
    )


def validate_required_columns(
    results: list[ValidationResult],
    dataset_name: str,
    dataframe: pd.DataFrame,
    required_columns: set[str],
) -> None:
    """Validate that all expected columns exist."""

    missing_columns = required_columns - set(dataframe.columns)

    _record(
        results,
        f"{dataset_name}: required columns",
        not missing_columns,
        "All required columns are present.",
        f"Missing columns: {sorted(missing_columns)}",
    )


def validate_primary_key(
    results: list[ValidationResult],
    dataset_name: str,
    dataframe: pd.DataFrame,
    key_column: str,
) -> None:
    """Validate that a primary key is populated and unique."""

    null_count = int(dataframe[key_column].isna().sum())
    duplicate_count = int(dataframe[key_column].duplicated().sum())

    _record(
        results,
        f"{dataset_name}: primary key {key_column}",
        null_count == 0 and duplicate_count == 0,
        f"{key_column} is populated and unique.",
        (
            f"{key_column} has {null_count} null values and "
            f"{duplicate_count} duplicate values."
        ),
    )


def validate_no_nulls(
    results: list[ValidationResult],
    dataset_name: str,
    dataframe: pd.DataFrame,
    required_columns: list[str],
) -> None:
    """Validate that required fields do not contain null values."""

    null_counts = dataframe[required_columns].isna().sum()
    failing_columns = {
        column: int(count)
        for column, count in null_counts.items()
        if count > 0
    }

    _record(
        results,
        f"{dataset_name}: required fields populated",
        not failing_columns,
        "Required fields contain no null values.",
        f"Null values found: {failing_columns}",
    )


def validate_range(
    results: list[ValidationResult],
    dataset_name: str,
    dataframe: pd.DataFrame,
    column: str,
    minimum: float,
    maximum: float,
) -> None:
    """Validate that numeric values fall within an inclusive range."""

    invalid_mask = ~dataframe[column].between(minimum, maximum, inclusive="both")
    invalid_count = int(invalid_mask.sum())

    _record(
        results,
        f"{dataset_name}: {column} range",
        invalid_count == 0,
        f"All values are between {minimum} and {maximum}.",
        f"{invalid_count} values fall outside [{minimum}, {maximum}].",
    )


def validate_foreign_key(
    results: list[ValidationResult],
    child_name: str,
    child_dataframe: pd.DataFrame,
    child_column: str,
    parent_name: str,
    parent_dataframe: pd.DataFrame,
    parent_column: str,
) -> None:
    """Validate a many-to-one foreign-key relationship."""

    unmatched = set(child_dataframe[child_column]) - set(
        parent_dataframe[parent_column]
    )

    _record(
        results,
        f"{child_name}.{child_column} -> {parent_name}.{parent_column}",
        not unmatched,
        "All foreign-key values reference existing parent rows.",
        f"Unmatched values: {sorted(unmatched)[:10]}",
    )


def validate_date_column(
    results: list[ValidationResult],
    dataset_name: str,
    dataframe: pd.DataFrame,
    column: str,
) -> None:
    """Validate that a column contains parseable dates."""

    parsed_dates = pd.to_datetime(dataframe[column], errors="coerce")
    invalid_count = int(parsed_dates.isna().sum())

    _record(
        results,
        f"{dataset_name}: {column} dates",
        invalid_count == 0,
        "All dates are parseable.",
        f"{invalid_count} values could not be parsed as dates.",
    )


def validate_synthetic_flag(
    results: list[ValidationResult],
    dataset_name: str,
    dataframe: pd.DataFrame,
) -> None:
    """Validate that every row is explicitly marked synthetic."""

    normalized = dataframe["is_synthetic"].astype(str).str.lower()
    invalid_count = int((normalized != "true").sum())

    _record(
        results,
        f"{dataset_name}: synthetic-data flag",
        invalid_count == 0,
        "Every row is marked as synthetic.",
        f"{invalid_count} rows are not marked as synthetic.",
    )


def load_raw_datasets(raw_dir: Path) -> dict[str, pd.DataFrame]:
    """Load the five local Raw CSV datasets."""

    paths = {
        "genes": raw_dir / "genes.csv",
        "crop_lines": raw_dir / "crop_lines.csv",
        "environments": raw_dir / "environments.csv",
        "candidates": raw_dir / "candidate_edits.csv",
        "experiments": raw_dir / "experiments.csv",
    }

    missing_files = [
        str(path)
        for path in paths.values()
        if not path.exists()
    ]

    if missing_files:
        raise FileNotFoundError(
            "Missing Raw dataset files:\n" + "\n".join(missing_files)
        )

    return {
        name: pd.read_csv(path)
        for name, path in paths.items()
    }


def run_validation() -> list[ValidationResult]:
    """Run all validation checks and return their results."""

    config = load_config()
    raw_dir = PROJECT_ROOT / config["data"]["local_raw_dir"]
    expected_counts = config["synthetic_data"]["row_counts"]

    datasets = load_raw_datasets(raw_dir)

    genes = datasets["genes"]
    crop_lines = datasets["crop_lines"]
    environments = datasets["environments"]
    candidates = datasets["candidates"]
    experiments = datasets["experiments"]

    results: list[ValidationResult] = []

    expected_columns = {
        "genes": {
            "gene_id",
            "gene_symbol",
            "chromosome",
            "start_position",
            "end_position",
            "gene_family",
            "synthetic_conservation_score",
            "is_synthetic",
        },
        "crop_lines": {
            "crop_line_id",
            "crop_type",
            "breeding_program",
            "maturity_group",
            "baseline_yield_index",
            "baseline_drought_tolerance",
            "is_synthetic",
        },
        "environments": {
            "environment_id",
            "location_code",
            "season",
            "soil_type",
            "rainfall_mm",
            "average_temperature_c",
            "synthetic_drought_index",
            "is_synthetic",
        },
        "candidates": {
            "candidate_id",
            "gene_id",
            "crop_line_id",
            "editing_method",
            "target_position",
            "predicted_edit_efficiency",
            "predicted_off_target_risk",
            "design_batch",
            "created_date",
            "is_synthetic",
        },
        "experiments": {
            "experiment_id",
            "candidate_id",
            "environment_id",
            "replicate_number",
            "observed_edit_success",
            "yield_change_percent",
            "drought_response_score",
            "measurement_quality_score",
            "experiment_date",
            "is_synthetic",
        },
    }

    for dataset_name, dataframe in datasets.items():
        validate_required_columns(
            results,
            dataset_name,
            dataframe,
            expected_columns[dataset_name],
        )

        _record(
            results,
            f"{dataset_name}: expected row count",
            len(dataframe) == expected_counts[dataset_name],
            f"Found expected {len(dataframe)} rows.",
            (
                f"Expected {expected_counts[dataset_name]} rows, "
                f"found {len(dataframe)}."
            ),
        )

        validate_synthetic_flag(results, dataset_name, dataframe)

    validate_primary_key(results, "genes", genes, "gene_id")
    validate_primary_key(results, "crop_lines", crop_lines, "crop_line_id")
    validate_primary_key(
        results,
        "environments",
        environments,
        "environment_id",
    )
    validate_primary_key(results, "candidates", candidates, "candidate_id")
    validate_primary_key(
        results,
        "experiments",
        experiments,
        "experiment_id",
    )

    validate_no_nulls(
        results,
        "genes",
        genes,
        [
            "gene_id",
            "gene_symbol",
            "chromosome",
            "start_position",
            "end_position",
        ],
    )
    validate_no_nulls(
        results,
        "crop_lines",
        crop_lines,
        [
            "crop_line_id",
            "crop_type",
            "baseline_yield_index",
        ],
    )
    validate_no_nulls(
        results,
        "environments",
        environments,
        [
            "environment_id",
            "rainfall_mm",
            "average_temperature_c",
        ],
    )
    validate_no_nulls(
        results,
        "candidates",
        candidates,
        [
            "candidate_id",
            "gene_id",
            "crop_line_id",
            "editing_method",
        ],
    )
    validate_no_nulls(
        results,
        "experiments",
        experiments,
        [
            "experiment_id",
            "candidate_id",
            "environment_id",
            "observed_edit_success",
        ],
    )

    validate_range(
        results,
        "genes",
        genes,
        "synthetic_conservation_score",
        0,
        1,
    )
    validate_range(
        results,
        "crop_lines",
        crop_lines,
        "baseline_drought_tolerance",
        0,
        1,
    )
    validate_range(
        results,
        "environments",
        environments,
        "synthetic_drought_index",
        0,
        1,
    )
    validate_range(
        results,
        "candidates",
        candidates,
        "predicted_edit_efficiency",
        0,
        1,
    )
    validate_range(
        results,
        "candidates",
        candidates,
        "predicted_off_target_risk",
        0,
        1,
    )
    validate_range(
        results,
        "experiments",
        experiments,
        "observed_edit_success",
        0,
        1,
    )
    validate_range(
        results,
        "experiments",
        experiments,
        "measurement_quality_score",
        0,
        1,
    )

    _record(
        results,
        "genes: genomic positions",
        bool((genes["end_position"] > genes["start_position"]).all()),
        "Every end position is greater than its start position.",
        "One or more gene end positions are not greater than start positions.",
    )

    validate_date_column(results, "candidates", candidates, "created_date")
    validate_date_column(
        results,
        "experiments",
        experiments,
        "experiment_date",
    )

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
    """Run validation, print results, and fail on any invalid dataset."""

    results = run_validation()
    failures = [result for result in results if not result.passed]

    print("Synthetic Raw data validation")
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
